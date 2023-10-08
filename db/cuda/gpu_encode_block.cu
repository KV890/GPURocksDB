//
// Created by jxx on 5/6/23.
//

#include <fcntl.h>

#include <csignal>

#include "gpu_encode_block.cuh"

namespace ROCKSDB_NAMESPACE {

__device__ void ProcessKeyValue(unsigned int kv_idx, unsigned int block_idx,
                                char* block_buffer, GPUKeyValue* keyValues,
                                InputFile* inputFiles,
                                size_t num_kv_data_block) {
  size_t shared = 0;
  size_t non_shared = keySize_ + 8;
  size_t encoded_size;

  // add encoded shared non_shared value_size to buffer
  GPUPutVarint32Varint32Varint32(
      block_buffer + kv_idx * key_value_size, static_cast<uint32_t>(shared),
      static_cast<uint32_t>(non_shared), static_cast<uint32_t>(valueSize_),
      encoded_size);

  // key.data
  memcpy(block_buffer + kv_idx * key_value_size + encoded_size,
         keyValues[block_idx * num_kv_data_block + kv_idx].key, keySize_ + 8);

  // 获得原始value
  __shared__ char value_buffer[1024];
  ExtractOriginalValue(keyValues[block_idx * num_kv_data_block + kv_idx],
                       inputFiles, num_files_d, value_buffer);

  // value.data
  memcpy(block_buffer + kv_idx * key_value_size + encoded_size + keySize_ + 8,
         value_buffer, valueSize_);
}

__device__ void ProcessRestartsChecksum(char* block_buffer,
                                        size_t num_kv_data_block,
                                        uint32_t num_restarts,
                                        uint32_t* restarts, size_t size_block) {
  for (int i = 0; i < num_restarts; ++i) {
    GPUPutFixed32(block_buffer + key_value_size * num_kv_data_block +
                      i * sizeof(uint32_t),
                  restarts[i]);
  }
  GPUPutFixed32(block_buffer + key_value_size * num_kv_data_block +
                    num_restarts * sizeof(uint32_t),
                num_restarts);

  // 计算trailer
  char trailer[5];
  char type = 0x0;
  trailer[0] = type;  // 表示kNoCompression
  uint32_t checksum =
      GPUComputeBuiltinChecksumWithLastByte(block_buffer, size_block - 5, type);
  GPUEncodeFixed32(trailer + 1, checksum);

  // add checksum to buffer
  memcpy(block_buffer + size_block - 5, trailer, 5);
}

__global__ void BuildKeyValueKernel(char* buffer, GPUKeyValue* keyValues,
                                    InputFile* inputFiles) {
  unsigned int kv_idx = threadIdx.x;
  unsigned int block_idx = blockIdx.x;

  char* block_buffer = buffer + block_idx * size_complete_data_block_d;

  ProcessKeyValue(kv_idx, block_idx, block_buffer, keyValues, inputFiles,
                  num_kv_data_block_d);
}

__global__ void BuildLastKeyValueKernel(char* buffer, GPUKeyValue* keyValues,
                                        InputFile* inputFiles) {
  unsigned int kv_idx = threadIdx.x;
  unsigned int block_idx = num_data_block_d - 1;

  char* key_value_buffer = buffer + block_idx * size_complete_data_block_d;

  ProcessKeyValue(kv_idx, block_idx, key_value_buffer, keyValues, inputFiles,
                  num_kv_last_data_block_d);
}

__global__ void BuildRestartsChecksumKernel(char* buffer, char* index_key,
                                            GPUKeyValue* keyValues_d) {
  unsigned int block_idx = blockIdx.x;
  unsigned int last_key_idx = (block_idx + 1) * num_kv_data_block_d - 1;

  memcpy(index_key + block_idx * keySize_, keyValues_d[last_key_idx].key,
         keySize_);

  char* block_buffer = buffer + block_idx * size_complete_data_block_d;

  ProcessRestartsChecksum(block_buffer, num_kv_data_block_d, num_restarts_d,
                          const_restarts_d, size_complete_data_block_d);
}

__global__ void BuildLastRestartsChecksumKernel(
    char* buffer, size_t num_kv_data_block, uint32_t num_restarts,
    uint32_t* restarts, char* index_key, GPUKeyValue* keyValues_d,
    size_t size_incomplete_data_block) {
  unsigned int block_idx = num_data_block_d - 1;
  unsigned int last_key_idx = (num_data_block_d - 1) * num_kv_data_block_d +
                              num_kv_last_data_block_d - 1;

  memcpy(index_key + (num_data_block_d - 1) * keySize_,
         keyValues_d[last_key_idx].key, keySize_);

  char* block_buffer = buffer + block_idx * size_complete_data_block_d;

  ProcessRestartsChecksum(block_buffer, num_kv_data_block, num_restarts,
                          restarts, size_incomplete_data_block);
}

void BuildDataBlock(char** buffer_d, GPUKeyValue* keyValues_d,
                    InputFile* inputFiles_d, size_t num_kv_data_block,
                    SSTableInfo info, size_t size_incomplete_data_block,
                    uint32_t last_num_restarts, uint32_t* last_restarts,
                    char* index_key, cudaStream_t* stream, cudaEvent_t start,
                    cudaEvent_t stop) {
  cudaEventRecord(start, nullptr);

  if (info.num_kv_last_data_block == 0) {
    dim3 block(num_kv_data_block);
    dim3 grid(info.num_data_block);

    BuildKeyValueKernel<<<grid, block, 0, stream[0]>>>(*buffer_d, keyValues_d,
                                                       inputFiles_d);

    BuildRestartsChecksumKernel<<<grid, 1, 0, stream[1]>>>(*buffer_d, index_key,
                                                           keyValues_d);
  } else {
    dim3 block(num_kv_data_block);
    dim3 grid(info.num_data_block - 1);

    BuildKeyValueKernel<<<grid, block, 0, stream[0]>>>(*buffer_d, keyValues_d,
                                                       inputFiles_d);

    BuildRestartsChecksumKernel<<<grid, 1, 0, stream[1]>>>(*buffer_d, index_key,
                                                           keyValues_d);

    BuildLastKeyValueKernel<<<1, info.num_kv_last_data_block, 0, stream[2]>>>(
        *buffer_d, keyValues_d, inputFiles_d);

    BuildLastRestartsChecksumKernel<<<1, 1, 0, stream[3]>>>(
        *buffer_d, info.num_kv_last_data_block, last_num_restarts,
        last_restarts, index_key, keyValues_d, size_incomplete_data_block);
  }

  cudaEventRecord(stop, nullptr);
  CHECK(cudaEventSynchronize(stop));

  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, stop);
  printf("GPU encode data block [kernel] time is: %f ms\n", elapsed_time);
}

__global__ void BuildIndexBlockKernel(char* buffer, char* index_keys,
                                      GPUBlockHandle* block_handle,
                                      uint32_t* restarts) {
  unsigned int block_idx = blockIdx.x;

  size_t shared = 0;
  size_t non_shared = keySize_;
  size_t encoded_size;

  GPUPutVarint32Varint32(buffer + size_index_entry * block_idx,
                         static_cast<uint32_t>(shared),
                         static_cast<uint32_t>(non_shared), encoded_size);

  memcpy(buffer + size_index_entry * block_idx + encoded_size,
         index_keys + block_idx * keySize_, keySize_);

  char handle_buffer[20];
  GPUPutFixed64Varint64(handle_buffer, block_handle[block_idx].offset(),
                        block_handle[block_idx].size());

  // value.data
  memcpy(buffer + size_index_entry * block_idx + encoded_size + keySize_,
         handle_buffer, encoded_index_entry);

  char* restarts_buffer = buffer + size_index_entry * num_data_block_d +
                          block_idx * sizeof(uint32_t);

  GPUPutFixed32(restarts_buffer, restarts[block_idx]);
}

void BuildIndexBlock(char** buffer_h, char** buffer_d, char* index_keys,
                     GPUBlockHandle* block_handle, size_t num_data_block,
                     uint32_t* restarts, size_t size_index_block,
                     cudaStream_t* stream, cudaEvent_t start,
                     cudaEvent_t stop) {
  dim3 block(1);
  dim3 grid(num_data_block);

  cudaEventRecord(start, nullptr);

  BuildIndexBlockKernel<<<grid, block, 0, stream[0]>>>(*buffer_d, index_keys,
                                                       block_handle, restarts);

  cudaEventRecord(stop, nullptr);
  CHECK(cudaEventSynchronize(stop));

  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, stop);
  printf("Encode index block [kernel] time is %f ms\n", elapsed_time);

  CHECK(cudaMemcpyAsync(*buffer_h, *buffer_d, size_index_block,
                        cudaMemcpyDeviceToHost, stream[0]));

  char last[9];
  // num_restarts
  GPUPutFixed32(last, num_data_block);

  // checksum
  char trailer[5];
  char type = 0x0;
  trailer[0] = type;  // 表示kNoCompression
  uint32_t checksum = ComputeBuiltinChecksumWithLastByte(
      kCRC32c, *buffer_h, size_index_block - 5, type);
  EncodeFixed32(trailer + 1, checksum);

  // add checksum to buffer
  memcpy(last + 4, trailer, 5);

  char* new_buffer_d =
      *buffer_d + (size_index_entry + sizeof(uint32_t)) * num_data_block;

  CHECK(cudaMemcpyAsync(new_buffer_d, last, 9, cudaMemcpyHostToDevice,
                        stream[0]));
}

__global__ void WriteBlocksKernel(char* buffer_d, char* block,
                                  size_t block_size, uint32_t checksum) {
  memcpy(buffer_d, block, block_size);
  char trailer[5];
  trailer[0] = 0x0;

  GPUEncodeFixed32(trailer + 1, checksum);

  memcpy(buffer_d + block_size, trailer, 5);
}

void WriteBlock(char** buffer, const Slice& block_contents, BlockHandle* handle,
                size_t last_offset, size_t& new_offset) {
  handle->set_offset(last_offset);
  handle->set_size(block_contents.size());

  memcpy(*buffer, block_contents.data(), block_contents.size());

  char trailer[5];
  char type = 0x0;
  trailer[0] = type;
  uint32_t checksum = ComputeBuiltinChecksumWithLastByte(
      kCRC32c, block_contents.data(), block_contents.size(), type);

  EncodeFixed32(trailer + 1, checksum);

  memcpy(*buffer + block_contents.size(), trailer, 5);

  new_offset = last_offset + block_contents.size() + 5;
}

void GPUWriteBlock(char** buffer_d, const Slice& block_contents,
                   BlockHandle* handle, size_t last_offset,
                   size_t& new_offset) {
  handle->set_offset(last_offset);
  handle->set_size(block_contents.size());

  char type = 0x0;
  uint32_t checksum = ComputeBuiltinChecksumWithLastByte(
      kCRC32c, block_contents.data(), block_contents.size(), type);

  char* block_d;
  cudaMalloc(&block_d, block_contents.size());
  cudaMemcpy(block_d, block_contents.data(), block_contents.size(),
             cudaMemcpyHostToDevice);

  WriteBlocksKernel<<<1, 1>>>(*buffer_d, block_d, block_contents.size(),
                              checksum);

  new_offset = last_offset + block_contents.size() + 5;
}

[[maybe_unused]] void BuildPropertiesBlock(
    char** buffer, FileMetaData* meta,
    const std::shared_ptr<TableBuilderOptions>& tboptions, size_t data_size,
    size_t index_size, MetaIndexBuilder* metaIndexBuilder, size_t& new_offset,
    TableProperties* props) {
  BlockHandle properties_block_handle;
  PropertyBlockBuilder propertyBlockBuilder;

  props->orig_file_number = tboptions->cur_file_num;
  props->data_size = data_size;
  props->raw_key_size = meta->raw_key_size;
  props->raw_value_size = meta->raw_value_size;
  props->num_data_blocks = meta->num_data_blocks;
  props->num_entries = meta->num_entries;
  props->filter_policy_name = "";
  props->index_size = index_size;

  props->creation_time = meta->file_creation_time;
  props->oldest_key_time = meta->oldest_ancester_time;
  props->file_creation_time = meta->file_creation_time;

  props->db_id = tboptions->db_id;
  props->db_session_id = tboptions->db_session_id;

  props->filter_policy_name = "";
  props->compression_name = "NoCompression";
  props->compression_options =
      "window_bits=-14; level=32767; strategy=0; max_dict_bytes=0; "
      "zstd_max_train_bytes=0; enabled=0; max_dict_buffer_bytes=0; "
      "use_zstd_dict_trainer=1;";
  props->comparator_name = "leveldb.BytewiseComparator";
  props->merge_operator_name = "nullptr";
  props->prefix_extractor_name = "nullptr";
  props->property_collectors_names = "[]";
  props->index_key_is_user_key = 1;
  props->index_value_is_delta_encoded = 1;

  props->db_host_id = "jxx";
  props->column_family_id = tboptions->column_family_id;
  props->column_family_name = tboptions->column_family_name;

  propertyBlockBuilder.AddTableProperty(*props);

  std::map<std::string, std::string> user_collected_properties;

  std::string val;
  PutFixed32(&val, static_cast<uint32_t>(0x00));
  user_collected_properties.insert(
      {BlockBasedTablePropertyNames::kIndexType, val});
  user_collected_properties.insert(
      {BlockBasedTablePropertyNames::kWholeKeyFiltering, kPropTrue});
  user_collected_properties.insert(
      {BlockBasedTablePropertyNames::kPrefixFiltering, kPropFalse});

  propertyBlockBuilder.Add(user_collected_properties);

  Slice block_data = propertyBlockBuilder.Finish();

  WriteBlock(buffer, block_data, &properties_block_handle,
             data_size + index_size, new_offset);

  const std::string* properties_block_meta = &kPropertiesBlockName;
  metaIndexBuilder->Add(*properties_block_meta, properties_block_handle);
}

void GPUBuildPropertiesBlock(
    char** buffer_d, FileMetaData* meta,
    const std::shared_ptr<TableBuilderOptions>& tboptions, size_t data_size,
    size_t index_size, MetaIndexBuilder* metaIndexBuilder, size_t& new_offset,
    TableProperties* props) {
  BlockHandle properties_block_handle;
  PropertyBlockBuilder propertyBlockBuilder;

  props->orig_file_number = tboptions->cur_file_num;
  props->data_size = data_size;
  props->raw_key_size = meta->raw_key_size;
  props->raw_value_size = meta->raw_value_size;
  props->num_data_blocks = meta->num_data_blocks;
  props->num_entries = meta->num_entries;
  props->filter_policy_name = "";
  props->index_size = index_size;

  props->creation_time = meta->file_creation_time;
  props->oldest_key_time = meta->oldest_ancester_time;
  props->file_creation_time = meta->file_creation_time;

  props->db_id = tboptions->db_id;
  props->db_session_id = tboptions->db_session_id;

  props->filter_policy_name = "";
  props->compression_name = "NoCompression";
  props->compression_options =
      "window_bits=-14; level=32767; strategy=0; max_dict_bytes=0; "
      "zstd_max_train_bytes=0; enabled=0; max_dict_buffer_bytes=0; "
      "use_zstd_dict_trainer=1;";
  props->comparator_name = "leveldb.BytewiseComparator";
  props->merge_operator_name = "nullptr";
  props->prefix_extractor_name = "nullptr";
  props->property_collectors_names = "[]";
  props->index_key_is_user_key = 1;
  props->index_value_is_delta_encoded = 1;

  props->db_host_id = "jxx";
  props->column_family_id = tboptions->column_family_id;
  props->column_family_name = tboptions->column_family_name;

  propertyBlockBuilder.AddTableProperty(*props);

  std::map<std::string, std::string> user_collected_properties;

  std::string val;
  PutFixed32(&val, static_cast<uint32_t>(0x00));
  user_collected_properties.insert(
      {BlockBasedTablePropertyNames::kIndexType, val});
  user_collected_properties.insert(
      {BlockBasedTablePropertyNames::kWholeKeyFiltering, kPropTrue});
  user_collected_properties.insert(
      {BlockBasedTablePropertyNames::kPrefixFiltering, kPropFalse});

  propertyBlockBuilder.Add(user_collected_properties);

  Slice block_data = propertyBlockBuilder.Finish();

  GPUWriteBlock(buffer_d, block_data, &properties_block_handle,
                data_size + index_size, new_offset);

  const std::string* properties_block_meta = &kPropertiesBlockName;
  metaIndexBuilder->Add(*properties_block_meta, properties_block_handle);
}

[[maybe_unused]] void BuildMetaIndexBlock(char** buffer,
                                          MetaIndexBuilder* metaIndexBuilder,
                                          BlockHandle* meta_index_block_handle,
                                          size_t last_offset,
                                          size_t& new_offset) {
  Slice block_data = metaIndexBuilder->Finish();

  WriteBlock(buffer, block_data, meta_index_block_handle, last_offset,
             new_offset);
}

void GPUBuildMetaIndexBlock(char** buffer_d, MetaIndexBuilder* metaIndexBuilder,
                            BlockHandle* meta_index_block_handle,
                            size_t last_offset, size_t& new_offset) {
  Slice block_data = metaIndexBuilder->Finish();

  GPUWriteBlock(buffer_d, block_data, meta_index_block_handle, last_offset,
                new_offset);
}

__global__ void BuilderFooterKernel([[maybe_unused]] char* buffer_d,
                                    char* footer_d, size_t footer_size) {
  memcpy(buffer_d, footer_d, footer_size);
}

[[maybe_unused]] void BuildFooter(char** buffer,
                                  BlockHandle& index_block_handle,
                                  BlockHandle& meta_index_block_handle,
                                  size_t last_offset, size_t& new_offset) {
  FooterBuilder footer;
  footer.Build(kBlockBasedTableMagicNumber, 5, last_offset, kCRC32c,
               meta_index_block_handle, index_block_handle);

  memcpy(*buffer, footer.GetSlice().data(), footer.GetSlice().size());

  new_offset = last_offset + footer.GetSlice().size();
}

void GPUBuildFooter(char** buffer_d, BlockHandle& index_block_handle,
                    BlockHandle& meta_index_block_handle, size_t last_offset,
                    size_t& new_offset) {
  FooterBuilder footer;
  footer.Build(kBlockBasedTableMagicNumber, 5, last_offset, kCRC32c,
               meta_index_block_handle, index_block_handle);

  char* footer_d;
  cudaMalloc(&footer_d, footer.GetSlice().size());
  cudaMemcpy(footer_d, footer.GetSlice().data(), footer.GetSlice().size(),
             cudaMemcpyHostToDevice);

  BuilderFooterKernel<<<1, 1>>>(*buffer_d, footer_d, footer.GetSlice().size());

  new_offset = last_offset + footer.GetSlice().size();

  cudaFree(footer_d);
}

char* BuildSSTable(const GPUKeyValue* keyValues, InputFile* inputFiles_d,
                   SSTableInfo& info, size_t num_kv_data_block,
                   FileMetaData& meta,
                   std::shared_ptr<TableBuilderOptions>& tboptions,
                   TableProperties& tp) {
  cudaStream_t stream[8];

  for (auto& s : stream) {
    cudaStreamCreate(&s);
  }

  cudaMemcpyToSymbolAsync(num_data_block_d, &info.num_data_block,
                          sizeof(size_t), 0, cudaMemcpyHostToDevice, stream[0]);
  cudaMemcpyToSymbolAsync(num_kv_last_data_block_d,
                          &info.num_kv_last_data_block, sizeof(size_t), 0,
                          cudaMemcpyHostToDevice, stream[1]);
  cudaMemcpyToSymbolAsync(num_restarts_d, &info.num_restarts, sizeof(uint32_t),
                          0, cudaMemcpyHostToDevice, stream[2]);

  size_t size_complete_data_block = key_value_size * num_kv_data_block +
                                    (info.num_restarts + 1) * sizeof(uint32_t) +
                                    5;

  cudaMemcpyToSymbolAsync(size_complete_data_block_d, &size_complete_data_block,
                          sizeof(uint32_t), 0, cudaMemcpyHostToDevice,
                          stream[3]);

  auto* restarts = new uint32_t[info.num_restarts];
  for (size_t i = 0; i < info.num_restarts; ++i) {
    restarts[i] = BlockRestartInterval * key_value_size * i;
  }

  cudaMemcpyToSymbolAsync(const_restarts_d, restarts,
                          info.num_restarts * sizeof(uint32_t), 0,
                          cudaMemcpyHostToDevice, stream[0]);

  uint32_t last_num_restarts =
      info.num_kv_last_data_block / BlockRestartInterval + 1;
  auto* last_restarts = new uint32_t[last_num_restarts];
  for (uint32_t i = 0; i < last_num_restarts; ++i) {
    last_restarts[i] = key_value_size * BlockRestartInterval * i;
  }

  size_t data_buffer_size =
      size_complete_data_block * (info.num_data_block - 1);

  size_t size_incomplete_data_block;
  if (info.num_kv_last_data_block == 0) {
    // 最后一个数据块的kv对数量为0，说明当前SSTable的大小为最大64MB，
    // 所有数据块的KV对数量最大且相同
    size_incomplete_data_block = size_complete_data_block;
  } else {
    // 如果不为0，说明最后一块中的键值对数量不足
    size_incomplete_data_block = key_value_size * info.num_kv_last_data_block +
                                 (last_num_restarts + 1) * sizeof(uint32_t) + 5;
  }

  data_buffer_size += size_incomplete_data_block;

  // 索引块
  // 最后的 +5 是 trailer
  size_t size_index_block =
      info.num_data_block * (size_index_entry + sizeof(uint32_t)) +
      sizeof(uint32_t) + 5;

  meta.num_entries = info.total_num_kv;
  meta.num_data_blocks = info.num_data_block;
  meta.raw_key_size = (keySize_ + 8) * info.total_num_kv;
  meta.raw_value_size = valueSize_ * info.total_num_kv;

  // 为了是blocks_buffer_size的大小是1024的整数倍
  size_t blocks_buffer_size = data_buffer_size + size_index_block + 2048 -
                              (data_buffer_size + size_index_block) % 1024;

  // 使用内存对齐
  char* blocks_buffer;  // 所有块的buffer
  cudaHostAlloc((void**)&blocks_buffer, blocks_buffer_size,
                cudaHostAllocWriteCombined);
  char* index_buffer;
  cudaHostAlloc((void**)&index_buffer, size_index_block,
                cudaHostAllocWriteCombined);
  auto* restarts_for_index = new uint32_t[info.num_data_block];
  auto* data_block_handle = new GPUBlockHandle[info.num_data_block];
  char* index_keys = new char[info.num_data_block * keySize_];

  char* blocks_buffer_d;
  uint32_t* last_restarts_d;
  GPUKeyValue* keyValues_d;
  char* index_buffer_d;
  char* index_keys_d;
  uint32_t* restarts_for_index_d;
  GPUBlockHandle* data_block_handle_d;

  cudaMalloc(&blocks_buffer_d, blocks_buffer_size);
  cudaMalloc(&last_restarts_d, sizeof(uint32_t) * last_num_restarts);
  cudaMalloc(&keyValues_d, info.total_num_kv * sizeof(GPUKeyValue));
  cudaMalloc(&index_buffer_d, size_index_block);
  cudaMalloc(&index_keys_d, info.num_data_block * keySize_);
  cudaMalloc(&restarts_for_index_d, info.num_data_block * sizeof(uint32_t));
  cudaMalloc(&data_block_handle_d,
             info.num_data_block * sizeof(GPUBlockHandle));

  CHECK(cudaMemcpyAsync(blocks_buffer_d, blocks_buffer, blocks_buffer_size,
                        cudaMemcpyHostToDevice, stream[0]));

  CHECK(cudaMemcpyAsync(last_restarts_d, last_restarts,
                        sizeof(uint32_t) * last_num_restarts,
                        cudaMemcpyHostToDevice, stream[1]));

  CHECK(cudaMemcpyAsync(keyValues_d, keyValues,
                        info.total_num_kv * sizeof(GPUKeyValue),
                        cudaMemcpyHostToDevice, stream[2]));

  CHECK(cudaMemcpyAsync(index_buffer_d, index_buffer, size_index_block,
                        cudaMemcpyHostToDevice, stream[3]));

  CHECK(cudaMemcpyAsync(index_keys_d, index_keys,
                        info.num_data_block * keySize_, cudaMemcpyHostToDevice,
                        stream[0]));

  for (size_t i = 0; i < info.num_data_block - 1; ++i) {
    restarts_for_index[i] = size_index_entry * i;

    data_block_handle[i].set_offset(size_complete_data_block * i);
    data_block_handle[i].set_size(size_complete_data_block - 5);
  }

  restarts_for_index[info.num_data_block - 1] =
      size_index_entry * (info.num_data_block - 1);

  data_block_handle[info.num_data_block - 1].set_offset(
      size_complete_data_block * (info.num_data_block - 1));
  data_block_handle[info.num_data_block - 1].set_size(
      size_incomplete_data_block - 5);

  CHECK(cudaMemcpyAsync(restarts_for_index_d, restarts_for_index,
                        info.num_data_block * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, stream[0]));

  CHECK(cudaMemcpyAsync(data_block_handle_d, data_block_handle,
                        info.num_data_block * sizeof(GPUBlockHandle),
                        cudaMemcpyHostToDevice, stream[1]));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  BuildDataBlock(&blocks_buffer_d, keyValues_d, inputFiles_d, num_kv_data_block,
                 info, size_incomplete_data_block, last_num_restarts,
                 last_restarts_d, index_keys_d, stream, start, stop);

  char* new_buffer_d = blocks_buffer_d + data_buffer_size;

  BuildIndexBlock(&index_buffer, &new_buffer_d, index_keys_d,
                  data_block_handle_d, info.num_data_block,
                  restarts_for_index_d, size_index_block, stream, start, stop);

  BlockHandle meta_index_block_handle, index_block_handle;
  MetaIndexBuilder metaIndexBuilder;

  size_t new_offset;

  new_buffer_d += size_index_block;

  //  GPUBuildPropertiesBlock(&new_buffer_d, meta, tboptions, data_buffer_size,
  //                          size_index_block, &metaIndexBuilder, new_offset,
  //                          tp);

  size_t props_size = new_offset - data_buffer_size - size_index_block;
  new_buffer_d += props_size;

  size_t last_offset = new_offset;

  GPUBuildMetaIndexBlock(&new_buffer_d, &metaIndexBuilder,
                         &meta_index_block_handle, last_offset, new_offset);

  index_block_handle.set_offset(data_buffer_size);
  index_block_handle.set_size(size_index_block - 5);

  size_t meta_index_size = new_offset - last_offset;
  new_buffer_d += meta_index_size;

  last_offset = new_offset;

  GPUBuildFooter(&new_buffer_d, index_block_handle, meta_index_block_handle,
                 last_offset, new_offset);

  CHECK(cudaMemcpyAsync(blocks_buffer, blocks_buffer_d, new_offset,
                        cudaMemcpyDeviceToHost, stream[0]));

  info.file_size = new_offset;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(index_keys_d);
  cudaFree(keyValues_d);
  cudaFree(index_buffer_d);
  cudaFree(data_block_handle_d);
  cudaFree(restarts_for_index_d);
  cudaFree(last_restarts_d);
  cudaFree(blocks_buffer_d);

  for (auto& s : stream) {
    cudaStreamSynchronize(s);
    cudaStreamDestroy(s);
  }

  delete[] restarts;
  delete[] last_restarts;
  delete[] index_keys;
  delete[] restarts_for_index;
  delete[] data_block_handle;
  cudaFreeHost(index_buffer);

  return blocks_buffer;
}

__device__ void BeginBuildDataBlock(
    uint32_t file_idx, uint32_t block_idx, char* current_block_buffer,
    GPUKeyValue* key_values_d, InputFile* input_files_d, char* index_keys_d,
    size_t num_kv_current_data_block, uint32_t num_current_restarts,
    uint32_t* current_restarts, size_t size_current_data_block) {
  size_t shared = 0;
  size_t non_shared = keySize_ + 8;
  size_t encoded_size;

  for (size_t i = 0; i < num_kv_current_data_block; ++i) {
    GPUPutVarint32Varint32Varint32(
        current_block_buffer + key_value_size * i,
        static_cast<uint32_t>(shared), static_cast<uint32_t>(non_shared),
        static_cast<uint32_t>(valueSize_), encoded_size);
    memcpy(current_block_buffer + key_value_size * i + encoded_size,
           key_values_d[file_idx * total_num_kv_front_file_d +
                        num_kv_data_block_d * block_idx + i]
               .key,
           non_shared);

    ExtractOriginalValue(
        key_values_d[file_idx * total_num_kv_front_file_d +
                     num_kv_data_block_d * block_idx + i],
        input_files_d, num_files_d,
        current_block_buffer + key_value_size * i + encoded_size + non_shared);
  }

  // 每个索引项中的key
  // num_data_block_d 指的是前面的文件的数据块数量
  memcpy(index_keys_d + (file_idx * num_data_block_d + block_idx) * keySize_,
         key_values_d[file_idx * total_num_kv_front_file_d +
                      num_kv_data_block_d * block_idx +
                      (num_kv_current_data_block - 1)]
             .key,
         keySize_);

  for (uint32_t i = 0; i < num_current_restarts; ++i) {
    GPUPutFixed32(current_block_buffer +
                      num_kv_current_data_block * key_value_size +
                      i * sizeof(uint32_t),
                  current_restarts[i]);
  }

  GPUPutFixed32(current_block_buffer +
                    num_kv_current_data_block * key_value_size +
                    num_current_restarts * sizeof(uint32_t),
                num_current_restarts);

  char trailer[5];
  char type = 0x0;
  trailer[0] = type;
  uint32_t checksum = GPUComputeBuiltinChecksumWithLastByte(
      current_block_buffer, size_current_data_block - 5, type);
  GPUEncodeFixed32(trailer + 1, checksum);

  memcpy(current_block_buffer + size_current_data_block - 5, trailer, 5);
}

__global__ void BuildDataBlocksFrontFileKernel(char* buffer_d,
                                               GPUKeyValue* key_values_d,
                                               InputFile* input_files_d,
                                               char* index_keys_d) {
  uint32_t file_idx = threadIdx.x;
  uint32_t block_idx = blockIdx.x;

  char* current_block_buffer = buffer_d + file_idx * size_front_file_d +
                               block_idx * size_complete_data_block_d;

  BeginBuildDataBlock(file_idx, block_idx, current_block_buffer, key_values_d,
                      input_files_d, index_keys_d, num_kv_data_block_d,
                      num_restarts_d, const_restarts_d,
                      size_complete_data_block_d);
}

__global__ void BuildDataBlocksLastFileKernel(char* buffer_d,
                                              GPUKeyValue* key_values_d,
                                              InputFile* input_files_d,
                                              char* index_keys_d) {
  uint32_t file_idx = num_outputs_d - 1;
  uint32_t block_idx = blockIdx.x;

  char* current_block_buffer = buffer_d + file_idx * size_front_file_d +
                               block_idx * size_complete_data_block_d;

  BeginBuildDataBlock(file_idx, block_idx, current_block_buffer, key_values_d,
                      input_files_d, index_keys_d, num_kv_data_block_d,
                      num_restarts_d, const_restarts_d,
                      size_complete_data_block_d);
}

__global__ void BuildLastDataBlockLastFileKernel(
    char* buffer_d, GPUKeyValue* key_values_d, InputFile* input_files_d,
    char* index_keys_d, size_t num_data_block_last_file,
    size_t num_kv_last_data_block_last_file,
    uint32_t num_restarts_last_data_block_last_file,
    uint32_t* restarts_last_data_block_last_file_d,
    size_t size_incomplete_data_block) {
  uint32_t file_idx = num_outputs_d - 1;
  uint32_t block_idx = num_data_block_last_file - 1;

  char* current_block_buffer = buffer_d + file_idx * size_front_file_d +
                               block_idx * size_complete_data_block_d;

  BeginBuildDataBlock(
      file_idx, block_idx, current_block_buffer, key_values_d, input_files_d,
      index_keys_d, num_kv_last_data_block_last_file,
      num_restarts_last_data_block_last_file,
      restarts_last_data_block_last_file_d, size_incomplete_data_block);
}

void BuildDataBlocks(char** buffer_d, GPUKeyValue* key_values_d,
                     InputFile* input_files_d, char* index_keys_d,
                     size_t size_incomplete_data_block,
                     size_t num_data_block_front_file,
                     size_t num_data_block_last_file,
                     size_t num_kv_last_data_block_last_file,
                     uint32_t num_restarts_last_data_block_last_file,
                     uint32_t* restarts_last_data_block_last_file_d,
                     size_t num_outputs, cudaStream_t* stream) {
  if (num_outputs > 1) {
    dim3 block(num_outputs - 1);
    dim3 grid(num_data_block_front_file);

    BuildDataBlocksFrontFileKernel<<<grid, block, 0, stream[3]>>>(
        *buffer_d, key_values_d, input_files_d, index_keys_d);
  }

  if (num_kv_last_data_block_last_file == 0) {
    dim3 block(1);
    dim3 grid(num_data_block_last_file);

    BuildDataBlocksLastFileKernel<<<grid, block, 0, stream[4]>>>(
        *buffer_d, key_values_d, input_files_d, index_keys_d);
  } else {
    dim3 block(1);
    dim3 grid(num_data_block_last_file - 1);

    BuildDataBlocksLastFileKernel<<<grid, block, 0, stream[4]>>>(
        *buffer_d, key_values_d, input_files_d, index_keys_d);

    BuildLastDataBlockLastFileKernel<<<1, 1, 0, stream[5]>>>(
        *buffer_d, key_values_d, input_files_d, index_keys_d,
        num_data_block_last_file, num_kv_last_data_block_last_file,
        num_restarts_last_data_block_last_file,
        restarts_last_data_block_last_file_d, size_incomplete_data_block);
  }
}

__device__ void BeginBuildIndexBlock(uint32_t file_idx, uint32_t block_idx,
                                     char* current_index_buffer,
                                     char* index_keys_d,
                                     GPUBlockHandle* block_handles_d,
                                     uint32_t* current_restarts,
                                     size_t current_num_data_block) {
  // 这里的 num_data_block_d 表示的是前面文件的一个文件的数据块数量

  size_t shared = 0;
  size_t non_shared = keySize_;
  size_t encoded_size;

  GPUPutVarint32Varint32(current_index_buffer + size_index_entry * block_idx,
                         static_cast<uint32_t>(shared),
                         static_cast<uint32_t>(non_shared), encoded_size);

  memcpy(current_index_buffer + size_index_entry * block_idx + encoded_size,
         index_keys_d + (file_idx * num_data_block_d + block_idx) * keySize_,
         keySize_);

  GPUPutFixed64Varint64(
      current_index_buffer + size_index_entry * block_idx + encoded_size +
          keySize_,
      block_handles_d[file_idx * num_data_block_d + block_idx].offset(),
      block_handles_d[file_idx * num_data_block_d + block_idx].size());

  char* restarts_buffer = current_index_buffer +
                          size_index_entry * current_num_data_block +
                          block_idx * sizeof(uint32_t);

  GPUPutFixed32(restarts_buffer, current_restarts[block_idx]);
}

__global__ void BuildIndexBlockFrontFileKernel(
    char* buffer_d, char* index_keys_d, GPUBlockHandle* block_handles_d,
    uint32_t* restarts_for_index_front_file_d) {
  uint32_t file_idx = threadIdx.x;
  uint32_t block_idx = blockIdx.x;

  char* current_index_buffer =
      buffer_d + file_idx * size_front_file_d + data_size_d;

  BeginBuildIndexBlock(file_idx, block_idx, current_index_buffer, index_keys_d,
                       block_handles_d, restarts_for_index_front_file_d,
                       num_data_block_d);
}

__global__ void BuildIndexBlockLastFileKernel(
    char* buffer_d, char* index_keys_d, GPUBlockHandle* block_handles_d,
    uint32_t* restarts_for_index_last_file_d, size_t num_data_block_last_file) {
  uint32_t file_idx = num_outputs_d - 1;
  uint32_t block_idx = blockIdx.x;

  char* current_index_buffer =
      buffer_d + file_idx * size_front_file_d + data_size_last_file_d;

  BeginBuildIndexBlock(file_idx, block_idx, current_index_buffer, index_keys_d,
                       block_handles_d, restarts_for_index_last_file_d,
                       num_data_block_last_file);
}

__device__ void BeginComputeChecksum(char* current_buffer,
                                     size_t num_data_block,
                                     size_t index_block_size) {
  GPUPutFixed32(current_buffer - 9, num_data_block);

  char trailer[5];
  char type = 0x0;
  trailer[0] = type;
  uint32_t checksum = GPUComputeBuiltinChecksumWithLastByte(
      current_buffer - index_block_size, index_block_size - 5, type);
  GPUEncodeFixed32(trailer + 1, checksum);

  memcpy(current_buffer - 5, trailer, 5);
}

__global__ void ComputeChecksumFrontFileKernel(char* buffer_d) {
  uint32_t file_idx = threadIdx.x;

  char* current_buffer =
      buffer_d + file_idx * size_front_file_d + data_size_d + index_size_d;

  BeginComputeChecksum(current_buffer, num_data_block_d, index_size_d);
}

__global__ void ComputeChecksumLastFileKernel(char* buffer_d,
                                              size_t num_data_block_last_file,
                                              size_t index_size_last_file) {
  uint32_t file_idx = num_outputs_d - 1;

  char* current_buffer = buffer_d + file_idx * size_front_file_d +
                         data_size_last_file_d + index_size_last_file;

  BeginComputeChecksum(current_buffer, num_data_block_last_file,
                       index_size_last_file);
}

__global__ void ComputeDataBlockHandleFrontFileKernel(
    GPUBlockHandle* block_handles_d,
    uint32_t* restarts_for_index_front_file_d) {
  uint32_t file_idx = threadIdx.x;
  uint32_t block_idx = blockIdx.x;

  block_handles_d[file_idx * num_data_block_d + block_idx].set_offset(
      block_idx * size_complete_data_block_d);
  block_handles_d[file_idx * num_data_block_d + block_idx].set_size(
      size_complete_data_block_d - 5);

  if (file_idx == 0)
    restarts_for_index_front_file_d[block_idx] = size_index_entry * block_idx;
}

__global__ void ComputeDataBlockHandleLastFileKernel(
    GPUBlockHandle* block_handles_d, uint32_t* restarts_for_index_last_file_d,
    size_t size_incomplete_data_block, size_t num_data_block_last_file) {
  uint32_t file_idx = num_outputs_d - 1;
  uint32_t block_idx = blockIdx.x;

  block_handles_d[file_idx * num_data_block_d + block_idx].set_offset(
      block_idx * size_complete_data_block_d);
  block_handles_d[file_idx * num_data_block_d + block_idx].set_size(
      size_complete_data_block_d - 5);

  restarts_for_index_last_file_d[block_idx] = size_index_entry * block_idx;

  if (block_idx == num_data_block_last_file - 1) {
    block_handles_d[file_idx * num_data_block_d + block_idx].set_size(
        size_incomplete_data_block - 5);
  }
}

void BuildIndexBlocks(char** buffer_d, char* index_keys_d,
                      GPUBlockHandle* block_handles_d, size_t num_outputs,
                      size_t num_data_block_front_file,
                      size_t num_data_block_last_file,
                      size_t index_size_last_file,
                      size_t size_incomplete_data_block,
                      uint32_t* restarts_for_index_front_file_d,
                      uint32_t* restarts_for_index_last_file_d,
                      cudaStream_t* stream) {
  if (num_outputs > 1) {
    dim3 block(num_outputs - 1);
    dim3 grid(num_data_block_front_file);

    ComputeDataBlockHandleFrontFileKernel<<<grid, block, 0, stream[6]>>>(
        block_handles_d, restarts_for_index_front_file_d);

    // 第二个核函数需要第一个核函数的结果，它们使用同一个流，所以不需要进行同步
    BuildIndexBlockFrontFileKernel<<<grid, block, 0, stream[6]>>>(
        *buffer_d, index_keys_d, block_handles_d,
        restarts_for_index_front_file_d);

    ComputeChecksumFrontFileKernel<<<1, num_outputs - 1, 0, stream[6]>>>(
        *buffer_d);
  }

  dim3 block(1);
  dim3 grid(num_data_block_last_file);

  ComputeDataBlockHandleLastFileKernel<<<grid, block, 0, stream[7]>>>(
      block_handles_d, restarts_for_index_last_file_d,
      size_incomplete_data_block, num_data_block_last_file);

  BuildIndexBlockLastFileKernel<<<grid, block, 0, stream[7]>>>(
      *buffer_d, index_keys_d, block_handles_d, restarts_for_index_last_file_d,
      num_data_block_last_file);

  ComputeChecksumLastFileKernel<<<1, 1, 0, stream[7]>>>(
      *buffer_d, num_data_block_last_file, index_size_last_file);
}

void WriteSSTable(char* buffer_d, CompactionJob* compaction_job,
                  const Compaction* compact,
                  const std::shared_ptr<WritableFileWriter>& file_writer,
                  const std::shared_ptr<TableBuilderOptions>& tbs,
                  FileMetaData* meta, TableProperties* tp, SSTableInfo* info,
                  size_t data_size, size_t index_size) {
  meta->num_entries = info->total_num_kv;
  meta->num_data_blocks = info->num_data_block;
  meta->raw_key_size = (keySize_ + 8) * info->total_num_kv;
  meta->raw_value_size = valueSize_ * info->total_num_kv;

  BlockHandle meta_index_block_handle, index_block_handle;
  MetaIndexBuilder meta_index_builder;

  char* new_buffer = buffer_d + data_size + index_size;

  size_t new_offset;

  GPUBuildPropertiesBlock(&new_buffer, meta, tbs, data_size, index_size,
                          &meta_index_builder, new_offset, tp);

  size_t props_size = new_offset - data_size - index_size;
  new_buffer += props_size;

  size_t last_offset = new_offset;

  GPUBuildMetaIndexBlock(&new_buffer, &meta_index_builder,
                         &meta_index_block_handle, last_offset, new_offset);

  index_block_handle.set_offset(data_size);
  index_block_handle.set_size(index_size - 5);

  size_t meta_index_size = new_offset - last_offset;
  new_buffer += meta_index_size;

  last_offset = new_offset;

  GPUBuildFooter(&new_buffer, index_block_handle, meta_index_block_handle,
                 last_offset, new_offset);

  info->file_size = new_offset;

//  auto start_time = std::chrono::high_resolution_clock::now();

  compaction_job->MyFinishCompactionOutputFile(compact, file_writer, meta, info,
                                               tp);

  std::string filename = file_writer->file_name();

  int fd = open(filename.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0664);

  CUfileDescr_t descr;
  descr.handle.fd = fd;
  descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

  CUfileHandle_t handle;
  cuFileHandleRegister(&handle, &descr);

  cuFileWrite(handle, buffer_d, new_offset, 0, 0);

  cuFileHandleDeregister(handle);
  close(fd);

//  auto end_time = std::chrono::high_resolution_clock::now();
//  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
//      end_time - start_time);
//  std::cout << "Write time: " << duration.count() << " us\n";
}

void BuildSSTables(
    CompactionJob* compaction_job, const Compaction* compact,
    GPUKeyValue* key_values_d, InputFile* input_files_d,
    std::vector<SSTableInfo>& infos, std::vector<FileMetaData>& metas,
    std::vector<std::shared_ptr<WritableFileWriter>>& file_writes,
    std::vector<std::shared_ptr<TableBuilderOptions>>& tbs,
    std::vector<TableProperties>& tps, size_t num_kv_data_block,
    cudaStream_t* stream) {
  size_t num_outputs = infos.size();  // 需要编码的文件数

  //  cudaEvent_t start, end;
  //  cudaEventCreate(&start);
  //  cudaEventCreate(&end);
  //  float elapsed_time;

  uint32_t num_restarts = infos[0].num_restarts;  // 非最后一块的restarts数量
  uint32_t num_restarts_last_data_block_last_file =
      infos[num_outputs - 1]
          .num_restarts;  // 最后一个文件最后一个数据块的restarts数量

  size_t total_estimate_file_size;  // 编码后所有SSTable的大小的估计值
  size_t total_num_all_data_blocks;  // index_keys 用

  // 最后一个文件的数据块数量
  size_t num_data_block_last_file = infos[num_outputs - 1].num_data_block;

  // 最后一个文件最后一个数据块的KV对数量
  size_t num_kv_last_data_block_last_file =
      infos[num_outputs - 1].num_kv_last_data_block;

  // 完整数据块大小
  size_t size_complete_data_block = key_value_size * num_kv_data_block +
                                    (num_restarts + 1) * sizeof(uint32_t) + 5;

  // 非完整数据块大小 -- 即最后一个文件的最后一个数据块的大小
  size_t size_incomplete_data_block;
  if (num_kv_last_data_block_last_file == 0) {
    size_incomplete_data_block = size_complete_data_block;
  } else {
    size_incomplete_data_block =
        key_value_size * num_kv_last_data_block_last_file +
        (num_restarts_last_data_block_last_file + 1) * sizeof(uint32_t) + 5;
  }

  // 数据块大小
  // (数据总数量 - 1) * size_complete_data_block + size_incomplete_data_block

  // 索引块大小
  // 数据块数量 * (size_index_entry + sizeof(uint32_t)) + sizeof(uint32_t) + 5

  // 一个SSTable大小的估计值
  // data_size + index_size + 2048 - (data_size + index_size) % 2048

  // 非最后一个SSTable的数据块、索引块和所有块的总大小
  size_t data_size = 0;
  size_t index_size = 0;
  size_t estimate_file_size = 0;

  size_t total_num_kv_front_file = 0;
  size_t num_data_block_front_file = 0;

  if (num_outputs > 1) {
    total_num_kv_front_file = infos[0].total_num_kv;
    num_data_block_front_file = infos[0].num_data_block;

    data_size = num_data_block_front_file * size_complete_data_block;
    index_size =
        num_data_block_front_file * (size_index_entry + sizeof(uint32_t)) +
        sizeof(uint32_t) + 5;
    estimate_file_size =
        data_size + index_size + 2048 - (data_size + index_size) % 1024;
  }

  // 最后一个SSTable的数据块、索引块和所有块的总大小
  size_t data_size_last_file =
      (num_data_block_last_file - 1) * size_complete_data_block +
      size_incomplete_data_block;
  size_t index_size_last_file =
      num_data_block_last_file * (size_index_entry + sizeof(uint32_t)) +
      sizeof(uint32_t) + 5;
  size_t estimate_last_file_size =
      data_size_last_file + index_size_last_file + 2048 -
      (data_size_last_file + index_size_last_file) % 1024;

  total_estimate_file_size =
      (num_outputs - 1) * estimate_file_size + estimate_last_file_size;

  total_num_all_data_blocks =
      (num_outputs - 1) * num_data_block_front_file + num_data_block_last_file;

  // 申请主机内存
  // 非最后一个数据块的restarts
  auto* restarts = new uint32_t[num_restarts];
  for (uint32_t i = 0; i < num_restarts; ++i) {
    restarts[i] = BlockRestartInterval * key_value_size * i;
  }

  // 最后一个文件的最后一个数据块的restarts信息
  auto* restarts_last_data_block_last_file =
      new uint32_t[num_restarts_last_data_block_last_file];
  for (uint32_t i = 0; i < num_restarts_last_data_block_last_file; ++i) {
    restarts_last_data_block_last_file[i] =
        BlockRestartInterval * key_value_size * i;
  }

  // 申请设备内存
  // 数据块
  char* all_files_buffer_d;
  char* index_keys_d;
  uint32_t* restarts_last_data_block_last_file_d;
  // 索引块
  GPUBlockHandle* block_handles_d;
  uint32_t* restarts_for_index_front_file_d;
  uint32_t* restarts_for_index_last_file_d;

  cudaMallocAsync(&all_files_buffer_d, total_estimate_file_size, stream[3]);
  cudaMallocAsync(&index_keys_d, total_num_all_data_blocks * keySize_,
                  stream[4]);
  cudaMallocAsync(&restarts_last_data_block_last_file_d,
                  num_restarts_last_data_block_last_file * sizeof(uint32_t),
                  stream[5]);
  cudaMallocAsync(&block_handles_d,
                  total_num_all_data_blocks * sizeof(GPUBlockHandle),
                  stream[6]);
  cudaMallocAsync(&restarts_for_index_front_file_d,
                  num_data_block_front_file * sizeof(uint32_t), stream[7]);
  cudaMallocAsync(&restarts_for_index_last_file_d,
                  num_data_block_last_file * sizeof(uint32_t), stream[5]);

  //  cudaEventRecord(start, nullptr);
  // 数据传输
  // 常量内存
  cudaMemcpyToSymbolAsync(num_data_block_d, &infos[0].num_data_block,
                          sizeof(size_t), 0, cudaMemcpyHostToDevice, stream[3]);
  cudaMemcpyToSymbolAsync(num_restarts_d, &num_restarts, sizeof(uint32_t), 0,
                          cudaMemcpyHostToDevice, stream[4]);
  cudaMemcpyToSymbolAsync(const_restarts_d, restarts,
                          num_restarts * sizeof(uint32_t), 0,
                          cudaMemcpyHostToDevice, stream[5]);
  cudaMemcpyToSymbolAsync(size_complete_data_block_d, &size_complete_data_block,
                          sizeof(size_t), 0, cudaMemcpyHostToDevice, stream[6]);
  cudaMemcpyToSymbolAsync(total_num_kv_front_file_d, &total_num_kv_front_file,
                          sizeof(size_t), 0, cudaMemcpyHostToDevice, stream[7]);
  cudaMemcpyToSymbolAsync(size_front_file_d, &estimate_file_size,
                          sizeof(size_t), 0, cudaMemcpyHostToDevice, stream[3]);
  cudaMemcpyToSymbolAsync(num_outputs_d, &num_outputs, sizeof(size_t), 0,
                          cudaMemcpyHostToDevice, stream[4]);
  cudaMemcpyToSymbolAsync(data_size_d, &data_size, sizeof(size_t), 0,
                          cudaMemcpyHostToDevice, stream[5]);
  cudaMemcpyToSymbolAsync(data_size_last_file_d, &data_size_last_file,
                          sizeof(size_t), 0, cudaMemcpyHostToDevice, stream[6]);
  cudaMemcpyToSymbolAsync(index_size_d, &index_size, sizeof(size_t), 0,
                          cudaMemcpyHostToDevice, stream[7]);

  // 全局内存
  CHECK(cudaMemcpyAsync(
      restarts_last_data_block_last_file_d, restarts_last_data_block_last_file,
      num_restarts_last_data_block_last_file * sizeof(uint32_t),
      cudaMemcpyHostToDevice, stream[3]));

  for (size_t i = 3; i < 8; i++) {
    CHECK(cudaStreamSynchronize(stream[i]));
  }

  //  cudaEventRecord(end, nullptr);
  //  cudaEventSynchronize(end);
  //  cudaEventElapsedTime(&elapsed_time, start, end);
  //  gpu_stats.transmission_time += elapsed_time;

  auto start_time = std::chrono::high_resolution_clock::now();

  // 构建所有文件的数据块
  BuildDataBlocks(&all_files_buffer_d, key_values_d, input_files_d,
                  index_keys_d, size_incomplete_data_block,
                  num_data_block_front_file, num_data_block_last_file,
                  num_kv_last_data_block_last_file,
                  num_restarts_last_data_block_last_file,
                  restarts_last_data_block_last_file_d, num_outputs, stream);

  // 构建所有文件的索引块
  BuildIndexBlocks(&all_files_buffer_d, index_keys_d, block_handles_d,
                   num_outputs, num_data_block_front_file,
                   num_data_block_last_file, index_size_last_file,
                   size_incomplete_data_block, restarts_for_index_front_file_d,
                   restarts_for_index_last_file_d, stream);

  for (size_t i = 3; i < 8; i++) {
    CHECK(cudaStreamSynchronize(stream[i]));
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);

  gpu_stats.gpu_all_micros += duration.count();

  // 写SSTable
  // 方法1
  /*size_t num_threads;
  if (num_outputs < num_threads_for_gpu) {
    num_threads = num_outputs;
  } else {
    num_threads = num_threads_for_gpu;
  }

  // 申请一个线程池，多线程执行写每个文件的最后三个块
  thread_pool_for_gpu.clear();
  thread_pool_for_gpu.reserve(num_threads - 1);

  // 使用分块策略将任务分配给线程
  size_t chunk_size =
      (num_outputs + num_threads_for_gpu - 1) / num_threads_for_gpu;

  for (size_t t = 0; t < num_threads - 1; ++t) {
    size_t begin = t * chunk_size;
    if (begin >= num_outputs) {
      break;
    }
    size_t end = std::min((t + 1) * chunk_size, num_outputs - 1);

    thread_pool_for_gpu.emplace_back([&, begin, end]() {
      for (size_t i = begin; i < end; ++i) {
        char* current_file_buffer = all_files_buffer + estimate_file_size * i;
        WriteSSTable(current_file_buffer, compaction_job, compact,
                     file_writes[i], tbs[i], &metas[i], &tps[i], &infos[i],
                     data_size, index_size);
      }
    });
  }

  WriteSSTable(all_files_buffer + estimate_file_size * (num_outputs - 1),
               compaction_job, compact, file_writes[num_outputs - 1],
               tbs[num_outputs - 1], &metas[num_outputs - 1],
               &tps[num_outputs - 1], &infos[num_outputs - 1],
               data_size_last_file, index_size_last_file);*/

  // 方法2
  /*thread_pool_for_gpu.clear();
  thread_pool_for_gpu.reserve(num_outputs - 1);

  for (size_t i = 0; i < num_outputs - 1; ++i) {
    char* current_file_buffer = all_files_buffer_d + estimate_file_size * i;
    thread_pool_for_gpu.emplace_back(&WriteSSTable, current_file_buffer,
                                     compaction_job, compact, file_writes[i],
                                     tbs[i], &metas[i], &tps[i], &infos[i],
                                     data_size, index_size);
  }

  char* current_file_buffer =
      all_files_buffer_d + estimate_file_size * (num_outputs - 1);
  WriteSSTable(current_file_buffer, compaction_job, compact,
               file_writes[num_outputs - 1], tbs[num_outputs - 1],
               &metas[num_outputs - 1], &tps[num_outputs - 1],
               &infos[num_outputs - 1], data_size_last_file,
               index_size_last_file);*/

  // 方法3
  /*size_t count = num_outputs % num_threads_for_gpu == 0
                     ? num_outputs / num_threads_for_gpu
                     : num_outputs / num_threads_for_gpu + 1;

  if (num_outputs < num_threads_for_gpu) {
    thread_pool_for_gpu.clear();
    thread_pool_for_gpu.reserve(num_outputs - 1);
  } else {
    thread_pool_for_gpu.clear();
    thread_pool_for_gpu.reserve(num_threads_for_gpu);
  }

  for (size_t t = 0; t < count; ++t) {
    for (size_t i = t * num_threads_for_gpu; i < (t + 1) * num_threads_for_gpu;
         ++i) {
      if (i >= num_outputs - 1) {
        break;
      }
      char* current_file_buffer = all_files_buffer + estimate_file_size * i;
      thread_pool_for_gpu.emplace_back(&WriteSSTable, current_file_buffer,
                                       compaction_job, compact, file_writes[i],
                                       tbs[i], &metas[i], &tps[i], &infos[i],
                                       data_size, index_size);
    }
  }

  WriteSSTable(all_files_buffer + estimate_file_size * (num_outputs - 1),
               compaction_job, compact, file_writes[num_outputs - 1],
               tbs[num_outputs - 1], &metas[num_outputs - 1],
               &tps[num_outputs - 1], &infos[num_outputs - 1],
               data_size_last_file, index_size_last_file);*/

  // 方法4
  for (size_t i = 0; i < num_outputs - 1; ++i) {
    char* current_file_buffer = all_files_buffer_d + estimate_file_size * i;
    WriteSSTable(current_file_buffer, compaction_job, compact, file_writes[i],
                 tbs[i], &metas[i], &tps[i], &infos[i], data_size, index_size);
  }

  char* current_file_buffer =
      all_files_buffer_d + estimate_file_size * (num_outputs - 1);
  WriteSSTable(current_file_buffer, compaction_job, compact,
               file_writes[num_outputs - 1], tbs[num_outputs - 1],
               &metas[num_outputs - 1], &tps[num_outputs - 1],
               &infos[num_outputs - 1], data_size_last_file,
               index_size_last_file);

  // 释放资源
  cudaFree(all_files_buffer_d);
  cudaFree(key_values_d);
  cudaFree(index_keys_d);
  cudaFree(restarts_last_data_block_last_file_d);
  cudaFree(block_handles_d);
  cudaFree(restarts_for_index_front_file_d);
  cudaFree(restarts_for_index_last_file_d);

  delete[] restarts;
  delete[] restarts_last_data_block_last_file;

  /*for (auto& thread : thread_pool_for_gpu) {
    thread.join();
  }*/
}

}  // namespace ROCKSDB_NAMESPACE
