//
// Created by jxx on 4/8/23.
//

#include "gpu_decode_block.cuh"

namespace ROCKSDB_NAMESPACE {

__global__ void PrepareDecode(InputFile* inputFiles_d, size_t num_file,
                              uint64_t* all_num_kv_d,
                              size_t* max_num_data_block_d) {
  for (size_t i = 0; i < num_file; ++i) {
    uint64_t num_kv = inputFiles_d[i].num_entries;
    *all_num_kv_d += num_kv;

    if (*max_num_data_block_d < inputFiles_d[i].num_data_blocks) {
      *max_num_data_block_d = inputFiles_d[i].num_data_blocks;
    }
  }
}

__global__ void DecodeFootersKernel(InputFile* inputFiles,
                                    GPUBlockHandle* footers) {
  unsigned int tid = threadIdx.x;

  const char* footer = inputFiles[tid].file + inputFiles[tid].file_size - 53;
  size_t footer_size = 53;
  footer += 1;
  footer_size -= 1;

  uint64_t offset, size;
  DecodeFrom(&footer, &footer_size, &offset, &size);  // meta
  DecodeFrom(&footer, &footer_size, &offset, &size);  // index

  footers[tid].set_offset(offset);
  footers[tid].set_size(size);
}

__global__ void ComputeRestartsKernel(InputFile* inputFiles,
                                      GPUBlockHandle* footer,
                                      uint32_t* restarts,
                                      uint64_t max_num_data_block_d) {
  unsigned int file_idx = threadIdx.x;
  unsigned int block_idx = blockIdx.x;

  const char* data = inputFiles[file_idx].file + footer[file_idx].offset();
  size_t data_size = footer[file_idx].size();

  if (block_idx < inputFiles[file_idx].num_data_blocks) {
    restarts[file_idx * max_num_data_block_d +
             inputFiles[file_idx].num_data_blocks - block_idx - 1] =
        GPUDecodeFixed32(data + data_size - (block_idx + 2) * sizeof(uint32_t));
  }
}

__global__ void DecodeIndexBlocksKernel(InputFile* inputFiles,
                                        const uint32_t* restarts,
                                        GPUBlockHandle* footer,
                                        GPUBlockHandle* index_block,
                                        uint64_t max_num_data_block_d) {
  unsigned int file_idx = threadIdx.x;
  unsigned int block_idx = blockIdx.x;

  const char* data = inputFiles[file_idx].file + footer[file_idx].offset();
  size_t data_size = footer[file_idx].size();
  uint32_t restarts_offset =
      RestartOffset(data_size, inputFiles[file_idx].num_data_blocks);

  if (block_idx < inputFiles[file_idx].num_data_blocks) {
    const char* p =
        data + restarts[file_idx * max_num_data_block_d + block_idx];
    const char* limit = data + restarts_offset;

    uint32_t shared, non_shared, value_length;
    p = DecodeIndexKey(p, limit, &shared, &non_shared, &value_length);

    const char* value = p + non_shared;
    const char* v = value;
    size_t v_size = data + restarts_offset - value;

    uint64_t offset, size;
    DecodeFromForIndex(&v, &v_size, &offset, &size);

    index_block[file_idx * max_num_data_block_d + block_idx].set_offset(offset);
    index_block[file_idx * max_num_data_block_d + block_idx].set_size(size);
  }
}

__global__ void DecodeDataBlocksKernel(InputFile* inputFiles,
                                       uint32_t* global_count,
                                       GPUBlockHandle* index_block,
                                       GPUKeyValue* keyValuePtr,
                                       uint64_t max_num_data_block_d) {
  unsigned int file_idx = threadIdx.x;
  unsigned int block_idx = blockIdx.x;
  //  unsigned int kv_idx = threadIdx.y;

  if (block_idx >= inputFiles[file_idx].num_data_blocks) return;

  const char* current_data_block =
      inputFiles[file_idx].file +
      index_block[file_idx * max_num_data_block_d + block_idx].offset();

  auto file_start = reinterpret_cast<uint64_t>(inputFiles[file_idx].file);

  size_t num_kv;
  if (block_idx < inputFiles[file_idx].num_data_blocks - 1) {
    num_kv = num_kv_data_block_d;
  } else {  // 最后一个数据块
    num_kv = inputFiles[file_idx].num_entries -
             num_kv_data_block_d * (inputFiles[file_idx].num_data_blocks - 1);
  }

  for (size_t i = 0; i < num_kv; ++i) {
    const char* p = current_data_block + key_value_size * i;

    uint32_t index = atomicAdd(global_count, 1);

    keyValuePtr[index].file_number = inputFiles[file_idx].file_number;

    const char* key = p + 2 + encoded_value_size;
    memcpy(keyValuePtr[index].key, key, keySize_ + 8);

    const char* value = key + keySize_ + 8;
    keyValuePtr[index].valuePtr =  // 获得value的偏移量（相对于文件开始位置）
        reinterpret_cast<uint64_t>(value) - file_start;

    GPUParseInternalKey(key, keySize_ + 8, keyValuePtr[index].sequence,
                        keyValuePtr[index].type);
  }

  /*if (kv_idx < num_data_blocks_d[file_idx] - 1) {
    const char* p = current_data_block + key_value_size_d * kv_idx;

  uint32_t index = atomicAdd(global_count, 1);

  keyValuePtr[index].file_number = inputFiles[file_idx].file_number;

  const char* key = p + 2 + encoded_value_size_d;
  memcpy(keyValuePtr[index].key, key, key_size_d);

  const char* value = key + key_size_d;
  keyValuePtr[index].valuePtr =  // 获得value的偏移量（相对于文件开始位置）
      reinterpret_cast<uint64_t>(value) - file_start;

  GPUParseInternalKey(key, key_size_d, keyValuePtr[index].sequence,
                      keyValuePtr[index].type);
  } else {
    if (kv_idx < num_kv_last_data_blocks_d[file_idx]) {
      const char* p = current_data_block + key_value_size_d * kv_idx;

      uint32_t index = atomicAdd(global_count, 1);

      keyValuePtr[index].file_number = inputFiles[file_idx].file_number;

      const char* key = p + 2 + encoded_value_size_d;
      memcpy(keyValuePtr[index].key, key, key_size_d);

      const char* value = key + key_size_d;
      keyValuePtr[index].valuePtr =  // 获得value的偏移量（相对于文件开始位置）
          reinterpret_cast<uint64_t>(value) - file_start;

      GPUParseInternalKey(key, key_size_d, keyValuePtr[index].sequence,
                          keyValuePtr[index].type);
    }
  }*/
}

GPUKeyValue* GetAndSort(size_t num_file, InputFile* inputFiles_d,
                        size_t num_kv_data_block, size_t& sorted_size,
                        cudaStream_t* stream) {
  cudaMemcpyToSymbolAsync(num_files_d, &num_file, sizeof(size_t), 0,
                          cudaMemcpyHostToDevice, stream[0]);
  cudaMemcpyToSymbolAsync(num_kv_data_block_d, &num_kv_data_block,
                          sizeof(size_t), 0, cudaMemcpyHostToDevice, stream[1]);

  // 准备输入数据
  // footer、索引块和数据块共有输入数据
  uint64_t* all_num_kv_d;
  cudaMallocAsync(&all_num_kv_d, sizeof(uint64_t), stream[0]);
  cudaMemsetAsync(all_num_kv_d, 0, sizeof(uint64_t), stream[0]);

  size_t* max_num_data_block_d;
  cudaMallocAsync(&max_num_data_block_d, sizeof(size_t), stream[1]);
  cudaMemsetAsync(max_num_data_block_d, 0, sizeof(size_t), stream[1]);

  for (size_t i = 0; i < 2; i++) {
    CHECK(cudaStreamSynchronize(stream[i]));
  }

  PrepareDecode<<<1, 1, 0, stream[0]>>>(inputFiles_d, num_file, all_num_kv_d,
                                        max_num_data_block_d);

  cudaStreamSynchronize(stream[0]);

  uint64_t all_num_kv;
  cudaMemcpyAsync(&all_num_kv, all_num_kv_d, sizeof(uint64_t),
                  cudaMemcpyDeviceToHost, stream[0]);

  size_t max_num_data_block;
  cudaMemcpyAsync(&max_num_data_block, max_num_data_block_d, sizeof(size_t),
                  cudaMemcpyDeviceToHost, stream[1]);

  uint32_t* all_restarts_d;
  cudaMallocAsync(&all_restarts_d,
                  num_file * max_num_data_block * sizeof(uint32_t), stream[2]);

  GPUBlockHandle* index_blocks_d;  // 索引块
  cudaMallocAsync(&index_blocks_d,
                  num_file * max_num_data_block * sizeof(GPUBlockHandle),
                  stream[0]);

  GPUBlockHandle* footers_d;
  cudaMallocAsync(&footers_d, sizeof(GPUBlockHandle) * num_file, stream[1]);

  uint32_t* global_count;
  cudaMallocAsync(&global_count, all_num_kv * sizeof(uint32_t), stream[2]);
  cudaMemsetAsync(global_count, 0, all_num_kv * sizeof(uint32_t), stream[2]);

  GPUKeyValue* key_value_d;
  cudaMallocAsync((void**)&key_value_d, all_num_kv * sizeof(GPUKeyValue),
                  stream[0]);

  for (size_t i = 0; i < 3; i++) {
    CHECK(cudaStreamSynchronize(stream[i]));
  }

  // 准备执行核函数
  dim3 block(num_file);
  dim3 grid(max_num_data_block);

  auto start_time = std::chrono::high_resolution_clock::now();

  DecodeFootersKernel<<<1, block, 0, stream[0]>>>(inputFiles_d, footers_d);

  ComputeRestartsKernel<<<grid, block, 0, stream[0]>>>(
      inputFiles_d, footers_d, all_restarts_d, max_num_data_block);

  DecodeIndexBlocksKernel<<<grid, block, 0, stream[0]>>>(
      inputFiles_d, all_restarts_d, footers_d, index_blocks_d,
      max_num_data_block);

  DecodeDataBlocksKernel<<<grid, block, 0, stream[0]>>>(
      inputFiles_d, global_count, index_blocks_d, key_value_d,
      max_num_data_block);

  CHECK(cudaStreamSynchronize(stream[0]));

  GPUSort(key_value_d, all_num_kv, sorted_size);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);

  gpu_stats.gpu_all_micros += duration.count();

  // 释放资源
  cudaFree(all_restarts_d);
  cudaFree(index_blocks_d);
  cudaFree(footers_d);
  cudaFree(global_count);
  cudaFree(max_num_data_block_d);
  cudaFree(all_num_kv_d);

  return key_value_d;
}

}  // namespace ROCKSDB_NAMESPACE
