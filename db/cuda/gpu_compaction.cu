#include <fcntl.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>

#include <csignal>

#include "gpu_compaction.cuh"

namespace ROCKSDB_NAMESPACE {

void MallocInputFiles(InputFile** input_files_d, size_t num_file) {
  cudaMalloc(input_files_d, num_file * sizeof(InputFile));
}

void AddInputFile(size_t level, const std::string& filename, FileMetaData* meta,
                  InputFile* input_file_d) {
  int fd = open(filename.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0664);

  CUfileDescr_t descr;
  descr.handle.fd = fd;
  descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

  CUfileHandle_t handle;
  cuFileHandleRegister(&handle, &descr);

  size_t file_size = meta->fd.file_size;
  char* file_d;
  cudaMalloc((void**)&file_d, file_size);

  cuFileBufRegister(file_d, file_size, 0);

  cuFileRead(handle, file_d, file_size, 0, 0);

  cuFileHandleDeregister(handle);
  close(fd);

  SetInputFile<<<1, 1>>>(input_file_d, level, file_d, file_size,
                         meta->fd.GetNumber(), meta->num_data_blocks,
                         meta->num_entries);
}

__global__ void PrepareOutputKernel(SSTableInfo* info_d, size_t info_size,
                                    GPUKeyValue* result_d, char* largest_key_d,
                                    char* smallest_key_d) {
  uint32_t tid = threadIdx.x;
  if (tid >= info_size) return;

  size_t curr_pos_first_key = 0;
  for (uint32_t i = 0; i < tid; ++i) {
    curr_pos_first_key += info_d[i].total_num_kv;
  }

  size_t curr_pos_last_key = curr_pos_first_key + info_d[tid].total_num_kv - 1;

  memcpy(largest_key_d + tid * (keySize_ + 8), result_d[curr_pos_last_key].key,
         keySize_ + 8);
  memcpy(smallest_key_d + tid * (keySize_ + 8),
         result_d[curr_pos_first_key].key, keySize_ + 8);
}

void InstallOutput(SSTableInfo* info, size_t info_size,
                   GPUKeyValue* key_value_d_tmp, char* largest_key,
                   char* smallest_key, uint64_t* largest_seqno,
                   uint64_t* smallest_seqno, cudaStream_t* stream) {
  size_t current_num_kv = 0;

  GPUKeyValue max_key;
  GPUKeyValue min_key;
  GPUKeyValue max_element;
  GPUKeyValue min_element;
  for (size_t i = 0; i < info_size; ++i) {
    cudaMemcpyAsync(&max_key,
                    key_value_d_tmp + current_num_kv + info[i].total_num_kv - 1,
                    sizeof(GPUKeyValue), cudaMemcpyDeviceToHost, stream[0]);
    cudaMemcpyAsync(&min_key, key_value_d_tmp + current_num_kv,
                    sizeof(GPUKeyValue), cudaMemcpyDeviceToHost, stream[0]);

    memcpy(largest_key + i * (keySize_ + 8), max_key.key, keySize_ + 8);
    memcpy(smallest_key + i * (keySize_ + 8), min_key.key, keySize_ + 8);

    thrust::sort(thrust::device, key_value_d_tmp + current_num_kv,
                 key_value_d_tmp + current_num_kv + info[i].total_num_kv,
                 MaxSequence());

    cudaMemcpyAsync(&max_element,
                    key_value_d_tmp + current_num_kv + info[i].total_num_kv - 1,
                    sizeof(GPUKeyValue), cudaMemcpyDeviceToHost, stream[1]);
    cudaMemcpyAsync(&min_element, key_value_d_tmp + current_num_kv,
                    sizeof(GPUKeyValue), cudaMemcpyDeviceToHost, stream[1]);

    largest_seqno[i] = max_element.sequence;
    smallest_seqno[i] = min_element.sequence;

    current_num_kv += info[i].total_num_kv;
  }
}

void ReadFile(std::string filename) {
  void* devPtr = nullptr;
  const size_t size = 128 * 1024;
  CUfileHandle_t cf_handle;

  int fd = open(filename.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0664);

  CUfileDescr_t cf_descr;
  memset((void*)&cf_descr, 0, sizeof(CUfileDescr_t));
  cf_descr.handle.fd = fd;
  cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

  cuFileHandleRegister(&cf_handle, &cf_descr);

  cudaMalloc(&devPtr, size);
  cudaMemset((void*)(devPtr), 0, size);
  cudaStreamSynchronize(nullptr);

  cuFileBufRegister(devPtr, size, 0);

  cuFileWrite(cf_handle, devPtr, size, 0, 0);

  cuFileBufDeregister(devPtr);

  cudaFree(devPtr);

  cuFileHandleDeregister(cf_handle);
  close(fd);
}

void CreateStream(cudaStream_t* stream, size_t stream_size) {
  for (size_t i = 0; i < stream_size; ++i) {
    cudaStreamCreate(&stream[i]);
  }
}

void DestroyStream(cudaStream_t* stream, size_t stream_size) {
  for (size_t i = 0; i < stream_size; ++i) {
    cudaStreamDestroy(stream[i]);
  }
}

GPUKeyValue* DecodeAndSort(size_t num_inputs, InputFile* inputFiles_d,
                           size_t num_kv_data_block, size_t& sorted_size,
                           cudaStream_t* stream) {
  // 记录整个解析SSTable的时间
  //  auto start_time = std::chrono::high_resolution_clock::now();

  GPUKeyValue* result_d = GetAndSort(num_inputs, inputFiles_d,
                                     num_kv_data_block, sorted_size, stream);

  //  auto end_time = std::chrono::high_resolution_clock::now();
  //  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
  //      end_time - start_time);
  //  std::cout << "GPU decode and sort time: " << duration.count() << " us"
  //            << std::endl;

  return result_d;
}

size_t ComputeNumKVDataBlock() {
  size_t num_kv_data_block = DataBlockSize / key_value_size;
  size_t remaining_size = DataBlockSize % key_value_size;  // 留给重启点的大小

  size_t num_restarts = num_kv_data_block / 16 + 1;
  size_t size_restarts = (num_restarts + 1) * 4 + 1 + encoded_value_size + 4;

  if (size_restarts > remaining_size) {
    // 留给重启点的大小不足，说明 num_kv_data_block 给大了
    num_kv_data_block--;
  }

  if (num_kv_data_block * key_value_size < BlockSizeDeviationLimit) {
    num_kv_data_block++;
  }

  return num_kv_data_block;
}

void EncodePrepare(size_t total_num_kv, std::vector<SSTableInfo>& info,
                   size_t num_kv_data_block) {
  uint32_t num_restarts = num_kv_data_block / BlockRestartInterval + 1;

  size_t size_complete_data_block = key_value_size * num_kv_data_block +
                                    (num_restarts + 1) * sizeof(uint32_t) + 5;

  auto max_num_data_block =
      static_cast<size_t>((MaxOutputFileSize - BlockSolidSize - 9) /
                          (size_complete_data_block + 32));

  size_t max_num_kv = max_num_data_block * num_kv_data_block;

  if (total_num_kv <= max_num_kv) {
    size_t num_data_block = total_num_kv / num_kv_data_block;
    size_t num_kv_last_data_block = total_num_kv % num_kv_data_block;
    if (num_kv_last_data_block > 0) {
      num_data_block++;
    }
    num_restarts = num_kv_last_data_block / BlockRestartInterval + 1;
    info.emplace_back(num_data_block, num_kv_last_data_block, num_restarts,
                      total_num_kv);
  } else {
    info.emplace_back(max_num_data_block, 0, num_restarts, max_num_kv);

    size_t remaining_num_kv = total_num_kv - max_num_kv;
    while (remaining_num_kv > max_num_kv) {
      info.emplace_back(max_num_data_block, 0, num_restarts, max_num_kv);
      remaining_num_kv -= max_num_kv;
    }

    size_t num_data_block = remaining_num_kv / num_kv_data_block;
    size_t num_kv_last_data_block = remaining_num_kv % num_kv_data_block;
    if (num_kv_last_data_block > 0) {
      num_data_block++;
    }
    num_restarts = num_kv_last_data_block / BlockRestartInterval + 1;
    info.emplace_back(num_data_block, num_kv_last_data_block, num_restarts,
                      remaining_num_kv);
  }
}

char* EncodeSSTable(const std::vector<GPUKeyValue>& keyValues,
                    InputFile* inputFiles_d, SSTableInfo& info,
                    size_t num_kv_data_block, FileMetaData& meta,
                    std::shared_ptr<TableBuilderOptions>& tboptions,
                    TableProperties& tp) {
  auto start_time = std::chrono::high_resolution_clock::now();

  char* result = BuildSSTable(keyValues.data(), inputFiles_d, info,
                              num_kv_data_block, meta, tboptions, tp);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  //  std::cout << "GPU Encode SSTable time: " << duration.count() << " us\n";
  //  printf("---------------------\n");

  return result;
}

void EncodeSSTables(
    GPUKeyValue* key_values_d, InputFile* input_files_d,
    std::vector<SSTableInfo>& infos, std::vector<FileMetaData>& metas,
    std::vector<std::shared_ptr<WritableFileWriter>>& file_writes,
    std::vector<std::shared_ptr<TableBuilderOptions>>& tbs,
    std::vector<TableProperties>& tps, size_t num_kv_data_block,
    cudaStream_t* stream) {
  auto start_time = std::chrono::high_resolution_clock::now();

  BuildSSTables(key_values_d, input_files_d, infos, metas, file_writes, tbs,
                tps, num_kv_data_block, stream);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  //  std::cout << "GPU Encode SSTable time: " << duration.count() << " us\n";
  //  printf("---------------------\n");
}

void ReleaseDevPtr(char** blocks_buffer_d) { cudaFreeHost(*blocks_buffer_d); }

void ReleaseSource(GPUKeyValue** key_value_h) { cudaFreeHost(*key_value_h); }

// 重点
void ReleaseSource(InputFile* inputFiles_d, GPUKeyValue* key_value_d,
                   size_t num_inputs) {
  cudaFree(key_value_d);

  InputFile inputFiles_h;

  for (size_t i = 0; i < num_inputs; ++i) {
    cudaMemcpy(&inputFiles_h, inputFiles_d + i, sizeof(InputFile),
               cudaMemcpyDeviceToHost);

    cuFileBufDeregister(inputFiles_h.file);
    cudaFree(inputFiles_h.file);
  }

  cudaFree(inputFiles_d);
}

}  // namespace ROCKSDB_NAMESPACE