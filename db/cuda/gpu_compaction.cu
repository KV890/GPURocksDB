#include "gpu_compaction.cuh"

namespace ROCKSDB_NAMESPACE {

GPUKeyValue* DecodeAndSort(const InputFile* inputFiles, size_t num_inputs,
                           InputFile** inputFiles_d, size_t num_kv_data_block,
                           GPUKeyValue** result_h, size_t& sorted_size) {
  //  cudaEvent_t start, end;
  //  cudaEventCreate(&start);
  //  cudaEventCreate(&end);
  //  float elapsed_time;

  // 记录整个解析SSTable的时间
  //  auto start_time = std::chrono::high_resolution_clock::now();
  GPUKeyValue* result_d = GetAndSort(inputFiles, num_inputs, inputFiles_d,
                                     num_kv_data_block, sorted_size);

  cudaHostAlloc((void**)result_h, sorted_size * sizeof(GPUKeyValue),
                cudaHostAllocDefault);
  //  *result_h = new GPUKeyValue[sorted_size];
  //  cudaEventRecord(start, nullptr);
  cudaMemcpy(*result_h, result_d, sorted_size * sizeof(GPUKeyValue),
             cudaMemcpyDeviceToHost);

  //  cudaEventRecord(end, nullptr);
  //  cudaEventSynchronize(end);
  //  cudaEventElapsedTime(&elapsed_time, start, end);
  //  gpu_stats.transmission_time += elapsed_time;

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
    CompactionJob* compaction_job, const Compaction* compact,
    GPUKeyValue* key_values_d, InputFile* input_files_d,
    std::vector<SSTableInfo>& infos, std::vector<FileMetaData>& metas,
    std::vector<std::shared_ptr<WritableFileWriter>>& file_writes,
    std::vector<std::shared_ptr<TableBuilderOptions>>& tbs,
    std::vector<TableProperties>& tps, size_t num_kv_data_block) {
  auto start_time = std::chrono::high_resolution_clock::now();

  BuildSSTables(compaction_job, compact, key_values_d, input_files_d, infos,
                metas, file_writes, tbs, tps, num_kv_data_block);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  //  std::cout << "GPU Encode SSTable time: " << duration.count() << " us\n";
  //  printf("---------------------\n");
}

void ReleaseDevPtr(char** blocks_buffer_d) { cudaFreeHost(*blocks_buffer_d); }

void ReleaseSource(GPUKeyValue** key_value_h) { cudaFreeHost(*key_value_h); }

void ReleaseSource(InputFile** inputFiles_d, size_t num_inputs) {
  for (size_t i = 0; i < num_inputs; ++i) {
    const char* file_host;
    cudaMemcpy(&file_host, &((*inputFiles_d)[i].file), sizeof(char*),
               cudaMemcpyDeviceToHost);
    cudaFree((void*)file_host);
  }
  cudaFree(inputFiles_d);
}

}  // namespace ROCKSDB_NAMESPACE