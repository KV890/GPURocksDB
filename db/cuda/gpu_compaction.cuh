#pragma once

#include <driver_types.h>
#include <snappy.h>
#include <sys/time.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "db/compaction/compaction.h"
#include "db/compaction/compaction_job.h"
#include "gpu_decode_block.cuh"
#include "gpu_encode_block.cuh"
#include "gpu_table_properties.cuh"
#include "gpu_transform_sort.cuh"

namespace ROCKSDB_NAMESPACE {

class CompactionJob;

void CreateStream(cudaStream_t* stream, size_t stream_size);
void DestroyStream(cudaStream_t* stream, size_t stream_size);

/**
 * 解码
 * @param inputFiles
 * @param num_inputs
 * @param inputFiles_d_ptr InputFile** 是因为要对外部的inputFiles_d进行修改
 *
 * @return
 */
GPUKeyValue* DecodeAndSort(const InputFile* inputFiles, size_t num_inputs,
                           InputFile** inputFiles_d, size_t num_kv_data_block,
                           GPUKeyValue** result_h, size_t& sorted_size,
                           cudaStream_t* stream);

/**
 * 计算一个数据块中键值对的数量
 *
 * @return
 */
size_t ComputeNumKVDataBlock();

/**
 *
 * @param total_num_kv
 * @param info
 * @param num_kv_data_block
 */
void EncodePrepare(size_t total_num_kv, std::vector<SSTableInfo>& info,
                   size_t num_kv_data_block);

/**
 * GPU 编码单SSTable
 *
 * @param keyValues
 * @param inputFiles_d
 * @param info
 * @param num_inputs
 * @param num_kv_data_block
 * @param meta
 * @param tboptions
 * @param tp
 * @return
 */
char* EncodeSSTable(const std::vector<GPUKeyValue>& keyValues,
                    InputFile* inputFiles_d, SSTableInfo& info,
                    size_t num_kv_data_block, FileMetaData& meta,
                    std::shared_ptr<TableBuilderOptions>& tboptions,
                    TableProperties& tp);

/**
 * GPU编码多SSTable
 *
 * @param key_values_d
 * @param input_files_d
 * @param infos
 * @param metas
 * @param file_writes
 * @param tbs
 * @param tps
 * @param num_kv_data_block
 */
void EncodeSSTables(
    CompactionJob* compaction_job, const Compaction* compact,
    GPUKeyValue* key_values_d, InputFile* input_files_d,
    std::vector<SSTableInfo>& infos, std::vector<FileMetaData>& metas,
    std::vector<std::shared_ptr<WritableFileWriter>>& file_writes,
    std::vector<std::shared_ptr<TableBuilderOptions>>& tbs,
    std::vector<TableProperties>& tps, size_t num_kv_data_block,
    cudaStream_t* stream);

/**
 * 释放资源
 *
 * @param blocks_buffer_d
 */
void ReleaseDevPtr(char** blocks_buffer_d);

void ReleaseSource(GPUKeyValue** key_value_h);

/**
 *
 * @param inputFiles_d
 * @param num_inputs
 */
void ReleaseSource(InputFile** inputFiles_d, size_t num_inputs);

}  // namespace ROCKSDB_NAMESPACE