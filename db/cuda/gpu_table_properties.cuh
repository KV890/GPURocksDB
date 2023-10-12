//
// Created by jxx on 4/24/23.
//
#pragma once

#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "gpu_slice.cuh"
#include "gpu_transform_sort.cuh"

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __constant__
#define __constant__
#endif

namespace ROCKSDB_NAMESPACE {

constexpr uint64_t MaxOutputFileSize = 128 << 20;
constexpr uint32_t DataBlockSize = 4096;
constexpr uint32_t BlockRestartInterval = 16;
// 在数据块的大小达到这个值之前，会继续插入键值对
constexpr uint64_t BlockSizeDeviationLimit = 3687;
constexpr uint64_t BlockSolidSize = 966;

inline std::mutex mutex_for_gpu_compaction;

// 获取CPU核心数
inline uint32_t num_threads_for_gpu = std::thread::hardware_concurrency();

inline std::vector<std::thread> thread_pool_for_gpu;

// 一个文件的数据块数量
inline __constant__ size_t num_data_block_d;
// 数据块中键值对数量
inline __constant__ size_t num_kv_data_block_d;
// 最后一个数据块键值对数量
inline __constant__ size_t num_kv_last_data_block_d;
// 重启点的数量
inline __constant__ uint32_t num_restarts_d;
// 重启点数组
inline __constant__ uint32_t const_restarts_d[32];
// 总文件数
inline __constant__ size_t num_files_d;
// 完整数据块的总大小
inline __constant__ size_t size_complete_data_block_d;

// GPU编码多SSTable专用
// 前面的文件中一个文件的KV对总数量
inline __constant__ size_t num_kv_front_file_d;
inline __constant__ size_t num_kv_last_file_d;
// 前面的文件中一个文件的预估大小
inline __constant__ size_t size_front_file_d;
// 输出文件的数量
inline __constant__ size_t num_outputs_d;
// 非最后一个文件的数据块的大小
inline __constant__ size_t data_size_d;
// 最后一个文件的数据块大小
inline __constant__ size_t data_size_last_file_d;
// 非最后一个文件索引块的大小
inline __constant__ size_t index_size_d;
// 非最后一个文件过滤块大小
inline __constant__ size_t filter_size_d;
// 最后一个文件过滤块大小
inline __constant__ size_t filter_size_last_file_d;
inline __constant__ uint32_t num_lines_d;
inline __constant__ uint32_t num_lines_last_file_d;

struct SSTableInfo {
  __host__ __device__ SSTableInfo() = default;

  __host__ __device__ SSTableInfo(size_t _num_data_block,
                                  size_t _num_kv_last_data_block,
                                  size_t _num_restarts, size_t _total_num_kv)
      : num_data_block(_num_data_block),
        num_kv_last_data_block(_num_kv_last_data_block),
        num_restarts(_num_restarts),
        total_num_kv(_total_num_kv) {}

  size_t num_data_block = 0;          // 该文件数据块数量
  size_t num_kv_last_data_block = 0;  // 该文件最后一个数据块KV对数量
  size_t num_restarts = 0;            // 该文件数据块的restarts大小
  size_t total_num_kv = 0;            // 该文件总 KV对数量
  size_t file_size = 0;               // 该文件大小
};

}  // namespace ROCKSDB_NAMESPACE