//
// Created by jxx on 7/7/23.
//
#pragma once

#include <cstdint>
#include <iostream>
#include <atomic>

#include "rocksdb/rocksdb_namespace.h"

namespace ROCKSDB_NAMESPACE {

class GPUCompactionStats {
 public:
  GPUCompactionStats();

  void PrintStats() const;

  void ResetStats();

  void OpenCuFileDriver();

  void CloseCuFileDriver();

  uint64_t gpu_total_input_bytes;
  uint64_t gpu_total_output_bytes;
  uint64_t gpu_all_micros;
  uint64_t gpu_compaction_count;

  uint64_t compaction_time;
  std::atomic<uint64_t> compaction_io_time;
  uint64_t flush_time;
  uint64_t flush_io_time;

  uint64_t transmission_and_malloc_time;

  uint64_t gpu_total_sort_time;

  uint64_t max_num_file = 0;
  uint64_t min_num_file = 0xffffffffffffffff;
  uint64_t total_num_file = 0;

  uint64_t max_file_size = 0;
  uint64_t min_file_size = 0xffffffffffffffff;
  uint64_t total_file_size = 0;

  uint64_t max_io_time = 0;
  uint64_t min_io_time = 0xffffffffffffffff;
  uint64_t total_io_time = 0;
};

extern GPUCompactionStats gpu_stats;

}  // namespace ROCKSDB_NAMESPACE
