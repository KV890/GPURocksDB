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

//  uint64_t gpu_total_input_bytes = 0;
//  uint64_t gpu_total_output_bytes = 0;
//  uint64_t gpu_all_micros = 0;
//  uint64_t gpu_compaction_count = 0;

  uint64_t compaction_time = 0;
  std::atomic_uint64_t compaction_io_time = 0;
//  uint64_t flush_time = 0;
//  uint64_t flush_io_time = 0;

//  uint64_t transmission_and_malloc_time = 0;
//  uint64_t gpu_total_sort_time = 0;
};

extern GPUCompactionStats gpu_stats;

}  // namespace ROCKSDB_NAMESPACE
