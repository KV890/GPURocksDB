//
// Created by jxx on 7/7/23.
//

#include "gpu_compaction_stats.h"

#include <cufile.h>

namespace ROCKSDB_NAMESPACE {

GPUCompactionStats gpu_stats;

GPUCompactionStats::GPUCompactionStats() {}

void GPUCompactionStats::PrintStats() const {
  std::cout << "-------------Stats-------------" << std::endl;

  std::cout << "Compaction 时间: " << compaction_time << " us, "
            << static_cast<double>(static_cast<double>(compaction_time) /
                                   1000000)
            << " sec" << std::endl;

  std::cout << "Compaction 计算时间: " << compaction_time - compaction_io_time
            << " us, "
            << static_cast<double>(
                   static_cast<double>(compaction_time - compaction_io_time) /
                   1000000)
            << " sec" << std::endl;

  std::cout << "Compaction I/O时间: " << compaction_io_time << " us, "
            << static_cast<double>(static_cast<double>(compaction_io_time) /
                                   1000000)
            << " sec" << std::endl;

  /*std::cout << "Flush时间: " << flush_time << " us, "
            << static_cast<double>(static_cast<double>(flush_time) / 1000000)
            << " sec" << std::endl;

  std::cout << "Flush 计算时间: " << flush_time - flush_io_time << " us, "
            << static_cast<double>(
                   static_cast<double>(flush_time - flush_io_time) / 1000000)
            << " sec" << std::endl;

  std::cout << "Flush I/O时间: " << flush_io_time << " us, "
            << static_cast<double>(static_cast<double>(flush_io_time) / 1000000)
            << " sec" << std::endl;

  std::cout << "总计算时间: "
            << compaction_time - compaction_io_time + flush_time - flush_io_time
            << " us, "
            << static_cast<double>(
                   static_cast<double>(compaction_time - compaction_io_time +
                                       flush_time - flush_io_time) /
                   1000000)
            << " sec" << std::endl;

  std::cout << "总I/O时间: " << compaction_io_time + flush_io_time << " us, "
            << static_cast<double>(
                   static_cast<double>(compaction_io_time + flush_io_time) /
                   1000000)
            << " sec" << std::endl;

  std::cout << "总计算时间/总I/O时间: "
            << static_cast<double>(
                   static_cast<double>(compaction_time - compaction_io_time +
                                       flush_time - flush_io_time) /
                   static_cast<double>(compaction_io_time + flush_io_time))
            << std::endl;*/

  /*std::cout << "GPU Compaction次数: " << gpu_compaction_count << std::endl;

  std::cout << "GPU Compaction总时间: " << gpu_all_micros << " us, "
            << static_cast<double>(static_cast<double>(gpu_all_micros) /
                                   1000000)
            << " sec" << std::endl;

  std::cout << "GPU Compaction总读入量: " << gpu_total_input_bytes << " bytes, "
            << static_cast<double>(static_cast<double>(gpu_total_input_bytes) /
                                   (1024 * 1024 * 1024))
            << " GB" << std::endl;
  std::cout << "GPU Compaction总写入量: " << gpu_total_output_bytes
            << " bytes, "
            << static_cast<double>(static_cast<double>(gpu_total_output_bytes) /
                                   (1024 * 1024 * 1024))
            << " GB" << std::endl;

  std::cout << "GPU Compaction 总吞吐量: "
            << static_cast<double>(
                   static_cast<double>(
                       (static_cast<double>(gpu_total_input_bytes) +
                        static_cast<double>(gpu_total_output_bytes)) /
                       (1024 * 1024)) /
                   (static_cast<double>(gpu_all_micros) / 1000000.0))
            << " MB/s" << std::endl;*/

  /*std::cout << "GPU Sort time: " << gpu_total_sort_time << " us, "
            << static_cast<double>(static_cast<double>(gpu_total_sort_time) /
                                   1000000.0)
            << " sec" << std::endl;*/

  std::cout << "============================" << std::endl;
}

void GPUCompactionStats::ResetStats() {
  compaction_time = 0;

  //  gpu_total_input_bytes = 0;
  //  gpu_total_output_bytes = 0;
  //  gpu_all_micros = 0;
  //  gpu_compaction_count = 0;
}

void GPUCompactionStats::OpenCuFileDriver() { cuFileDriverOpen(); }

void GPUCompactionStats::CloseCuFileDriver() { cuFileDriverClose(); }

}  // namespace ROCKSDB_NAMESPACE
