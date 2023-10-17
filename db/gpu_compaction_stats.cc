//
// Created by jxx on 7/7/23.
//

#include "gpu_compaction_stats.h"

#include <cufile.h>

namespace ROCKSDB_NAMESPACE {

GPUCompactionStats gpu_stats;

GPUCompactionStats::GPUCompactionStats()
    : cpu_all_micros(0),
      gpu_total_input_bytes(0),
      gpu_total_output_bytes(0),
      gpu_all_micros(0),
      gpu_compaction_count(0),
      flush_time(0),
      gpu_total_sort_time(0),
      transmission_time(0) {}

void GPUCompactionStats::PrintStats() const {
  std::cout << "CPU 总compaction时间: " << cpu_all_micros << " us, "
            << static_cast<double>(static_cast<double>(cpu_all_micros) /
                                   1000000)
            << " sec" << std::endl;

  std::cout << "-------------GPU-------------" << std::endl;

  std::cout << "GPU compaction次数: " << gpu_compaction_count << std::endl;

  std::cout << "GPU 总compaction时间: " << gpu_all_micros << " us, "
            << static_cast<double>(static_cast<double>(gpu_all_micros) /
                                   1000000)
            << " sec" << std::endl;

  std::cout << "总Flush时间: " << flush_time << " us, "
            << static_cast<double>(static_cast<double>(flush_time) / 1000000)
            << " sec" << std::endl;

  std::cout << "GPU sort总时间: " << gpu_total_sort_time << " us, "
            << static_cast<double>(static_cast<double>(gpu_total_sort_time) /
                                   1000000)
            << " sec" << std::endl;

  std::cout << "GPU compaction总读入量: " << gpu_total_input_bytes << " bytes, "
            << static_cast<double>(static_cast<double>(gpu_total_input_bytes) /
                                   (1024 * 1024 * 1024))
            << " GB" << std::endl;
  std::cout << "GPU compaction总写入量: " << gpu_total_output_bytes
            << " bytes, "
            << static_cast<double>(static_cast<double>(gpu_total_output_bytes) /
                                   (1024 * 1024 * 1024))
            << " GB" << std::endl;

  std::cout << "GPU compaction 总吞吐量: "
            << static_cast<double>(
                   static_cast<double>(
                       (static_cast<double>(gpu_total_input_bytes) +
                        static_cast<double>(gpu_total_output_bytes)) /
                       (1024 * 1024)) /
                   (static_cast<double>(gpu_all_micros) / 1000000.0))
            << " MB/s" << std::endl;

  std::cout << "Transmission time between GPU and CPU: " << transmission_time
            << " ms, " << transmission_time / 1000 << " sec" << std::endl;
  std::cout << "The ratio of transfer time to total compaction time: "
            << (transmission_time / 1000) /
                   (static_cast<double>(static_cast<double>(cpu_all_micros) /
                                        1000000))
            << std::endl;

  std::cout << "============================" << std::endl;
}

void GPUCompactionStats::ResetStats() {
  cpu_all_micros = 0;

  gpu_total_input_bytes = 0;
  gpu_total_output_bytes = 0;
  gpu_all_micros = 0;
  gpu_compaction_count = 0;

  flush_time = 0;
  gpu_total_sort_time = 0;
  transmission_time = 0;
}

void GPUCompactionStats::OpenCuFileDriver() { cuFileDriverOpen(); }

void GPUCompactionStats::CloseCuFileDriver() { cuFileDriverClose(); }

}  // namespace ROCKSDB_NAMESPACE
