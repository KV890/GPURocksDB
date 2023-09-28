//
// Created by jxx on 4/20/23.
//

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "gpu_transform_sort.cuh"

namespace ROCKSDB_NAMESPACE {

void GPUSort(GPUKeyValue* key_value_d, size_t num_element,
             size_t& sorted_size) {
  thrust::sort(thrust::device, key_value_d, key_value_d + num_element);

  sorted_size =
      thrust::unique(thrust::device, key_value_d, key_value_d + num_element) -
      key_value_d;
}

}  // namespace ROCKSDB_NAMESPACE