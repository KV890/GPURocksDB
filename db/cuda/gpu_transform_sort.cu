//
// Created by jxx on 4/20/23.
//

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "gpu_transform_sort.cuh"

namespace ROCKSDB_NAMESPACE {

/*__global__ void GPUTransform(GPUKeyValue* key_value_d, size_t num_element) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= num_element) return;

  for (int k = 0; k < keySize_; k += 8) {
    uint64_t result = 0;
    for (int i = 0; i < 8; ++i) {
      char c;
      if (k + i < keySize_) {
        c = key_value_d[idx].key[k + i];
      } else {
        c = '\0';
      }
      result |= (static_cast<uint64_t>(static_cast<unsigned char>(c))
                 << ((7 - i) * 8));
    }
    key_value_d[idx].key_int[k / 8] = result;
  }
}*/

void GPUSort(GPUKeyValue* key_value_d, size_t num_element,
             size_t& sorted_size) {
  /*size_t blockSize = 256;
  size_t gridSize = (num_element + blockSize - 1) / blockSize;
  GPUTransform<<<gridSize, blockSize>>>(key_value_d, num_element);*/

  thrust::sort(thrust::device, key_value_d, key_value_d + num_element);

  sorted_size =
      thrust::unique(thrust::device, key_value_d, key_value_d + num_element) -
      key_value_d;
}

__global__ void GPURadixSortAssertion(GPUKeyValue* key_value_d,
                                      size_t num_element) {
  for (size_t i = 1; i < num_element; ++i) {
    if (key_value_d[i - 1] > key_value_d[i]) {
      printf("Sort error\n");
    }
  }
}

__global__ void HistogramKernel(GPUKeyValue* key_value_d, size_t num_element,
                                uint32_t bit, uint32_t* histogram_d) {
  uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= num_element) return;

  atomicAdd(&histogram_d[static_cast<uint32_t>(
                static_cast<unsigned char>(key_value_d[tid].key[bit]))],
            1);
}

// inclusive prefix sum
__global__ void ScanKernel(uint32_t* histogram_d, uint32_t* prefix_sum,
                           size_t size) {
  extern __shared__ unsigned int temp[];

  int tid = threadIdx.x;
  int pout = 0, pin = 1;

  temp[size * pout + tid] = histogram_d[tid];
  __syncthreads();

  for (int offset = 1; offset < size; offset <<= 1) {
    pout = 1 - pout;  // // 交换输入输出缓冲区
    pin = 1 - pout;

    if (tid >= offset) {
      temp[pout * size + tid] =
          temp[pin * size + tid] + temp[pin * size + tid - offset];
    } else {
      temp[pout * size + tid] = temp[pin * size + tid];
    }
    __syncthreads();
  }

  prefix_sum[tid] = temp[pout * size + tid];
}

__global__ void ReorderKernel(GPUKeyValue* key_value_d, size_t num_element,
                              uint32_t bit, uint32_t* prefix_sum,
                              GPUKeyValue* output_d) {
  for (int i = num_element - 1; i >= 0; --i) {
    output_d[--prefix_sum[static_cast<uint32_t>(
        static_cast<unsigned char>(key_value_d[i].key[bit]))]] = key_value_d[i];
  }
}

void GPURadixSort(GPUKeyValue* key_value_d, size_t num_element,
                  size_t& sorted_size) {
  uint32_t* histogram_d;
  uint32_t* prefix_sum;
  GPUKeyValue* output_d;
  cudaMalloc(&histogram_d, sizeof(int) * 256);
  cudaMalloc(&prefix_sum, sizeof(int) * 256);
  cudaMalloc(&output_d, sizeof(GPUKeyValue) * num_element);

  dim3 blockSize(256);
  dim3 gridSize((num_element + blockSize.x - 1) / blockSize.x);

  for (int bit = keySize_ - 1; bit >= 0; --bit) {
    cudaMemset(histogram_d, 0, 256 * sizeof(int));

    HistogramKernel<<<gridSize, blockSize>>>(key_value_d, num_element, bit,
                                             histogram_d);

    ScanKernel<<<1, blockSize, 2 * 1024>>>(histogram_d, prefix_sum, 256);

    ReorderKernel<<<1, 1>>>(key_value_d, num_element, bit, prefix_sum,
                            output_d);

    thrust::swap(key_value_d, output_d);
  }

  cudaFree(output_d);
  cudaFree(prefix_sum);
  cudaFree(histogram_d);

  sorted_size =
      thrust::unique(thrust::device, key_value_d, key_value_d + num_element) -
      key_value_d;

  //  GPURadixSortAssertion<<<1, 1>>>(key_value_d, sorted_size);
}

void CPUCountingSort(GPUKeyValue* key_value_h, int idx, size_t num_element) {
  std::vector<GPUKeyValue> output(num_element);
  std::vector<int> count(256, 0);

  for (int i = 0; i < num_element; ++i) {
    count[(int)(unsigned char)key_value_h[i].key[idx] + 1]++;
  }

  for (int i = 1; i <= 256; i++) {
    count[i] += count[i - 1];
  }

  for (int i = (int)num_element - 1; i >= 0; i--) {
    output[--count[(int)(unsigned char)key_value_h[i].key[idx] + 1]] =
        key_value_h[i];
  }

  for (int i = 0; i < num_element; ++i) {
    key_value_h[i] = output[i];
  }
}

void CPURadixSort(GPUKeyValue* key_value_h, size_t num_element) {
  for (int i = keySize_ - 1; i >= 0; --i) {
    CPUCountingSort(key_value_h, i, num_element);
  }
}

}  // namespace ROCKSDB_NAMESPACE