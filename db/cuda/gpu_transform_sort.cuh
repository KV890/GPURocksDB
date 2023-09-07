//
// Created by jxx on 4/20/23.
//
#pragma once

/**
 * 日志
 *
 * 排序去重的修改是随着解码的修改而修改的
 *
 * 4月份基本上没有太大改动
 *
 * 5月，在完成键值分离后，设计了一个新的结构体以适应于键值分离
 *
 * 5月11日，由于采用了GPU并行解析多个SSTable，会使得新旧key错乱，
 * 因此在重载运算符函数中增加一个sequence的判断，使得判断最多有4次
 *
 * 6.3
 *  解决学长现存的问题：key大小是有限且固定的
 *
 * 6.17
 *  尝试用不同的排序算法实现
 *  由于key集合中部分key是有序的，所以采取归并排序算法
 *
 *  每个线程处理每个集合中的一个key，关键点就在怎么找到这些线程中的最小key
 *
 *  由于每个集合中的元素肯定超过1024，所以用网格存储集合
 *
 *  集合0: 1000, 集合1: 1100, 集合2: 900, 集合3: 1000
 *
 *  last_size[] = {0, 1000, 2100, 3000}
 *  key_value[last_size[file_idx] + kv_idx] -- 定位到第
 * file_idx个文件的第kv_idx个key
 *
 *  if kv_idx >= all_size[file_idx] ? // 如果kv_idx所指向的位置大于某个集合大小
 *  结果是不对其进行比较
 *  过程就是会有很多判断语句，用一种不需要判断的方法，如果某个集合大小小于最大集合大小，
 *  则扩充该集合大小到最大集合大小，以保持相同大小，用 '\x7f' 填充
 *
 *  如果所有集合大小相同，即通过key_value[last_size[file_idx] + kv_idx]
 *  可以获得所有集合的所有key，并且不需要在核函数中进行判断
 *
 *  kv_idx指向每个集合中第 kv_idx 的位置
 *  通过key_value[last_size[file_idx] + kv_idx] 这个语句就可以获得所有集合中
 *  第 kv_idx 的位置的key，并且每个线程块中的一个线程拥有某个集合中的一个key
 *
 *  接下来就是要对这些线程拥有的key进行比较，按字典顺序选择一个最小的key(难点和重点)
 *  比较方式是要按照字典顺序进行排序
 *  目前先用简单的实现方式，在比较的时候进行for循环，找到最小key
 *
 *  或者另一种方式：转换一下？
 *
 *  找到最小key后，将其放到结果数组中，结果数组中的index需要用到 atomicAdd
 *
 */

#include <vector>

#include "gpu_format.cuh"

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#define CHECK(call)                                           \
  do {                                                        \
    cudaError_t error = call;                                 \
    if (error != cudaSuccess) {                               \
      printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(error));                      \
      exit(1);                                                \
    }                                                         \
  } while (0)

namespace ROCKSDB_NAMESPACE {

class GPUKeyValue {
 public:
  __host__ __device__ GPUKeyValue()
      : key{}, valuePtr(0), sequence(0), type{}, file_number(0) {}

  __host__ __device__ GPUKeyValue(const GPUKeyValue &other)
      : key{},
        valuePtr(other.valuePtr),
        sequence(other.sequence),
        type(other.type),
        file_number(other.file_number) {
    memcpy(key, other.key, keySize_ + 8);
  }

  char key[keySize_ + 8 + 1];
  uint64_t valuePtr;

  uint64_t sequence;  // 序列号
  unsigned char type;
  uint64_t file_number;

  __host__ __device__ GPUKeyValue &operator=(const GPUKeyValue &other) {
    if (this == &other) {
      return *this;
    }

    memcpy(key, other.key, keySize_ + 8);
    valuePtr = other.valuePtr;
    sequence = other.sequence;
    type = other.type;
    file_number = other.file_number;

    return *this;
  }

  __host__ __device__ bool operator<(const GPUKeyValue &other) const {
    const auto *p1 = (const unsigned char *)key;
    const auto *p2 = (const unsigned char *)other.key;

    for (size_t i = 0; i < keySize_; ++i) {
      if (static_cast<int>(p1[i]) != static_cast<int>(p2[i])) {
        return static_cast<int>(p1[i]) < static_cast<int>(p2[i]);
      }
    }

    return sequence > other.sequence;
  }

  __host__ __device__ bool operator>(const GPUKeyValue &other) const {
    const auto *p1 = (const unsigned char *)key;
    const auto *p2 = (const unsigned char *)other.key;

    for (size_t i = 0; i < keySize_; ++i) {
      if (static_cast<int>(p1[i]) != static_cast<int>(p2[i])) {
        return static_cast<int>(p1[i]) > static_cast<int>(p2[i]);
      }
    }

    return sequence < other.sequence;
  }

  __host__ __device__ bool operator==(const GPUKeyValue &other) const {
    const auto *p1 = (const unsigned char *)key;
    const auto *p2 = (const unsigned char *)other.key;

    for (size_t i = 0; i < keySize_; ++i) {
      if (static_cast<int>(p1[i]) != static_cast<int>(p2[i])) {
        return false;
      }
    }

    return true;
  }
};

//__global__ void GPUTransform(GPUKeyValue *key_value_d, size_t num_element);

void GPUSort(GPUKeyValue *key_value_d, size_t num_element, size_t &sorted_size);

__global__ void GPURadixSortAssertion(GPUKeyValue *key_value_d,
                                      size_t num_element);

__global__ void HistogramKernel(GPUKeyValue *key_value_d, size_t num_element,
                                uint32_t bit, uint32_t *histogram_d);

__global__ void ScanKernel(uint32_t *histogram_d, uint32_t *prefix_sum,
                           size_t size);

__global__ void ReorderKernel(GPUKeyValue *key_value_d, size_t num_element,
                              uint32_t bit, uint32_t *prefix_sum,
                              GPUKeyValue *output_d);

void GPURadixSort(GPUKeyValue *key_value_d, size_t num_element,
                  size_t &sorted_size);

void CPUCountingSort(GPUKeyValue *key_value_h, int idx, size_t num_element);

void CPURadixSort(GPUKeyValue *key_value_h, size_t num_element);

}  // namespace ROCKSDB_NAMESPACE
