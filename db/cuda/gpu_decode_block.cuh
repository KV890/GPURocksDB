//
// Created by jxx on 4/8/23.
//
#pragma once

/**
 * 日志
 *
 * 4月中旬，完成GPU解析多文件的footer和单文件的索引块，CPU解析单文件的数据块
 * 然后，在GPU解析多文件的footer和单文件的索引块的基础上，完成GPU解析单文件的数据块
 * 在次基础上，利用CPU多线程，异步解析多文件的索引块和数据块
 *
 * 5月初，完成键值分离，决定去掉压缩
 *
 * 5月11日，完成GPU并行解析多个SSTable，解析SSTable暂告一段落
 *
 * 5月17日
 *  在编码阶段，将shared设置为0，出现了很多常量，这样可以将并行粒度从数据块降到键值对，
 *  所以写出一个每个线程处理一个键值对的核函数是必要的
 *
 * 5月18日
 *  实现了，但是效果不明显，回到以前的实现
 *
 */

#include <driver_types.h>
#include <snappy.h>

#include <cstdlib>
#include <memory>
#include <vector>

#include "db/cuda/gpu_transform_sort.cuh"
#include "db/gpu_compaction_stats.h"
#include "gpu_format.cuh"
#include "gpu_slice.cuh"
#include "gpu_table_properties.cuh"

#ifndef __global__
#define __global__
#endif

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

enum GPUCompressionType : unsigned char {
  kGPUNoCompression [[maybe_unused]] = 0x0,
  kGPUSnappyCompression = 0x1
};

/**
 * 辅助函数
 * @param block_data
 * @param block_size
 * @return
 */
__host__ __device__ inline GPUCompressionType GPUGetBlockCompressionType(
    const char* block_data, size_t block_size) {
  return static_cast<GPUCompressionType>(block_data[block_size]);
}

/**
 * 辅助函数
 * @param p
 * @param limit
 * @param shared
 * @param non_shared
 * @param value_length
 * @return
 */
__host__ __device__ inline const char* DecodeIndexKey(const char* p,
                                                      const char* limit,
                                                      uint32_t* shared,
                                                      uint32_t* non_shared,
                                                      uint32_t* value_length) {
  *value_length = 0;
  if (limit - p < 3) {
    return nullptr;
  }
  *shared = reinterpret_cast<const unsigned char*>(p)[0];
  *non_shared = reinterpret_cast<const unsigned char*>(p)[1];
  if ((*shared | *non_shared) < 128) {
    p += 2;
  } else {
    if ((p = GPUGetVarint32Ptr(p, limit, shared)) == nullptr) return nullptr;
    if ((p = GPUGetVarint32Ptr(p, limit, non_shared)) == nullptr)
      return nullptr;
  }
  return p;
}

/**
 * 辅助函数: 检查该块是否要解压
 * @param block_data 使用指针的指针对block_data重新赋值
 * @param block_size
 */
[[maybe_unused]] __host__ __device__ inline std::pair<const char*, bool>
GetUncompressedBlock(const char* block_data, size_t& block_size) {
  bool allocated = false;
  auto type = GPUGetBlockCompressionType(block_data, block_size);
  if (type == kGPUSnappyCompression) {
    // 获得未压缩时的大小
    size_t uncompressed_block_data_size = 0;
    snappy::GetUncompressedLength(block_data, block_size,
                                  &uncompressed_block_data_size);

    // 获得未压缩时的数据
    char* data_block = new char[uncompressed_block_data_size];
    snappy::RawUncompress(block_data, block_size, data_block);

    block_data = data_block;
    block_size = uncompressed_block_data_size;
    allocated = true;
  }
  return std::make_pair(block_data, allocated);
}

/**
 * 辅助函数
 *
 * @param data
 * @param size
 * @return
 */
[[maybe_unused]] __host__ __device__ inline uint32_t NumberRestarts(
    const char* data, size_t size) {
  uint32_t block_footer = GPUDecodeFixed32(data + size - sizeof(uint32_t));
  uint32_t num_restarts = block_footer;
  if (size > 1U << 16) {
    return num_restarts;
  }
  if (num_restarts) {
    num_restarts = block_footer & ((1U << 31) - 1U);
  }
  return num_restarts;
}

/**
 * 辅助函数
 *
 * @param size
 * @param num_restarts
 * @return
 */
[[maybe_unused]] __host__ __device__ inline uint32_t RestartOffset(
    size_t size, uint32_t num_restarts) {
  return static_cast<uint32_t>(size) - (1 + num_restarts) * sizeof(uint32_t);
}

/**
 * 辅助函数
 *
 * static_cast：这是最常用的类型转换操作符。它在编译时进行类型检查，因此相对安全。
 * static_cast 用于将一种类型转换为另一种类型，例如将 int 转换为
 * float，或将基类指针转换为派生类指针。
 * 它不能用于转换与目标类型无关的类型，例如将指针转换为整数。
 *
 * reinterpret_cast：这是一个低级别的类型转换操作符，通常用于将一种类型的指针或引用转换为另一种类型的指针或引用。
 * 它基本上只是告诉编译器将给定的内存重新解释为其他类型。这种转换可能导致未定义的行为，因此应谨慎使用。
 *
 * @param data
 * @param value
 * @param value_size
 * @return
 */
[[maybe_unused]] __host__ __device__ inline uint32_t GPUNextEntryOffset(
    const char* data, const char* value, size_t value_size) {
  return static_cast<uint32_t>(value + value_size - data);
}

/**
 * 辅助函数
 *
 * @param p
 * @param limit
 * @param shared
 * @param non_shared
 * @param value_length
 * @return
 */
[[maybe_unused]] __host__ __device__ inline const char* GPUDecodeDataEntry(
    const char* p, const char* limit, uint32_t* shared, uint32_t* non_shared,
    uint32_t* value_length) {
  if (limit - p < 3) {
    return nullptr;
  }
  *shared = reinterpret_cast<const unsigned char*>(p)[0];
  *non_shared = reinterpret_cast<const unsigned char*>(p)[1];
  *value_length = reinterpret_cast<const unsigned char*>(p)[2];
  if ((*shared | *non_shared | *value_length) < 128) {
    p += 3;
  } else {
    if ((p = GPUGetVarint32Ptr(p, limit, shared)) == nullptr) return nullptr;
    if ((p = GPUGetVarint32Ptr(p, limit, non_shared)) == nullptr)
      return nullptr;
    if ((p = GPUGetVarint32Ptr(p, limit, value_length)) == nullptr)
      return nullptr;
  }
  return p;
}

__host__ __device__ inline void DecodeFrom(const char** data, size_t* data_size,
                                           uint64_t* offset, uint64_t* size) {
  if (GPUGetVarint64(data, data_size, offset) &&
      GPUGetVarint64(data, data_size, size)) {
  } else {
    *offset = 0;
    *size = 0;
    printf("bad block handle\n");
  }
}

/**
 * 辅助函数
 *
 * @param data
 * @param data_size
 * @param offset
 * @param size
 */
__host__ __device__ inline void DecodeFromForIndex(const char** data,
                                                   size_t* data_size,
                                                   uint64_t* offset,
                                                   uint64_t* size) {
  if (GPUGetFixed64(data, data_size, offset) &&
      GPUGetVarint64(data, data_size, size)) {
  } else {
    *offset = 0;
    *size = 0;
    printf("bad block handle\n");
  }
}

/**
 *
 * @param internal_key
 * @param internal_key_size
 * @param sequence
 * @param type
 */
__host__ __device__ inline void GPUParseInternalKey(const char* internal_key,
                                                    size_t internal_key_size,
                                                    uint64_t& sequence,
                                                    unsigned char& type) {
  uint64_t num = GPUDecodeFixed64(internal_key + internal_key_size - 8);
  unsigned char c = num & 0xff;
  sequence = num >> 8;
  type = c;
}

__global__ void PrepareDecode(InputFile* inputFiles_d, size_t num_file,
                              uint64_t* all_num_kv_d,
                              size_t* max_num_data_block_d);

/**
 * GPU并行解码footer
 *
 * @param inputFiles
 * @param footers
 */
__global__ void DecodeFootersKernel(InputFile* inputFiles,
                                    GPUBlockHandle* footers);

/**
 * 计算Restarts数组核函数
 *
 * @param inputFiles
 * @param footer
 * @param restarts
 */
__global__ void ComputeRestartsKernel(InputFile* inputFiles,
                                      GPUBlockHandle* footer,
                                      uint32_t* restarts,
                                      uint64_t max_num_data_block_d);

/**
 * GPU并行解析索引块核函数
 *
 * @param inputFiles
 * @param restarts
 * @param footer
 * @param index_block
 */
__global__ void DecodeIndexBlocksKernel(InputFile* inputFiles,
                                        const uint32_t* restarts,
                                        GPUBlockHandle* footer,
                                        GPUBlockHandle* index_block,
                                        uint64_t max_num_data_block_d);

/**
 * GPU并行解析数据块核函数
 *
 * @param inputFiles
 * @param global_count
 * @param index_block
 * @param keyValuePtr
 */
__global__ void DecodeDataBlocksKernel(InputFile* inputFiles,
                                       uint32_t* global_count,
                                       GPUBlockHandle* index_block,
                                       GPUKeyValue* keyValuePtr,
                                       uint64_t max_num_data_block_d);

/**
 * GPU并行解析SSTable控制函数
 *
 * @param inputFiles
 * @param num_file
 * @param inputFiles_d
 * @return
 */
GPUKeyValue* GetAndSort(size_t num_file, InputFile* inputFiles_d,
                        size_t num_kv_data_block, size_t& sorted_size,
                        cudaStream_t* stream);

}  // namespace ROCKSDB_NAMESPACE