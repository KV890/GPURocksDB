//
// Created by jxx on 4/6/23.
//
#pragma once

#include "gpu_coding.cuh"
#include "gpu_options.cuh"

#ifndef __constant__
#define __constant__
#endif

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

namespace ROCKSDB_NAMESPACE {

// 表示一个文件指针
class GPUBlockHandle {
 public:
  __host__ __device__ GPUBlockHandle() : offset_(0), size_(0) {}
  __host__ __device__ GPUBlockHandle(uint64_t offset, uint64_t size)
      : offset_(offset), size_(size) {}

  __host__ __device__ uint64_t offset() const { return offset_; }
  __host__ __device__ void set_offset(uint64_t _offset) { offset_ = _offset; }

  __host__ __device__ uint64_t size() const { return size_; }
  __host__ __device__ void set_size(uint64_t _size) { size_ = _size; }

 private:
  uint64_t offset_;  // 指向文件的偏移量
  uint64_t size_;    // 表示Block的大小
};

class GPUKeyValuePtr {
 public:
  __host__ __device__ GPUKeyValuePtr()
      : file_number(0), key{}, valuePtr(0), sequence(0), type{} {}

  uint64_t file_number;  // 文件编号

  char key[keySize_ + 8 + 1];  // 内部key
  // 定位原始value的方法
  // char value[valueSize_ + 1] = {};
  // const char* valuePtr = reinterpret_cast<const char*>(file + item.valuePtr);
  // memcpy(value, valuePtr, valueSize_);
  uint64_t valuePtr;

  uint64_t sequence;  // 序列号
  unsigned char type;
};

struct InputFile {
  __host__ __device__ InputFile()
      : level(0),
        file{},
        file_size(0),
        file_number(0),
        num_data_blocks(0),
        num_entries(0) {}

  ~InputFile() {
    delete[] file;  // 释放 file_char 内存
  }

  size_t level;
  char* file;
  size_t file_size;
  uint64_t file_number;
  uint64_t num_data_blocks;
  uint64_t num_entries;
};

}  // namespace ROCKSDB_NAMESPACE