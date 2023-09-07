//
// Created by jxx on 4/8/23.
//
#pragma once

#include <cstdio>
#include <string>

#include "rocksdb/rocksdb_namespace.h"

#ifndef __global__
#define __global__
#endif

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

namespace ROCKSDB_NAMESPACE {

class GPUSlice {
 public:
  __host__ __device__ GPUSlice() : data_{}, size_(0) {}
  __host__ __device__ GPUSlice(const char* d, size_t n) : data_(d), size_(n) {}

  __host__ explicit GPUSlice(std::string& s)
      : data_(s.data()), size_(s.size()) {}

  __host__ __device__ explicit GPUSlice(const char* s) : data_(s) {
    int len = 0;
    while (s[len] != '\0') {
      len++;
    }
    size_ = len;
  }

  __host__ __device__ const char* data() const { return data_; }
  __host__ __device__ size_t size() const { return size_; }

  __host__ __device__ bool empty() const { return size_ == 0; }

  __host__ __device__ char operator[](size_t n) const { return data_[n]; }

  __host__ __device__ void clear() {
    data_ = {};
    size_ = 0;
  }

  __host__ __device__ void remove_prefix(size_t n) {
    data_ += n;
    size_ -= n;
  }

  [[nodiscard]] std::string ToString(bool hex = false) const;

  __host__ GPUSlice& append_host(const char* s, size_t n);

  __host__ __device__ GPUSlice& append_device(const char* s, size_t n);

 private:
  const char* data_;
  size_t size_;
};

}  // namespace ROCKSDB_NAMESPACE