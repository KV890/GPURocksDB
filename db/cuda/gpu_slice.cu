//
// Created by jxx on 4/8/23.
//

#include "gpu_slice.cuh"

namespace ROCKSDB_NAMESPACE {

char toHex(unsigned char v) {
  if (v <= 9) {
    return '0' + v;
  }
  return 'A' + v - 10;
}

std::string GPUSlice::ToString(bool hex) const {
  std::string result;
  if (hex) {
    result.reserve(2 * size_);
    for (size_t i = 0; i < size_; i++) {
      unsigned char c = data_[i];
      result.push_back(toHex(c >> 4));
      result.push_back(toHex(c & 0xf));  // c & 00001111
    }
    return result;
  } else {
    result.assign(data_, size_);
    return result;
  }
}

__host__ GPUSlice& GPUSlice::append_host(const char* s, size_t n) {
  size_t new_size = size_ + n;
  char* new_data = new char[new_size + 1];
  memcpy(new_data, data_, size_);
  memcpy(new_data + size_, s, n);
  new_data[new_size] = '\0';

  delete[] data_;
  data_ = new_data;
  size_ = new_size;

  return *this;
}

__host__ __device__ GPUSlice& GPUSlice::append_device(const char* s, size_t n) {
  size_t new_size = size_ + n;
  char* new_data;
  cudaMallocManaged(&new_data, new_size + 1);
  memcpy(new_data, data_, size_);
  memcpy(new_data + size_, s, n);
  new_data[new_size] = '\0';

  cudaFree((void*)data_);
  data_ = new_data;
  size_ = new_size;

  return *this;
}

}  // namespace ROCKSDB_NAMESPACE