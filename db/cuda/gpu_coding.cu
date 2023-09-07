//
// Created by jxx on 4/7/23.
//

#include "gpu_coding.cuh"

namespace ROCKSDB_NAMESPACE {

__host__ __device__ char* GPUEncodeVarint32(char* dst, uint32_t v) {
  auto* ptr = reinterpret_cast<unsigned char*>(dst);
  static const int B = 128;
  if (v <
      (1 << 7)) {  // 编码为一个字节, 最高位（第 7 位）为 0, 其余 7 位表示数值
    *(ptr++) = v;
  } else if (v < (1 << 14)) {
    // 编码为两个字节，第一个字节的最高位（第 7 位）为 1，其余 7 位表示 v 的低 7
    // 位； 第二个字节的最高位（第 7 位）为 0，其余 7 位表示 v 的高 7 位；
    *(ptr++) = v | B;
    *(ptr++) = v >> 7;
  } else if (v < (1 << 21)) {
    *(ptr++) = v | B;
    *(ptr++) = (v >> 7) | B;
    *(ptr++) = v >> 14;
  } else if (v < (1 << 28)) {
    *(ptr++) = v | B;
    *(ptr++) = (v >> 7) | B;
    *(ptr++) = (v >> 14) | B;
    *(ptr++) = v >> 21;
  } else {
    *(ptr++) = v | B;
    *(ptr++) = (v >> 7) | B;
    *(ptr++) = (v >> 14) | B;
    *(ptr++) = (v >> 21) | B;
    *(ptr++) = v >> 28;
  }
  return reinterpret_cast<char*>(ptr);
}

__host__ __device__ char* GPUEncodeVarint64(char* dst, uint32_t v) {
  static const unsigned int B = 128;
  auto* ptr = reinterpret_cast<unsigned char*>(dst);
  while (v >= B) {
    *(ptr++) = (v & (B - 1)) | B;
    v >>= 7;
  }
  *(ptr++) = static_cast<unsigned char>(v);
  return reinterpret_cast<char*>(ptr);
}

__global__ void ConcatenateKernel(const char* data1, const char* data2,
                                  size_t data1_size, size_t data2_size,
                                  char* result) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < data1_size) {
    result[tid] = data1[tid];
  } else if (tid < data1_size + data2_size) {
    result[tid] = data2[tid - data1_size];
  }
}

__host__ __device__ const char* GPUGetVarint32Ptr(const char* p,
                                                  const char* limit,
                                                  uint32_t* value) {
  if (p < limit) {
    uint32_t result = *(reinterpret_cast<const unsigned char*>(p));
    if ((result & 128) == 0) {
      *value = result;
      return p + 1;
    }
  }
  return GPUGetVarint32PtrFallback(p, limit, value);
}

__host__ __device__ const char* GPUGetVarsignedint64Ptr(const char* p,
                                                        const char* limit,
                                                        int64_t* value) {
  uint64_t u = 0;
  const char* ret = GPUGetVarint64Ptr(p, limit, &u);
  *value = GPUZigzagToI64(u);
  return ret;
}

__host__ __device__ const char* GPUGetVarint32PtrFallback(const char* p,
                                                          const char* limit,
                                                          uint32_t* value) {
  uint32_t result = 0;
  for (uint32_t shift = 0; shift <= 28 && p < limit; shift += 7) {
    uint32_t byte = *(reinterpret_cast<const unsigned char*>(p));
    p++;
    if (byte & 128) {
      // More bytes are present
      result |= ((byte & 127) << shift);
    } else {
      result |= (byte << shift);
      *value = result;
      return reinterpret_cast<const char*>(p);
    }
  }
  return nullptr;
}

__host__ __device__ const char* GPUGetVarint64Ptr(const char* p,
                                                  const char* limit,
                                                  uint64_t* value) {
  uint64_t result = 0;
  for (uint32_t shift = 0; shift <= 63 && p < limit; shift += 7) {
    uint64_t byte = *(reinterpret_cast<const unsigned char*>(p));
    p++;
    if (byte & 128) {
      // More bytes are present
      result |= ((byte & 127) << shift);
    } else {
      result |= (byte << shift);
      *value = result;
      return reinterpret_cast<const char*>(p);
    }
  }
  return nullptr;
}

__host__ __device__ const char* GPUGetVarint64Ptr(const char* p,
                                                  const char* limit,
                                                  uint64_t& value) {
  uint64_t result = 0;
  for (uint32_t shift = 0; shift <= 63 && p < limit; shift += 7) {
    uint64_t byte = *(reinterpret_cast<const unsigned char*>(p));
    p++;
    if (byte & 128) {
      // More bytes are present
      result |= ((byte & 127) << shift);
    } else {
      result |= (byte << shift);
      value = result;
      return p;
    }
  }
  return nullptr;
}

__host__ __device__ size_t GPUGetVarint64Length(uint64_t value) {
  size_t length = 0;
  while (value >= 128) {
    value >>= 7;
    length++;
  }
  length++;
  return length;
}

}  // namespace ROCKSDB_NAMESPACE
