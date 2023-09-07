//
// Created by jxx on 4/7/23.
//
#pragma once

#include <cstdint>

#include "gpu_slice.cuh"
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

#ifndef __shared__
#define __shared__
#endif

namespace ROCKSDB_NAMESPACE {

__host__ __device__ inline bool LittleEndian() {
  return __BYTE_ORDER == __LITTLE_ENDIAN;
}

__host__ __device__ inline void GPUEncodeFixed32(char* buf, uint64_t value);
__host__ __device__ inline void GPUEncodeFixed64(char* buf, uint64_t value);
__host__ __device__ extern int GPUVarintLength(uint64_t v);

__host__ __device__ extern char* GPUEncodeVarint32(char* dst, uint32_t value);
__host__ __device__ extern char* GPUEncodeVarint64(char* dst, uint32_t v);

__host__ __device__ extern void GPUPutVarint32Varint32Varint32(GPUSlice* dst,
                                                               uint32_t v1,
                                                               uint32_t v2,
                                                               uint32_t v3);

__host__ __device__ inline int64_t GPUZigzagToI64(uint64_t n) {
  // ^ 异或操作
  return (n >> 1) ^ -static_cast<int64_t>(n & 1);
}

__host__ inline void GPUPutFixed32Host(GPUSlice& dst, uint32_t value) {
  if (LittleEndian()) {
    dst.append_host(const_cast<const char*>(reinterpret_cast<char*>(&value)),
                    sizeof(value));
  } else {
    char buf[sizeof(value)];
    GPUEncodeFixed32(buf, value);
    dst.append_host(buf, sizeof(buf));
  }
}

__device__ inline void GPUPutFixed32Device(GPUSlice& dst, uint32_t value) {
  if (LittleEndian()) {
    dst.append_host(const_cast<const char*>(reinterpret_cast<char*>(&value)),
                    sizeof(value));
  } else {
    __shared__ char buf[sizeof(value)];
    GPUEncodeFixed32(buf, value);
    dst.append_device(buf, sizeof(buf));
  }
}

// extern 关键字用于声明一个变量或函数的定义在其他地方（通常在其他源文件中）
__host__ __device__ extern const char* GPUGetVarint32Ptr(const char* p,
                                                         const char* limit,
                                                         uint32_t* value);

__host__ __device__ extern const char* GPUGetVarint64Ptr(const char* p,
                                                         const char* limit,
                                                         uint64_t* value);

__host__ __device__ extern const char* GPUGetVarint64Ptr(const char* p,
                                                         const char* limit,
                                                         uint64_t& value);

__host__ __device__ extern const char* GPUGetVarsignedint64Ptr(
    const char* p, const char* limit, int64_t* value);

__host__ __device__ extern const char* GPUGetVarint32PtrFallback(
    const char* p, const char* limit, uint32_t* value);

__host__ __device__ extern size_t GPUGetVarint64Length(uint64_t value);

// inline是C++中的关键字，用于指示编译器将函数的代码插入到函数调用点处，而不是执行函数调用。
// 这样可以避免函数调用的开销，提高程序的执行效率。使用inline修饰的函数被称为内联函数。

__host__ __device__ inline int GPUVarintLength(uint64_t v) {
  int len = 1;
  while (v >= 128) {
    v >>= 7;  // v /= 128
    len++;
  }
  return len;
}

__host__ __device__ inline bool GPUGetVarint32(GPUSlice* input,
                                               uint32_t* value) {
  const char* p = input->data();
  const char* limit = p + input->size();
  const char* q = GPUGetVarint32Ptr(p, limit, value);
  if (q == nullptr) {
    return false;
  } else {
    *input = GPUSlice(q, static_cast<size_t>(limit - q));
    return true;
  }
}

__host__ __device__ inline bool GPUGetVarint64(GPUSlice*& input,
                                               uint64_t* value) {
  const char* p = input->data();
  const char* limit = p + input->size();
  const char* q = GPUGetVarint64Ptr(p, limit, value);
  if (q == nullptr) {
    return false;
  } else {
    *input = GPUSlice(q, static_cast<size_t>(limit - q));
    return true;
  }
}

__host__ __device__ inline bool GPUGetVarint64(const char** input_char,
                                               size_t* input_size,
                                               uint64_t* value) {
  const char* p = *input_char;
  const char* limit = p + *input_size;
  const char* q = GPUGetVarint64Ptr(p, limit, value);

  if (q == nullptr) {
    return false;
  } else {
    *input_char = q;
    *input_size = static_cast<size_t>(limit - q);
    return true;
  }
}

__host__ __device__ inline void GPUPutVarint32Varint32Varint32(GPUSlice* dst,
                                                               uint32_t v1,
                                                               uint32_t v2,
                                                               uint32_t v3) {
  char buf[15];
  char* ptr = GPUEncodeVarint32(buf, v1);
  ptr = GPUEncodeVarint32(ptr, v2);
  ptr = GPUEncodeVarint32(ptr, v3);
  dst->append_host(buf, static_cast<size_t>(ptr - buf));
}

__host__ __device__ inline bool GPUGetVarsignedint64(GPUSlice* input,
                                                     int64_t* value) {
  const char* p = input->data();
  const char* limit = p + input->size();
  const char* q = GPUGetVarsignedint64Ptr(p, limit, value);
  if (q != nullptr) {
    return false;
  } else {
    *input = GPUSlice(q, static_cast<size_t>(limit - q));
    return true;
  }
}

__host__ __device__ inline bool GPUGetLengthPrefixedSlice(GPUSlice* input,
                                                          GPUSlice* result) {
  uint32_t len = 0;
  if (GPUGetVarint32(input, &len) && input->size() >= len) {
    *result = GPUSlice(input->data(), len);
    input->remove_prefix(len);
    return true;
  } else {
    return false;
  }
}

/**
 *
 * @param dst
 * @param value
 */
__host__ __device__ inline void GPUPutFixed64(char* dst, uint32_t value) {
  memcpy(dst, const_cast<const char*>(reinterpret_cast<char*>(&value)),
         sizeof(uint64_t));
}

__host__ __device__ inline uint32_t GPUDecodeFixed32(const char* ptr) {
  uint32_t result;
  memcpy(&result, ptr, sizeof(uint32_t));
  return result;
}

__host__ __device__ inline uint64_t GPUDecodeFixed64(const char* ptr) {
  if (LittleEndian()) {
    uint64_t result;
    memcpy(&result, ptr, sizeof(uint64_t));
    return result;
  } else {
    uint64_t lo = GPUDecodeFixed32(ptr);
    uint64_t hi = GPUDecodeFixed32(ptr + 4);
    return (hi << 32) | lo;
  }
}

__host__ __device__ inline void GPUEncodeFixed32(char* buf, uint64_t value) {
  if (LittleEndian()) {
    memcpy(buf, &value, sizeof(value));
  } else {
    buf[0] = value & 0xff;
    buf[1] = (value >> 8) & 0xff;
    buf[2] = (value >> 16) & 0xff;
    buf[3] = (value >> 24) & 0xff;
  }
}

__host__ __device__ inline void GPUEncodeFixed64(char* buf, uint64_t value) {
  if (LittleEndian()) {
    memcpy(buf, &value, sizeof(value));
  } else {
    buf[0] = value & 0xff;
    buf[1] = (value >> 8) & 0xff;
    buf[2] = (value >> 16) & 0xff;
    buf[3] = (value >> 24) & 0xff;
    buf[4] = (value >> 32) & 0xff;
    buf[5] = (value >> 40) & 0xff;
    buf[6] = (value >> 48) & 0xff;
    buf[7] = (value >> 56) & 0xff;
  }
}

__host__ __device__ inline bool GPUGetFixed32(GPUSlice* input,
                                              uint32_t* value) {
  if (input->size() < sizeof(uint32_t)) {
    return false;
  }
  *value = GPUDecodeFixed32(input->data());
  input->remove_prefix(sizeof(uint32_t));
  return true;
}

__host__ __device__ inline bool GPUGetFixed64(const char** input_char,
                                              size_t* input_size,
                                              uint64_t* value) {
  *value = GPUDecodeFixed64(*input_char);
  *input_char += sizeof(uint64_t);
  *input_size -= sizeof(uint64_t);
  return true;
}

}  // namespace ROCKSDB_NAMESPACE
