//
// Created by d306 on 10/10/23.
//
#include "gpu_bloom.cuh"

namespace ROCKSDB_NAMESPACE {

__device__ uint32_t GPUBloomHash(const char* key, size_t key_size) {
  return GPUHash(key, key_size, 0xbc9f1d34);
}

__device__ uint32_t GPUHash(const char* data, size_t n, uint32_t seed) {
  // MurmurHash1 - fast but mediocre quality
  // https://github.com/aappleby/smhasher/wiki/MurmurHash1
  //
  const uint32_t m = 0xc6a4a793;
  const uint32_t r = 24;
  const char* limit = data + n;
  auto h = static_cast<uint32_t>(seed ^ (n * m));

  // Pick up four bytes at a time
  while (data + 4 <= limit) {
    uint32_t w = GPUDecodeFixed32(data);
    data += 4;
    h += w;
    h *= m;
    h ^= (h >> 16);
  }

  // Pick up remaining bytes
  switch (limit - data) {
    // Note: The original hash implementation used data[i] << shift, which
    // promotes the char to int and then performs the shift. If the char is
    // negative, the shift is undefined behavior in C++. The hash algorithm is
    // part of the format definition, so we cannot change it; to obtain the same
    // behavior in a legal way we just cast to uint32_t, which will do
    // sign-extension. To guarantee compatibility with architectures where chars
    // are unsigned we first cast the char to int8_t.
    case 3:
      h += static_cast<uint32_t>(static_cast<int8_t>(data[2])) << 16;
      // FALLTHROUGH_INTENDED;
    case 2:
      h += static_cast<uint32_t>(static_cast<int8_t>(data[1])) << 8;
      // FALLTHROUGH_INTENDED;
    case 1:
      h += static_cast<uint32_t>(static_cast<int8_t>(data[0]));
      h *= m;
      h ^= (h >> r);
      break;
  }
  return h;
}

}  // namespace ROCKSDB_NAMESPACE