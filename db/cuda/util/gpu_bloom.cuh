//
// Created by d306 on 10/10/23.
//
#pragma once

#include "db/cuda/gpu_coding.cuh"

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

namespace ROCKSDB_NAMESPACE {

__device__ extern uint32_t GPUBloomHash(const char* key, size_t key_size);

__device__ extern uint32_t GPUHash(const char* data, size_t n, uint32_t seed);

}  // namespace ROCKSDB_NAMESPACE