//
// Created by jxx on 3/25/23.
//

#pragma once

namespace ROCKSDB_NAMESPACE {

constexpr int keySize_ = 16;
constexpr int valueSize_ = 1024;

constexpr size_t encoded_value_size = valueSize_ < 128 ? 1 : 2;
constexpr size_t key_value_size =
    2 + encoded_value_size + keySize_ + 8 + valueSize_;
constexpr size_t encoded_index_entry = 10;
constexpr size_t size_index_entry = keySize_ + encoded_index_entry + 2;

}  // namespace ROCKSDB_NAMESPACE