//
// Created by jxx on 8/31/23.
//
#pragma once

#include "rocksdb/db.h"
#include "rocksdb/rocksdb_namespace.h"
#include "rocksdb/write_batch.h"

namespace ROCKSDB_NAMESPACE {

class MyWriteBatch {
 public:
  MyWriteBatch();

  Status MyPut(ColumnFamilyHandle* column_family, const Slice& key,
               const Slice& value);

  Status MyWriteLast();

  uint32_t Count() const;

  void Clear();

  size_t num_ops;
  size_t batch_size;
  size_t current_ops;
  DB* db;
  WriteBatch total_batch;
};

extern MyWriteBatch my_write_batch;

}  // namespace ROCKSDB_NAMESPACE
