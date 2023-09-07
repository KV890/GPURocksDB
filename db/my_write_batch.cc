//
// Created by jxx on 8/31/23.
//

#include "my_write_batch.h"

#include "db/cuda/gpu_options.cuh"

namespace ROCKSDB_NAMESPACE {

MyWriteBatch::MyWriteBatch()
    : num_ops(0),
      batch_size(1000),
      current_ops(0),
      db(nullptr),
      total_batch(WriteBatch((keySize_ + valueSize_ + 24) * batch_size, 0, 0, 0)) {}

Status MyWriteBatch::MyPut(ColumnFamilyHandle* column_family, const Slice& key,
                           const Slice& value) {
  current_ops++;
  return total_batch.Put(column_family, key, value);
}

Status MyWriteBatch::MyWriteLast() {
  assert(Count() > 0);

  Status s = db->Write(WriteOptions(), &total_batch);
  if (!s.ok()) {
    return s;
  }

  Clear();
  return s;
}

uint32_t MyWriteBatch::Count() const { return total_batch.Count(); }

void MyWriteBatch::Clear() { total_batch.Clear(); }

MyWriteBatch my_write_batch;

}  // namespace ROCKSDB_NAMESPACE
