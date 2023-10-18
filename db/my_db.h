//
// Created by d306 on 10/18/23.
//
#pragma once

#include <string>
#include <vector>

#include "rocksdb/db.h"
#include "rocksdb/slice.h"
#include "rocksdb/write_batch.h"

namespace ROCKSDB_NAMESPACE {

class MyDB {
 public:
  explicit MyDB(DB *db);

  void MyGet(const std::string &key);

  void MyUpdate(const std::string &key, const std::string &value);

  void MyInsert(const std::string &key, const std::string &value);

  void MyScan(const std::string &key, int len);

  void MyModify(const std::string &key, const std::string &value);

  void Finish();

 protected:
  void TransactionRead(std::vector<Slice> &keys);

  void TransactionUpdate(WriteBatch &batch);

  void TransactionInsert(WriteBatch &batch);

  void TransactionScan(std::vector<Slice> &keys, std::vector<int> &lens);

  void TransactionReadModifyWrite(std::vector<Slice> &keys, WriteBatch &batch);

 private:
  void ReadBatch(std::vector<Slice> &keys);

  void ScanBatch(std::vector<Slice> &keys, std::vector<int> &lens);

  void UpdateBatch(WriteBatch &batch);

  void InsertBatch(WriteBatch &batch);

  DB *db_;
  size_t batch_size_;

  WriteBatch batch_update_;
  WriteBatch batch_insert_;
  WriteBatch batch_modify_;

  std::vector<Slice> keys_read_;
  std::vector<Slice> keys_scan_;
  std::vector<int> lens_;
  std::vector<Slice> keys_modify_;
};

}  // namespace ROCKSDB_NAMESPACE
