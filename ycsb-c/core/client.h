//
//  client.h
//  YCSB-C
//
//  Created by Jinglei Ren on 12/10/14.
//  Copyright (c) 2014 Jinglei Ren <jinglei@ren.systems>.
//
#pragma once

#include <string>
#include <utility>

#include "core_workload.h"
#include "db.h"
#include "utils.h"

namespace ycsbc {

class Client {
 public:
  Client(DB& db, CoreWorkload& wl) : db_(db), workload_(wl) {}

  virtual bool DoInsert(bool is_running);
  virtual size_t DoInsert(rocksdb::WriteBatch& batch, size_t batch_size);
  virtual bool DoTransaction();
  virtual size_t DoTransaction(
      rocksdb::WriteBatch& batch_update, rocksdb::WriteBatch& batch_insert,
      rocksdb::WriteBatch& batch_f, std::vector<rocksdb::Slice>& keys,
      std::vector<rocksdb::Slice>& keys_scan, std::vector<int>& lens,
      std::vector<rocksdb::Slice>& keys_f, size_t batch_size);

  virtual ~Client() = default;

 protected:
  virtual int TransactionRead();
  virtual void TransactionRead(std::vector<rocksdb::Slice>& keys);
  virtual int TransactionReadModifyWrite();
  virtual void TransactionReadModifyWrite(std::vector<rocksdb::Slice>& keys,
                                          rocksdb::WriteBatch& batch);
  virtual int TransactionScan();
  virtual void TransactionScan(std::vector<rocksdb::Slice>& keys,
                               std::vector<int>& lens);
  virtual int TransactionUpdate();
  virtual int TransactionInsert();
  virtual void TransactionUpdate(rocksdb::WriteBatch& batch);
  virtual void TransactionInsert(rocksdb::WriteBatch& batch);

  DB& db_;
  CoreWorkload& workload_;
};

inline bool Client::DoInsert(bool is_running) {
  std::string key = workload_.NextSequenceKey();
  std::vector<DB::KVPair> pairs;
  workload_.BuildValues(pairs);
  if (!is_running) {
    return db_.Insert(workload_.NextTable(), key, pairs) == DB::kOK;
  } else {
    return true;
  }
}

inline size_t Client::DoInsert(rocksdb::WriteBatch& batch, size_t batch_size) {
  for (size_t j = 0; j < batch_size; ++j) {
    std::string key = workload_.NextSequenceKey();
    workload_.NextTable();
    std::vector<DB::KVPair> pairs;
    workload_.BuildValues(pairs);

    batch.Put(key, pairs[0].second);
  }

  db_.InsertBatch(batch);

  batch.Clear();

  return batch_size;
}

inline bool Client::DoTransaction() {
  int status;
  switch (workload_.NextOperation()) {
    case READ:
      status = TransactionRead();
      break;
    case UPDATE:
      status = TransactionUpdate();
      break;
    case INSERT:
      status = TransactionInsert();
      break;
    case SCAN:
      status = TransactionScan();
      break;
    case READMODIFYWRITE:
      status = TransactionReadModifyWrite();
      break;
    default:
      throw utils::Exception("Operation request is not recognized!");
  }
  assert(status >= 0);
  return (status == DB::kOK);
}

inline size_t Client::DoTransaction(
    rocksdb::WriteBatch& batch_update, rocksdb::WriteBatch& batch_insert,
    rocksdb::WriteBatch& batch_f, std::vector<rocksdb::Slice>& keys,
    std::vector<rocksdb::Slice>& keys_scan, std::vector<int>& lens,
    std::vector<rocksdb::Slice>& keys_f, size_t batch_size) {
  for (size_t i = 0; i < batch_size; ++i) {
    Operation op = workload_.NextOperation();
    if (op == READ) {
      std::string key_str = workload_.NextTransactionKey();

      char* key_char = new char[rocksdb::keySize_];
      memcpy(key_char, key_str.c_str(), rocksdb::keySize_);

      keys.emplace_back(key_char, rocksdb::keySize_);
    } else if (op == UPDATE) {
      std::vector<DB::KVPair> pairs;
      workload_.BuildValues(pairs);

      batch_update.Put(workload_.NextSequenceKey(), pairs[0].second);
    } else if (op == INSERT) {
      std::vector<DB::KVPair> pairs;
      workload_.BuildValues(pairs);

      batch_insert.Put(workload_.NextSequenceKey(), pairs[0].second);
    } else if (op == SCAN) {
      std::string key_str = workload_.NextTransactionKey();
      int len = workload_.NextScanLength();

      char* key_char = new char[rocksdb::keySize_];
      memcpy(key_char, key_str.c_str(), rocksdb::keySize_);

      keys_scan.emplace_back(key_char, rocksdb::keySize_);
      lens.emplace_back(len);
    } else if (op == READMODIFYWRITE) {
      std::string key_str = workload_.NextTransactionKey();
      char* key_char = new char[rocksdb::keySize_];
      memcpy(key_char, key_str.c_str(), rocksdb::keySize_);

      keys_f.emplace_back(key_char, rocksdb::keySize_);

      std::vector<DB::KVPair> values_update;
      workload_.BuildUpdate(values_update);

      batch_f.Put(key_str, values_update[0].second);
    } else {
      throw utils::Exception("Operation request is not recognized!");
    }
  }

  if (!keys.empty()) {
    TransactionRead(keys);
  }

  if (batch_update.Count() > 0) {
    TransactionUpdate(batch_update);
  }

  if (batch_insert.Count() > 0) {
    TransactionInsert(batch_insert);
  }

  if (!keys_scan.empty()) {
    TransactionScan(keys_scan, lens);
  }

  if (!keys_f.empty()) {
    TransactionReadModifyWrite(keys_f, batch_f);
  }

  return batch_size;
}

inline int Client::TransactionRead() {
  const std::string& table = workload_.NextTable();
  const std::string& key = workload_.NextTransactionKey();

  std::vector<DB::KVPair> result;
  if (!workload_.read_all_fields()) {
    std::vector<std::string> fields;
    fields.push_back("field" + workload_.NextFieldName());
    return db_.Read(table, key, &fields, result);
  } else {
    return db_.Read(table, key, nullptr, result);
  }
}

inline void Client::TransactionRead(std::vector<rocksdb::Slice>& keys) {
  db_.ReadBatch(keys);

  for (auto& key : keys) {
    delete[] key.data();
  }

  keys.clear();
}

inline int Client::TransactionReadModifyWrite() {
  const std::string& table = workload_.NextTable();
  const std::string& key = workload_.NextTransactionKey();
  std::vector<DB::KVPair> result;

  if (!workload_.read_all_fields()) {
    std::vector<std::string> fields;
    fields.push_back("field" + workload_.NextFieldName());
    db_.Read(table, key, &fields, result);
  } else {
    db_.Read(table, key, nullptr, result);
  }

  std::vector<DB::KVPair> values;
  if (workload_.write_all_fields()) {
    workload_.BuildValues(values);
  } else {
    workload_.BuildUpdate(values);
  }
  return db_.Update(table, key, values);
}

inline void Client::TransactionReadModifyWrite(
    std::vector<rocksdb::Slice>& keys, rocksdb::WriteBatch& batch) {
  db_.ReadBatch(keys);
  db_.UpdateBatch(batch);

  for (auto& key : keys) {
    delete[] key.data();
  }

  keys.clear();
  batch.Clear();
}

inline int Client::TransactionScan() {
  const std::string& table = workload_.NextTable();
  const std::string& key = workload_.NextTransactionKey();
  int len = workload_.NextScanLength();
  std::vector<std::vector<DB::KVPair>> result;
  if (!workload_.read_all_fields()) {
    std::vector<std::string> fields;
    fields.push_back("field" + workload_.NextFieldName());
    return db_.Scan(table, key, len, &fields, result);
  } else {
    return db_.Scan(table, key, len, nullptr, result);
  }
}

inline void Client::TransactionScan(std::vector<rocksdb::Slice>& keys,
                                    std::vector<int>& lens) {
  db_.ScanBatch(keys, lens);

  for (auto& key : keys) {
    delete[] key.data();
  }

  keys.clear();
}

inline int Client::TransactionUpdate() {
  const std::string& table = workload_.NextTable();
  const std::string& key = workload_.NextTransactionKey();
  std::vector<DB::KVPair> values;
  if (workload_.write_all_fields()) {
    workload_.BuildValues(values);
  } else {
    workload_.BuildUpdate(values);
  }
  return db_.Update(table, key, values);
}

inline int Client::TransactionInsert() {
  const std::string& table = workload_.NextTable();
  const std::string& key = workload_.NextSequenceKey();
  std::vector<DB::KVPair> values;
  workload_.BuildValues(values);
  return db_.Insert(table, key, values);
}

inline void Client::TransactionUpdate(rocksdb::WriteBatch& batch) {
  db_.UpdateBatch(batch);

  batch.Clear();
}

inline void Client::TransactionInsert(rocksdb::WriteBatch& batch) {
  db_.InsertBatch(batch);

  batch.Clear();
}

}  // namespace ycsbc
