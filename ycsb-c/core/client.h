//
//  client.h
//  YCSB-C
//
//  Created by Jinglei Ren on 12/10/14.
//  Copyright (c) 2014 Jinglei Ren <jinglei@ren.systems>.
//
#pragma once

#include <string>

#include "core_workload.h"
#include "db.h"
#include "utils.h"

namespace ycsbc {

class Client {
 public:
  Client(DB& db, CoreWorkload& wl) : db_(db), workload_(wl) {}

  virtual bool DoInsert(bool is_running);
  size_t DoInsert(rocksdb::WriteBatch batch, size_t batch_size);
  virtual bool DoTransaction();

  virtual ~Client() = default;

 protected:
  virtual int TransactionRead();
  virtual int TransactionReadModifyWrite();
  virtual int TransactionScan();
  virtual int TransactionUpdate();
  virtual int TransactionInsert();

  DB& db_;
  CoreWorkload& workload_;
};

inline bool Client::DoInsert(bool is_running) {
  uint64_t key_num;
  std::string key = workload_.NextSequenceKey(key_num);
  std::vector<DB::KVPair> pairs;
  workload_.BuildValues(pairs);
  if (!is_running) {
    return db_.Insert(workload_.NextTable(), key, pairs, key_num) == DB::kOK;
  } else {
    return true;
  }
}

inline size_t Client::DoInsert(rocksdb::WriteBatch batch, size_t batch_size) {
  uint64_t key_num;

  for (size_t j = 0; j < batch_size; ++j) {
    std::string key = workload_.NextSequenceKey(key_num);
    workload_.NextTable();
    std::vector<DB::KVPair> pairs;
    workload_.BuildValues(pairs);

    batch.Put(db_.db_with_cf.GetCfh(static_cast<int64_t>(key_num)), key,
              pairs[0].second);
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

inline int Client::TransactionRead() {
  uint64_t key_num;
  const std::string& table = workload_.NextTable();
  const std::string& key = workload_.NextTransactionKey(key_num);

  std::vector<DB::KVPair> result;
  if (!workload_.read_all_fields()) {
    std::vector<std::string> fields;
    fields.push_back("field" + workload_.NextFieldName());
    return db_.Read(table, key, &fields, result, key_num);
  } else {
    return db_.Read(table, key, nullptr, result, key_num);
  }
}

inline int Client::TransactionReadModifyWrite() {
  uint64_t key_num;

  const std::string& table = workload_.NextTable();
  const std::string& key = workload_.NextTransactionKey(key_num);
  std::vector<DB::KVPair> result;

  if (!workload_.read_all_fields()) {
    std::vector<std::string> fields;
    fields.push_back("field" + workload_.NextFieldName());
    db_.Read(table, key, &fields, result, key_num);
  } else {
    db_.Read(table, key, nullptr, result, key_num);
  }

  std::vector<DB::KVPair> values;
  if (workload_.write_all_fields()) {
    workload_.BuildValues(values);
  } else {
    workload_.BuildUpdate(values);
  }
  return db_.Update(table, key, values, key_num);
}

inline int Client::TransactionScan() {
  uint64_t key_num;

  const std::string& table = workload_.NextTable();
  const std::string& key = workload_.NextTransactionKey(key_num);
  int len = workload_.NextScanLength();
  std::vector<std::vector<DB::KVPair>> result;
  if (!workload_.read_all_fields()) {
    std::vector<std::string> fields;
    fields.push_back("field" + workload_.NextFieldName());
    return db_.Scan(table, key, len, &fields, result, key_num);
  } else {
    return db_.Scan(table, key, len, nullptr, result, key_num);
  }
}

inline int Client::TransactionUpdate() {
  uint64_t key_num;

  const std::string& table = workload_.NextTable();
  const std::string& key = workload_.NextTransactionKey(key_num);
  std::vector<DB::KVPair> values;
  if (workload_.write_all_fields()) {
    workload_.BuildValues(values);
  } else {
    workload_.BuildUpdate(values);
  }
  return db_.Update(table, key, values, key_num);
}

inline int Client::TransactionInsert() {
  uint64_t key_num;

  const std::string& table = workload_.NextTable();
  const std::string& key = workload_.NextSequenceKey(key_num);
  std::vector<DB::KVPair> values;
  workload_.BuildValues(values);
  return db_.Insert(table, key, values, key_num);
}

}  // namespace ycsbc
