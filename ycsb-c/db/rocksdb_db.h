#pragma once

#include <iostream>
#include <string>

#include "../core/core_workload.h"
#include "../core/db.h"
#include "../core/properties.h"

namespace ycsbc {

class RocksDB : public DB {
 public:
  explicit RocksDB(const char *dbfilename);

  RocksDB(const char *dbfilename, int num_column_family);

  int Read(const std::string &table, const std::string &key,
           const std::vector<std::string> *fields, std::vector<KVPair> &result,
           uint64_t sequence) override;

  int Scan(const std::string &table, const std::string &key, int len,
           const std::vector<std::string> *fields,
           std::vector<std::vector<KVPair>> &result,
           uint64_t sequence) override;

  int Update(const std::string &table, const std::string &key,
             std::vector<KVPair> &values, uint64_t sequence) override;

  int Insert(const std::string &table, const std::string &key,
             std::vector<KVPair> &values, uint64_t sequence) override;

  int InsertBatch(rocksdb::WriteBatch batch) override;

  int Delete(const std::string &table, const std::string &key,
             uint64_t sequence) override;

  void PrintStats();

  void PrintMyStats() override;

  ~RocksDB() override;

 private:
  std::shared_ptr<rocksdb::Cache> cache_;
  std::shared_ptr<rocksdb::Statistics> db_stats_;
  size_t not_found_;

  void SetOptions(rocksdb::Options *options);
};

}  // namespace ycsbc
