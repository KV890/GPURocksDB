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

  RocksDB(const char *dbfilename, int max_background_jobs);

  int Read(const std::string &table, const std::string &key,
           const std::vector<std::string> *fields,
           std::vector<KVPair> &result) override;

  int ReadBatch(std::vector<rocksdb::Slice> keys,
                std::vector<std::string>& values) override;

  int Scan(const std::string &table, const std::string &key, int len,
           const std::vector<std::string> *fields,
           std::vector<std::vector<KVPair>> &result) override;

  int Update(const std::string &table, const std::string &key,
             std::vector<KVPair> &values) override;

  int UpdateBatch(rocksdb::WriteBatch batch) override;

  int Insert(const std::string &table, const std::string &key,
             std::vector<KVPair> &values) override;

  int InsertBatch(rocksdb::WriteBatch batch) override;

  int Delete(const std::string &table, const std::string &key) override;

  void PrintStats();

  void PrintMyStats() override;

  ~RocksDB() override;

 private:
  rocksdb::DB *db_;
  std::shared_ptr<rocksdb::Cache> cache_;
  std::shared_ptr<rocksdb::Statistics> db_stats_;
  size_t not_found_;

  void SetOptions(rocksdb::Options *options, int max_background_jobs);
};

}  // namespace ycsbc
