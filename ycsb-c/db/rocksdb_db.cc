#include "rocksdb_db.h"

#include <iostream>

#include "rocksdb/rate_limiter.h"

namespace ycsbc {

RocksDB::RocksDB(const char *dbfilename)
    : db_(nullptr),
      my_db_(nullptr),
      cache_(nullptr),
      db_stats_(nullptr),
      not_found_(0) {
  // set option
  rocksdb::Options options;
  SetOptions(&options);

  rocksdb::Status s = rocksdb::DB::Open(options, dbfilename, &db_);

  rocksdb::MyDB my_db(db_);
  my_db_ = my_db;

  if (!s.ok()) {
    std::cerr << "Can't open rocksdb " << dbfilename << " " << s.ToString()
              << std::endl;
    exit(0);
  }
}

void RocksDB::SetOptions(rocksdb::Options *options) {
  options->create_if_missing = true;
  options->compression = rocksdb::CompressionType::kNoCompression;

  auto table_options =
      options->table_factory->GetOptions<rocksdb::BlockBasedTableOptions>();

  // cache
  size_t capacity = 8 << 20;
  std::shared_ptr<rocksdb::Cache> cache = rocksdb::NewLRUCache(capacity);
  table_options->block_cache = cache;

  // bloom filter
  table_options->filter_policy.reset(rocksdb::NewBloomFilterPolicy(10));

  db_stats_ = rocksdb::CreateDBStatistics();
  options->statistics = db_stats_;
}

int RocksDB::Read(const std::string &table, const std::string &key,
                  const std::vector<std::string> *fields,
                  std::vector<KVPair> &result) {
//  my_db_.MyGet(key);

  std::string value;
  db_->Get(rocksdb::ReadOptions(), key, &value);

  return 0;
}

int RocksDB::Scan(const std::string &table, const std::string &key, int len,
                  const std::vector<std::string> *fields,
                  std::vector<std::vector<KVPair>> &result) {
//  my_db_.MyScan(key, len);

  auto it = db_->NewIterator(rocksdb::ReadOptions());
  it->Seek(key);

  for (int i = 0; i < len && it->Valid(); i++) {
    it->Next();
  }

  delete it;

  return DB::kOK;
}

int RocksDB::Insert(const std::string &table, const std::string &key,
                    std::vector<KVPair> &values) {
  my_db_.MyInsert(key, values[0].second);

  return DB::kOK;
}

int RocksDB::Update(const std::string &table, const std::string &key,
                    std::vector<KVPair> &values) {
  my_db_.MyUpdate(key, values[0].second);

  return 0;
}

int RocksDB::Modify(const std::string &table, const std::string &key,
                    std::vector<KVPair> &values) {
  my_db_.MyModify(key, values[0].second);

  return 0;
}

int RocksDB::Delete(const std::string &table, const std::string &key) {
  std::vector<DB::KVPair> values;
  return Insert(table, key, values);
}

void RocksDB::FinishRun() { my_db_.Finish(); }

void RocksDB::PrintStats() {
  std::string stats;
  db_->GetProperty("rocksdb.stats", &stats);
  std::cout << stats << std::endl;

  fprintf(stdout, "STATISTICS:\n%s\n", db_stats_->ToString().c_str());

  if (not_found_) std::cerr << "read not found: " << not_found_ << std::endl;
}

void RocksDB::PrintMyStats() { PrintStats(); }

RocksDB::~RocksDB() {}

}  // namespace ycsbc
