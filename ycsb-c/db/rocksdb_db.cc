#include "rocksdb_db.h"

#include <iostream>

#include "rocksdb/rate_limiter.h"

namespace ycsbc {

RocksDB::RocksDB(const char *dbfilename)
    : db_(nullptr), cache_(nullptr), db_stats_(nullptr), not_found_(0) {
  // set option
  rocksdb::Options options;

  rocksdb::Status s = rocksdb::DB::Open(options, dbfilename, &db_);

  if (!s.ok()) {
    std::cerr << "Can't open rocksdb " << dbfilename << " " << s.ToString()
              << std::endl;
    exit(0);
  }
}

RocksDB::RocksDB(const char *dbfilename, int max_background_jobs)
    : db_(nullptr), cache_(nullptr), db_stats_(nullptr), not_found_(0) {
  rocksdb::Options options;
  SetOptions(&options, max_background_jobs);

  rocksdb::Status s = rocksdb::DB::Open(options, dbfilename, &db_);

  if (!s.ok()) {
    std::cerr << "Can't open rocksdb " << dbfilename << " " << s.ToString()
              << std::endl;
    exit(1);
  }
}

void RocksDB::SetOptions(rocksdb::Options *options, int max_background_jobs) {
  options->create_if_missing = true;
  options->compression = rocksdb::CompressionType::kNoCompression;
  options->max_background_jobs = max_background_jobs;

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
  std::string value;

  rocksdb::Status s = db_->Get(rocksdb::ReadOptions(), key, &value);
  if (s.ok()) {
    return DB::kOK;
  }
  if (s.IsNotFound()) {
    not_found_++;
    db_->Get(rocksdb::ReadOptions(), key, &value);
    return DB::kOK;
  } else {
    std::cerr << "read error!" << std::endl;
    std::cout << "key: " << key << std::endl;
    std::cout << "value: " << value << std::endl;
    return 0;
  }
}

int RocksDB::ReadBatch(std::vector<rocksdb::Slice> keys) {
  const size_t thread_num = 3;
  std::vector<std::thread> thread_pool;
  thread_pool.reserve(thread_num);

  size_t num_per_thread = keys.size() / (thread_num + 1);

  std::vector<std::vector<rocksdb::Slice>> keys_tmps;
  keys_tmps.reserve(thread_num);

  for (size_t i = 0; i < thread_num; ++i) {
    // 创建每个线程需要处理的 key 的临时子集
    keys_tmps.emplace_back(keys.begin() + i * num_per_thread,
                           keys.begin() + (i + 1) * num_per_thread);
    // 将任务添加到线程池
    thread_pool.emplace_back([this, &keys_tmp = keys_tmps.back()] {
      std::vector<std::string> values_tmp;
      std::vector<rocksdb::Status> status =
          db_->MultiGet(rocksdb::ReadOptions(), keys_tmp, &values_tmp);

      for (const auto &s : status) {
        if (s.IsNotFound()) {
          not_found_++;
        }
      }
    });
  }

  std::vector<rocksdb::Slice> keys_last(
      keys.begin() + thread_num * num_per_thread, keys.end());
  std::vector<std::string> values_tmp;

  std::vector<rocksdb::Status> status =
      db_->MultiGet(rocksdb::ReadOptions(), keys_last, &values_tmp);

  for (const auto &s : status) {
    if (s.IsNotFound()) {
      not_found_++;
    }
  }

  for (auto &thread : thread_pool) {
    thread.join();
  }

  return DB::kOK;
}

int RocksDB::Scan(const std::string &table, const std::string &key, int len,
                  const std::vector<std::string> *fields,
                  std::vector<std::vector<KVPair>> &result) {
  auto it = db_->NewIterator(rocksdb::ReadOptions());
  it->Seek(key);
  std::string val;
  std::string k;
  for (int i = 0; i < len && it->Valid(); i++) {
    k = it->key().ToString();
    val = it->value().ToString();
    KVPair pair(k, val);

    it->Next();
  }
  delete it;
  return DB::kOK;
}

int RocksDB::ScanBatch(std::vector<rocksdb::Slice> &keys,
                       std::vector<int> &lens) {
  const size_t thread_num = 3;
  std::vector<std::thread> thread_pool;
  thread_pool.reserve(thread_num);

  size_t num_per_thread = keys.size() / (thread_num + 1);

  std::vector<std::vector<rocksdb::Slice>> keys_tmps;
  std::vector<std::vector<int>> lens_tmps;
  keys_tmps.reserve(thread_num);
  lens_tmps.reserve(thread_num);

  for (size_t i = 0; i < thread_num; ++i) {
    // 创建每个线程需要处理的 key 的临时子集
    keys_tmps.emplace_back(keys.begin() + i * num_per_thread,
                           keys.begin() + (i + 1) * num_per_thread);
    lens_tmps.emplace_back(lens.begin() + i * num_per_thread,
                           lens.begin() + (i + 1) * num_per_thread);

    // 将任务添加到线程池
    thread_pool.emplace_back(
        [this, &keys_tmp = keys_tmps.back(), &lens_tmp = lens_tmps.back()] {
          for (size_t j = 0; j < keys_tmp.size(); ++j) {
            auto it = db_->NewIterator(rocksdb::ReadOptions());
            it->Seek(keys_tmp[j]);

            for (int k = 0; k < lens_tmp[j] && it->Valid(); k++) {
              it->Next();
            }

            delete it;
          }
        });
  }

  std::vector<rocksdb::Slice> keys_last(
      keys.begin() + thread_num * num_per_thread, keys.end());
  std::vector<int> lens_last(lens.begin() + thread_num * num_per_thread,
                             lens.end());

  for (size_t j = 0; j < keys_last.size(); ++j) {
    auto it = db_->NewIterator(rocksdb::ReadOptions());
    it->Seek(keys_last[j]);

    std::string val;
    std::string k;
    for (int i = 0; i < lens_last[j] && it->Valid(); i++) {
      it->Next();
    }

    delete it;
  }

  for (auto &thread : thread_pool) {
    thread.join();
  }

  return 0;
}

int RocksDB::Insert(const std::string &table, const std::string &key,
                    std::vector<KVPair> &values) {
  rocksdb::Status s;

  for (KVPair &p : values) {
    s = db_->Put(rocksdb::WriteOptions(), key, p.second);
    if (!s.ok()) {
      fprintf(stderr, "insert error!\n");
      std::cout << s.ToString() << std::endl;
      exit(0);
    }
  }

  return DB::kOK;
}

int RocksDB::InsertBatch(rocksdb::WriteBatch &batch) {
  rocksdb::Status s = db_->Write(rocksdb::WriteOptions(), &batch);

  if (!s.ok()) {
    fprintf(stderr, "insert error!\n");
    std::cout << s.ToString() << std::endl;
    exit(0);
  }

  return 0;
}

int RocksDB::Update(const std::string &table, const std::string &key,
                    std::vector<KVPair> &values) {
  return Insert(table, key, values);
}

int RocksDB::UpdateBatch(rocksdb::WriteBatch &batch) {
  return InsertBatch(batch);
}

int RocksDB::Delete(const std::string &table, const std::string &key) {
  std::vector<DB::KVPair> values;
  return Insert(table, key, values);
}

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
