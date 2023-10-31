//
// Created by d306 on 10/18/23.
//

#include "my_db.h"

#include <iostream>
#include <thread>

#include "db/cuda/gpu_options.cuh"

namespace ROCKSDB_NAMESPACE {

MyDB::MyDB(DB *db) : db_(db), batch_size_(1000) {}

void MyDB::MyGet(const std::string &key) {
  char *key_char = new char[keySize_];
  memcpy(key_char, key.c_str(), keySize_);

  keys_read_.emplace_back(key_char, keySize_);

  if (keys_read_.size() == batch_size_) {
    TransactionRead(keys_read_);
  }
}

void MyDB::MyUpdate(const std::string &key, const std::string &value) {
  batch_update_.Put(key, value);

  if (batch_update_.Count() == batch_size_) {
    TransactionUpdate(batch_update_);
  }
}

void MyDB::MyInsert(const std::string &key, const std::string &value) {
  batch_insert_.Put(key, value);

  if (batch_insert_.Count() == batch_size_) {
    TransactionInsert(batch_insert_);
  }
}

void MyDB::MyScan(const std::string &key, int len) {
  char *key_char = new char[keySize_];
  memcpy(key_char, key.c_str(), keySize_);

  keys_scan_.emplace_back(key_char, keySize_);
  lens_.emplace_back(len);

  if (keys_scan_.size() == batch_size_) {
    TransactionScan(keys_scan_, lens_);
  }
}

void MyDB::MyModify(const std::string &key, const std::string &value) {
  char *key_char = new char[rocksdb::keySize_];
  memcpy(key_char, key.c_str(), keySize_);

  keys_modify_.emplace_back(key_char, keySize_);

  batch_modify_.Put(key, value);
  if (keys_modify_.size() == batch_size_) {
    TransactionReadModifyWrite(keys_modify_, batch_modify_);
  }
}

void MyDB::Finish() {
  if (!keys_read_.empty()) {
    TransactionRead(keys_read_);
  }

  if (batch_update_.Count() > 0) {
    TransactionUpdate(batch_update_);
  }

  if (batch_insert_.Count() > 0) {
    TransactionInsert(batch_insert_);
  }

  if (!keys_scan_.empty()) {
    TransactionScan(keys_scan_, lens_);
  }

  if (!keys_modify_.empty()) {
    TransactionReadModifyWrite(keys_modify_, batch_modify_);
  }
}

void MyDB::TransactionRead(std::vector<Slice> &keys) {
  ReadBatch(keys);

  for (auto &key : keys) {
    delete[] key.data();
  }

  keys.clear();
}

void MyDB::TransactionUpdate(WriteBatch &batch) {
  UpdateBatch(batch);

  batch.Clear();
}

void MyDB::TransactionInsert(WriteBatch &batch) {
  InsertBatch(batch);

  batch.Clear();
}

void MyDB::TransactionScan(std::vector<Slice> &keys, std::vector<int> &lens) {
  ScanBatch(keys, lens);

  for (auto &key : keys) {
    delete[] key.data();
  }

  keys.clear();
}

void MyDB::TransactionReadModifyWrite(std::vector<Slice> &keys,
                                      WriteBatch &batch) {
  ReadBatch(keys);
  UpdateBatch(batch);

  for (auto &key : keys) {
    delete[] key.data();
  }

  keys.clear();
  batch.Clear();
}

void MyDB::ReadBatch(std::vector<Slice> &keys) {
  const size_t thread_num = 3;
  std::vector<std::thread> thread_pool;
  thread_pool.reserve(thread_num);

  size_t num_per_thread = keys.size() / (thread_num + 1);

  std::vector<std::vector<Slice>> keys_tmps;
  keys_tmps.reserve(thread_num);

  for (size_t i = 0; i < thread_num; ++i) {
    // 创建每个线程需要处理的 key 的临时子集
    keys_tmps.emplace_back(keys.begin() + i * num_per_thread,
                           keys.begin() + (i + 1) * num_per_thread);
    // 将任务添加到线程池
    thread_pool.emplace_back([this, &keys_tmp = keys_tmps.back()] {
      std::vector<std::string> values_tmp;
      std::vector<Status> status =
          db_->MultiGet(ReadOptions(), keys_tmp, &values_tmp);

      for (const auto &s : status) {
        if (s.IsNotFound()) {
          printf("not found\n");
        }
      }
    });
  }

  std::vector<Slice> keys_last(keys.begin() + thread_num * num_per_thread,
                               keys.end());
  std::vector<std::string> values_tmp;

  std::vector<Status> status =
      db_->MultiGet(ReadOptions(), keys_last, &values_tmp);

  for (const auto &s : status) {
    if (s.IsNotFound()) {
      printf("not found\n");
    }
  }

  for (auto &thread : thread_pool) {
    thread.join();
  }
}

void MyDB::ScanBatch(std::vector<Slice> &keys, std::vector<int> &lens) {
  const size_t thread_num = 3;
  std::vector<std::thread> thread_pool;
  thread_pool.reserve(thread_num);

  size_t num_per_thread = keys.size() / (thread_num + 1);

  std::vector<std::vector<Slice>> keys_tmps;
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
            auto it = db_->NewIterator(ReadOptions());
            it->Seek(keys_tmp[j]);

            for (int k = 0; k < lens_tmp[j] && it->Valid(); k++) {
              it->Next();
            }

            delete it;
          }
        });
  }

  std::vector<Slice> keys_last(keys.begin() + thread_num * num_per_thread,
                               keys.end());
  std::vector<int> lens_last(lens.begin() + thread_num * num_per_thread,
                             lens.end());

  for (size_t j = 0; j < keys_last.size(); ++j) {
    auto it = db_->NewIterator(ReadOptions());
    it->Seek(keys_last[j]);

    for (int i = 0; i < lens_last[j] && it->Valid(); i++) {
      it->Next();
    }

    delete it;
  }

  for (auto &thread : thread_pool) {
    thread.join();
  }
}

void MyDB::UpdateBatch(WriteBatch &batch) { InsertBatch(batch); }

void MyDB::InsertBatch(WriteBatch &batch) {
  rocksdb::Status s = db_->Write(WriteOptions(), &batch);

  if (!s.ok()) {
    std::cerr << "insert error\n" << std::endl;
    std::cout << s.ToString() << std::endl;
    exit(1);
  }
}

}  // namespace ROCKSDB_NAMESPACE