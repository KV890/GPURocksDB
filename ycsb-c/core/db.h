//
//  db.h
//  YCSB-C
//
//  Created by Jinglei Ren on 12/10/14.
//  Copyright (c) 2014 Jinglei Ren <jinglei@ren.systems>.
//

#ifndef YCSB_C_DB_H_
#define YCSB_C_DB_H_

#include <string>
#include <vector>

#include "port/port_posix.h"
#include "rocksdb/cache.h"
#include "rocksdb/db.h"
#include "rocksdb/filter_policy.h"
#include "rocksdb/options.h"
#include "rocksdb/statistics.h"
#include "rocksdb/table.h"
#include "rocksdb/utilities/optimistic_transaction_db.h"
#include "util/mutexlock.h"

namespace ycsbc {

static std::string ColumnFamilyName(size_t i) {
  if (i == 0) {
    return rocksdb::kDefaultColumnFamilyName;
  } else {
    char name[100];
    snprintf(name, sizeof(name), "column_family_name_%06zu", i);
    return name;
  }
}

struct DBWithColumnFamilies {
  std::vector<rocksdb::ColumnFamilyHandle *> cfh;
  rocksdb::DB *db;
  rocksdb::OptimisticTransactionDB *opt_txn_db;
  std::atomic<size_t> num_created;  // Need to be updated after all the
                                    // new entries in cfh are set.
  size_t num_hot;  // Number of column families to be queried at each moment.
                   // After each CreateNewCf(), another num_hot number of new
                   // Column families will be created and used to be queried.
  rocksdb::port::Mutex
      create_cf_mutex;  // Only one thread can execute CreateNewCf()
  std::vector<int> cfh_idx_to_prob;  // ith index holds probability of operating
                                     // on cfh[i].

  DBWithColumnFamilies() : db(nullptr), opt_txn_db(nullptr) {
    cfh.clear();
    num_created = 0;
    num_hot = 0;
  }

  DBWithColumnFamilies(const DBWithColumnFamilies &other)
      : cfh(other.cfh),
        db(other.db),
        opt_txn_db(other.opt_txn_db),
        num_created(other.num_created.load()),
        num_hot(other.num_hot),
        cfh_idx_to_prob(other.cfh_idx_to_prob) {}

  void DeleteDBs() {
    std::for_each(cfh.begin(), cfh.end(),
                  [](rocksdb::ColumnFamilyHandle *cfhi) { delete cfhi; });
    cfh.clear();
    if (opt_txn_db) {
      delete opt_txn_db;
      opt_txn_db = nullptr;
    } else {
      delete db;
      db = nullptr;
    }
  }

  rocksdb::ColumnFamilyHandle *GetCfh(int64_t rand_num) {
    assert(num_hot > 0);
    size_t rand_offset = 0;
    if (!cfh_idx_to_prob.empty()) {
      assert(cfh_idx_to_prob.size() == num_hot);
      int sum = 0;
      while (sum + cfh_idx_to_prob[rand_offset] < rand_num % 100) {
        sum += cfh_idx_to_prob[rand_offset];
        ++rand_offset;
      }
      assert(rand_offset < cfh_idx_to_prob.size());
    } else {
      rand_offset = rand_num % num_hot;
    }
    return cfh[num_created.load(std::memory_order_acquire) - num_hot +
               rand_offset];
  }

  // stage: assume CF from 0 to stage * num_hot has be created. Need to create
  //        stage * num_hot + 1 to stage * (num_hot + 1).
  void CreateNewCf(rocksdb::ColumnFamilyOptions options, int64_t stage) {
    rocksdb::MutexLock l(&create_cf_mutex);
    if ((stage + 1) * num_hot <= num_created) {
      // Already created.
      return;
    }
    auto new_num_created = num_created + num_hot;
    assert(new_num_created <= cfh.size());
    for (size_t i = num_created; i < new_num_created; i++) {
      rocksdb::Status s =
          db->CreateColumnFamily(options, ColumnFamilyName(i), &(cfh[i]));
      if (!s.ok()) {
        fprintf(stderr, "create column family error: %s\n",
                s.ToString().c_str());
        abort();
      }
    }
    num_created.store(new_num_created, std::memory_order_release);
  }
};

class DB {
 public:
  DBWithColumnFamilies db_with_cf;

  typedef std::pair<std::string, std::string> KVPair;
  static const int kOK = 0;
  static const int kErrorNoData = 1;
  static const int kErrorConflict = 2;

  ///
  /// Initializes any state for accessing this DB.
  /// Called once per DB client (thread); there is a single DB instance
  /// globally.
  ///
  virtual void Init() {}
  ///
  /// Clears any state for accessing this DB.
  /// Called once per DB client (thread); there is a single DB instance
  /// globally.
  ///
  virtual void Close() {}
  ///
  /// Reads a record from the database.
  /// Field/value pairs from the result are stored in a vector.
  ///
  /// @param table The name of the table.
  /// @param key The key of the record to read.
  /// @param fields The list of fields to read, or NULL for all of them.
  /// @param result A vector of field/value pairs for the result.
  /// @return Zero on success, or a non-zero error code on error/record-miss.
  ///
  virtual int Read(const std::string &table, const std::string &key,
                   const std::vector<std::string> *fields,
                   std::vector<KVPair> &result, uint64_t sequence) = 0;

  ///
  /// Performs a range scan for a set of records in the database.
  /// Field/value pairs from the result are stored in a vector.
  ///
  /// @param table The name of the table.
  /// @param key The key of the first record to read.
  /// @param record_count The number of records to read.
  /// @param fields The list of fields to read, or NULL for all of them.
  /// @param result A vector of vector, where each vector contains field/value
  ///        pairs for one record
  /// @return Zero on success, or a non-zero error code on error.
  ///
  virtual int Scan(const std::string &table, const std::string &key,
                   int record_count, const std::vector<std::string> *fields,
                   std::vector<std::vector<KVPair>> &result, uint64_t sequence) = 0;
  ///
  /// Updates a record in the database.
  /// Field/value pairs in the specified vector are written to the record,
  /// overwriting any existing values with the same field names.
  ///
  /// @param table The name of the table.
  /// @param key The key of the record to write.
  /// @param values A vector of field/value pairs to update in the record.
  /// @return Zero on success, a non-zero error code on error.
  ///
  virtual int Update(const std::string &table, const std::string &key,
                     std::vector<KVPair> &values, uint64_t sequence) = 0;
  ///
  /// Inserts a record into the database.
  /// Field/value pairs in the specified vector are written into the record.
  ///
  /// @param table The name of the table.
  /// @param key The key of the record to insert.
  /// @param values A vector of field/value pairs to insert in the record.
  /// @return Zero on success, a non-zero error code on error.
  ///
  virtual int Insert(const std::string &table, const std::string &key,
                     std::vector<KVPair> &values, uint64_t sequence) = 0;

  /**
   * Batch insertion
   *
   * @param batch
   * @return
   */
  virtual int InsertBatch(rocksdb::WriteBatch batch) = 0;

  ///
  /// Deletes a record from the database.
  ///
  /// @param table The name of the table.
  /// @param key The key of the record to delete.
  /// @return Zero on success, a non-zero error code on error.
  ///
  virtual int Delete(const std::string &table, const std::string &key, uint64_t sequence) = 0;

  virtual void PrintMyStats() = 0;

  virtual ~DB(){};
};

}  // namespace ycsbc

#endif  // YCSB_C_DB_H_
