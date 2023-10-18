//
//  basic_db.cc
//  YCSB-C
//
//  Created by Jinglei Ren on 12/17/14.
//  Copyright (c) 2014 Jinglei Ren <jinglei@ren.systems>.
//

#include "db_factory.h"

#include <string>

#include "rocksdb_db.h"

using namespace std;
using ycsbc::DB;
using ycsbc::DBFactory;
using ycsbc::RocksDB;

DB* DBFactory::CreateDB(utils::Properties& props) {
  if (props["dbname"] == "rocksdb") {
    return new RocksDB(props["dbfilename"].c_str());
  }

  return nullptr;
}
