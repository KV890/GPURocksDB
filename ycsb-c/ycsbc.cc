//
//  ycsb-c.cc
//  YCSB-C
// -
//  Created by Jinglei Ren on 12/19/14.
//  Copyright (c) 2014 Jinglei Ren <jinglei@ren.systems>.
//

#include <chrono>
#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <string>
#include <thread>

#include "core/client.h"
#include "core/core_workload.h"
#include "core/timer.h"
#include "db/db_factory.h"
#include "db/gpu_compaction_stats.h"

using namespace std;

void UsageMessage(const char *command);
bool StrStartWith(const char *str, const char *pre);
std::string ParseCommandLine(int argc, const char *argv[],
                             utils::Properties &props);

int DelegateClient(ycsbc::DB *db, ycsbc::CoreWorkload *wl, size_t num_ops,
                   bool is_loading) {
  time_t now = time(nullptr);
  fprintf(stderr, "Time: %s", ctime(&now));  // ctime() adds newline
  db->Init();
  ycsbc::Client client(*db, *wl);
  int oks = 0;
  int ops_stage = 100;

  for (size_t i = 0; i < num_ops; ++i) {
    if (is_loading) {
      oks += client.DoInsert();
    } else {
      if (oks == 33800000) {
        printf("\n");
      }

      oks += client.DoTransaction();
    }

    if (oks >= ops_stage) {
      if (ops_stage < 1000)
        ops_stage += 100;
      else if (ops_stage < 5000)
        ops_stage += 500;
      else if (ops_stage < 10000)
        ops_stage += 1000;
      else if (ops_stage < 50000)
        ops_stage += 5000;
      else if (ops_stage < 100000)
        ops_stage += 10000;
      else if (ops_stage < 500000)
        ops_stage += 50000;
      else
        ops_stage += 100000;
      fprintf(stderr, "... finished %d ops\n", oks);
      fflush(stderr);
    }
  }

  db->Close();
  return oks;
}

int Run(utils::Properties props, const std::string &filename) {
  ycsbc::DB *db = ycsbc::DBFactory::CreateDB(props);

  if (!db) {
    cout << "Unknown database name " << props["dbname"] << endl;
    exit(0);
  }

  ycsbc::CoreWorkload wl;
  wl.Init(props);

  // Loads data
  cerr << "--------------------Loading--------------------" << endl;

  size_t total_ops = stoll(props[ycsbc::CoreWorkload::RECORD_COUNT_PROPERTY]);

  utils::Timer<double> timer_load;
  timer_load.Start();

  size_t sum = DelegateClient(db, &wl, total_ops, true);

  db->FinishRun();

  double duration_load = timer_load.End();

  std::this_thread::sleep_for(std::chrono::seconds(1));

  rocksdb::gpu_stats.PrintStats();
  rocksdb::gpu_stats.ResetStats();

  cerr << "\n----------------------Loading-----------------------" << endl;
  cerr << "Loading records : " << sum << endl;
  cerr << "Loading duration: " << duration_load << " sec" << endl;
  cerr << "Loading throughput (KTPS): "
       << static_cast<double>(total_ops) / duration_load / 1000 << endl;
  cerr << "----------------------------------------------------" << endl;

  std::this_thread::sleep_for(std::chrono::seconds(1));

  cerr << endl;

  // Performs transactions
  cerr << "--------------------Transaction--------------------" << endl;

  size_t num_ops = stoll(props[ycsbc::CoreWorkload::OPERATION_COUNT_PROPERTY]);

  utils::Timer<double> timer;
  timer.Start();

  size_t sum_transaction = DelegateClient(db, &wl, num_ops, false);

  db->FinishRun();

  double duration = timer.End();

  std::this_thread::sleep_for(std::chrono::seconds(1));

  db->PrintMyStats();
  rocksdb::gpu_stats.PrintStats();

  std::this_thread::sleep_for(std::chrono::seconds(2));
  cerr << endl;

  cerr << "DB name: " << props["dbname"] << endl;
  cerr << "Workload filename: " << filename << endl;

  cerr << "----------------------Loading-----------------------" << endl;
  cerr << "Loading records : " << sum << endl;
  cerr << "Loading duration: " << duration_load << " sec" << endl;
  cerr << "Loading throughput (KTPS): "
       << static_cast<double>(total_ops) / duration_load / 1000 << endl;
  cerr << "----------------------------------------------------" << endl;

  cerr << "--------------------Transaction---------------------" << endl;
  cerr << "Read proportion: "
       << std::stod(props[ycsbc::CoreWorkload::READ_PROPORTION_PROPERTY])
       << endl;
  cerr << "Transactions records : " << sum_transaction << endl;
  cerr << "Transactions duration: " << duration << " sec" << endl;
  cerr << "Transaction throughput (KTPS): "
       << static_cast<double>(num_ops) / duration / 1000 << endl;
  cerr << "----------------------------------------------------" << endl;

  return 0;
}

std::string ParseCommandLine(int argc, const char *argv[],
                             utils::Properties &props) {
  std::string filename;
  int argindex = 1;
  while (argindex < argc && StrStartWith(argv[argindex], "-")) {
    if (strcmp(argv[argindex], "-db") == 0) {
      argindex++;
      if (argindex >= argc) {
        UsageMessage(argv[0]);
        exit(0);
      }
      props.SetProperty("dbname", argv[argindex]);
      argindex++;
    } else if (strcmp(argv[argindex], "-host") == 0) {
      argindex++;
      if (argindex >= argc) {
        UsageMessage(argv[0]);
        exit(0);
      }
      props.SetProperty("host", argv[argindex]);
      argindex++;
    } else if (strcmp(argv[argindex], "-port") == 0) {
      argindex++;
      if (argindex >= argc) {
        UsageMessage(argv[0]);
        exit(0);
      }
      props.SetProperty("port", argv[argindex]);
      argindex++;
    } else if (strcmp(argv[argindex], "-slaves") == 0) {
      argindex++;
      if (argindex >= argc) {
        UsageMessage(argv[0]);
        exit(0);
      }
      props.SetProperty("slaves", argv[argindex]);
      argindex++;
    } else if (strcmp(argv[argindex], "-filename") == 0) {
      argindex++;
      if (argindex >= argc) {
        UsageMessage(argv[0]);
        exit(0);
      }

      props.SetProperty("dbfilename", argv[argindex]);
      argindex++;
    } else if (strcmp(argv[argindex], "-configpath") == 0) {
      argindex++;
      if (argindex >= argc) {
        UsageMessage(argv[0]);
        exit(0);
      }
      props.SetProperty("configpath", argv[argindex]);
      argindex++;
    } else if (strcmp(argv[argindex], "-P") == 0) {
      argindex++;
      if (argindex >= argc) {
        UsageMessage(argv[0]);
        exit(0);
      }
      filename.assign(argv[argindex]);
      ifstream input(filename);
      try {
        props.Load(input);
      } catch (const string &message) {
        cout << message << endl;
        exit(0);
      }
      input.close();
      argindex++;
    } else {
      cout << "Unknown option '" << argv[argindex] << "'" << endl;
      exit(0);
    }
  }

  std::error_code ec;
  for (auto &entry : filesystem::directory_iterator(
           props.GetProperty("dbfilename"), ec)) {
    filesystem::remove_all(entry, ec);
  }

  if (argindex == 1 || argindex != argc) {
    UsageMessage(argv[0]);
    exit(0);
  }

  return filename;
}

void UsageMessage(const char *command) {
  cout << "Usage: " << command << " [options]" << endl;
  cout << "Options:" << endl;
  cout << "  -threads n: execute using n threads (default: 1)" << endl;
  cout << "  -db dbname: specify the name of the DB to use (default: basic)"
       << endl;
  cout << "  -P propertyfile: load properties from the given file. Multiple "
          "files can"
       << endl;
  cout << "                   be specified, and will be processed in the order "
          "specified"
       << endl;
}

inline bool StrStartWith(const char *str, const char *pre) {
  return strncmp(str, pre, strlen(pre)) == 0;
}

int main(const int argc, const char *argv[]) {
  utils::Properties props;
  string filename = ParseCommandLine(argc, argv, props);

  rocksdb::gpu_stats.OpenCuFileDriver();

  Run(props, filename);

  rocksdb::gpu_stats.CloseCuFileDriver();

  return 0;
}
