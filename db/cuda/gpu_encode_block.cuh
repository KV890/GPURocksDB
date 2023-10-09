//
// Created by jxx on 5/6/23.
//
#pragma once

/**
 * 日志
 *
 * 5.11
 * 起步阶段，目前思路，模仿compaction_job中prepare函数，将结果划分成小块，每一块并行编码，
 * 并行粒度大小定为数据块
 *
 * 数据块的结构不变，1个数据块大小为4KB，每16个键值对插入一个重启点
 * 关键：确定一个数据块中有多少键值对，重启点之间的键值对有多少
 * 再细分就是一个键值对有多大，因为数据块大小是固定的
 *
 * 注意：这里的key是内部key, 内部key包括用户key，sequence和value type
 * 在重启点上的键值对大小，没有共享部分的键
 *  - shared, non_shared, value size
 *      如果Value size 小于2^7，则每个值大小为1B，若小于2^14，则每个值大小为2B,
 * 以此类推 具体细节看EncodeVarint32函数
 *  - internal key size
 *  - value size
 *
 *  没有共享部分的键值对大小 = 4 + internal key size + value size
 *
 * 不在重启点上的键值对大小，有共享部分的键
 *  - shared, non_shared, value size
 *      如果value size
 * 小于2^7，则每个值大小为1B，若小于2^14，则每个值大小为2B，以此类推
 *  - internal key size - shared
 *  - value size
 *
 *  有共享部分的键值对大小 =
 *      EncodeVarint32(shared, non_shared, value size) +
 *              internal key size - shared + value size
 *
 *
 * 非共享和共享键值对总和 = 没有共享部分的键值对大小 + 有共享部分的键值对大小
 *
 * 每16个键值对插入一个重启点，所以
 *  没有共享部分的键值对大小占比 = 1/16
 *  有共享部分的键值对大小占比 = 15/16
 *
 * 数据块中键值对的个数 = 4096 /
 *                       (没有共享部分的键值对大小占比 *
 * 没有共享部分的键值对大小 + 有共享部分的键值对大小占比 *
 * 有共享部分的键值对大小)
 *
 *  比如 internal key size = 24, value size = 32, shared = 6
 *
 *  经过试验，key大小不同时，shared的大小绝大多数是6，少部分是5或7，平均下来也是6
 *  所以确定shared的大小为6，这些仅在db_bench中
 *
 *  没有共享部分的键值对大小 = 3 + 24 + 32 = 59
 *  有共享部分的键值对大小 = 3 + 24 - 6 + 32 = 53
 *  数据块中键值对的个数 = 4096 / ((1/16) * 59 + (15/16) * 53) = 76.74
 *  因为一个数据块的大小不能超过4096，所以需要将结果向下取整，所以该例的键值对数量是
 * 76 76 / 16 = 4.75,
 * 所以76个键值对中有5个是没有共享部分的键值对，71个有共享部分的键值对
 *
 *  两个重启点之间的大小为 59 + 53 * 15 =  854
 *  一个数据块的大小为 5 * 59 + 71 * 53 = 4058 < 4096
 *
 *  以上就是一个数据块中的大致内容，一个GPU线程对应一个数据块，细节还需继续补充
 *
 *  索引块需要每个数据块的最后一个key，所以需要记录每个数据块的最后一个key
 *
 *  对于一个SSTable
 *
 *
 *  5.12
 *   去掉前缀压缩，也就是把shared设为0
 *   这样会使得一个数据块中的键值对数量始终保持相等
 *
 *   这样的话，键值对的大小始终保持在 3 + 24 + 32 = 59
 *   数据块中的键值对数量 = 4096 / 59 = 69
 *
 *   假设是69，此时键值对大小占数据块大小 = 69 * 59 = 4071
 *   69个键值对有重启点 = 69 / 16 = 4 + 1 = 5
 *   5个重启点占4B * 5 =
 * 20字节，重启点数量占4B，重启点数组偏移量占4B，总共28字节 键值对大小 +
 * 重启点相关大小 = 4071 + 24 = 4099 > 4096，所有键值对数量不能为69
 *
 *
 * 5.13
 * 在数据块的大小没有达到这个阈值之前，尽管预测值达到4096也会继续插入键值对
 *  block_size_deviation_limit_(
 *  ((block_size * (100 - block_size_deviation)) + 99) / 100)
 *
 *  准备工作 基本结束
 *
 * 5.15
 *  数据块编码即将结束，现在主要解决trailer的编码问题
 *
 * 5.16
 *  校验码采用crc32算法，目前已解决
 *
 *  对于RocksDB采取xxHash算法，在GPU中较难实现
 *
 *  数据块编码工作基本上结束
 *
 * 优化阶段
 *  5.17
 *   在准备阶段不进行划分文件，当做一个文件来处理
 *   在编码完成后再进行文件划分
 *
 *  编码数据块的优化工作已基本结束
 *
 * 5.22
 *  开始编码索引块
 *
 *  问题：索引块中 索引项的大小不一，有两个原因
 *   第一，索引块中的key用的是缩短之后的key
 *   第二，对offset和size编码用的是可变长编码
 *  以上两个原因是导致索引项大小不一
 *
 *  解决方案：
 *   对于第一个，可以采用全长的key，很容易解决
 *   对于第二个，可能需要扩充字符，或者使用定长编码
 *
 *  对于第二个问题的修改是在较底层进行的，所以可能会导致较多的问题，这需要大量的时间来调试
 *
 *  接下来主要解决第二个问题
 *   思路1：使用全长key的前提下，对索引项中的key进行字符填充，使得每个索引项的大小相等
 *
 * 5.25
 *  将checksum放到CPU来做，对于数据块来说性能提升似乎不是那么明显，甚至在下降
 *  已解决
 *
 * 5.26
 *  注意：索引块的偏移量的值就是数据块的大小
 *  在我的代码中就是 data_buffer_size
 *
 * 5.28
 *  索引块最后一项的size不正确
 *  数据块最后一块不正确
 *
 * 5.30
 *  编码SSTable已结束，进入优化阶段
 *
 * 6.3
 *  GPU并行编码多SSTable
 *
 *  画系统图
 *
 * 6.5
 *  GPU并行编码多SSTable
 *
 *  思路：
 *   数据块编码比较简单，可以直接将所有KV对编码成一个数据块
 *   每个SSTable的数据块数量是确定的，偏移量也是确定的，所以可以根据偏移量对数据块进行划分，
 *   然后对划分好的数据块编码对应的索引块、属性块、meta index block 和footer
 *
 * 6.9
 *  目前对数据传输和写SSTable进行了优化
 *  数据传输主要是用到了内存对齐，写SSTable的优化主要对SSTable进行了压缩
 *
 *  但目前性能还不是特别好，与RocksDB还有差距
 *
 *  主要性能开销，多SSTable进行编码时，虽然是CPU多线程跑的，
 *  但是设备只有一个，所以需要等待，压缩也需要设备，所以也需要等待
 *
 *  经验：能放到GPU常量内存的还是放进去吧
 *
 *  GPU并行解码多SSTable:
 *   一次编码所有数据块肯定要做的
 *   编码数据块时，需要保留每个数据块的最后一个key，也需要确定数据块的偏移量和大小
 *
 *   一个SSTable的数据块确定，索引块也就确定
 *
 * 6.12
 *  目前优化阶段1已结束
 *
 *
 */

#include <cuda_runtime.h>
#include <driver_types.h>
#include <fcntl.h>

#include <csignal>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>

#include "db/compaction/compaction_job.h"
#include "db/cuda/gpu_transform_sort.cuh"
#include "db/cuda/util/gpu_crc32c.cuh"
#include "db/gpu_compaction_stats.h"
#include "gpu_format.cuh"
#include "gpu_slice.cuh"
#include "gpu_table_properties.cuh"
#include "table/block_based/block_based_table_builder.h"
#include "table/block_based/block_based_table_factory.h"
#include "table/meta_blocks.h"

#ifndef __global__
#define __global__
#endif

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __constant__
#define __constant__
#endif

#define CHECK(call)                                           \
  do {                                                        \
    cudaError_t error = call;                                 \
    if (error != cudaSuccess) {                               \
      printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(error));                      \
      exit(1);                                                \
    }                                                         \
  } while (0)

namespace ROCKSDB_NAMESPACE {

class CompactionJob;

/**
 *
 * @param data
 * @param data_size
 * @param last_type
 * @return
 */
__host__ __device__ inline uint32_t GPUComputeBuiltinChecksumWithLastByte(
    const char* data, size_t data_size, char last_type) {
  uint32_t crc = GPUValue(data, data_size);
  crc = GPUExtend(crc, &last_type, 1);
  return GPUMask(crc);
}

/**
 *
 * @param dst
 * @param v1
 * @param v2
 * @param v3
 * @param encoded_size
 */
__host__ __device__ inline void GPUPutVarint32Varint32Varint32(
    char* dst, uint32_t v1, uint32_t v2, uint32_t v3, size_t& encoded_size) {
  char buf[15];
  char* ptr = GPUEncodeVarint32(buf, v1);
  ptr = GPUEncodeVarint32(ptr, v2);
  ptr = GPUEncodeVarint32(ptr, v3);
  encoded_size = static_cast<size_t>(ptr - buf);
  memcpy(dst, buf, encoded_size);
}

/**
 *
 * @param dst
 * @param v1
 * @param v2
 * @param encoded_size
 */
__host__ __device__ inline void GPUPutVarint32Varint32(char* dst, uint32_t v1,
                                                       uint32_t v2,
                                                       size_t& encoded_size) {
  char buf[10];
  char* ptr = GPUEncodeVarint32(buf, v1);
  ptr = GPUEncodeVarint32(ptr, v2);
  encoded_size = static_cast<size_t>(ptr - buf);
  memcpy(dst, buf, encoded_size);
}

/**
 *
 * @param dst
 * @param value
 */
__host__ __device__ inline void GPUPutFixed32(char* dst, uint32_t value) {
  memcpy(dst, const_cast<const char*>(reinterpret_cast<char*>(&value)),
         sizeof(uint32_t));
}

/**
 *
 * @param dst
 * @param v1
 * @param v2
 */
__host__ __device__ inline void GPUPutFixed64Varint64(char* dst, uint64_t v1,
                                                      uint64_t v2) {
  char buf[20];
  GPUEncodeFixed64(buf, v1);
  char* ptr = GPUEncodeVarint64(buf + sizeof(uint64_t), v2);

  memcpy(dst, buf, static_cast<size_t>(ptr - buf));
}

/**
 *
 * @param buffer
 * @param size_block
 */
__host__ inline void ComputeChecksum(char* buffer, size_t num_data_block,
                                     size_t size_block) {
  GPUPutFixed32(buffer - 9, num_data_block);

  // 计算trailer
  char trailer[5];
  char type = 0x0;
  trailer[0] = type;  // 表示kNoCompression
  uint32_t checksum = ComputeBuiltinChecksumWithLastByte(
      ChecksumType::kCRC32c, buffer - size_block, size_block - 5, type);
  EncodeFixed32(trailer + 1, checksum);

  // add checksum to buffer
  memcpy(buffer - 5, trailer, 5);
}

/**
 *
 * @param keyValue
 * @param inputFiles
 * @param num_inputs
 * @param value
 */
__host__ __device__ inline void ExtractOriginalValue(
    const GPUKeyValue& keyValue, const InputFile* inputFiles, size_t num_inputs,
    char* value) {
  const char* file = nullptr;
  for (size_t i = 0; i < num_inputs; ++i) {
    if (inputFiles[i].file_number == keyValue.file_number) {
      file = inputFiles[i].file;
      break;
    }
  }

  if (file != nullptr) {
    const char* p = file;
    const char* valuePtr = reinterpret_cast<const char*>(p + keyValue.valuePtr);
    memcpy(value, valuePtr, valueSize_);
  }
}

/**
 *
 * @param dst
 * @param offset
 * @param size
 * @return
 */
__device__ inline char* GPUEncodeTo(char* dst, uint64_t offset, uint64_t size) {
  char* cur = GPUEncodeVarint64(dst, offset);
  cur = GPUEncodeVarint64(cur, size);
  return cur;
}

inline void WriteSSTable(char* buffer_d, const std::string& filename,
                         size_t file_size) {
  int fd = open(filename.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0664);

  CUfileDescr_t descr;
  descr.handle.fd = fd;
  descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

  CUfileHandle_t handle;
  cuFileHandleRegister(&handle, &descr);

  cuFileWrite(handle, buffer_d, file_size, 0, 0);

  cuFileHandleDeregister(handle);
  close(fd);
}

/**
 * 并行方式
 * 采用
 *
 * @param kv_idx
 * @param block_idx
 * @param block_buffer
 * @param keyValues
 * @param inputFiles
 * @param num_kv_data_block
 */
__device__ void ProcessKeyValue(unsigned int kv_idx, unsigned int block_idx,
                                char* block_buffer, GPUKeyValue* keyValues,
                                InputFile* inputFiles,
                                size_t num_kv_data_block);

/**
 *
 * @param block_buffer
 * @param num_kv_data_block
 * @param num_restarts
 * @param restarts
 * @param size_block
 */
__device__ void ProcessRestartsChecksum(char* block_buffer,
                                        size_t num_kv_data_block,
                                        uint32_t num_restarts,
                                        uint32_t* restarts, size_t size_block);

/**
 *
 * @param buffer
 * @param keyValues
 * @param inputFiles
 */
__global__ void BuildKeyValueKernel(char* buffer, GPUKeyValue* keyValues,
                                    InputFile* inputFiles);

/**
 *
 * @param buffer
 * @param keyValues
 * @param inputFiles
 */
__global__ void BuildLastKeyValueKernel(char* buffer, GPUKeyValue* keyValues,
                                        InputFile* inputFiles);

/**
 *
 * @param buffer
 * @param index_key
 * @param keyValues_d
 */
__global__ void BuildRestartsChecksumKernel(char* buffer, char* index_key,
                                            GPUKeyValue* keyValues_d);

/**
 *
 * @param buffer
 * @param num_kv_data_block
 * @param num_restarts
 * @param restarts
 * @param index_key
 * @param keyValues_d
 * @param block_size
 */
__global__ void BuildLastRestartsChecksumKernel(
    char* buffer, size_t num_kv_data_block, uint32_t num_restarts,
    uint32_t* restarts, char* index_key, GPUKeyValue* keyValues_d,
    size_t size_incomplete_data_block);

/**
 *
 * @param buffer_d
 * @param keyValues_d
 * @param inputFiles_d
 * @param num_kv_data_block
 * @param info
 * @param size_incomplete_data_block
 * @param buffer_size
 * @param last_num_restarts
 * @param last_restarts
 * @param index_key
 * @param stream
 * @param start
 * @param stop
 */
void BuildDataBlock(char** buffer_d, GPUKeyValue* keyValues_d,
                    InputFile* inputFiles_d, size_t num_kv_data_block,
                    SSTableInfo info, size_t size_incomplete_data_block,
                    uint32_t last_num_restarts, uint32_t* last_restarts,
                    char* index_key, cudaStream_t* stream, cudaEvent_t start,
                    cudaEvent_t stop);

/**
 *
 * @param buffer
 * @param index_keys
 * @param block_handle
 * @param restarts
 */
__global__ void BuildIndexBlockKernel(char* buffer, char* index_keys,
                                      GPUBlockHandle* block_handle,
                                      uint32_t* restarts);

/**
 *
 * @param buffer_h
 * @param buffer_d
 * @param index_keys
 * @param block_handle
 * @param restarts
 * @param size_index_block
 * @param info
 * @param stream
 * @param start
 * @param stop
 */
void BuildIndexBlock(char** buffer_h, char** buffer_d, char* index_keys,
                     GPUBlockHandle* block_handle, size_t num_data_block,
                     uint32_t* restarts, size_t size_index_block,
                     cudaStream_t* stream, cudaEvent_t start, cudaEvent_t stop);

/**
 * 一个线程即可
 *
 * @param buffer_d
 * @param block
 * @param block_size
 * @param checksum
 */
__global__ void WriteBlocksKernel(char* buffer_d, char* block,
                                  size_t block_size, uint32_t checksum);

/**
 * CPU执行
 *
 * @param buffer
 * @param block_contents
 * @param handle
 * @param last_offset
 * @param new_offset
 */
void WriteBlock(char** buffer, const Slice& block_contents, BlockHandle* handle,
                size_t last_offset, size_t& new_offset);

/**
 *
 * @param buffer_d
 * @param block_contents
 * @param handle
 * @param last_offset
 * @param new_offset
 */
void GPUWriteBlock(char** buffer_d, const Slice& block_contents,
                   BlockHandle* handle, size_t last_offset, size_t& new_offset);

/**
 * CPU执行
 *
 * @param buffer
 * @param meta
 * @param tboptions
 * @param data_size
 * @param index_size
 * @param metaIndexBuilder
 * @param new_offset
 * @param props
 */
[[maybe_unused]] void BuildPropertiesBlock(
    char** buffer, FileMetaData* meta,
    const std::shared_ptr<TableBuilderOptions>& tboptions, size_t data_size,
    size_t index_size, MetaIndexBuilder* metaIndexBuilder, size_t& new_offset,
    TableProperties* props);

/**
 *
 * @param buffer_d
 * @param meta
 * @param tboptions
 * @param data_size
 * @param index_size
 * @param metaIndexBuilder
 * @param new_offset
 * @param props
 */
void GPUBuildPropertiesBlock(
    char** buffer_d, FileMetaData* meta,
    const std::shared_ptr<TableBuilderOptions>& tboptions, size_t data_size,
    size_t index_size, MetaIndexBuilder* metaIndexBuilder, size_t& new_offset,
    TableProperties* props);

/**
 * CPU执行
 *
 * @param buffer
 * @param metaIndexBuilder
 * @param meta_index_block_handle
 * @param last_offset
 * @param new_offset
 */
[[maybe_unused]] [[maybe_unused]] [[maybe_unused]] void BuildMetaIndexBlock(
    char** buffer, MetaIndexBuilder* metaIndexBuilder,
    BlockHandle* meta_index_block_handle, size_t last_offset,
    size_t& new_offset);

/**
 *
 * @param buffer_d
 * @param metaIndexBuilder
 * @param meta_index_block_handle
 * @param last_offset
 * @param new_offset
 */
void GPUBuildMetaIndexBlock(char** buffer_d, MetaIndexBuilder* metaIndexBuilder,
                            BlockHandle* meta_index_block_handle,
                            size_t last_offset, size_t& new_offset);

/**
 *
 * @param buffer_d
 * @param index_block_offset
 * @param index_block_size
 * @param meta_index_offset
 * @param meta_index_size
 * @param checksum_type
 * @param format_version
 * @param magic_number
 */
__global__ void BuilderFooterKernel(char* buffer_d, char* footer_d,
                                    size_t footer_size);

[[maybe_unused]] [[maybe_unused]] [[maybe_unused]] void BuildFooter(
    char** buffer, BlockHandle& index_block_handle,
    BlockHandle& meta_index_block_handle, size_t last_offset,
    size_t& new_offset);

/**
 *
 * @param buffer_d
 * @param index_block_handle
 * @param meta_index_block_handle
 * @param last_offset
 * @param new_offset
 */
void GPUBuildFooter(char** buffer_d, BlockHandle& index_block_handle,
                    BlockHandle& meta_index_block_handle, size_t last_offset,
                    size_t& new_offset);

/**
 *
 * @param keyValues
 * @param inputFiles_d
 * @param info
 * @param num_kv_data_block
 * @param num_inputs
 * @param meta
 * @param tboptions
 * @param tp
 * @return
 */
char* BuildSSTable(const GPUKeyValue* keyValues, InputFile* inputFiles_d,
                   SSTableInfo& info, size_t num_kv_data_block,
                   FileMetaData& meta,
                   std::shared_ptr<TableBuilderOptions>& tboptions,
                   TableProperties& tp);

// 以下是GPU编码多SSTable的函数声明
/**
 *
 * @param file_idx
 * @param block_idx
 * @param current_block_buffer
 * @param key_values_d
 * @param input_files_d
 * @param index_keys_d
 * @param num_kv_current_data_block
 * @param total_num_kv_front_file
 * @param num_current_restarts
 * @param current_restarts
 * @param size_current_data_block
 */
__device__ void BeginBuildDataBlock(
    uint32_t file_idx, uint32_t block_idx, char* current_block_buffer,
    GPUKeyValue* key_values_d, InputFile* input_files_d, char* index_keys_d,
    size_t num_kv_current_data_block, uint32_t num_current_restarts,
    uint32_t* current_restarts, size_t size_current_data_block);

/**
 * 编码前几个文件(不包含最后一个文件)的数据块
 *
 * @param buffer_d
 * @param key_values_d
 * @param input_files_d
 * @param index_keys_d
 */
__global__ void BuildDataBlocksFrontFileKernel(char* buffer_d,
                                               GPUKeyValue* key_values_d,
                                               InputFile* input_files_d,
                                               char* index_keys_d);

/**
 * 编码最后一个文件的前面的数据块(不包含最后一个数据块)
 *
 * @param buffer_d
 * @param key_values_d
 * @param input_files_d
 * @param index_keys_d
 */
__global__ void BuildDataBlocksLastFileKernel(char* buffer_d,
                                              GPUKeyValue* key_values_d,
                                              InputFile* input_files_d,
                                              char* index_keys_d);

/**
 * 编码最后一个文件的最后一个的数据块
 *
 * @param buffer_d
 * @param key_values_d
 * @param input_files_d
 * @param index_keys_d
 * @param num_data_block_last_file
 * @param num_kv_last_data_block_last_file
 * @param num_restarts_last_data_block_last_file
 * @param restarts_last_data_block_last_file_d
 * @param size_incomplete_data_block
 */
__global__ void BuildLastDataBlockLastFileKernel(
    char* buffer_d, GPUKeyValue* key_values_d, InputFile* input_files_d,
    char* index_keys_d, size_t num_data_block_last_file,
    size_t num_kv_last_data_block_last_file,
    uint32_t num_restarts_last_data_block_last_file,
    uint32_t* restarts_last_data_block_last_file_d,
    size_t size_incomplete_data_block);

/**
 * 编码索引文件的数据块
 *
 * @param buffer_d
 * @param key_values_d
 * @param input_files_d
 * @param index_keys_d
 * @param size_incomplete_data_block
 * @param num_data_block_front_file
 * @param num_data_block_last_file
 * @param num_kv_last_data_block_last_file
 * @param num_restarts_last_data_block_last_file
 * @param restarts_last_data_block_last_file_d
 * @param num_outputs
 * @param stream
 */
void BuildDataBlocks(char** buffer_d, GPUKeyValue* key_values_d,
                     InputFile* input_files_d, char* index_keys_d,
                     size_t size_incomplete_data_block,
                     size_t num_data_block_front_file,
                     size_t num_data_block_last_file,
                     size_t num_kv_last_data_block_last_file,
                     uint32_t num_restarts_last_data_block_last_file,
                     uint32_t* restarts_last_data_block_last_file_d,
                     size_t num_outputs, cudaStream_t* stream);

/**
 * 构建索引块的主要逻辑
 *
 * @param file_idx
 * @param block_idx
 * @param current_index_buffer
 * @param index_keys_d
 * @param block_handles_d
 * @param current_restarts
 * @param current_num_data_block
 */
__device__ void BeginBuildIndexBlock(uint32_t file_idx, uint32_t block_idx,
                                     char* current_index_buffer,
                                     char* index_keys_d,
                                     GPUBlockHandle* block_handles_d,
                                     uint32_t* current_restarts,
                                     size_t current_num_data_block);

/**
 * 并行构建所有前面文件的索引块
 * @param buffer_d
 * @param index_keys_d
 * @param block_handles_d
 * @param restarts_for_index_front_file_d
 */
__global__ void BuildIndexBlockFrontFileKernel(
    char* buffer_d, char* index_keys_d, GPUBlockHandle* block_handles_d,
    uint32_t* restarts_for_index_front_file_d);

/**
 * 并行构建最后一个文件的索引块
 *
 * @param buffer_d
 * @param index_keys_d
 * @param block_handles_d
 * @param restarts_for_index_last_file_d
 * @param num_data_block_last_file
 */
__global__ void BuildIndexBlockLastFileKernel(
    char* buffer_d, char* index_keys_d, GPUBlockHandle* block_handles_d,
    uint32_t* restarts_for_index_last_file_d, size_t num_data_block_last_file);

/**
 * 计算checksum主要逻辑
 *
 * @param current_buffer
 * @param num_data_block
 * @param index_block_size
 */
__device__ void BeginComputeChecksum(char* current_buffer,
                                     size_t num_data_block,
                                     size_t index_block_size);

/**
 * 计算前面文件的索引块的checksum
 *
 * @param buffer_d
 */
[[maybe_unused]] __global__ void ComputeChecksumFrontFileKernel(char* buffer_d);

/**
 * 计算最后一个文件的索引块的checksum
 *
 * @param buffer_d
 * @param num_data_block_last_file
 * @param index_size_last_file
 */
[[maybe_unused]] __global__ void ComputeChecksumLastFileKernel(
    char* buffer_d, size_t num_data_block_last_file,
    size_t index_size_last_file);

/**
 * 并行计算前面文件的所有数据块的偏移量和大小
 *
 * @param block_handles_d
 * @param restarts_for_index_front_file_d
 */
__global__ void ComputeDataBlockHandleFrontFileKernel(
    GPUBlockHandle* block_handles_d, uint32_t* restarts_for_index_front_file_d);

/**
 * 并行计算最后一个文件的所有数据块的偏移量和大小
 *
 * @param block_handles_d
 * @param restarts_for_index_last_file_d
 * @param size_incomplete_data_block
 */
__global__ void ComputeDataBlockHandleLastFileKernel(
    GPUBlockHandle* block_handles_d, uint32_t* restarts_for_index_last_file_d,
    size_t size_incomplete_data_block, size_t num_data_block_last_file);

/**
 * 编码所有SSTable的索引块
 *
 * @param buffer_d 指向设备内存
 * @param index_keys_d
 * @param block_handles_d
 * @param num_outputs
 * @param num_data_block_front_file
 * @param num_data_block_last_file
 * @param index_size_last_file
 * @param size_incomplete_data_block
 * @param restarts_front_file
 * @param restarts_last_file
 * @param stream
 */
void BuildIndexBlocks(char** buffer_d, char* index_keys_d,
                      GPUBlockHandle* block_handles_d, size_t num_outputs,
                      size_t num_data_block_front_file,
                      size_t num_data_block_last_file,
                      size_t index_size_last_file,
                      size_t size_incomplete_data_block,
                      uint32_t* restarts_front_file,
                      uint32_t* restarts_last_file, cudaStream_t* stream);

/**
 * 写其他三个块
 *
 * @param buffer_d 指向主机内存，表示每个SSTable首地址
 * @param file_writer
 * @param meta
 * @param tbs
 * @param tp
 * @param info
 * @param data_size
 * @param index_size
 */
void WriteOtherBlocks(char* buffer_d,
                      const std::shared_ptr<TableBuilderOptions>& tbs,
                      FileMetaData* meta, TableProperties* tp,
                      SSTableInfo* info, size_t data_size, size_t index_size);

/**
 * GPU并行编码多SSTable
 *
 * @param key_values_d
 * @param input_files_d
 * @param infos
 * @param metas
 * @param file_writes
 * @param tbs
 * @param tps
 * @param num_kv_data_block
 */
void BuildSSTables(
    GPUKeyValue* key_values_d, InputFile* input_files_d,
    std::vector<SSTableInfo>& infos, std::vector<FileMetaData>& metas,
    std::vector<std::shared_ptr<WritableFileWriter>>& file_writes,
    std::vector<std::shared_ptr<TableBuilderOptions>>& tbs,
    std::vector<TableProperties>& tps, size_t num_kv_data_block,
    cudaStream_t* stream);

}  // namespace ROCKSDB_NAMESPACE
