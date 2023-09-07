//
// Created by jxx on 4/6/23.
//

#include "gpu_format.cuh"

namespace ROCKSDB_NAMESPACE {

// Footer format, in three parts:
// * Part1
//   -> format_version == 0 (inferred from legacy magic number)
//      <empty> (0 bytes)
//   -> format_version >= 1
//      checksum type (char, 1 byte)
// * Part2
//      metaindex handle (varint64 offset, varint64 size)
//      index handle     (varint64 offset, varint64 size)
//      <zero padding> for part2 size = 2 * BlockHandle::kMaxEncodedLength = 40
// * Part3
//   -> format_version == 0 (inferred from legacy magic number)
//      legacy magic number (8 bytes)
//   -> format_version >= 1 (inferred from NOT legacy magic number)
//      format_version (uint32LE, 4 bytes), also called "footer version"
//      newer magic number (8 bytes)


}  // namespace ROCKSDB_NAMESPACE