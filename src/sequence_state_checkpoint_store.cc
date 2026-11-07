// Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "sequence_state_checkpoint_store.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "crc32c/crc32c.h"
#include "model_config_utils.h"

namespace triton { namespace core {
namespace {

constexpr std::array<char, 8> kMagic{'T', 'R', 'T', 'N', 'S', 'T', 'A', 'T'};
constexpr uint32_t kFormatVersion = 1;
constexpr size_t kHeaderByteSize = 56;
constexpr size_t kChecksumOffset = 52;
constexpr char kHex[] = "0123456789abcdef";
std::atomic<uint64_t> next_temporary_id{0};

class ScopedFd {
 public:
  explicit ScopedFd(int fd = -1) : fd_(fd) {}
  ~ScopedFd()
  {
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  int Get() const { return fd_; }

  void Reset(int fd)
  {
    if (fd_ >= 0) {
      close(fd_);
    }
    fd_ = fd;
  }

 private:
  int fd_;
};

Status
StoreError(Status::Code code, const std::string& message)
{
  return Status(code, "sequence state checkpoint: " + message);
}

Status
SystemError(const std::string& operation, int error)
{
  return StoreError(
      Status::Code::INTERNAL, operation + " failed: " + std::strerror(error));
}

bool
AddSize(size_t value, size_t increment, size_t* result)
{
  if (increment > (std::numeric_limits<size_t>::max() - value)) {
    return false;
  }
  *result = value + increment;
  return true;
}

void
AppendUint32(uint32_t value, std::vector<char>* output)
{
  for (size_t index = 0; index < sizeof(value); ++index) {
    output->push_back(static_cast<char>(value >> (index * 8)));
  }
}

void
AppendUint64(uint64_t value, std::vector<char>* output)
{
  for (size_t index = 0; index < sizeof(value); ++index) {
    output->push_back(static_cast<char>(value >> (index * 8)));
  }
}

bool
ReadUint32(const std::vector<char>& input, size_t* offset, uint32_t* value)
{
  if ((*offset > input.size()) || ((input.size() - *offset) < sizeof(*value))) {
    return false;
  }
  *value = 0;
  for (size_t index = 0; index < sizeof(*value); ++index) {
    *value |= static_cast<uint32_t>(
                  static_cast<unsigned char>(input[*offset + index]))
              << (index * 8);
  }
  *offset += sizeof(*value);
  return true;
}

bool
ReadUint64(const std::vector<char>& input, size_t* offset, uint64_t* value)
{
  if ((*offset > input.size()) || ((input.size() - *offset) < sizeof(*value))) {
    return false;
  }
  *value = 0;
  for (size_t index = 0; index < sizeof(*value); ++index) {
    *value |= static_cast<uint64_t>(
                  static_cast<unsigned char>(input[*offset + index]))
              << (index * 8);
  }
  *offset += sizeof(*value);
  return true;
}

int64_t
DecodeInt64(uint64_t value)
{
  if (value <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return static_cast<int64_t>(value);
  }
  return -1 -
         static_cast<int64_t>(std::numeric_limits<uint64_t>::max() - value);
}

Status
WriteAll(int fd, const char* data, size_t byte_size)
{
  while (byte_size != 0) {
    const size_t request_size = std::min(
        byte_size, static_cast<size_t>(std::numeric_limits<ssize_t>::max()));
    const ssize_t written = write(fd, data, request_size);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      return SystemError("write", errno);
    }
    if (written == 0) {
      return StoreError(Status::Code::INTERNAL, "write made no progress");
    }
    data += written;
    byte_size -= static_cast<size_t>(written);
  }
  return Status::Success;
}

Status
ReadAll(int fd, char* data, size_t byte_size)
{
  while (byte_size != 0) {
    const size_t request_size = std::min(
        byte_size, static_cast<size_t>(std::numeric_limits<ssize_t>::max()));
    const ssize_t read_size = read(fd, data, request_size);
    if (read_size < 0) {
      if (errno == EINTR) {
        continue;
      }
      return SystemError("read", errno);
    }
    if (read_size == 0) {
      return StoreError(Status::Code::INVALID_ARG, "checkpoint is truncated");
    }
    data += read_size;
    byte_size -= static_cast<size_t>(read_size);
  }
  return Status::Success;
}

Status
SnapshotMetadataSize(
    const SequenceStateSnapshot& snapshot, const std::string& sequence_id,
    const std::string& model_name, size_t* metadata_byte_size)
{
  if (snapshot.Tensors().empty()) {
    return StoreError(Status::Code::INVALID_ARG, "snapshot has no state");
  }
  if ((sequence_id.size() > std::numeric_limits<uint32_t>::max()) ||
      (model_name.size() > std::numeric_limits<uint32_t>::max()) ||
      (snapshot.Tensors().size() > std::numeric_limits<uint32_t>::max())) {
    return StoreError(Status::Code::INVALID_ARG, "metadata is too large");
  }

  size_t size = sequence_id.size();
  if (!AddSize(size, model_name.size(), &size)) {
    return StoreError(Status::Code::INVALID_ARG, "metadata is too large");
  }

  size_t expected_offset = 0;
  std::string previous_name;
  for (const auto& tensor : snapshot.Tensors()) {
    if (tensor.input_name_.empty() ||
        (!previous_name.empty() && (tensor.input_name_ <= previous_name)) ||
        (tensor.input_name_.size() > std::numeric_limits<uint32_t>::max()) ||
        (tensor.shape_.size() > std::numeric_limits<uint32_t>::max()) ||
        (tensor.offset_ != expected_offset) ||
        (tensor.offset_ > snapshot.Data().size()) ||
        (tensor.byte_size_ > (snapshot.Data().size() - tensor.offset_))) {
      return StoreError(
          Status::Code::INVALID_ARG, "snapshot layout is invalid");
    }

    int64_t byte_size = 0;
    RETURN_IF_ERROR(GetByteSize(
        tensor.datatype_, tensor.shape_, tensor.input_name_, &byte_size));
    if ((byte_size < 0) ||
        (static_cast<uint64_t>(byte_size) != tensor.byte_size_)) {
      return StoreError(
          Status::Code::INVALID_ARG, "snapshot layout is invalid");
    }

    size_t tensor_size = 20;
    if (!AddSize(tensor_size, tensor.input_name_.size(), &tensor_size) ||
        (tensor.shape_.size() >
         ((std::numeric_limits<size_t>::max() - tensor_size) /
          sizeof(int64_t))) ||
        !AddSize(
            tensor_size, tensor.shape_.size() * sizeof(int64_t),
            &tensor_size) ||
        !AddSize(size, tensor_size, &size) ||
        !AddSize(expected_offset, tensor.byte_size_, &expected_offset)) {
      return StoreError(Status::Code::INVALID_ARG, "snapshot is too large");
    }
    previous_name = tensor.input_name_;
  }
  if (expected_offset != snapshot.Data().size()) {
    return StoreError(Status::Code::INVALID_ARG, "snapshot layout is invalid");
  }

  *metadata_byte_size = size;
  return Status::Success;
}

Status
ParseMetadata(
    const std::vector<char>& metadata, uint32_t sequence_size,
    uint32_t model_size, uint32_t tensor_count,
    const std::string& expected_sequence_id,
    const std::string& expected_model_name, uint32_t expected_layout_checksum,
    uint64_t payload_byte_size,
    std::vector<SequenceStateSnapshot::Tensor>* tensors)
{
  const size_t identity_size =
      static_cast<size_t>(sequence_size) + static_cast<size_t>(model_size);
  if ((identity_size > metadata.size()) || (tensor_count == 0) ||
      (tensor_count > ((metadata.size() - identity_size) / 20))) {
    return StoreError(Status::Code::INVALID_ARG, "metadata is invalid");
  }

  size_t offset = 0;
  const std::string sequence_id(metadata.data(), sequence_size);
  offset += sequence_size;
  const std::string model_name(metadata.data() + offset, model_size);
  offset += model_size;
  if ((sequence_id != expected_sequence_id) ||
      (model_name != expected_model_name)) {
    return StoreError(
        Status::Code::INVALID_ARG, "checkpoint identity does not match");
  }

  const uint32_t layout_checksum =
      crc32c::Crc32c(metadata.data() + offset, metadata.size() - offset);
  if (layout_checksum != expected_layout_checksum) {
    return StoreError(Status::Code::INVALID_ARG, "state layout is corrupt");
  }

  std::vector<SequenceStateSnapshot::Tensor> parsed;
  parsed.reserve(tensor_count);
  size_t payload_offset = 0;
  std::string previous_name;
  for (uint32_t index = 0; index < tensor_count; ++index) {
    uint32_t name_size = 0;
    uint32_t datatype = 0;
    uint32_t rank = 0;
    uint64_t tensor_byte_size = 0;
    if (!ReadUint32(metadata, &offset, &name_size) ||
        !ReadUint32(metadata, &offset, &datatype) ||
        !ReadUint32(metadata, &offset, &rank) ||
        !ReadUint64(metadata, &offset, &tensor_byte_size) ||
        (name_size > (metadata.size() - offset))) {
      return StoreError(Status::Code::INVALID_ARG, "state layout is invalid");
    }

    std::string name(metadata.data() + offset, name_size);
    offset += name_size;
    if (name.empty() || (!previous_name.empty() && (name <= previous_name)) ||
        (rank > ((metadata.size() - offset) / sizeof(int64_t)))) {
      return StoreError(Status::Code::INVALID_ARG, "state layout is invalid");
    }

    std::vector<int64_t> shape;
    shape.reserve(rank);
    for (uint32_t dimension = 0; dimension < rank; ++dimension) {
      uint64_t encoded_dimension = 0;
      if (!ReadUint64(metadata, &offset, &encoded_dimension)) {
        return StoreError(Status::Code::INVALID_ARG, "state layout is invalid");
      }
      shape.push_back(DecodeInt64(encoded_dimension));
    }

    const auto data_type = static_cast<inference::DataType>(datatype);
    int64_t expected_byte_size = 0;
    const Status byte_size_status =
        GetByteSize(data_type, shape, name, &expected_byte_size);
    if ((data_type == inference::DataType::TYPE_INVALID) ||
        (data_type == inference::DataType::TYPE_STRING) ||
        !byte_size_status.IsOk() || (expected_byte_size < 0) ||
        (static_cast<uint64_t>(expected_byte_size) != tensor_byte_size) ||
        (tensor_byte_size > (payload_byte_size - payload_offset)) ||
        (tensor_byte_size > std::numeric_limits<size_t>::max())) {
      return StoreError(Status::Code::INVALID_ARG, "state layout is invalid");
    }

    parsed.emplace_back(
        SequenceStateSnapshot::Tensor{
            std::move(name), data_type, std::move(shape), payload_offset,
            static_cast<size_t>(tensor_byte_size)});
    payload_offset += static_cast<size_t>(tensor_byte_size);
    previous_name = parsed.back().input_name_;
  }
  if ((offset != metadata.size()) || (payload_offset != payload_byte_size)) {
    return StoreError(Status::Code::INVALID_ARG, "state layout is invalid");
  }

  *tensors = std::move(parsed);
  return Status::Success;
}

}  // namespace

Status
SequenceStateCheckpointStore::Create(
    const std::string& directory, const std::string& model_name,
    int64_t model_version, size_t max_checkpoint_byte_size,
    std::unique_ptr<SequenceStateCheckpointStore>* store)
{
  if ((store == nullptr) || directory.empty() || model_name.empty() ||
      (max_checkpoint_byte_size < kHeaderByteSize)) {
    return StoreError(Status::Code::INVALID_ARG, "invalid store configuration");
  }

  const int directory_fd =
      open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (directory_fd < 0) {
    return SystemError("open checkpoint directory", errno);
  }

  *store = std::unique_ptr<SequenceStateCheckpointStore>(
      new SequenceStateCheckpointStore(
          directory_fd, model_name, model_version, max_checkpoint_byte_size));
  return Status::Success;
}

SequenceStateCheckpointStore::SequenceStateCheckpointStore(
    int directory_fd, std::string model_name, int64_t model_version,
    size_t max_checkpoint_byte_size)
    : directory_fd_(directory_fd), model_name_(std::move(model_name)),
      model_version_(model_version),
      max_checkpoint_byte_size_(max_checkpoint_byte_size)
{
}

SequenceStateCheckpointStore::~SequenceStateCheckpointStore()
{
  close(directory_fd_);
}

Status
SequenceStateCheckpointStore::CheckpointName(
    const std::string& sequence_id, std::string* name) const
{
  if ((name == nullptr) || sequence_id.empty() ||
      (sequence_id.size() > ((std::numeric_limits<size_t>::max() - 9) / 2))) {
    return StoreError(Status::Code::INVALID_ARG, "invalid sequence ID");
  }

  name->clear();
  name->reserve(4 + (sequence_id.size() * 2) + 5);
  name->append("seq-");
  for (const unsigned char byte : sequence_id) {
    name->push_back(kHex[byte >> 4]);
    name->push_back(kHex[byte & 0x0f]);
  }
  name->append(".ckpt");

  errno = 0;
  const long name_max = fpathconf(directory_fd_, _PC_NAME_MAX);
  if (((name_max < 0) && (errno != 0)) ||
      ((name_max >= 0) &&
       (name->size() > static_cast<unsigned long>(name_max)))) {
    return StoreError(Status::Code::INVALID_ARG, "sequence ID is too long");
  }
  return Status::Success;
}

Status
SequenceStateCheckpointStore::Save(
    const std::string& sequence_id, const SequenceStateSnapshot& snapshot) const
{
  std::string checkpoint_name;
  RETURN_IF_ERROR(CheckpointName(sequence_id, &checkpoint_name));

  size_t metadata_byte_size = 0;
  RETURN_IF_ERROR(SnapshotMetadataSize(
      snapshot, sequence_id, model_name_, &metadata_byte_size));
  size_t checkpoint_byte_size = kHeaderByteSize;
  if (!AddSize(
          checkpoint_byte_size, metadata_byte_size, &checkpoint_byte_size) ||
      !AddSize(
          checkpoint_byte_size, snapshot.Data().size(),
          &checkpoint_byte_size) ||
      (checkpoint_byte_size > max_checkpoint_byte_size_)) {
    return StoreError(
        Status::Code::INVALID_ARG, "checkpoint exceeds the configured limit");
  }

  std::vector<char> metadata;
  metadata.reserve(metadata_byte_size);
  metadata.insert(metadata.end(), sequence_id.begin(), sequence_id.end());
  metadata.insert(metadata.end(), model_name_.begin(), model_name_.end());
  const size_t layout_offset = metadata.size();
  for (const auto& tensor : snapshot.Tensors()) {
    AppendUint32(tensor.input_name_.size(), &metadata);
    AppendUint32(static_cast<uint32_t>(tensor.datatype_), &metadata);
    AppendUint32(tensor.shape_.size(), &metadata);
    AppendUint64(tensor.byte_size_, &metadata);
    metadata.insert(
        metadata.end(), tensor.input_name_.begin(), tensor.input_name_.end());
    for (const int64_t dimension : tensor.shape_) {
      AppendUint64(static_cast<uint64_t>(dimension), &metadata);
    }
  }

  const uint32_t layout_checksum = crc32c::Crc32c(
      metadata.data() + layout_offset, metadata.size() - layout_offset);
  std::vector<char> header;
  header.reserve(kHeaderByteSize);
  header.insert(header.end(), kMagic.begin(), kMagic.end());
  AppendUint32(kFormatVersion, &header);
  AppendUint32(sequence_id.size(), &header);
  AppendUint32(model_name_.size(), &header);
  AppendUint32(snapshot.Tensors().size(), &header);
  AppendUint64(static_cast<uint64_t>(model_version_), &header);
  AppendUint64(metadata.size(), &header);
  AppendUint64(snapshot.Data().size(), &header);
  AppendUint32(layout_checksum, &header);

  uint32_t checksum = crc32c::Crc32c(header.data(), header.size());
  checksum = crc32c::Extend(
      checksum, reinterpret_cast<const uint8_t*>(metadata.data()),
      metadata.size());
  checksum = crc32c::Extend(
      checksum, reinterpret_cast<const uint8_t*>(snapshot.Data().data()),
      snapshot.Data().size());
  AppendUint32(checksum, &header);

  std::string temporary_name;
  ScopedFd file;
  for (size_t attempt = 0; attempt < 100; ++attempt) {
    temporary_name = ".tmp-" + std::to_string(getpid()) + "-" +
                     std::to_string(next_temporary_id.fetch_add(1));
    const int fd = openat(
        directory_fd_, temporary_name.c_str(),
        O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd >= 0) {
      file.Reset(fd);
      break;
    }
    if (errno != EEXIST) {
      return SystemError("create temporary checkpoint", errno);
    }
  }
  if (file.Get() < 0) {
    return StoreError(
        Status::Code::INTERNAL, "cannot create a temporary checkpoint");
  }

  Status status = WriteAll(file.Get(), header.data(), header.size());
  if (status.IsOk()) {
    status = WriteAll(file.Get(), metadata.data(), metadata.size());
  }
  if (status.IsOk()) {
    status =
        WriteAll(file.Get(), snapshot.Data().data(), snapshot.Data().size());
  }
  if (status.IsOk() && (fsync(file.Get()) != 0)) {
    status = SystemError("synchronize checkpoint", errno);
  }
  if (!status.IsOk()) {
    unlinkat(directory_fd_, temporary_name.c_str(), 0);
    return status;
  }

  if (linkat(
          directory_fd_, temporary_name.c_str(), directory_fd_,
          checkpoint_name.c_str(), 0) != 0) {
    const int error = errno;
    unlinkat(directory_fd_, temporary_name.c_str(), 0);
    if (error == EEXIST) {
      return StoreError(
          Status::Code::ALREADY_EXISTS, "checkpoint already exists");
    }
    return SystemError("publish checkpoint", error);
  }
  unlinkat(directory_fd_, temporary_name.c_str(), 0);
  if (fsync(directory_fd_) != 0) {
    return SystemError("synchronize checkpoint directory", errno);
  }
  return Status::Success;
}

Status
SequenceStateCheckpointStore::Load(
    const std::string& sequence_id, SequenceStateSnapshot* snapshot) const
{
  if (snapshot == nullptr) {
    return StoreError(Status::Code::INVALID_ARG, "output must not be null");
  }

  std::string checkpoint_name;
  RETURN_IF_ERROR(CheckpointName(sequence_id, &checkpoint_name));
  const int fd = openat(
      directory_fd_, checkpoint_name.c_str(),
      O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) {
    if (errno == ELOOP) {
      return StoreError(
          Status::Code::INVALID_ARG, "checkpoint file is invalid");
    }
    return StoreError(
        (errno == ENOENT) ? Status::Code::NOT_FOUND : Status::Code::INTERNAL,
        (errno == ENOENT)
            ? "checkpoint does not exist"
            : "open checkpoint failed: " + std::string(std::strerror(errno)));
  }
  ScopedFd file(fd);

  struct stat file_status;
  if (fstat(fd, &file_status) != 0) {
    return SystemError("inspect checkpoint", errno);
  }
  if (!S_ISREG(file_status.st_mode) || (file_status.st_size < 0) ||
      (static_cast<uint64_t>(file_status.st_size) >
       max_checkpoint_byte_size_) ||
      (static_cast<uint64_t>(file_status.st_size) < kHeaderByteSize)) {
    return StoreError(Status::Code::INVALID_ARG, "checkpoint file is invalid");
  }

  std::vector<char> header(kHeaderByteSize);
  RETURN_IF_ERROR(ReadAll(fd, header.data(), header.size()));
  if (!std::equal(kMagic.begin(), kMagic.end(), header.begin())) {
    return StoreError(
        Status::Code::INVALID_ARG, "checkpoint format is invalid");
  }

  size_t header_offset = kMagic.size();
  uint32_t version = 0;
  uint32_t sequence_size = 0;
  uint32_t model_size = 0;
  uint32_t tensor_count = 0;
  uint64_t encoded_model_version = 0;
  uint64_t metadata_byte_size = 0;
  uint64_t payload_byte_size = 0;
  uint32_t layout_checksum = 0;
  uint32_t stored_checksum = 0;
  if (!ReadUint32(header, &header_offset, &version) ||
      !ReadUint32(header, &header_offset, &sequence_size) ||
      !ReadUint32(header, &header_offset, &model_size) ||
      !ReadUint32(header, &header_offset, &tensor_count) ||
      !ReadUint64(header, &header_offset, &encoded_model_version) ||
      !ReadUint64(header, &header_offset, &metadata_byte_size) ||
      !ReadUint64(header, &header_offset, &payload_byte_size) ||
      !ReadUint32(header, &header_offset, &layout_checksum) ||
      !ReadUint32(header, &header_offset, &stored_checksum) ||
      (header_offset != kHeaderByteSize) || (version != kFormatVersion) ||
      (DecodeInt64(encoded_model_version) != model_version_) ||
      (metadata_byte_size > std::numeric_limits<size_t>::max()) ||
      (payload_byte_size > std::numeric_limits<size_t>::max())) {
    return StoreError(
        Status::Code::INVALID_ARG, "checkpoint header is invalid");
  }

  size_t expected_file_size = kHeaderByteSize;
  if (!AddSize(
          expected_file_size, static_cast<size_t>(metadata_byte_size),
          &expected_file_size) ||
      !AddSize(
          expected_file_size, static_cast<size_t>(payload_byte_size),
          &expected_file_size) ||
      (expected_file_size != static_cast<size_t>(file_status.st_size))) {
    return StoreError(Status::Code::INVALID_ARG, "checkpoint size is invalid");
  }

  std::vector<char> metadata(static_cast<size_t>(metadata_byte_size));
  RETURN_IF_ERROR(ReadAll(fd, metadata.data(), metadata.size()));
  std::vector<SequenceStateSnapshot::Tensor> tensors;
  SequenceStateSnapshot loaded;
  RETURN_IF_ERROR(ParseMetadata(
      metadata, sequence_size, model_size, tensor_count, sequence_id,
      model_name_, layout_checksum, payload_byte_size, &tensors));

  loaded.tensors_ = std::move(tensors);
  loaded.data_.resize(static_cast<size_t>(payload_byte_size));
  RETURN_IF_ERROR(ReadAll(fd, loaded.data_.data(), loaded.data_.size()));
  if ((fstat(fd, &file_status) != 0) ||
      (static_cast<uint64_t>(file_status.st_size) != expected_file_size)) {
    return StoreError(Status::Code::INVALID_ARG, "checkpoint size changed");
  }
  uint32_t checksum = crc32c::Crc32c(header.data(), kChecksumOffset);
  checksum = crc32c::Extend(
      checksum, reinterpret_cast<const uint8_t*>(metadata.data()),
      metadata.size());
  checksum = crc32c::Extend(
      checksum, reinterpret_cast<const uint8_t*>(loaded.data_.data()),
      loaded.data_.size());
  if (checksum != stored_checksum) {
    return StoreError(Status::Code::INVALID_ARG, "checkpoint checksum failed");
  }

  *snapshot = std::move(loaded);
  return Status::Success;
}

Status
SequenceStateCheckpointStore::Remove(const std::string& sequence_id) const
{
  std::string checkpoint_name;
  RETURN_IF_ERROR(CheckpointName(sequence_id, &checkpoint_name));

  struct stat file_status;
  if (fstatat(
          directory_fd_, checkpoint_name.c_str(), &file_status,
          AT_SYMLINK_NOFOLLOW) != 0) {
    return StoreError(
        (errno == ENOENT) ? Status::Code::NOT_FOUND : Status::Code::INTERNAL,
        (errno == ENOENT) ? "checkpoint does not exist"
                          : "inspect checkpoint failed: " +
                                std::string(std::strerror(errno)));
  }
  if (!S_ISREG(file_status.st_mode)) {
    return StoreError(Status::Code::INVALID_ARG, "checkpoint file is invalid");
  }
  if (unlinkat(directory_fd_, checkpoint_name.c_str(), 0) != 0) {
    return SystemError("remove checkpoint", errno);
  }
  if (fsync(directory_fd_) != 0) {
    return SystemError("synchronize checkpoint directory", errno);
  }
  return Status::Success;
}

}}  // namespace triton::core
