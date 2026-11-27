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

#include "sequence_state_checkpoint_executor.h"

#include <limits>
#include <utility>

namespace triton { namespace core {
namespace {

Status
ExecutorError(Status::Code code, const std::string& message)
{
  return Status(code, "sequence state checkpoint executor: " + message);
}

}  // namespace

Status
SequenceStateCheckpointExecutor::Create(
    std::unique_ptr<SequenceStateCheckpointStore>&& store,
    const Options& options,
    std::unique_ptr<SequenceStateCheckpointExecutor>* executor)
{
  if ((store == nullptr) || (executor == nullptr) ||
      (options.worker_count_ == 0) || (options.max_job_count_ == 0) ||
      (options.max_reserved_byte_size_ == 0)) {
    return ExecutorError(Status::Code::INVALID_ARG, "invalid configuration");
  }

  *executor = std::unique_ptr<SequenceStateCheckpointExecutor>(
      new SequenceStateCheckpointExecutor(std::move(store), options));
  return Status::Success;
}

SequenceStateCheckpointExecutor::SequenceStateCheckpointExecutor(
    std::unique_ptr<SequenceStateCheckpointStore>&& store,
    const Options& options)
    : store_(std::move(store)),
      workers_(new triton::common::ThreadPool(options.worker_count_)),
      max_job_count_(options.max_job_count_),
      max_reserved_byte_size_(options.max_reserved_byte_size_),
      accepting_(true), queued_job_count_(0), active_job_count_(0),
      queued_byte_size_(0), active_byte_size_(0), completion_byte_size_(0),
      reserved_byte_size_(0), stored_byte_size_(0)
{
}

SequenceStateCheckpointExecutor::~SequenceStateCheckpointExecutor()
{
  Shutdown();
}

Status
SequenceStateCheckpointExecutor::Enqueue(
    const std::string& sequence_id, size_t reserved_byte_size,
    std::function<void()>&& task)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (!accepting_) {
    return ExecutorError(
        Status::Code::UNAVAILABLE, "executor is shutting down");
  }
  if ((queued_job_count_ + active_job_count_ + completions_.size()) >=
      max_job_count_) {
    return ExecutorError(Status::Code::UNAVAILABLE, "job limit exceeded");
  }
  if ((reserved_byte_size > max_reserved_byte_size_) ||
      (reserved_byte_size_ > (max_reserved_byte_size_ - reserved_byte_size))) {
    return ExecutorError(
        Status::Code::UNAVAILABLE, "reserved byte limit exceeded");
  }

  if (!pending_sequences_.insert(sequence_id).second) {
    return ExecutorError(
        Status::Code::UNAVAILABLE, "sequence already has a pending operation");
  }
  ++queued_job_count_;
  queued_byte_size_ += reserved_byte_size;
  reserved_byte_size_ += reserved_byte_size;
  workers_->Enqueue(std::move(task));
  return Status::Success;
}

Status
SequenceStateCheckpointExecutor::SubmitSave(
    std::string sequence_id, SequenceStateSnapshot&& snapshot)
{
  const size_t reserved_byte_size = snapshot.Data().size();
  auto captured = std::make_shared<SequenceStateSnapshot>(std::move(snapshot));
  return Enqueue(
      sequence_id, reserved_byte_size,
      [this, sequence_id, captured, reserved_byte_size]() mutable {
        StartJob(reserved_byte_size);
        size_t checkpoint_byte_size = 0;
        const Status status =
            store_->Save(sequence_id, *captured, &checkpoint_byte_size);
        FinishJob(
            Operation::SAVE, std::move(sequence_id), status, nullptr,
            checkpoint_byte_size, reserved_byte_size);
      });
}

Status
SequenceStateCheckpointExecutor::SubmitLoad(std::string sequence_id)
{
  const size_t reserved_byte_size = store_->MaxCheckpointByteSize();
  return Enqueue(
      sequence_id, reserved_byte_size,
      [this, sequence_id, reserved_byte_size]() mutable {
        StartJob(reserved_byte_size);
        auto snapshot = std::make_unique<SequenceStateSnapshot>();
        size_t checkpoint_byte_size = 0;
        const Status status =
            store_->Load(sequence_id, snapshot.get(), &checkpoint_byte_size);
        if (!status.IsOk()) {
          snapshot.reset();
        }
        FinishJob(
            Operation::LOAD, std::move(sequence_id), status,
            std::move(snapshot), checkpoint_byte_size, reserved_byte_size);
      });
}

Status
SequenceStateCheckpointExecutor::SubmitRemove(std::string sequence_id)
{
  return Enqueue(sequence_id, 0, [this, sequence_id]() mutable {
    StartJob(0);
    size_t checkpoint_byte_size = 0;
    const Status status = store_->Remove(sequence_id, &checkpoint_byte_size);
    FinishJob(
        Operation::REMOVE, std::move(sequence_id), status, nullptr,
        checkpoint_byte_size, 0);
  });
}

void
SequenceStateCheckpointExecutor::StartJob(size_t reserved_byte_size)
{
  std::lock_guard<std::mutex> lock(mutex_);
  --queued_job_count_;
  ++active_job_count_;
  queued_byte_size_ -= reserved_byte_size;
  active_byte_size_ += reserved_byte_size;
}

void
SequenceStateCheckpointExecutor::FinishJob(
    Operation operation, std::string sequence_id, const Status& status,
    std::unique_ptr<SequenceStateSnapshot>&& snapshot,
    size_t checkpoint_byte_size, size_t reserved_byte_size)
{
  std::lock_guard<std::mutex> lock(mutex_);
  --active_job_count_;
  active_byte_size_ -= reserved_byte_size;
  completion_byte_size_ += reserved_byte_size;
  const bool durability_uncertain =
      status.StatusCode() == Status::Code::UNAVAILABLE;
  if (durability_uncertain) {
    accepting_ = false;
  }
  if (checkpoint_byte_size != 0) {
    if (operation == Operation::SAVE) {
      if (checkpoint_byte_size <=
          (std::numeric_limits<size_t>::max() - stored_byte_size_)) {
        stored_byte_size_ += checkpoint_byte_size;
        stored_checkpoints_[sequence_id] = checkpoint_byte_size;
      }
    } else if (operation == Operation::REMOVE) {
      auto it = stored_checkpoints_.find(sequence_id);
      if (it != stored_checkpoints_.end()) {
        stored_byte_size_ -= it->second;
        stored_checkpoints_.erase(it);
      }
    }
  }
  completions_.emplace_back(CompletedJob{
      Completion{
          operation, std::move(sequence_id), status, std::move(snapshot),
          checkpoint_byte_size, durability_uncertain},
      reserved_byte_size});
}

bool
SequenceStateCheckpointExecutor::PopCompletion(Completion* completion)
{
  if (completion == nullptr) {
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (completions_.empty()) {
    return false;
  }
  reserved_byte_size_ -= completions_.front().reserved_byte_size_;
  completion_byte_size_ -= completions_.front().reserved_byte_size_;
  pending_sequences_.erase(completions_.front().completion_.sequence_id_);
  *completion = std::move(completions_.front().completion_);
  completions_.pop_front();
  return true;
}

SequenceStateCheckpointExecutor::Statistics
SequenceStateCheckpointExecutor::GetStatistics() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return Statistics{queued_job_count_,   active_job_count_,
                    completions_.size(), queued_byte_size_,
                    active_byte_size_,   completion_byte_size_,
                    reserved_byte_size_, stored_byte_size_};
}

void
SequenceStateCheckpointExecutor::Shutdown()
{
  std::lock_guard<std::mutex> shutdown_lock(shutdown_mutex_);
  std::unique_ptr<triton::common::ThreadPool> workers;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!accepting_ && (workers_ == nullptr)) {
      return;
    }
    accepting_ = false;
    workers = std::move(workers_);
  }
  workers.reset();
}

}}  // namespace triton::core
