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

#pragma once

#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "sequence_state_checkpoint_store.h"
#include "triton/common/thread_pool.h"

namespace triton { namespace core {

class SequenceStateCheckpointExecutor {
 public:
  enum class Operation { SAVE, LOAD, REMOVE };

  struct Options {
    size_t worker_count_;
    size_t max_job_count_;
    // Payload budget; excludes metadata and allocator overhead.
    size_t max_reserved_byte_size_;
  };

  struct Completion {
    Operation operation_;
    std::string sequence_id_;
    Status status_;
    std::unique_ptr<SequenceStateSnapshot> snapshot_;
    size_t checkpoint_byte_size_;
    bool durability_uncertain_;
  };

  struct Statistics {
    size_t queued_job_count_;
    size_t active_job_count_;
    size_t completion_count_;
    size_t queued_byte_size_;
    size_t active_byte_size_;
    size_t completion_byte_size_;
    size_t reserved_byte_size_;
    // Bytes published by this executor; excludes pre-existing files.
    size_t stored_byte_size_;
  };

  static Status Create(
      std::unique_ptr<SequenceStateCheckpointStore>&& store,
      const Options& options,
      std::unique_ptr<SequenceStateCheckpointExecutor>* executor);

  ~SequenceStateCheckpointExecutor();

  SequenceStateCheckpointExecutor(const SequenceStateCheckpointExecutor&) =
      delete;
  SequenceStateCheckpointExecutor& operator=(
      const SequenceStateCheckpointExecutor&) = delete;

  // Consumes the snapshot, including on rejection.
  Status SubmitSave(std::string sequence_id, SequenceStateSnapshot&& snapshot);
  Status SubmitLoad(std::string sequence_id);
  Status SubmitRemove(std::string sequence_id);

  bool PopCompletion(Completion* completion);
  Statistics GetStatistics() const;
  void Shutdown();

 private:
  struct CompletedJob {
    Completion completion_;
    size_t reserved_byte_size_;
  };

  SequenceStateCheckpointExecutor(
      std::unique_ptr<SequenceStateCheckpointStore>&& store,
      const Options& options);

  Status Enqueue(
      const std::string& sequence_id, size_t reserved_byte_size,
      std::function<void()>&& task);
  void StartJob(size_t reserved_byte_size);
  void FinishJob(
      Operation operation, std::string sequence_id, const Status& status,
      std::unique_ptr<SequenceStateSnapshot>&& snapshot,
      size_t checkpoint_byte_size, size_t reserved_byte_size);

  std::unique_ptr<SequenceStateCheckpointStore> store_;
  std::unique_ptr<triton::common::ThreadPool> workers_;
  const size_t max_job_count_;
  const size_t max_reserved_byte_size_;

  mutable std::mutex mutex_;
  std::mutex shutdown_mutex_;
  std::unordered_set<std::string> pending_sequences_;
  std::unordered_map<std::string, size_t> stored_checkpoints_;
  bool accepting_;
  size_t queued_job_count_;
  size_t active_job_count_;
  size_t queued_byte_size_;
  size_t active_byte_size_;
  size_t completion_byte_size_;
  size_t reserved_byte_size_;
  size_t stored_byte_size_;
  std::deque<CompletedJob> completions_;
};

}}  // namespace triton::core
