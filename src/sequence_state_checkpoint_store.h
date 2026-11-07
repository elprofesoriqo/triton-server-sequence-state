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
#include <cstdint>
#include <memory>
#include <string>

#include "sequence_state_snapshot.h"

namespace triton { namespace core {

class SequenceStateCheckpointStore {
 public:
  static Status Create(
      const std::string& directory, const std::string& model_name,
      int64_t model_version, size_t max_checkpoint_byte_size,
      std::unique_ptr<SequenceStateCheckpointStore>* store);

  ~SequenceStateCheckpointStore();

  SequenceStateCheckpointStore(const SequenceStateCheckpointStore&) = delete;
  SequenceStateCheckpointStore& operator=(const SequenceStateCheckpointStore&) =
      delete;

  Status Save(
      const std::string& sequence_id,
      const SequenceStateSnapshot& snapshot) const;
  Status Load(
      const std::string& sequence_id, SequenceStateSnapshot* snapshot) const;
  Status Remove(const std::string& sequence_id) const;

 private:
  SequenceStateCheckpointStore(
      int directory_fd, std::string model_name, int64_t model_version,
      size_t max_checkpoint_byte_size);

  Status CheckpointName(
      const std::string& sequence_id, std::string* name) const;

  int directory_fd_;
  std::string model_name_;
  int64_t model_version_;
  size_t max_checkpoint_byte_size_;
};

}}  // namespace triton::core
