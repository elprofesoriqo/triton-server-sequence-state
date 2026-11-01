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

#include "sequence_state_snapshot.h"

#include <cstring>
#include <limits>
#include <memory>
#include <utility>

#include "model_config_utils.h"
#include "triton_memory.h"

namespace triton { namespace core {
namespace {

Status
InvalidSnapshot(const std::string& message)
{
  return Status(
      Status::Code::INVALID_ARG, "sequence state snapshot: " + message);
}

bool
IsHostMemory(TRITONSERVER_MemoryType memory_type)
{
  return (memory_type == TRITONSERVER_MEMORY_CPU) ||
         (memory_type == TRITONSERVER_MEMORY_CPU_PINNED);
}

Status
StateByteSize(
    const std::string& input_name, const std::shared_ptr<SequenceState>& state,
    size_t* byte_size)
{
  if (state == nullptr) {
    return InvalidSnapshot("input state '" + input_name + "' is unavailable");
  }
  if (state->Name() != input_name) {
    return InvalidSnapshot("input state name does not match its map key");
  }
  if (state->DType() == inference::DataType::TYPE_STRING) {
    return InvalidSnapshot(
        "string state '" + input_name + "' is not supported");
  }
  if (state->UseGrowableMemory()) {
    return InvalidSnapshot(
        "growable state '" + input_name + "' is not supported");
  }
  if (state->Data() == nullptr) {
    return InvalidSnapshot("state buffer for '" + input_name + "' is missing");
  }

  int64_t expected_byte_size = 0;
  RETURN_IF_ERROR(GetByteSize(
      state->DType(), state->Shape(), state->Name(), &expected_byte_size));
  if (expected_byte_size < 0) {
    return InvalidSnapshot(
        "state '" + input_name + "' does not have a fixed size");
  }

  const size_t expected_size = static_cast<size_t>(expected_byte_size);
  if (state->Data()->TotalByteSize() != expected_size) {
    return InvalidSnapshot(
        "buffer size for state '" + input_name + "' does not match its shape");
  }

  *byte_size = expected_size;
  return Status::Success;
}

Status
CopyStateData(const SequenceState& state, size_t byte_size, char* destination)
{
  size_t offset = 0;
  for (size_t index = 0; index < state.Data()->BufferCount(); ++index) {
    size_t buffer_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    const char* buffer = state.Data()->BufferAt(
        index, &buffer_size, &memory_type, &memory_type_id);
    if (!IsHostMemory(memory_type)) {
      return InvalidSnapshot(
          "state '" + state.Name() + "' is not stored in CPU memory");
    }
    if ((buffer_size != 0) && (buffer == nullptr)) {
      return InvalidSnapshot(
          "state buffer for '" + state.Name() + "' is unavailable");
    }
    if (buffer_size > (byte_size - offset)) {
      return InvalidSnapshot(
          "buffer size for state '" + state.Name() + "' is inconsistent");
    }
    if (buffer_size != 0) {
      std::memcpy(destination + offset, buffer, buffer_size);
    }
    offset += buffer_size;
  }
  if (offset != byte_size) {
    return InvalidSnapshot(
        "buffer size for state '" + state.Name() + "' is incomplete");
  }
  return Status::Success;
}

struct RestoreTarget {
  std::shared_ptr<SequenceState> input_state_;
  SequenceState* output_state_;
  std::shared_ptr<Memory> data_;
};

}  // namespace

Status
SequenceStateSnapshot::Capture(
    const SequenceStates& states, SequenceStateSnapshot* snapshot)
{
  if (snapshot == nullptr) {
    return InvalidSnapshot("output must not be null");
  }
  if (states.InputStates().empty()) {
    return InvalidSnapshot("sequence has no implicit state");
  }

  SequenceStateSnapshot captured;
  size_t total_byte_size = 0;
  captured.tensors_.reserve(states.InputStates().size());
  for (const auto& [input_name, state] : states.InputStates()) {
    size_t byte_size = 0;
    RETURN_IF_ERROR(StateByteSize(input_name, state, &byte_size));
    if (byte_size > (std::numeric_limits<size_t>::max() - total_byte_size)) {
      return InvalidSnapshot("total state size exceeds the addressable range");
    }

    captured.tensors_.emplace_back(
        Tensor{
            state->Name(), state->DType(), state->Shape(), total_byte_size,
            byte_size});
    total_byte_size += byte_size;
  }

  captured.data_.resize(total_byte_size);
  size_t index = 0;
  for (const auto& state_entry : states.InputStates()) {
    const auto& state = state_entry.second;
    const Tensor& tensor = captured.tensors_[index++];
    char* destination = tensor.byte_size_ == 0
                            ? nullptr
                            : captured.data_.data() + tensor.offset_;
    RETURN_IF_ERROR(CopyStateData(*state, tensor.byte_size_, destination));
  }

  *snapshot = std::move(captured);
  return Status::Success;
}

Status
SequenceStateSnapshot::Restore(SequenceStates* states) const
{
  if (states == nullptr) {
    return InvalidSnapshot("target must not be null");
  }
  if (tensors_.empty()) {
    return InvalidSnapshot("snapshot contains no state");
  }
  if (states->InputStates().size() != tensors_.size()) {
    return InvalidSnapshot("target state count does not match the snapshot");
  }

  std::vector<RestoreTarget> targets;
  targets.reserve(tensors_.size());
  size_t expected_offset = 0;
  size_t tensor_index = 0;
  for (const auto& [input_name, state] : states->InputStates()) {
    const Tensor& tensor = tensors_[tensor_index++];
    size_t target_byte_size = 0;
    RETURN_IF_ERROR(StateByteSize(input_name, state, &target_byte_size));
    if ((tensor.input_name_ != input_name) ||
        (tensor.datatype_ != state->DType()) ||
        (tensor.shape_ != state->Shape()) ||
        (tensor.byte_size_ != target_byte_size) ||
        (tensor.offset_ != expected_offset) ||
        (tensor.offset_ > data_.size()) ||
        (tensor.byte_size_ > (data_.size() - tensor.offset_))) {
      return InvalidSnapshot(
          "target layout for state '" + input_name +
          "' does not match the snapshot");
    }
    expected_offset += tensor.byte_size_;

    SequenceState* output_state = nullptr;
    if (state->UseSingleBuffer()) {
      output_state = state->OtherState();
      if (output_state == nullptr) {
        return InvalidSnapshot(
            "output state for '" + input_name + "' is unavailable");
      }
    }

    auto restored_data = std::make_shared<AllocatedMemory>(
        tensor.byte_size_, TRITONSERVER_MEMORY_CPU, 0);
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    char* buffer = restored_data->MutableBuffer(&memory_type, &memory_type_id);
    if ((restored_data->TotalByteSize() != tensor.byte_size_) ||
        !IsHostMemory(memory_type) ||
        ((tensor.byte_size_ != 0) && (buffer == nullptr))) {
      return Status(
          Status::Code::INTERNAL,
          "sequence state snapshot: failed to allocate restored state");
    }
    if (tensor.byte_size_ != 0) {
      std::memcpy(buffer, data_.data() + tensor.offset_, tensor.byte_size_);
    }

    targets.emplace_back(
        RestoreTarget{state, output_state, std::move(restored_data)});
  }
  if (expected_offset != data_.size()) {
    return InvalidSnapshot("snapshot data size does not match its layout");
  }

  for (auto& target : targets) {
    RETURN_IF_ERROR(target.input_state_->RemoveAllData());
    RETURN_IF_ERROR(target.input_state_->SetData(target.data_));
    if (target.output_state_ != nullptr) {
      RETURN_IF_ERROR(target.output_state_->RemoveAllData());
      RETURN_IF_ERROR(target.output_state_->SetData(target.data_));
    }
  }

  return Status::Success;
}

}}  // namespace triton::core
