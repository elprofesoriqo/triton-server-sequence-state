// Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <charconv>
#include <system_error>

#include "sequence_batch_scheduler.h"
#include "triton/common/triton_json.h"

namespace triton { namespace core {
namespace {

uint64_t
NowMicros()
{
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::string
StorageId(const InferenceRequest::SequenceId& id)
{
  return id.Type() == InferenceRequest::SequenceId::DataType::STRING
             ? "s:" + id.StringValue()
             : "u:" + std::to_string(id.UnsignedIntValue());
}

Status
ReadLimit(const inference::ModelConfig& config, const char* name, size_t* value)
{
  const auto it = config.parameters().find(name);
  if (it != config.parameters().end()) {
    const auto& text = it->second.string_value();
    const auto parsed =
        std::from_chars(text.data(), text.data() + text.size(), *value);
    if ((parsed.ec == std::errc()) &&
        (parsed.ptr == text.data() + text.size()) && (*value > 0)) {
      return Status::Success;
    }
  }
  return Status(
      Status::Code::INVALID_ARG,
      std::string("invalid checkpoint limit: ") + name);
}

}  // namespace

Status
SequenceBatchScheduler::ConfigureCheckpointing()
{
  const auto& config = model_->Config();
  const auto directory =
      config.parameters().find("sequence_checkpoint_directory");
  if (directory == config.parameters().end()) {
    return Status::Success;
  }
  if ((batchers_.size() != 1) || config.sequence_batching().has_oldest() ||
      config.sequence_batching().iterative_sequence() ||
      state_output_config_map_.empty() ||
      (batchers_.begin()->first->Kind() !=
       TRITONSERVER_INSTANCEGROUPKIND_CPU)) {
    return Status(
        Status::Code::UNSUPPORTED,
        "checkpointing requires one CPU Direct instance with non-iterative "
        "implicit state");
  }
  for (const auto& state : config.sequence_batching().state()) {
    if (std::any_of(
            state.dims().begin(), state.dims().end(),
            [](int64_t dimension) { return dimension < 0; })) {
      return Status(
          Status::Code::UNSUPPORTED,
          "checkpoint state dimensions must be fixed");
    }
  }
  SequenceStateCheckpointExecutor::Options options{};
  RETURN_IF_ERROR(
      ReadLimit(config, "sequence_checkpoint_workers", &options.worker_count_));
  RETURN_IF_ERROR(ReadLimit(
      config, "sequence_checkpoint_max_jobs", &options.max_job_count_));
  checkpoint_max_controls_ = options.max_job_count_;
  RETURN_IF_ERROR(ReadLimit(
      config, "sequence_checkpoint_max_bytes",
      &options.max_reserved_byte_size_));
  RETURN_IF_ERROR(ReadLimit(
      config, "sequence_checkpoint_max_file_bytes", &checkpoint_file_limit_));
  RETURN_IF_ERROR(ReadLimit(
      config, "sequence_checkpoint_max_count", &checkpoint_max_count_));
  RETURN_IF_ERROR(ReadLimit(
      config, "sequence_checkpoint_max_disk_bytes", &checkpoint_max_bytes_));
  if ((checkpoint_file_limit_ > options.max_reserved_byte_size_) ||
      (checkpoint_file_limit_ > checkpoint_max_bytes_)) {
    return Status(
        Status::Code::INVALID_ARG,
        "checkpoint file limit exceeds memory or disk budget");
  }
  // Validate the supported tensor layout before accepting any inference.
  SequenceStates initial;
  RETURN_IF_ERROR(initial.Initialize(
      state_output_config_map_, max_batch_size_, initial_state_,
      TRITONSERVER_INSTANCEGROUPKIND_CPU, 0, {}));
  SequenceStateSnapshot layout;
  RETURN_IF_ERROR(SequenceStateSnapshot::Capture(initial, &layout));
  checkpoint_layout_ = layout.Tensors();
  std::unique_ptr<SequenceStateCheckpointStore> store;
  RETURN_IF_ERROR(SequenceStateCheckpointStore::Create(
      directory->second.string_value(), model_->Name(), model_->Version(),
      checkpoint_file_limit_, &store));
  size_t fixed_size = 0;
  RETURN_IF_ERROR(store->CheckpointByteSize("u:1", layout, &fixed_size));
  checkpoint_base_bytes_ = fixed_size - 3;
  std::vector<SequenceStateCheckpointStore::Checkpoint> recovered;
  RETURN_IF_ERROR(store->Recover(
      checkpoint_max_count_, checkpoint_max_bytes_, &recovered,
      &checkpoint_count_, &checkpoint_bytes_, &checkpoint_layout_));
  for (const auto& entry : recovered) {
    InferenceRequest::SequenceId id;
    if ((entry.sequence_id_.size() > 2) &&
        (entry.sequence_id_.compare(0, 2, "s:") == 0)) {
      id = InferenceRequest::SequenceId(entry.sequence_id_.substr(2));
    } else if (entry.sequence_id_.compare(0, 2, "u:") == 0) {
      uint64_t numeric = 0;
      const auto& text = entry.sequence_id_;
      const auto parsed =
          std::from_chars(text.data() + 2, text.data() + text.size(), numeric);
      if ((parsed.ec != std::errc()) ||
          (parsed.ptr != text.data() + text.size()) || (numeric == 0) ||
          (StorageId(InferenceRequest::SequenceId(numeric)) != text)) {
        continue;
      }
      id = InferenceRequest::SequenceId(numeric);
    } else {
      continue;
    }
    checkpoints_.emplace(
        id, CheckpointRecord{CheckpointPhase::SUSPENDED, entry.byte_size_, {}});
    paused_sequences_.insert(id);
  }
  return SequenceStateCheckpointExecutor::Create(
      std::move(store), options, &checkpoint_executor_);
}

Status
SequenceBatchScheduler::SequenceCheckpoint(
    TRITONSERVER_SequenceCheckpointOperation operation,
    InferenceRequest::SequenceId id, std::shared_ptr<Model> model,
    CheckpointCompleteFn complete)
{
  if (!CheckpointEnabled()) {
    return Status(
        Status::Code::UNSUPPORTED, "sequence checkpointing is disabled");
  }
  if (!id.InSequence() || !complete ||
      (operation < TRITONSERVER_SEQUENCE_CHECKPOINT_SUSPEND) ||
      (operation > TRITONSERVER_SEQUENCE_CHECKPOINT_STATUS)) {
    return Status(
        Status::Code::INVALID_ARG, "invalid sequence checkpoint operation");
  }
  if (operation == TRITONSERVER_SEQUENCE_CHECKPOINT_STATUS) {
    std::string result;
    const auto status = CheckpointStatus(id, &result);
    complete(status, result);
    return Status::Success;
  }
  if (StorageId(id).size() > checkpoint_file_limit_ - checkpoint_base_bytes_) {
    return Status(
        Status::Code::INVALID_ARG, "sequence ID exceeds checkpoint file limit");
  }
  std::lock_guard<std::mutex> lock(checkpoint_control_mu_);
  if (checkpoint_controls_.size() + (checkpoint_busy_ ? 1 : 0) >=
      checkpoint_max_controls_) {
    return Status(
        Status::Code::UNAVAILABLE, "checkpoint control queue is full");
  }
  checkpoint_controls_.push_back(
      {operation, std::move(id), std::move(complete)});
  if (checkpoint_busy_) {
    return Status::Success;
  }
  checkpoint_busy_ = true;
  try {
    std::thread([this, model = std::move(model)] {
      for (;;) {
        CheckpointControl control;
        {
          std::lock_guard<std::mutex> lock(checkpoint_control_mu_);
          if (checkpoint_controls_.empty()) {
            checkpoint_busy_ = false;
            break;
          }
          control = std::move(checkpoint_controls_.front());
          checkpoint_controls_.pop_front();
        }
        const auto status =
            RunCheckpointOperation(control.operation_, control.id_);
        std::string result;
        CheckpointStatus(control.id_, &result);
        control.complete_(status, result);
      }
    }).detach();
  }
  catch (const std::system_error& error) {
    checkpoint_busy_ = false;
    checkpoint_controls_.clear();
    return Status(Status::Code::INTERNAL, error.what());
  }
  return Status::Success;
}

Status
SequenceBatchScheduler::RunCheckpointOperation(
    TRITONSERVER_SequenceCheckpointOperation operation,
    const InferenceRequest::SequenceId& id)
{
  const auto storage_id = StorageId(id);
  DirectSequenceBatch* batcher;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (stop_ || checkpoint_fenced_) {
      return Status(
          Status::Code::UNAVAILABLE,
          "checkpoint storage or scheduler is stopped");
    }
    batcher =
        static_cast<DirectSequenceBatch*>(batchers_.begin()->second.get());
  }
  auto finish_io =
      [this, &storage_id](SequenceStateCheckpointExecutor::Completion* result) {
        if (!checkpoint_executor_->WaitCompletion(storage_id, result)) {
          return Status(Status::Code::INTERNAL, "checkpoint result is missing");
        }
        if (result->durability_uncertain_) {
          std::lock_guard<std::mutex> lock(mu_);
          checkpoint_fenced_ = true;
        }
        return result->status_;
      };
  auto fail = [this, &id](const Status& status, CheckpointPhase phase) {
    std::lock_guard<std::mutex> lock(mu_);
    auto& record = checkpoints_.at(id);
    record.error_ = status.Message();
    record.phase_ = checkpoint_fenced_ ? CheckpointPhase::ERROR : phase;
    return status;
  };
  auto account_remove =
      [this, &id](const SequenceStateCheckpointExecutor::Completion& result) {
        if (result.checkpoint_byte_size_ != 0) {
          std::lock_guard<std::mutex> lock(mu_);
          checkpoint_bytes_ -= checkpoints_.at(id).bytes_;
          checkpoints_.at(id).bytes_ = 0;
          --checkpoint_count_;
        }
      };

  if (operation == TRITONSERVER_SEQUENCE_CHECKPOINT_SUSPEND) {
    BatcherSequenceSlot slot;
    RETURN_IF_ERROR(PauseSequenceAdmission(id, &slot));
    {
      std::lock_guard<std::mutex> lock(mu_);
      if ((checkpoint_count_ >= checkpoint_max_count_) ||
          (checkpoint_base_bytes_ + storage_id.size() >
           checkpoint_max_bytes_ - checkpoint_bytes_)) {
        paused_sequences_.erase(id);
        return Status(
            Status::Code::UNAVAILABLE, "stored checkpoint limit exceeded");
      }
      checkpoints_.emplace(
          id, CheckpointRecord{CheckpointPhase::SUSPENDING, 0, {}});
    }
    SequenceStateSnapshot snapshot;
    auto status = batcher->CheckpointState(
        DirectSequenceBatch::StateOperation::CAPTURE, slot.seq_slot_, id,
        &snapshot);
    if (status.IsOk()) {
      status =
          checkpoint_executor_->SubmitSave(storage_id, std::move(snapshot));
      if (status.IsOk()) {
        SequenceStateCheckpointExecutor::Completion result;
        status = finish_io(&result);
        if (result.checkpoint_byte_size_ != 0) {
          std::lock_guard<std::mutex> lock(mu_);
          checkpoints_.at(id).bytes_ = result.checkpoint_byte_size_;
          checkpoint_bytes_ += result.checkpoint_byte_size_;
          ++checkpoint_count_;
        }
      }
    }
    if (!status.IsOk()) {
      std::lock_guard<std::mutex> lock(mu_);
      if (checkpoint_fenced_) {
        checkpoints_.at(id).phase_ = CheckpointPhase::ERROR;
        checkpoints_.at(id).error_ = status.Message();
      } else {
        checkpoints_.erase(id);
        paused_sequences_.erase(id);
        if (sequence_to_batcherseqslot_map_.count(id) != 0) {
          correlation_id_timestamps_[id] = NowMicros();
        }
      }
      return status;
    }
    status = batcher->CheckpointState(
        DirectSequenceBatch::StateOperation::RELEASE, slot.seq_slot_, id,
        nullptr);
    if (!status.IsOk()) {
      std::lock_guard<std::mutex> lock(mu_);
      checkpoint_fenced_ = true;
      checkpoints_.at(id).phase_ = CheckpointPhase::ERROR;
      checkpoints_.at(id).error_ = status.Message();
      return status;
    }
    std::lock_guard<std::mutex> lock(mu_);
    checkpoints_.at(id).phase_ = CheckpointPhase::SUSPENDED;
    return Status::Success;
  }

  {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = checkpoints_.find(id);
    if (it == checkpoints_.end()) {
      return Status(Status::Code::NOT_FOUND, "sequence has no checkpoint");
    }
    if (it->second.phase_ != CheckpointPhase::SUSPENDED) {
      return Status(Status::Code::UNAVAILABLE, "checkpoint is not suspended");
    }
  }

  if (operation == TRITONSERVER_SEQUENCE_CHECKPOINT_RESUME) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      checkpoints_.at(id).phase_ = CheckpointPhase::RESUMING;
    }
    auto status = checkpoint_executor_->SubmitLoad(storage_id);
    SequenceStateCheckpointExecutor::Completion loaded;
    if (status.IsOk()) {
      status = finish_io(&loaded);
    }
    if (!status.IsOk()) {
      return fail(status, CheckpointPhase::SUSPENDED);
    }
    BatcherSequenceSlot slot;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (stop_ || ready_batcher_seq_slots_.empty()) {
        checkpoints_.at(id).phase_ = CheckpointPhase::SUSPENDED;
        return Status(Status::Code::UNAVAILABLE, "no sequence slot available");
      }
      slot = ready_batcher_seq_slots_.top();
      ready_batcher_seq_slots_.pop();
    }
    status = batcher->CheckpointState(
        DirectSequenceBatch::StateOperation::RESTORE, slot.seq_slot_, id,
        loaded.snapshot_.get());
    if (!status.IsOk()) {
      {
        std::lock_guard<std::mutex> lock(mu_);
        ready_batcher_seq_slots_.push(slot);
      }
      return fail(status, CheckpointPhase::SUSPENDED);
    }
    status = checkpoint_executor_->SubmitRemove(storage_id);
    SequenceStateCheckpointExecutor::Completion removed;
    if (status.IsOk()) {
      status = finish_io(&removed);
    }
    account_remove(removed);
    if (!status.IsOk()) {
      if (!removed.durability_uncertain_) {
        const auto released = batcher->CheckpointState(
            DirectSequenceBatch::StateOperation::RELEASE, slot.seq_slot_, id,
            nullptr);
        if (!released.IsOk()) {
          std::lock_guard<std::mutex> lock(mu_);
          checkpoint_fenced_ = true;
        }
      }
      return fail(status, CheckpointPhase::SUSPENDED);
    }
    std::lock_guard<std::mutex> lock(mu_);
    sequence_to_batcherseqslot_map_[id] = slot;
    correlation_id_timestamps_[id] =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();
    checkpoints_.erase(id);
    paused_sequences_.erase(id);
    return Status::Success;
  }

  auto status = checkpoint_executor_->SubmitRemove(storage_id);
  SequenceStateCheckpointExecutor::Completion removed;
  if (status.IsOk()) {
    status = finish_io(&removed);
  }
  account_remove(removed);
  if (!status.IsOk()) {
    return fail(status, CheckpointPhase::SUSPENDED);
  }
  std::lock_guard<std::mutex> lock(mu_);
  checkpoints_.erase(id);
  paused_sequences_.erase(id);
  return Status::Success;
}

Status
SequenceBatchScheduler::CheckpointStatus(
    const InferenceRequest::SequenceId& id, std::string* json)
{
  std::lock_guard<std::mutex> lock(mu_);
  const auto it = checkpoints_.find(id);
  const char* phase = "active";
  if (it != checkpoints_.end()) {
    switch (it->second.phase_) {
      case CheckpointPhase::SUSPENDING:
        phase = "suspending";
        break;
      case CheckpointPhase::SUSPENDED:
        phase = "suspended";
        break;
      case CheckpointPhase::RESUMING:
        phase = "resuming";
        break;
      case CheckpointPhase::ERROR:
        phase = "error";
        break;
    }
  } else if (sequence_to_batcherseqslot_map_.count(id) == 0) {
    return Status(Status::Code::NOT_FOUND, "sequence does not exist");
  }
  triton::common::TritonJson::Value result(
      triton::common::TritonJson::ValueType::OBJECT);
  RETURN_IF_ERROR(result.AddString("state", phase));
  RETURN_IF_ERROR(result.AddUInt(
      "checkpoint_bytes", it == checkpoints_.end() ? 0 : it->second.bytes_));
  RETURN_IF_ERROR(result.AddUInt("stored_bytes", checkpoint_bytes_));
  RETURN_IF_ERROR(result.AddUInt("stored_count", checkpoint_count_));
  RETURN_IF_ERROR(result.AddBool("fenced", checkpoint_fenced_));
  const auto io = checkpoint_executor_->GetStatistics();
  RETURN_IF_ERROR(result.AddUInt("queued_jobs", io.queued_job_count_));
  RETURN_IF_ERROR(result.AddUInt("active_jobs", io.active_job_count_));
  RETURN_IF_ERROR(
      result.AddUInt("reserved_payload_bytes", io.reserved_byte_size_));
  RETURN_IF_ERROR(result.AddString(
      "last_error", it == checkpoints_.end() ? "" : it->second.error_));
  triton::common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(result.Write(&buffer));
  *json = buffer.Contents();
  return Status::Success;
}

}}  // namespace triton::core
