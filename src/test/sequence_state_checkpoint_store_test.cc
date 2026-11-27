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

#include <unistd.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"
#include "pinned_memory_manager.h"
#include "sequence_state_checkpoint_executor.h"
#include "triton_memory.h"

namespace triton { namespace core { namespace {

using StateConfig = inference::ModelSequenceBatching_State;
using StateConfigMap = std::unordered_map<std::string, const StateConfig&>;

StateConfig
StateConfiguration()
{
  StateConfig config;
  config.set_input_name("STATE_INPUT");
  config.set_output_name("STATE_OUTPUT");
  config.set_data_type(inference::DataType::TYPE_INT32);
  config.add_dims(2);
  return config;
}

Status
InitializeStates(SequenceStates* states)
{
  const StateConfig config = StateConfiguration();
  const StateConfigMap configs{{config.output_name(), config}};
  const std::unordered_map<std::string, SequenceStates::InitialStateData>
      initial_state;
  return states->Initialize(
      configs, 0, initial_state, TRITONSERVER_INSTANCEGROUPKIND_CPU, 0, {});
}

void
SetValues(SequenceStates* states, int32_t first, int32_t second)
{
  auto memory = std::static_pointer_cast<MutableMemory>(
      states->InputStates().at("STATE_INPUT")->Data());
  const int32_t values[]{first, second};
  std::memcpy(memory->MutableBuffer(), values, sizeof(values));
}

std::vector<int32_t>
Values(const SequenceStates& states)
{
  const auto& memory = states.InputStates().at("STATE_INPUT")->Data();
  std::vector<int32_t> values(2);
  size_t byte_size;
  TRITONSERVER_MemoryType memory_type;
  int64_t memory_type_id;
  const char* data =
      memory->BufferAt(0, &byte_size, &memory_type, &memory_type_id);
  std::memcpy(values.data(), data, byte_size);
  return values;
}

std::string
CheckpointName(const std::string& sequence_id)
{
  constexpr char hex[] = "0123456789abcdef";
  std::string name = "seq-";
  for (const unsigned char byte : sequence_id) {
    name.push_back(hex[byte >> 4]);
    name.push_back(hex[byte & 0x0f]);
  }
  return name + ".ckpt";
}

bool
WaitForCompletion(
    SequenceStateCheckpointExecutor* executor,
    SequenceStateCheckpointExecutor::Completion* completion)
{
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline) {
    if (executor->PopCompletion(completion)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

class SequenceStateCheckpointStoreTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite()
  {
    const Status status =
        PinnedMemoryManager::Create(PinnedMemoryManager::Options{});
    ASSERT_TRUE(status.IsOk()) << status.Message();
  }

  static void TearDownTestSuite() { PinnedMemoryManager::Reset(); }

  void SetUp() override
  {
    char directory[] = "/tmp/triton-sequence-checkpoint-XXXXXX";
    ASSERT_NE(mkdtemp(directory), nullptr);
    directory_ = directory;
    const Status status = SequenceStateCheckpointStore::Create(
        directory_, "counter", 1, 1024 * 1024, &store_);
    ASSERT_TRUE(status.IsOk()) << status.Message();
  }

  void TearDown() override { std::filesystem::remove_all(directory_); }

  SequenceStateSnapshot Snapshot(int32_t first = 4, int32_t second = 7)
  {
    SequenceStates states;
    EXPECT_TRUE(InitializeStates(&states).IsOk());
    SetValues(&states, first, second);
    SequenceStateSnapshot snapshot;
    EXPECT_TRUE(SequenceStateSnapshot::Capture(states, &snapshot).IsOk());
    return snapshot;
  }

  std::filesystem::path Path(const std::string& sequence_id) const
  {
    return std::filesystem::path(directory_) / CheckpointName(sequence_id);
  }

  std::string directory_;
  std::unique_ptr<SequenceStateCheckpointStore> store_;
};

TEST_F(SequenceStateCheckpointStoreTest, SavesAndLoadsCheckpoint)
{
  const SequenceStateSnapshot source = Snapshot();
  ASSERT_TRUE(store_->Save("42", source).IsOk());

  SequenceStateSnapshot loaded;
  const Status load_status = store_->Load("42", &loaded);
  ASSERT_TRUE(load_status.IsOk()) << load_status.Message();
  SequenceStates restored;
  ASSERT_TRUE(InitializeStates(&restored).IsOk());
  ASSERT_TRUE(loaded.Restore(&restored).IsOk());
  EXPECT_EQ(Values(restored), (std::vector<int32_t>{4, 7}));
  EXPECT_EQ(
      store_->Save("42", source).StatusCode(), Status::Code::ALREADY_EXISTS);
}

TEST_F(SequenceStateCheckpointStoreTest, RejectsInvalidCheckpoint)
{
  ASSERT_TRUE(store_->Save("corrupt", Snapshot()).IsOk());
  {
    std::fstream file(
        Path("corrupt"), std::ios::binary | std::ios::in | std::ios::out);
    ASSERT_TRUE(file.is_open());
    file.seekg(-1, std::ios::end);
    char byte;
    file.read(&byte, 1);
    file.seekp(-1, std::ios::end);
    byte ^= 1;
    file.write(&byte, 1);
  }
  SequenceStateSnapshot loaded;
  EXPECT_EQ(
      store_->Load("corrupt", &loaded).StatusCode(), Status::Code::INVALID_ARG);

  ASSERT_TRUE(store_->Save("truncated", Snapshot()).IsOk());
  std::filesystem::resize_file(
      Path("truncated"), std::filesystem::file_size(Path("truncated")) - 1);
  EXPECT_EQ(
      store_->Load("truncated", &loaded).StatusCode(),
      Status::Code::INVALID_ARG);

  ASSERT_TRUE(store_->Save("trailing", Snapshot()).IsOk());
  std::ofstream(Path("trailing"), std::ios::binary | std::ios::app).put('\0');
  EXPECT_EQ(
      store_->Load("trailing", &loaded).StatusCode(),
      Status::Code::INVALID_ARG);
}

TEST_F(SequenceStateCheckpointStoreTest, ValidatesIdentityAndRemoval)
{
  ASSERT_TRUE(store_->Save("7", Snapshot()).IsOk());

  std::unique_ptr<SequenceStateCheckpointStore> incompatible;
  ASSERT_TRUE(SequenceStateCheckpointStore::Create(
                  directory_, "other-model", 1, 1024 * 1024, &incompatible)
                  .IsOk());
  SequenceStateSnapshot loaded;
  EXPECT_EQ(
      incompatible->Load("7", &loaded).StatusCode(), Status::Code::INVALID_ARG);

  ASSERT_TRUE(store_->Remove("7").IsOk());
  EXPECT_EQ(store_->Load("7", &loaded).StatusCode(), Status::Code::NOT_FOUND);
  EXPECT_EQ(store_->Remove("7").StatusCode(), Status::Code::NOT_FOUND);
}

TEST_F(SequenceStateCheckpointStoreTest, RejectsSymlink)
{
  const std::filesystem::path target =
      std::filesystem::path(directory_) / "target";
  std::ofstream(target).put('\0');
  std::filesystem::create_symlink(target, Path("link"));
  ASSERT_TRUE(std::filesystem::is_symlink(Path("link")));

  SequenceStateSnapshot loaded;
  EXPECT_NE(store_->Load("link", &loaded).StatusCode(), Status::Code::SUCCESS);
  EXPECT_EQ(store_->Remove("link").StatusCode(), Status::Code::INVALID_ARG);
}

TEST_F(SequenceStateCheckpointStoreTest, ExecutesBoundedCheckpointJobs)
{
  std::unique_ptr<SequenceStateCheckpointExecutor> executor;
  ASSERT_TRUE(SequenceStateCheckpointExecutor::Create(
                  std::move(store_), {1, 1, 1024 * 1024}, &executor)
                  .IsOk());

  ASSERT_TRUE(executor->SubmitSave("42", Snapshot()).IsOk());
  EXPECT_EQ(
      executor->SubmitSave("43", Snapshot()).StatusCode(),
      Status::Code::UNAVAILABLE);

  SequenceStateCheckpointExecutor::Completion completion;
  ASSERT_TRUE(WaitForCompletion(executor.get(), &completion));
  ASSERT_TRUE(completion.status_.IsOk()) << completion.status_.Message();
  EXPECT_EQ(
      completion.operation_, SequenceStateCheckpointExecutor::Operation::SAVE);
  EXPECT_GT(completion.checkpoint_byte_size_, 0);
  auto statistics = executor->GetStatistics();
  EXPECT_EQ(statistics.queued_byte_size_, 0);
  EXPECT_EQ(statistics.active_byte_size_, 0);
  EXPECT_EQ(statistics.completion_byte_size_, 0);
  EXPECT_EQ(statistics.reserved_byte_size_, 0);
  EXPECT_EQ(statistics.stored_byte_size_, completion.checkpoint_byte_size_);

  ASSERT_TRUE(executor->SubmitLoad("42").IsOk());
  ASSERT_TRUE(WaitForCompletion(executor.get(), &completion));
  ASSERT_TRUE(completion.status_.IsOk()) << completion.status_.Message();
  ASSERT_NE(completion.snapshot_, nullptr);
  SequenceStates restored;
  ASSERT_TRUE(InitializeStates(&restored).IsOk());
  ASSERT_TRUE(completion.snapshot_->Restore(&restored).IsOk());
  EXPECT_EQ(Values(restored), (std::vector<int32_t>{4, 7}));

  ASSERT_TRUE(executor->SubmitRemove("42").IsOk());
  ASSERT_TRUE(WaitForCompletion(executor.get(), &completion));
  ASSERT_TRUE(completion.status_.IsOk()) << completion.status_.Message();
  EXPECT_EQ(executor->GetStatistics().stored_byte_size_, 0);
}

TEST_F(SequenceStateCheckpointStoreTest, EnforcesByteLimitAndShutdown)
{
  std::unique_ptr<SequenceStateCheckpointExecutor> limited;
  ASSERT_TRUE(SequenceStateCheckpointExecutor::Create(
                  std::move(store_), {1, 1, 1}, &limited)
                  .IsOk());
  EXPECT_EQ(
      limited->SubmitSave("42", Snapshot()).StatusCode(),
      Status::Code::UNAVAILABLE);
  EXPECT_EQ(limited->GetStatistics().queued_job_count_, 0);

  limited->Shutdown();
  EXPECT_EQ(
      limited->SubmitRemove("42").StatusCode(), Status::Code::UNAVAILABLE);

  std::unique_ptr<SequenceStateCheckpointStore> store;
  ASSERT_TRUE(SequenceStateCheckpointStore::Create(
                  directory_, "counter", 1, 1024 * 1024, &store)
                  .IsOk());
  std::unique_ptr<SequenceStateCheckpointExecutor> executor;
  ASSERT_TRUE(SequenceStateCheckpointExecutor::Create(
                  std::move(store), {1, 1, 1024 * 1024}, &executor)
                  .IsOk());
  ASSERT_TRUE(executor->SubmitSave("shutdown", Snapshot()).IsOk());
  executor->Shutdown();
  SequenceStateCheckpointExecutor::Completion completion;
  ASSERT_TRUE(executor->PopCompletion(&completion));
  EXPECT_TRUE(completion.status_.IsOk()) << completion.status_.Message();
  EXPECT_EQ(
      executor->SubmitRemove("shutdown").StatusCode(),
      Status::Code::UNAVAILABLE);
}

TEST_F(SequenceStateCheckpointStoreTest, SerializesSequenceOperations)
{
  std::unique_ptr<SequenceStateCheckpointExecutor> executor;
  ASSERT_TRUE(SequenceStateCheckpointExecutor::Create(
                  std::move(store_), {2, 4, 4 * 1024 * 1024}, &executor)
                  .IsOk());
  ASSERT_TRUE(executor->SubmitSave("42", Snapshot()).IsOk());
  EXPECT_EQ(
      executor->SubmitRemove("42").StatusCode(), Status::Code::UNAVAILABLE);
  EXPECT_EQ(executor->SubmitLoad("42").StatusCode(), Status::Code::UNAVAILABLE);
  ASSERT_TRUE(executor->SubmitSave("43", Snapshot()).IsOk());
  executor->Shutdown();
  const auto statistics = executor->GetStatistics();
  EXPECT_EQ(statistics.completion_count_, 2);
  EXPECT_GT(statistics.reserved_byte_size_, 0);
  SequenceStateCheckpointExecutor::Completion completion;
  while (executor->PopCompletion(&completion)) {
    EXPECT_TRUE(completion.status_.IsOk());
    EXPECT_FALSE(completion.durability_uncertain_);
  }
  EXPECT_EQ(executor->GetStatistics().reserved_byte_size_, 0);
}

TEST_F(SequenceStateCheckpointStoreTest, AccountsOnlyPublishedCheckpoints)
{
  ASSERT_TRUE(store_->Save("old", Snapshot()).IsOk());
  std::unique_ptr<SequenceStateCheckpointExecutor> executor;
  ASSERT_TRUE(SequenceStateCheckpointExecutor::Create(
                  std::move(store_), {2, 4, 4 * 1024 * 1024}, &executor)
                  .IsOk());
  ASSERT_TRUE(executor->SubmitSave("new", Snapshot()).IsOk());
  SequenceStateCheckpointExecutor::Completion completion;
  ASSERT_TRUE(WaitForCompletion(executor.get(), &completion));
  ASSERT_TRUE(completion.status_.IsOk());
  const size_t published_size = completion.checkpoint_byte_size_;
  ASSERT_TRUE(executor->SubmitRemove("old").IsOk());
  ASSERT_TRUE(WaitForCompletion(executor.get(), &completion));
  ASSERT_TRUE(completion.status_.IsOk());
  EXPECT_EQ(executor->GetStatistics().stored_byte_size_, published_size);
}

TEST_F(SequenceStateCheckpointStoreTest, JoinsConcurrentShutdown)
{
  std::unique_ptr<SequenceStateCheckpointExecutor> executor;
  ASSERT_TRUE(SequenceStateCheckpointExecutor::Create(
                  std::move(store_), {2, 16, 1024 * 1024}, &executor)
                  .IsOk());
  for (size_t i = 0; i < 16; ++i) {
    ASSERT_TRUE(executor->SubmitSave(std::to_string(i), Snapshot()).IsOk());
  }
  size_t completed[2] = {};
  auto shutdown = [&](size_t index) {
    executor->Shutdown();
    completed[index] = executor->GetStatistics().completion_count_;
  };
  std::thread first(shutdown, 0);
  std::thread second(shutdown, 1);
  first.join();
  second.join();
  EXPECT_EQ(completed[0], 16);
  EXPECT_EQ(completed[1], 16);
}

}}}  // namespace triton::core
