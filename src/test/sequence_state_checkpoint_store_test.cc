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

#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"
#include "pinned_memory_manager.h"
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
  ASSERT_TRUE(
      SequenceStateCheckpointStore::Create(
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

}}}  // namespace triton::core
