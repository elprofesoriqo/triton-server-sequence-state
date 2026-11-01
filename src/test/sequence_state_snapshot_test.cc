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

#include <array>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"
#include "pinned_memory_manager.h"
#include "triton_memory.h"

namespace triton { namespace core { namespace {

using StateConfig = inference::ModelSequenceBatching_State;
using StateConfigMap = std::unordered_map<std::string, const StateConfig&>;

StateConfig
MakeStateConfig(
    const std::string& input_name, const std::string& output_name,
    inference::DataType datatype, std::initializer_list<int64_t> shape,
    bool use_single_buffer = false)
{
  StateConfig config;
  config.set_input_name(input_name);
  config.set_output_name(output_name);
  config.set_data_type(datatype);
  for (const int64_t dimension : shape) {
    config.add_dims(dimension);
  }
  config.set_use_same_buffer_for_input_output(use_single_buffer);
  return config;
}

Status
InitializeStates(
    const std::vector<StateConfig>& configs, SequenceStates* states)
{
  StateConfigMap config_map;
  for (const StateConfig& config : configs) {
    config_map.emplace(config.output_name(), config);
  }
  const std::unordered_map<std::string, SequenceStates::InitialStateData>
      initial_state;
  return states->Initialize(
      config_map, 0, initial_state, TRITONSERVER_INSTANCEGROUPKIND_CPU, 0, {});
}

template <typename T>
void
SetValues(
    SequenceStates* states, const std::string& name,
    const std::vector<T>& values)
{
  auto state = states->InputStates().at(name);
  auto memory = std::static_pointer_cast<MutableMemory>(state->Data());
  ASSERT_EQ(memory->TotalByteSize(), values.size() * sizeof(T));
  char* buffer = memory->MutableBuffer();
  ASSERT_NE(buffer, nullptr);
  std::memcpy(buffer, values.data(), memory->TotalByteSize());
}

template <typename T>
std::vector<T>
Values(const SequenceStates& states, const std::string& name)
{
  const auto& memory = states.InputStates().at(name)->Data();
  std::vector<T> values(memory->TotalByteSize() / sizeof(T));
  size_t offset = 0;
  for (size_t index = 0; index < memory->BufferCount(); ++index) {
    size_t byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    const char* buffer =
        memory->BufferAt(index, &byte_size, &memory_type, &memory_type_id);
    std::memcpy(
        reinterpret_cast<char*>(values.data()) + offset, buffer, byte_size);
    offset += byte_size;
  }
  return values;
}

class SequenceStateSnapshotTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite()
  {
    const Status status =
        PinnedMemoryManager::Create(PinnedMemoryManager::Options{});
    ASSERT_TRUE(status.IsOk()) << status.Message();
  }

  static void TearDownTestSuite() { PinnedMemoryManager::Reset(); }
};

TEST_F(SequenceStateSnapshotTest, CaptureAndRestore)
{
  const std::vector<StateConfig> configs{
      MakeStateConfig(
          "Z_INPUT", "Z_OUTPUT", inference::DataType::TYPE_FP32, {1}),
      MakeStateConfig(
          "A_INPUT", "A_OUTPUT", inference::DataType::TYPE_INT32, {2})};

  SequenceStates source;
  ASSERT_TRUE(InitializeStates(configs, &source).IsOk());

  const std::array<int32_t, 2> a_values{4, 7};
  auto fragmented = std::make_shared<MemoryReference>();
  fragmented->AddBuffer(
      reinterpret_cast<const char*>(&a_values[0]), sizeof(a_values[0]),
      TRITONSERVER_MEMORY_CPU, 0);
  fragmented->AddBuffer(
      reinterpret_cast<const char*>(&a_values[1]), sizeof(a_values[1]),
      TRITONSERVER_MEMORY_CPU, 0);
  auto a_state = source.InputStates().at("A_INPUT");
  ASSERT_TRUE(a_state->RemoveAllData().IsOk());
  ASSERT_TRUE(a_state->SetData(fragmented).IsOk());
  SetValues<float>(&source, "Z_INPUT", {1.25F});

  SequenceStateSnapshot snapshot;
  const Status capture_status =
      SequenceStateSnapshot::Capture(source, &snapshot);
  ASSERT_TRUE(capture_status.IsOk()) << capture_status.Message();
  ASSERT_EQ(snapshot.Tensors().size(), 2);
  EXPECT_EQ(snapshot.Tensors()[0].input_name_, "A_INPUT");
  EXPECT_EQ(snapshot.Tensors()[0].offset_, 0);
  EXPECT_EQ(snapshot.Tensors()[1].input_name_, "Z_INPUT");
  EXPECT_EQ(snapshot.Tensors()[1].offset_, 2 * sizeof(int32_t));

  SequenceStates restored;
  ASSERT_TRUE(InitializeStates(configs, &restored).IsOk());
  const Status restore_status = snapshot.Restore(&restored);
  ASSERT_TRUE(restore_status.IsOk()) << restore_status.Message();
  EXPECT_EQ(Values<int32_t>(restored, "A_INPUT"), (std::vector<int32_t>{4, 7}));
  EXPECT_EQ(Values<float>(restored, "Z_INPUT"), (std::vector<float>{1.25F}));
}

TEST_F(SequenceStateSnapshotTest, RestorePreservesSharedBuffer)
{
  const std::vector<StateConfig> configs{MakeStateConfig(
      "STATE_INPUT", "STATE_OUTPUT", inference::DataType::TYPE_INT64, {2},
      true)};
  SequenceStates source;
  ASSERT_TRUE(InitializeStates(configs, &source).IsOk());
  SetValues<int64_t>(&source, "STATE_INPUT", {13, 21});

  SequenceStateSnapshot snapshot;
  ASSERT_TRUE(SequenceStateSnapshot::Capture(source, &snapshot).IsOk());

  SequenceStates restored;
  ASSERT_TRUE(InitializeStates(configs, &restored).IsOk());
  ASSERT_TRUE(snapshot.Restore(&restored).IsOk());
  const auto input = restored.InputStates().at("STATE_INPUT");
  const auto output = restored.OutputStates().at("STATE_OUTPUT");
  EXPECT_EQ(input->Data().get(), output->Data().get());
  EXPECT_EQ(
      Values<int64_t>(restored, "STATE_INPUT"), (std::vector<int64_t>{13, 21}));
}

TEST_F(SequenceStateSnapshotTest, RejectsUnsupportedState)
{
  const std::vector<StateConfig> string_config{MakeStateConfig(
      "STATE_INPUT", "STATE_OUTPUT", inference::DataType::TYPE_STRING, {1})};
  SequenceStates string_states;
  ASSERT_TRUE(InitializeStates(string_config, &string_states).IsOk());
  SequenceStateSnapshot snapshot;
  EXPECT_EQ(
      SequenceStateSnapshot::Capture(string_states, &snapshot).StatusCode(),
      Status::Code::INVALID_ARG);

  const std::vector<StateConfig> numeric_config{MakeStateConfig(
      "STATE_INPUT", "STATE_OUTPUT", inference::DataType::TYPE_INT32, {1})};
  SequenceStates gpu_states;
  ASSERT_TRUE(InitializeStates(numeric_config, &gpu_states).IsOk());
  const int32_t value = 1;
  auto gpu_memory = std::make_shared<MemoryReference>();
  gpu_memory->AddBuffer(
      reinterpret_cast<const char*>(&value), sizeof(value),
      TRITONSERVER_MEMORY_GPU, 0);
  auto state = gpu_states.InputStates().at("STATE_INPUT");
  ASSERT_TRUE(state->RemoveAllData().IsOk());
  ASSERT_TRUE(state->SetData(gpu_memory).IsOk());
  EXPECT_EQ(
      SequenceStateSnapshot::Capture(gpu_states, &snapshot).StatusCode(),
      Status::Code::INVALID_ARG);
}

TEST_F(SequenceStateSnapshotTest, RejectsDifferentLayout)
{
  const std::vector<StateConfig> source_config{MakeStateConfig(
      "STATE_INPUT", "STATE_OUTPUT", inference::DataType::TYPE_INT32, {2})};
  SequenceStates source;
  ASSERT_TRUE(InitializeStates(source_config, &source).IsOk());
  SetValues<int32_t>(&source, "STATE_INPUT", {1, 2});
  SequenceStateSnapshot snapshot;
  ASSERT_TRUE(SequenceStateSnapshot::Capture(source, &snapshot).IsOk());

  const std::vector<StateConfig> target_config{MakeStateConfig(
      "STATE_INPUT", "STATE_OUTPUT", inference::DataType::TYPE_INT32, {3})};
  SequenceStates target;
  ASSERT_TRUE(InitializeStates(target_config, &target).IsOk());
  EXPECT_EQ(snapshot.Restore(&target).StatusCode(), Status::Code::INVALID_ARG);
}

}}}  // namespace triton::core
