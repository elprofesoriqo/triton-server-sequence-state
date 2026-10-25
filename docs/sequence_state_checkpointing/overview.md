# Sequence state checkpointing in NVIDIA Triton Inference Server

## Purpose

Triton keeps model state between requests in a sequence. An active sequence
uses memory and occupies a scheduler slot. Its state is normally lost when the
sequence ends or the server restarts.

The project adds explicit state checkpointing. A sequence can be suspended,
removed from an execution slot, and restored without replaying previous
requests.

## Operations

The server will provide four operations:

- `suspend` saves a sequence and releases its memory and slot.
- `resume` restores a saved sequence.
- `discard` removes a saved checkpoint.
- `status` reports session and resource usage.

## Suspend Flow

1. Stop accepting new requests for the selected sequence.
2. Finish requests that were already accepted.
3. Wait until the backend no longer uses the state.
4. Copy the state into a separate memory buffer.
5. Send the snapshot to a bounded queue.
6. A worker writes the checkpoint to disk.
7. Release the live state and scheduler slot after the checkpoint is durably
   committed.

If the write fails, the original sequence remains active.

## Resume Flow

1. Read the checkpoint from disk.
2. Check its format, size, checksum, sequence ID, model, and state layout.
3. Reserve a scheduler slot.
4. Restore the state.
5. Continue the sequence without processing its earlier requests again.

The same checkpoint can be used after a planned server restart.

## Components

- Scheduler integration for request admission, slot ownership, and safe state
  capture.
- A bounded queue and worker threads for file operations.
- A checkpoint file format with metadata and a checksum.
- Resource limits for queued jobs, memory, saved sessions, and disk usage.
- Startup recovery that finds valid checkpoints and ignores incomplete files.
- Safe shutdown that stops workers and completes or cancels accepted jobs.

## Scope

The first version supports:

- Linux and C++;
- one model instance on CPU;
- ONNX Runtime for integration tests;
- Direct Sequence Batching;
- fixed-size numeric implicit state.

## Correctness Requirements

- State is not copied or released while the backend uses it.
- Accepted requests are not lost or executed twice.
- A successful suspend releases the state and slot.
- A failed suspend leaves the sequence usable.
- Resumed execution matches uninterrupted execution.
- Corrupt or incompatible checkpoints are rejected.
- Valid checkpoints remain available after a server restart.
- Queue length, memory use, session count, and disk use have fixed limits.
- File operations do not block unrelated inference.
- Shutdown leaves no running workers or incomplete published checkpoints.
- The complete demonstration runs on Linux or WSL2 with CPU.
