#!/usr/bin/env python3
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Run against a built Core library and an installed ONNX Runtime backend."""

import argparse
import concurrent.futures
import ctypes as C
import json
import logging
import tempfile
import threading
import time
from pathlib import Path

import onnx
from onnx import TensorProto, helper

P = C.c_void_p
PP = C.POINTER(P)
ALLOC = C.CFUNCTYPE(
    P,
    P,
    C.c_char_p,
    C.c_size_t,
    C.c_int,
    C.c_int64,
    P,
    PP,
    PP,
    C.POINTER(C.c_int),
    C.POINTER(C.c_int64),
)
FREE = C.CFUNCTYPE(P, P, P, P, C.c_size_t, C.c_int, C.c_int64)
RESPONSE = C.CFUNCTYPE(None, P, C.c_uint32, P)
RELEASE = C.CFUNCTYPE(None, P, C.c_uint32, P)
CONTROL = C.CFUNCTYPE(None, P, P, P)


class Server:
    def __init__(self, library, backends, repository):
        self.lib = C.CDLL(str(library), mode=C.RTLD_GLOBAL)
        self.buffers = {}
        self.callbacks = []
        self.retained_requests = []
        self.server = P()
        self.allocator = P()
        self.bind("ErrorMessage", [P], C.c_char_p)
        self.bind("ErrorDelete", [P], None)
        self.bind("ServerOptionsNew", [PP])
        self.bind("ServerOptionsDelete", [P])
        self.bind("ServerOptionsSetModelRepositoryPath", [P, C.c_char_p])
        self.bind("ServerOptionsSetBackendDirectory", [P, C.c_char_p])
        self.bind("ServerOptionsSetExitTimeout", [P, C.c_uint32])
        self.bind("ServerNew", [PP, P])
        self.bind("ServerDelete", [P])
        self.bind("ServerIsReady", [P, C.POINTER(C.c_bool)])
        self.bind("ResponseAllocatorNew", [PP, ALLOC, FREE, P])
        self.bind("ResponseAllocatorDelete", [P])
        self.bind("InferenceRequestNew", [PP, P, C.c_char_p, C.c_int64])
        self.bind("InferenceRequestDelete", [P])
        self.bind("InferenceRequestSetCorrelationId", [P, C.c_uint64])
        self.bind("InferenceRequestSetCorrelationIdString", [P, C.c_char_p])
        self.bind("InferenceRequestSetFlags", [P, C.c_uint32])
        self.bind(
            "InferenceRequestAddInput",
            [P, C.c_char_p, C.c_int, C.POINTER(C.c_int64), C.c_uint64],
        )
        self.bind(
            "InferenceRequestAppendInputData",
            [P, C.c_char_p, P, C.c_size_t, C.c_int, C.c_int64],
        )
        self.bind("InferenceRequestAddRequestedOutput", [P, C.c_char_p])
        self.bind("InferenceRequestSetReleaseCallback", [P, RELEASE, P])
        self.bind("InferenceRequestSetResponseCallback", [P, P, P, RESPONSE, P])
        self.bind("ServerInferAsync", [P, P, P])
        self.bind("InferenceResponseError", [P])
        self.bind("InferenceResponseDelete", [P])
        self.bind(
            "InferenceResponseOutput",
            [
                P,
                C.c_uint32,
                C.POINTER(C.c_char_p),
                C.POINTER(C.c_int),
                C.POINTER(C.POINTER(C.c_int64)),
                C.POINTER(C.c_uint64),
                PP,
                C.POINTER(C.c_size_t),
                C.POINTER(C.c_int),
                C.POINTER(C.c_int64),
                PP,
            ],
        )
        self.bind(
            "ServerSequenceCheckpoint",
            [P, C.c_char_p, C.c_int64, C.c_uint64, C.c_char_p, C.c_int, CONTROL, P],
        )
        self.bind(
            "MessageSerializeToJson", [P, C.POINTER(C.c_char_p), C.POINTER(C.c_size_t)]
        )
        self.bind("MessageDelete", [P])

        @ALLOC
        def allocate(
            allocator,
            name,
            size,
            kind,
            device,
            user,
            base,
            buffer_user,
            actual,
            actual_id,
        ):
            buffer = C.create_string_buffer(max(1, size))
            address = C.addressof(buffer)
            self.buffers[address] = buffer
            base[0], buffer_user[0] = address, None
            actual[0], actual_id[0] = 0, 0
            return None

        @FREE
        def free(allocator, base, user, size, kind, device):
            self.buffers.pop(base, None)
            return None

        self.callbacks += [allocate, free]
        self.check(
            self.lib.TRITONSERVER_ResponseAllocatorNew(
                C.byref(self.allocator), allocate, free, None
            )
        )
        options = P()
        self.call("ServerOptionsNew", C.byref(options))
        try:
            self.call(
                "ServerOptionsSetModelRepositoryPath", options, str(repository).encode()
            )
            self.call(
                "ServerOptionsSetBackendDirectory", options, str(backends).encode()
            )
            self.call("ServerOptionsSetExitTimeout", options, 5)
            self.call("ServerNew", C.byref(self.server), options)
        finally:
            self.call("ServerOptionsDelete", options)
        ready = C.c_bool()
        self.call("ServerIsReady", self.server, C.byref(ready))
        assert ready.value, "server is not ready"

    def bind(self, name, args, result=P):
        fn = getattr(self.lib, "TRITONSERVER_" + name)
        fn.argtypes, fn.restype = args, result

    def check(self, error):
        if error:
            message = self.lib.TRITONSERVER_ErrorMessage(error).decode()
            self.lib.TRITONSERVER_ErrorDelete(error)
            raise RuntimeError(message)

    def call(self, name, *args):
        self.check(getattr(self.lib, "TRITONSERVER_" + name)(*args))

    def infer(self, sequence_id, value, start=False, end=False, retain=False):
        response_done, released = threading.Event(), threading.Event()
        result = {}
        request = P()
        self.call("InferenceRequestNew", C.byref(request), self.server, b"acc", 1)

        @RESPONSE
        def response_ready(response, flags, user):
            try:
                error = self.lib.TRITONSERVER_InferenceResponseError(response)
                if error:
                    raise RuntimeError(
                        self.lib.TRITONSERVER_ErrorMessage(error).decode()
                    )
                name, datatype, shape = C.c_char_p(), C.c_int(), C.POINTER(C.c_int64)()
                dims, base, size = C.c_uint64(), P(), C.c_size_t()
                kind, device, output_user = C.c_int(), C.c_int64(), P()
                self.call(
                    "InferenceResponseOutput",
                    response,
                    0,
                    C.byref(name),
                    C.byref(datatype),
                    C.byref(shape),
                    C.byref(dims),
                    C.byref(base),
                    C.byref(size),
                    C.byref(kind),
                    C.byref(device),
                    C.byref(output_user),
                )
                assert name.value == b"Y" and size.value == 4
                result["value"] = C.cast(base, C.POINTER(C.c_int32))[0]
            except Exception as error:
                result["error"] = error
            finally:
                if response:
                    self.call("InferenceResponseDelete", response)
                response_done.set()

        @RELEASE
        def request_released(req, flags, user):
            # Retain the released request until both callbacks finish.
            released.set()

        try:
            if isinstance(sequence_id, str):
                self.call(
                    "InferenceRequestSetCorrelationIdString",
                    request,
                    sequence_id.encode(),
                )
            else:
                self.call("InferenceRequestSetCorrelationId", request, sequence_id)
            self.call("InferenceRequestSetFlags", request, int(start) | (int(end) << 1))
            shape = (C.c_int64 * 2)(1, 1)
            data = C.c_int32(value)
            self.call("InferenceRequestAddInput", request, b"X", 8, shape, 2)
            self.call(
                "InferenceRequestAppendInputData", request, b"X", C.byref(data), 4, 0, 0
            )
            self.call("InferenceRequestAddRequestedOutput", request, b"Y")
            self.call(
                "InferenceRequestSetReleaseCallback", request, request_released, None
            )
            self.call(
                "InferenceRequestSetResponseCallback",
                request,
                self.allocator,
                None,
                response_ready,
                None,
            )
            self.call("ServerInferAsync", self.server, request, None)
            assert response_done.wait(20), "response timed out"
            assert released.wait(20), "request release timed out"
            if "error" in result:
                raise result["error"]
            logging.info(
                "infer sequence=%r input=%d output=%d",
                sequence_id,
                value,
                result["value"],
            )
            if retain:
                self.retained_requests.append(request)
                request = P()
            return result["value"]
        finally:
            if request:
                self.call("InferenceRequestDelete", request)

    def control(self, sequence_id, operation):
        done, result = threading.Event(), {}

        @CONTROL
        def completed(error, message, user):
            try:
                self.check(error)
                if message:
                    data, size = C.c_char_p(), C.c_size_t()
                    self.call(
                        "MessageSerializeToJson", message, C.byref(data), C.byref(size)
                    )
                    result["value"] = json.loads(C.string_at(data, size.value))
            except Exception as failure:
                result["error"] = failure
            finally:
                if message:
                    self.call("MessageDelete", message)
                done.set()

        self.call(
            "ServerSequenceCheckpoint",
            self.server,
            b"acc",
            1,
            0 if isinstance(sequence_id, str) else sequence_id,
            sequence_id.encode() if isinstance(sequence_id, str) else None,
            operation,
            completed,
            None,
        )
        assert done.wait(20), "checkpoint operation timed out"
        if "error" in result:
            raise result["error"]
        logging.info(
            "checkpoint sequence=%r operation=%d result=%s",
            sequence_id,
            operation,
            result.get("value"),
        )
        return result.get("value")

    def close(self):
        for request in self.retained_requests:
            self.call("InferenceRequestDelete", request)
        self.retained_requests.clear()
        self.call("ServerDelete", self.server)
        self.call("ResponseAllocatorDelete", self.allocator)


def check_control_queue(server):
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    results = []

    def record(error, message):
        results.append(
            server.lib.TRITONSERVER_ErrorMessage(error).decode() if error else "success"
        )
        if error:
            server.lib.TRITONSERVER_ErrorDelete(error)
        if message:
            server.call("MessageDelete", message)
        if len(results) == 4:
            done.set()

    @CONTROL
    def blocked(error, message, user):
        entered.set()
        release.wait(20)
        record(error, message)

    @CONTROL
    def completed(error, message, user):
        record(error, message)

    server.call(
        "ServerSequenceCheckpoint",
        server.server,
        b"acc",
        1,
        888,
        None,
        2,
        blocked,
        None,
    )
    try:
        assert entered.wait(20), "management callback timed out"
        threads = len(list(Path("/proc/self/task").iterdir()))
        for _ in range(3):
            server.call(
                "ServerSequenceCheckpoint",
                server.server,
                b"acc",
                1,
                888,
                None,
                2,
                completed,
                None,
            )
        assert len(list(Path("/proc/self/task").iterdir())) <= threads
        try:
            server.call(
                "ServerSequenceCheckpoint",
                server.server,
                b"acc",
                1,
                888,
                None,
                2,
                completed,
                None,
            )
            raise AssertionError("management queue limit was ignored")
        except RuntimeError as error:
            assert "queue is full" in str(error)
    finally:
        release.set()
        assert done.wait(20), "queued management completions timed out"
    assert all("no checkpoint" in result for result in results), results


def create_model(root, single_buffer=False, batch_size=1):
    model = root / "models" / "acc"
    (model / "1").mkdir(parents=True)
    inputs = [
        helper.make_tensor_value_info(name, dtype, [None, 1])
        for name, dtype in [
            ("X", TensorProto.INT32),
            ("S_IN", TensorProto.INT32),
            ("START", TensorProto.BOOL),
        ]
    ]
    outputs = [
        helper.make_tensor_value_info(name, TensorProto.INT32, [None, 1])
        for name in ["Y", "S_OUT"]
    ]
    graph = helper.make_graph(
        [
            helper.make_node("Where", ["START", "ZERO", "S_IN"], ["OLD"]),
            helper.make_node("Add", ["X", "OLD"], ["Y"]),
            helper.make_node("Identity", ["Y"], ["S_OUT"]),
        ],
        "acc",
        inputs,
        outputs,
        [helper.make_tensor("ZERO", TensorProto.INT32, [1], [0])],
    )
    onnx.save(
        helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10
        ),
        model / "1" / "model.onnx",
    )
    checkpoints = root / "checkpoints"
    checkpoints.mkdir()
    config = """name: "acc"
backend: "onnxruntime"
max_batch_size: 1
input [{ name: "X" data_type: TYPE_INT32 dims: [1] }]
output [{ name: "Y" data_type: TYPE_INT32 dims: [1] }]
instance_group [{ count: 1 kind: KIND_CPU }]
sequence_batching {
  max_sequence_idle_microseconds: 60000000
  direct {}
  control_input [{ name: "START" control [{ kind: CONTROL_SEQUENCE_START bool_false_true: [false, true] }] }]
  state [{ input_name: "S_IN" output_name: "S_OUT" data_type: TYPE_INT32 dims: [1]
    initial_state { name: "zero" data_type: TYPE_INT32 dims: [1] zero_data: true } }]
}
"""
    limits = {
        "directory": str(checkpoints),
        "workers": 2,
        "max_jobs": 4,
        "max_bytes": 65536,
        "max_file_bytes": 4096,
        "max_count": 1,
        "max_disk_bytes": 32768,
    }
    if single_buffer:
        config = config.replace(
            'initial_state { name: "zero"',
            'use_same_buffer_for_input_output: true\n    initial_state { name: "zero"',
        )
    config = config.replace("max_batch_size: 1", f"max_batch_size: {batch_size}")
    for key, value in limits.items():
        config += f'parameters {{ key: "sequence_checkpoint_{key}" value {{ string_value: "{value}" }} }}\n'
    config += (
        'parameters { key: "intra_op_thread_count" value { string_value: "1" } }\n'
    )
    (model / "config.pbtxt").write_text(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--backend-directory", type=Path, required=True)
    parser.add_argument("--inject-io-errors", action="store_true")
    parser.add_argument("--single-buffer", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--batch-size", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )
    with tempfile.TemporaryDirectory(prefix="triton-checkpoint-") as directory:
        root = Path(directory)
        create_model(root, args.single_buffer, args.batch_size)
        server = Server(args.library, args.backend_directory, root / "models")
        try:
            baseline = 0
            for i, value in enumerate([4, 7, 1, 3, 5]):
                baseline = server.infer(100, value, start=i == 0, end=i == 4)
            assert server.infer(42, 4, start=True, retain=True) == 4
            assert server.infer(42, 7, retain=True) == 11
            (root / "checkpoints").chmod(0o500)
            try:
                try:
                    server.control(42, 0)
                    raise AssertionError("write to read-only directory succeeded")
                except RuntimeError as error:
                    assert "temporary checkpoint" in str(error)
            finally:
                (root / "checkpoints").chmod(0o700)
            assert server.infer(42, 1) == 12
            assert server.control(42, 0)["state"] == "suspended"
            # A single-slot model can now run another sequence.
            assert server.infer(43, 9, start=True) == 9
            try:
                server.control(43, 0)
                raise AssertionError("checkpoint count limit was ignored")
            except RuntimeError as error:
                assert "limit" in str(error)
            assert server.infer(43, 2, end=True) == 11
            try:
                server.infer(42, 1)
                raise AssertionError("suspended inference was accepted")
            except RuntimeError as error:
                assert "paused" in str(error)
            assert server.control(42, 1)["state"] == "active"
            assert server.infer(42, 3) == 15
            server.control(42, 0)
        finally:
            server.close()
        server = Server(args.library, args.backend_directory, root / "models")
        try:
            assert server.control(42, 3)["state"] == "suspended"
            server.control(42, 1)
            assert server.infer(42, 5, end=True) == baseline
            assert server.infer(44, 2, start=True) == 2
            server.control(44, 0)
            server.control(44, 2)
            assert not list((root / "checkpoints").glob("*.ckpt"))
            check_control_queue(server)
            assert server.infer(50, 0, start=True) == 0
            barrier = threading.Barrier(17)

            def submit():
                barrier.wait()
                try:
                    return server.infer(50, 1)
                except RuntimeError as error:
                    assert "paused" in str(error)
                    return None

            with concurrent.futures.ThreadPoolExecutor(max_workers=17) as workers:
                requests = [workers.submit(submit) for _ in range(16)]
                barrier.wait()
                server.control(50, 0)
                values = [request.result() for request in requests]
            accepted = sorted(value for value in values if value is not None)
            assert accepted == list(range(1, len(accepted) + 1)), accepted
            server.control(50, 1)
            assert server.infer(50, 1, end=True) == len(accepted) + 1
            assert server.infer("session/42", 6, start=True) == 6
            server.control("session/42", 0)
        finally:
            server.close()
        server = Server(args.library, args.backend_directory, root / "models")
        try:
            server.control("session/42", 1)
            assert server.infer("session/42", 1, end=True) == 7
            assert server.infer(60, 5, start=True) == 5
            server.control(60, 0)
        finally:
            server.close()
        damaged = root / "checkpoints" / ("seq-" + b"u:60".hex() + ".ckpt")
        contents = bytearray(damaged.read_bytes())
        contents[-1] ^= 1
        damaged.write_bytes(contents)
        server = Server(args.library, args.backend_directory, root / "models")
        try:
            try:
                server.control(60, 3)
                raise AssertionError("corrupt checkpoint was recovered")
            except RuntimeError as error:
                assert "does not exist" in str(error)
        finally:
            server.close()
        damaged.unlink()
        config_path = root / "models" / "acc" / "config.pbtxt"
        config = config_path.read_text()
        config = (
            config.replace(
                '"sequence_checkpoint_max_count" value { string_value: "1"',
                '"sequence_checkpoint_max_count" value { string_value: "8"',
            )
            .replace(
                '"sequence_checkpoint_max_disk_bytes" value { string_value: "32768"',
                '"sequence_checkpoint_max_disk_bytes" value { string_value: "128"',
            )
            .replace(
                '"sequence_checkpoint_max_file_bytes" value { string_value: "4096"',
                '"sequence_checkpoint_max_file_bytes" value { string_value: "128"',
            )
        )
        config_path.write_text(config)
        server = Server(args.library, args.backend_directory, root / "models")
        try:
            assert server.infer(70, 3, start=True) == 3
            server.control(70, 0)
            assert server.infer(71, 4, start=True) == 4
            try:
                server.control(71, 0)
                raise AssertionError("disk reservation limit was ignored")
            except RuntimeError as error:
                assert "limit" in str(error)
            assert server.infer(71, 1, end=True) == 5
            server.control(70, 2)
        finally:
            server.close()
        if args.inject_io_errors:
            hooks = C.CDLL(None)
            fail_sync = hooks.TritonCheckpointFailDirectorySyncOnce
            fail_sync.argtypes, fail_sync.restype = [], None
            hold = hooks.TritonCheckpointHoldNextRequestRelease
            held = hooks.TritonCheckpointRequestReleaseIsHeld
            release = hooks.TritonCheckpointReleaseRequest
            hold.argtypes, hold.restype = [], None
            held.argtypes, held.restype = [], C.c_bool
            release.argtypes, release.restype = [], None
            server = Server(args.library, args.backend_directory, root / "models")
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as workers:
                    hold()
                    request = workers.submit(server.infer, 77, 8, True)
                    control = None
                    try:
                        deadline = time.monotonic() + 10
                        while not held():
                            assert time.monotonic() < deadline, "request was not held"
                            time.sleep(0.001)
                        control = workers.submit(server.control, 77, 0)
                        while server.control(77, 3)["state"] != "suspending":
                            assert time.monotonic() < deadline, "suspend did not start"
                            time.sleep(0.001)
                        assert not control.done(), "captured backend-owned state"
                        assert not list((root / "checkpoints").glob("*.ckpt"))
                        if args.batch_size == 2:
                            assert server.infer(78, 2, start=True, end=True) == 2
                            assert not control.done(), "held sequence was not isolated"
                    finally:
                        release()
                    assert request.result() == 8
                    assert control is not None
                    assert control.result()["state"] == "suspended"
                server.control(77, 1)
                assert server.infer(77, 1, end=True) == 9
                assert server.infer(55, 9, start=True) == 9
                fail_sync()
                try:
                    server.control(55, 0)
                    raise AssertionError("uncertain publication was accepted")
                except RuntimeError as error:
                    assert "durability is uncertain" in str(error)
                status = server.control(55, 3)
                assert status["state"] == "error" and status["fenced"]
                try:
                    server.infer(55, 1)
                    raise AssertionError("fenced sequence advanced")
                except RuntimeError as error:
                    assert "paused" in str(error)
            finally:
                server.close()
            server = Server(args.library, args.backend_directory, root / "models")
            try:
                server.control(55, 1)
                assert server.infer(55, 1, end=True) == 10
                assert server.infer(56, 3, start=True) == 3
                server.control(56, 0)
                fail_sync()
                try:
                    server.control(56, 1)
                    raise AssertionError("uncertain deletion was accepted")
                except RuntimeError as error:
                    assert "durability is uncertain" in str(error)
                status = server.control(56, 3)
                assert status["state"] == "error" and status["fenced"]
                assert status["stored_bytes"] == 0
            finally:
                server.close()
        print(
            "PASS: suspend, slot reuse, admission, resume equivalence, write failure, limits, restart, discard"
        )


if __name__ == "__main__":
    main()
