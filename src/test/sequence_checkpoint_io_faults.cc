// Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace {
std::atomic<bool> fail_directory_sync{false};
std::atomic<bool> hold_next_release{false};
std::atomic<bool> request_held{false};
std::mutex release_mutex;
std::condition_variable release_cv;
bool allow_release = false;
}  // namespace

extern "C" void
TritonCheckpointHoldNextRequestRelease()
{
  std::lock_guard<std::mutex> lock(release_mutex);
  allow_release = false;
  hold_next_release = true;
}

extern "C" bool
TritonCheckpointRequestReleaseIsHeld()
{
  return request_held;
}

extern "C" void
TritonCheckpointReleaseRequest()
{
  {
    std::lock_guard<std::mutex> lock(release_mutex);
    allow_release = true;
  }
  release_cv.notify_all();
}

struct TRITONSERVER_Error;
struct TRITONBACKEND_Request;

extern "C" TRITONSERVER_Error*
TRITONBACKEND_RequestRelease(TRITONBACKEND_Request* request, uint32_t flags)
{
  static const auto release =
      reinterpret_cast<TRITONSERVER_Error* (*)(TRITONBACKEND_Request*,
                                               uint32_t)>(
          dlsym(RTLD_NEXT, "TRITONBACKEND_RequestRelease"));
  if (!hold_next_release.exchange(false)) {
    return release(request, flags);
  }
  std::thread([request, flags] {
    {
      std::unique_lock<std::mutex> lock(release_mutex);
      request_held = true;
      release_cv.wait(lock, [] { return allow_release; });
    }
    auto* error = release(request, flags);
    if (error != nullptr) {
      static const auto delete_error =
          reinterpret_cast<void (*)(TRITONSERVER_Error*)>(
              dlsym(RTLD_NEXT, "TRITONSERVER_ErrorDelete"));
      delete_error(error);
    }
    request_held = false;
  }).detach();
  return nullptr;
}

extern "C" void
TritonCheckpointFailDirectorySyncOnce()
{
  fail_directory_sync = true;
}

extern "C" int
fsync(int fd)
{
  static const auto sync =
      reinterpret_cast<int (*)(int)>(dlsym(RTLD_NEXT, "fsync"));
  struct stat info;
  if ((fstat(fd, &info) == 0) && S_ISDIR(info.st_mode) &&
      fail_directory_sync.exchange(false)) {
    errno = EIO;
    return -1;
  }
  return sync(fd);
}
