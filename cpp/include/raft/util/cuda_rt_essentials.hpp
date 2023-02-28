/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// This file provides a few essential functions that wrap the CUDA runtime API.
// The scope is necessarily limited to ensure that compilation times are
// minimized. Please make sure not to include large / expensive files from here.
// Specifically, the code below has been architected to prevent inclusion of
// <string> and <exception>, which can slow down device code compilation (even
// when they are not used in device code).

#include <cuda_runtime.h>

namespace raft {

struct raft_cuda_error_t {
  const cudaError_t status;
  const char* call_site;
  const char* cuda_error_name;
  const char* cuda_error_string;

  // TODO: add conversions from/to cudaError_t

  explicit operator cudaError_t() const { return status; }
};

inline raft_cuda_error_t raft_success()
{
  raft_cuda_error_t status{cudaSuccess, nullptr, nullptr, nullptr};
  return status;
}

}  // namespace raft

/**
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call, if the call does not return
 * cudaSuccess, invokes cudaGetLastError() to clear the error and returns
 * `raft_cuda_error_t` detailing the CUDA error that occurred.
 *
 */
#define RAFT_CUDA_RETURN_ON_ERROR(call)                                                      \
  do {                                                                                       \
    auto const status             = call;                                                    \
    cudaError_t const cuda_status = static_cast<cudaError_t>(status);                        \
    if (cuda_status != cudaSuccess) {                                                        \
      raft::raft_cuda_error_t error{                                                         \
        cuda_status, #call, cudaGetErrorName(cuda_status), cudaGetErrorString(cuda_status)}; \
      /* Reset error to cudaSuccess */                                                       \
      cudaGetLastError();                                                                    \
      return error;                                                                          \
    }                                                                                        \
  } while (0)
