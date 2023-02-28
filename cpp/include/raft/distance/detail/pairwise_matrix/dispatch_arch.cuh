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

/* This file has two responsibilities:
 *
 * 1. Dispatch to the correct implementation of a kernel based on the
 *    architecture of the device on which the kernel will be launched. For
 *    instance, the cosine distance has a CUTLASS-based implementation that can
 *    be used on SM80+ and a legacy implementation.
 *
 * 2. Provide concise function templates that can be instantiated in
 *    src/distance/distance/specializations/detail/. Previously,
 *    raft::distance::detail::distance was instantiated. The function
 *    necessarily required a large set of include files, which slowed down the
 *    build. The raft::distance::detail::pairwise_matrix_arch_dispatch functions
 *    do not require as large an include files set, which speeds up the build.
 */

#include <raft/distance/detail/distance_ops/all_ops.cuh>
#include <raft/util/arch.cuh>
#include <raft/util/cuda_rt_essentials.hpp>
// NOTE: to minimize compile times, we do not include dispatch_sm60.cuh and
// dispatch_sm80.cuh. Especially dispatch_sm80.cuh can slow down compile times
// (due to CUTLASS). Therefore, it is the caller's responsibility to include the
// correct dispatch_smXX.cuh headers, as is done in
// raft/distance/detail/distance.cuh and the specializations in
// src/distance/distance/specializations/detail/.

namespace raft::distance::detail {

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT>
struct pairwise_matrix_dispatch_params {
  IdxT m;
  IdxT n;
  IdxT k;
  const DataT* x;
  const DataT* y;
  const DataT* x_norm;
  const DataT* y_norm;
  OutT* out;
  FinOpT fin_op;
  cudaStream_t stream;
  bool is_row_major;
};

template <typename AccT, typename IdxT, typename DataT, typename OutT, typename FinOpT>
pairwise_matrix_dispatch_params<DataT, AccT, OutT, FinOpT, IdxT>
make_pairwise_matrix_dispatch_params(IdxT m,
                                     IdxT n,
                                     IdxT k,
                                     const DataT* x,
                                     const DataT* y,
                                     const DataT* x_norm,
                                     const DataT* y_norm,
                                     OutT* out,
                                     FinOpT fin_op,
                                     cudaStream_t stream,
                                     bool is_row_major)
{
  pairwise_matrix_dispatch_params<DataT, AccT, OutT, FinOpT, IdxT> params{
    m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major};
  return params;
}

template <typename OpT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename FinOpT,
          typename IdxT>
raft::raft_cuda_error_t pairwise_matrix_dispatch_any_to_sm60(
  OpT distance_op, pairwise_matrix_dispatch_params<DataT, AccT, OutT, FinOpT, IdxT> params)
{
  auto any_arch = raft::arch::SM_range(raft::arch::SM_min(), raft::arch::SM_future());
  return pairwise_matrix_sm60_dispatch(distance_op, any_arch, params);
}

template <typename OpT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename FinOpT,
          typename IdxT>
raft::raft_cuda_error_t pairwise_matrix_dispatch_split_at_sm80(
  OpT distance_op, pairwise_matrix_dispatch_params<DataT, AccT, OutT, FinOpT, IdxT> params)
{
  // On CUDA 12:
  // - always execute normal kernel
  //
  // On CUDA 11 and below:
  // - execute CUTLASS-based kernel on SM_80 and above
  // - execute normal kernel otherwise.

  if constexpr (__CUDACC_VER_MAJOR__ == 12) {
    // Always execute legacy kernels on CUDA 12
    auto any_arch = raft::arch::SM_range(raft::arch::SM_min(), raft::arch::SM_future());

    return pairwise_matrix_sm60_dispatch(distance_op, any_arch, params);
  } else {
    auto runtime_arch  = raft::arch::kernel_runtime_arch();
    auto cutlass_range = raft::arch::SM_range(raft::arch::SM_80(), raft::arch::SM_future());
    auto legacy_range  = raft::arch::SM_range(raft::arch::SM_min(), raft::arch::SM_80());

    if (cutlass_range.contains(runtime_arch)) {
      // If device is SM_80 or later, use CUTLASS-based kernel.
      return pairwise_matrix_sm80_dispatch(distance_op, cutlass_range, params);
    } else {
      // Else use "legacy" L2. Compile *only* for architectures in the legacy
      // range. For newer architectures, compile empty kernels.
      return pairwise_matrix_sm60_dispatch(distance_op, legacy_range, params);
    }
  }
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT>
raft::raft_cuda_error_t pairwise_matrix_arch_dispatch(
  ops::canberra_distance_op<DataT, AccT, IdxT> distance_op,
  pairwise_matrix_dispatch_params<DataT, AccT, OutT, FinOpT, IdxT> params)
{
  return pairwise_matrix_dispatch_any_to_sm60(distance_op, params);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT>
raft::raft_cuda_error_t pairwise_matrix_arch_dispatch(
  ops::correlation_distance_op<DataT, AccT, IdxT> distance_op,
  pairwise_matrix_dispatch_params<DataT, AccT, OutT, FinOpT, IdxT> params)
{
  return pairwise_matrix_dispatch_any_to_sm60(distance_op, params);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT>
raft::raft_cuda_error_t pairwise_matrix_arch_dispatch(
  ops::cosine_distance_op<DataT, AccT, IdxT> distance_op,
  pairwise_matrix_dispatch_params<DataT, AccT, OutT, FinOpT, IdxT> params)
{
  return pairwise_matrix_dispatch_split_at_sm80(distance_op, params);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT>
raft::raft_cuda_error_t pairwise_matrix_arch_dispatch(
  ops::hamming_distance_op<DataT, AccT, IdxT> distance_op,
  pairwise_matrix_dispatch_params<DataT, AccT, OutT, FinOpT, IdxT> params)
{
  return pairwise_matrix_dispatch_any_to_sm60(distance_op, params);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT>
raft::raft_cuda_error_t pairwise_matrix_arch_dispatch(
  ops::hellinger_distance_op<DataT, AccT, IdxT> distance_op,
  pairwise_matrix_dispatch_params<DataT, AccT, OutT, FinOpT, IdxT> params)
{
  return pairwise_matrix_dispatch_any_to_sm60(distance_op, params);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT>
raft::raft_cuda_error_t pairwise_matrix_arch_dispatch(
  ops::jensen_shannon_distance_op<DataT, AccT, IdxT> distance_op,
  pairwise_matrix_dispatch_params<DataT, AccT, OutT, FinOpT, IdxT> params)
{
  return pairwise_matrix_dispatch_any_to_sm60(distance_op, params);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT>
raft::raft_cuda_error_t pairwise_matrix_arch_dispatch(
  ops::kl_divergence_op<DataT, AccT, IdxT> distance_op,
  pairwise_matrix_dispatch_params<DataT, AccT, OutT, FinOpT, IdxT> params)
{
  return pairwise_matrix_dispatch_any_to_sm60(distance_op, params);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT>
raft::raft_cuda_error_t pairwise_matrix_arch_dispatch(
  ops::l1_distance_op<DataT, AccT, IdxT> distance_op,
  pairwise_matrix_dispatch_params<DataT, AccT, OutT, FinOpT, IdxT> params)
{
  return pairwise_matrix_dispatch_any_to_sm60(distance_op, params);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT>
raft::raft_cuda_error_t pairwise_matrix_arch_dispatch(
  ops::l2_exp_distance_op<DataT, AccT, IdxT> distance_op,
  pairwise_matrix_dispatch_params<DataT, AccT, OutT, FinOpT, IdxT> params)
{
  return pairwise_matrix_dispatch_split_at_sm80(distance_op, params);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT>
raft::raft_cuda_error_t pairwise_matrix_arch_dispatch(
  ops::l2_unexp_distance_op<DataT, AccT, IdxT> distance_op,
  pairwise_matrix_dispatch_params<DataT, AccT, OutT, FinOpT, IdxT> params)
{
  return pairwise_matrix_dispatch_any_to_sm60(distance_op, params);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT>
raft::raft_cuda_error_t pairwise_matrix_arch_dispatch(
  ops::l_inf_distance_op<DataT, AccT, IdxT> distance_op,
  pairwise_matrix_dispatch_params<DataT, AccT, OutT, FinOpT, IdxT> params)
{
  return pairwise_matrix_dispatch_any_to_sm60(distance_op, params);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT>
raft::raft_cuda_error_t pairwise_matrix_arch_dispatch(
  ops::lp_unexp_distance_op<DataT, AccT, IdxT> distance_op,
  pairwise_matrix_dispatch_params<DataT, AccT, OutT, FinOpT, IdxT> params)
{
  return pairwise_matrix_dispatch_any_to_sm60(distance_op, params);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT>
raft::raft_cuda_error_t pairwise_matrix_arch_dispatch(
  ops::russel_rao_distance_op<DataT, AccT, IdxT> distance_op,
  pairwise_matrix_dispatch_params<DataT, AccT, OutT, FinOpT, IdxT> params)
{
  return pairwise_matrix_dispatch_any_to_sm60(distance_op, params);
}

}  // namespace raft::distance::detail
