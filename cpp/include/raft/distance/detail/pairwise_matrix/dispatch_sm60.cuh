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

#include "dispatch_common.cuh"
#include "kernel_sm60.cuh"
#include <raft/linalg/contractions.cuh>
#include <raft/util/arch.cuh>

namespace raft::distance::detail {

template <typename OpT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename FinOpT,
          typename IdxT        = int,
          typename SM_compat_t = raft::arch::SM_range<raft::arch::SM_min, raft::arch::SM_future>>
void distance_matrix_dispatch(
  OpT distance_op,
  IdxT m,
  IdxT n,
  IdxT k,
  const DataT* x,
  const DataT* y,
  const DataT* x_norm,
  const DataT* y_norm,
  OutT* out,
  FinOpT fin_op,
  cudaStream_t stream,
  bool is_row_major,
  SM_compat_t sm_compat_range = raft::arch::SM_range(raft::arch::SM_min(), raft::arch::SM_future()))
{
  // Determine leading dimensions and, if column-major, flip order of passing x
  // and y.
  IdxT ldx, ldy, ld_out;
  if (is_row_major) {
    ldx = k, ldy = k, ld_out = n;
  } else {
    // Flip x, y, and m, n.
    std::swap<const DataT*>(x, y);
    std::swap<const DataT*>(x_norm, y_norm);
    std::swap(m, n);
    ldx = m, ldy = n, ld_out = n;
  }

  size_t align_x        = alignment_of_2d_array(x, ldx);
  size_t align_y        = alignment_of_2d_array(y, ldy);
  size_t byte_alignment = min(align_x, align_y);

  // Since alignment is in bytes, it could be smaller than sizeof(DataT).
  // Handle this (unlikely) case here.
  RAFT_EXPECTS(sizeof(DataT) <= byte_alignment,
               "Input matrix must be aligned to size of elements.");

  // Compute number of elements that can be loaded in one instruction
  // without causing misalignent errors.
  int vec_len_aligned = (byte_alignment % sizeof(DataT) == 0) ? byte_alignment / sizeof(DataT) : 1;

  dispatch_common(is_row_major, vec_len_aligned, [&](auto row_major, auto vec_len_aligned) {
    // row_major and vec_len are std::integral_constants of type bool and int
    // respectively.

    // To keep compile times in check, we only specialize on veclen > 1 when
    // the inner loop is relatively cheap (< 5 flops).
    constexpr int vec_len_op = distance_op.expensive_inner_loop ? 1 : vec_len_aligned();

    // Prevent double, vec_len=4 combination (this is not supported)
    constexpr int vec_len = std::min(vec_len_op, static_cast<int>(16 / sizeof(DataT)));

    typedef typename raft::linalg::Policy4x4<DataT, vec_len>::Policy RowPolicy;
    typedef typename raft::linalg::Policy4x4<DataT, vec_len>::ColPolicy ColPolicy;
    typedef typename std::conditional<row_major(), RowPolicy, ColPolicy>::type Policy;

    return pairwise_matrix<Policy, row_major(), DataT, AccT, OutT, IdxT, OpT, FinOpT>(
      distance_op,
      fin_op,
      x,
      y,
      x_norm,
      y_norm,
      m,
      n,
      k,
      ldx,
      ldy,
      ld_out,
      out,
      stream,
      sm_compat_range);
  });
}

};  // namespace raft::distance::detail
