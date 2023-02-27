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

#include <raft/distance/detail/pairwise_distance_cutlass_base.cuh>
#include <raft/distance/detail/pairwise_matrix/dispatch_arch.cuh>
#include <raft/distance/detail/pairwise_matrix/dispatch_common.cuh>

namespace raft::distance::detail {

template <typename OpT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename FinOpT,
          typename IdxT,
          typename SM_compat_t>
void pairwise_matrix_sm80_dispatch(
  OpT distance_op,
  SM_compat_t,
  pairwise_matrix_dispatch_params<DataT, AccT, OutT, FinOpT, IdxT> params)
{
  auto cutlass_op = distance_op.get_cutlass_op();

  // Determine leading dimensions and possibly flip order of passing x and y if
  // column_major.
  IdxT ldx, ldy, ld_out;
  if (params.is_row_major) {
    ldx = params.k, ldy = params.k, ld_out = params.n;
  } else {
    // Flip x, y, and m, n.
    std::swap<const DataT*>(params.x, params.y);
    std::swap<const DataT*>(params.x_norm, params.y_norm);
    std::swap(params.m, params.n);
    ldx = params.m, ldy = params.n, ld_out = params.n;
  }
  size_t align_x        = alignment_of_2d_array(params.x, ldx);
  size_t align_y        = alignment_of_2d_array(params.y, ldy);
  size_t byte_alignment = min(align_x, align_y);

  // Since alignment is in bytes, it could be smaller than sizeof(DataT).
  // Handle this (unlikely) case here.
  RAFT_EXPECTS(sizeof(DataT) <= byte_alignment,
               "Input matrix must be aligned to size of elements.");

  // Compute number of elements that can be loaded in one instruction
  // without causing misalignent errors.
  int vec_len_aligned = (byte_alignment % sizeof(DataT) == 0) ? byte_alignment / sizeof(DataT) : 1;

  dispatch_common(params.is_row_major, vec_len_aligned, [&](auto row_major, auto vec_len_aligned) {
    // row_major and vec_len are std::integral_constants of type bool and int
    // respectively.

    // Prevent double, vec_len=4 combination (this is not supported)
    constexpr int vec_len = std::min(vec_len_aligned(), static_cast<int>(16 / sizeof(DataT)));

    cutlassDistanceKernel<DataT,
                          AccT,
                          OutT,
                          IdxT,
                          vec_len,
                          FinOpT,
                          decltype(cutlass_op),
                          row_major()>(params.x,
                                       params.y,
                                       params.x_norm,
                                       params.y_norm,
                                       params.m,
                                       params.n,
                                       params.k,
                                       ldx,
                                       ldy,
                                       ld_out,
                                       params.out,
                                       params.fin_op,
                                       cutlass_op,
                                       params.stream);
  });
}

}  // namespace raft::distance::detail
