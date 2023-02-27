/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <raft/core/operators.hpp>
#include <raft/distance/detail/distance_ops/canberra.cuh>
#include <raft/distance/detail/distance_ops/correlation.cuh>
#include <raft/distance/detail/distance_ops/cosine.cuh>
#include <raft/distance/detail/distance_ops/hamming.cuh>
#include <raft/distance/detail/distance_ops/hellinger.cuh>
#include <raft/distance/detail/distance_ops/jensen_shannon.cuh>
#include <raft/distance/detail/distance_ops/kl_divergence.cuh>
#include <raft/distance/detail/distance_ops/l1.cuh>
#include <raft/distance/detail/distance_ops/l2_exp.cuh>
#include <raft/distance/detail/distance_ops/l2_unexp.cuh>
#include <raft/distance/detail/distance_ops/l_inf.cuh>
#include <raft/distance/detail/distance_ops/lp_unexp.cuh>
#include <raft/distance/detail/distance_ops/russel_rao.cuh>

#include <raft/distance/detail/pairwise_matrix/dispatch_sm60.cuh>
#include <raft/util/arch.cuh>

namespace raft::distance::detail {

template void
distance_matrix_dispatch<ops::lp_unexp_distance_op<float, float, int>,
                         float,
                         float,
                         float,
                         decltype(raft::identity_op()),
                         int,
                         raft::arch::SM_range<raft::arch::SM_min, raft::arch::SM_future>>(
  ops::lp_unexp_distance_op<float, float, int>,
  int,
  int,
  int,
  const float*,
  const float*,
  const float*,
  const float*,
  float*,
  decltype(raft::identity_op()),
  cudaStream_t,
  bool,
  raft::arch::SM_range<raft::arch::SM_min, raft::arch::SM_future>);

}  // namespace raft::distance::detail
