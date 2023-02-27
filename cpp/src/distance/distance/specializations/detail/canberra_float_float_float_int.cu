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
#include <raft/distance/detail/distance_ops/all_ops.cuh>

#include <raft/distance/detail/pairwise_matrix/dispatch_arch.cuh>
#include <raft/distance/detail/pairwise_matrix/dispatch_sm60.cuh>
#include <raft/util/arch.cuh>

namespace raft::distance::detail {

template void
pairwise_matrix_arch_dispatch<float, float, float, decltype(raft::identity_op()), int>(
  ops::canberra_distance_op<float, float, int> distance_op,
  pairwise_matrix_dispatch_params<float, float, float, decltype(raft::identity_op()), int> params);

}  // namespace raft::distance::detail
