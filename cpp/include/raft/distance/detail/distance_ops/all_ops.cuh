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
#include <type_traits>

namespace raft::distance::detail::ops {

// https://en.cppreference.com/w/cpp/types/void_t
// Primary template handles types that do not support CUTLASS
template <typename, typename = void>
struct has_cutlass_op : std::false_type {
};

// Specialization recognizes types that do support CUTLASS
template <typename T>
struct has_cutlass_op<T, std::void_t<decltype(&T::get_cutlass_op)>> : std::true_type {
};

}  // namespace raft::distance::detail::ops
