#!/usr/bin/env python3

# NOTE: this template is not perfectly formatted. Use pre-commit to get
# everything in shape again.
template = """/*
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
distance_matrix_dispatch<OpT,
                         DataT,
                         AccT,
                         OutT,
                         FinopT,
                         IdxT,
                         SM_compat_t>(
    OpT,
    IdxT,
    IdxT,
    IdxT,
    const DataT*,
    const DataT*,
    const DataT*,
    const DataT*,
    OutT*,
    FinopT,
    cudaStream_t ,
    bool,
    SM_compat_t);

}  // namespace raft::distance::detail
"""

data_type_instances = [
    dict(
        DataT="float",
        AccT="float",
        OutT="float",
        IdxT="int",
    ),
    dict(
        DataT="double",
        AccT="double",
        OutT="double",
        IdxT="int",
    ),

]




op_instances = [
    dict(
        path_prefix="canberra",
        OpT="ops::canberra_distance_op<DataT, AccT, IdxT>",
        SM_compat_t="raft::arch::SM_range<raft::arch::SM_min, raft::arch::SM_future>",
    ),
    dict(
        path_prefix="correlation",
        OpT="ops::correlation_distance_op<DataT, AccT, IdxT>",
        SM_compat_t="raft::arch::SM_range<raft::arch::SM_min, raft::arch::SM_future>",
    ),
    dict(
        path_prefix="cosine",
        OpT="ops::cosine_distance_op<DataT, AccT, IdxT>",
        # cosine uses CUTLASS for SM80+
        SM_compat_t="raft::arch::SM_range<raft::arch::SM_min, raft::arch::SM_80>",
    ),
    dict(
        path_prefix="hamming_unexpanded",
        OpT="ops::hamming_distance_op<DataT, AccT, IdxT>",
        SM_compat_t="raft::arch::SM_range<raft::arch::SM_min, raft::arch::SM_future>",
    ),
    dict(
        path_prefix="hellinger_expanded",
        OpT="ops::hellinger_distance_op<DataT, AccT, IdxT>",
        SM_compat_t="raft::arch::SM_range<raft::arch::SM_min, raft::arch::SM_future>",
    ),
    # inner product is handled by cublas.
    dict(
        path_prefix="jensen_shannon",
        OpT="ops::jensen_shannon_distance_op<DataT, AccT, IdxT>",
        SM_compat_t="raft::arch::SM_range<raft::arch::SM_min, raft::arch::SM_future>",
    ),
    dict(
        path_prefix="kl_divergence",
        OpT="ops::kl_divergence_op<DataT, AccT, IdxT>",
        SM_compat_t="raft::arch::SM_range<raft::arch::SM_min, raft::arch::SM_future>",
    ),
    dict(
        path_prefix="l1",
        OpT="ops::l1_distance_op<DataT, AccT, IdxT>",
        SM_compat_t="raft::arch::SM_range<raft::arch::SM_min, raft::arch::SM_future>",
    ),
    dict(
        path_prefix="l2_expanded",
        OpT="ops::l2_exp_distance_op<DataT, AccT, IdxT>",
        # L2 expanded uses CUTLASS for SM80+
        SM_compat_t="raft::arch::SM_range<raft::arch::SM_min, raft::arch::SM_80>",
    ),
    dict(
        path_prefix="l2_unexpanded",
        OpT="ops::l2_unexp_distance_op<DataT, AccT, IdxT>",
        SM_compat_t="raft::arch::SM_range<raft::arch::SM_min, raft::arch::SM_future>",
    ),
    dict(
        path_prefix="l_inf",
        OpT="ops::l_inf_distance_op<DataT, AccT, IdxT>",
        SM_compat_t="raft::arch::SM_range<raft::arch::SM_min, raft::arch::SM_future>",
    ),
    dict(
        path_prefix="lp_unexpanded",
        OpT="ops::lp_unexp_distance_op<DataT, AccT, IdxT>",
        SM_compat_t="raft::arch::SM_range<raft::arch::SM_min, raft::arch::SM_future>",
    ),
    dict(
        path_prefix="russel_rao",
        OpT="ops::russel_rao_distance_op<DataT, AccT, IdxT>",
        SM_compat_t="raft::arch::SM_range<raft::arch::SM_min, raft::arch::SM_future>",
    ),
]

def fill_in(s, template):
    for k, v in template.items():
        s = s.replace(k, v)
    return s

for op_instance in op_instances:
    for data_type_instance in data_type_instances:
        op_data_instance = {
            k : fill_in(v, data_type_instance)
            for k, v in op_instance.items()
        }
        instance = {
            **op_data_instance,
            **data_type_instance,
            "FinopT": "decltype(raft::identity_op())",
        }

        text = fill_in(template, instance)

        path = fill_in("path_prefix_DataT_AccT_OutT_IdxT.cu", instance)
        with open(path, "w") as f:
            f.write(text)
