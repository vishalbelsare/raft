/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include "../test_utils.cuh"
#include <gtest/gtest.h>

#include <raft/core/device_mdspan.hpp>
#include <raft/distance/distance.cuh>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/brute_force.cuh>
#include <raft/random/rng.cuh>
#include <raft/spatial/knn/knn.cuh>

#include <rmm/device_buffer.hpp>

namespace raft::spatial::knn {
template <typename IdxT, typename DistT, typename compareDist>
struct idx_dist_pair {
  IdxT idx;
  DistT dist;
  compareDist eq_compare;
  bool operator==(const idx_dist_pair<IdxT, DistT, compareDist>& a) const
  {
    if (idx == a.idx) return true;
    if (eq_compare(dist, a.dist)) return true;
    return false;
  }
  idx_dist_pair(IdxT x, DistT y, compareDist op) : idx(x), dist(y), eq_compare(op) {}
};

template <typename T, typename DistT>
testing::AssertionResult devArrMatchKnnPair(const T* expected_idx,
                                            const T* actual_idx,
                                            const DistT* expected_dist,
                                            const DistT* actual_dist,
                                            size_t rows,
                                            size_t cols,
                                            const DistT eps,
                                            cudaStream_t stream = 0)
{
  size_t size = rows * cols;
  std::unique_ptr<T[]> exp_idx_h(new T[size]);
  std::unique_ptr<T[]> act_idx_h(new T[size]);
  std::unique_ptr<DistT[]> exp_dist_h(new DistT[size]);
  std::unique_ptr<DistT[]> act_dist_h(new DistT[size]);
  raft::update_host<T>(exp_idx_h.get(), expected_idx, size, stream);
  raft::update_host<T>(act_idx_h.get(), actual_idx, size, stream);
  raft::update_host<DistT>(exp_dist_h.get(), expected_dist, size, stream);
  raft::update_host<DistT>(act_dist_h.get(), actual_dist, size, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  for (size_t i(0); i < rows; ++i) {
    for (size_t j(0); j < cols; ++j) {
      auto idx      = i * cols + j;  // row major assumption!
      auto exp_idx  = exp_idx_h.get()[idx];
      auto act_idx  = act_idx_h.get()[idx];
      auto exp_dist = exp_dist_h.get()[idx];
      auto act_dist = act_dist_h.get()[idx];
      idx_dist_pair exp_kvp(exp_idx, exp_dist, raft::CompareApprox<DistT>(eps));
      idx_dist_pair act_kvp(act_idx, act_dist, raft::CompareApprox<DistT>(eps));
      if (!(exp_kvp == act_kvp)) {
        return testing::AssertionFailure()
               << "actual=" << act_kvp.idx << "," << act_kvp.dist << "!="
               << "expected" << exp_kvp.idx << "," << exp_kvp.dist << " @" << i << "," << j;
      }
    }
  }
  return testing::AssertionSuccess();
}
}  // namespace raft::spatial::knn