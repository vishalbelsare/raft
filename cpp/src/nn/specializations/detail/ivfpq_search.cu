/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <raft/neighbors/ivf_pq.cuh>
#include <raft/neighbors/specializations/detail/ivf_pq_search.cuh>
#include <raft/neighbors/specializations/ivf_pq_specialization.hpp>

namespace raft::neighbors::ivf_pq {

#define RAFT_SEARCH_INST(T, IdxT)                                                          \
  void search(const handle_t& handle,                                                      \
              const search_params& params,                                                 \
              const index<IdxT>& idx,                                                      \
              const T* queries,                                                            \
              uint32_t n_queries,                                                          \
              uint32_t k,                                                                  \
              IdxT* neighbors,                                                             \
              float* distances,                                                            \
              rmm::mr::device_memory_resource* mr)                                         \
  {                                                                                        \
    search<T, IdxT>(handle, params, idx, queries, n_queries, k, neighbors, distances, mr); \
  }

RAFT_SEARCH_INST(float, uint64_t);
RAFT_SEARCH_INST(int8_t, uint64_t);
RAFT_SEARCH_INST(uint8_t, uint64_t);

#undef RAFT_INST_SEARCH

}  // namespace raft::neighbors::ivf_pq
