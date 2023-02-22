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

#include "../ivf_flat_types.hpp"
#include "ann_utils.cuh"

#include <raft/cluster/kmeans_balanced.cuh>
#include <raft/core/device_resources.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/serialize.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/stats/histogram.cuh>
#include <raft/util/pow2_utils.cuh>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/extrema.h>

#include <cstdint>
#include <fstream>

namespace raft::spatial::knn::ivf_flat::detail {

template <typename T, typename IdxT>
__global__ void get_data_ptr_kernel(const uint32_t* list_sizes,
                                    const IdxT* list_offsets,
                                    const T* data,
                                    const IdxT* indices,
                                    uint32_t dim,
                                    uint32_t veclen,
                                    uint32_t n_list,
                                    T** ptrs_to_data)
{
  const IdxT list_id = IdxT(blockDim.x) * IdxT(blockIdx.x) + threadIdx.x;
  if (list_id >= n_list) { return; }
  const IdxT inlist_id     = IdxT(blockDim.y) * IdxT(blockIdx.y) + threadIdx.y;
  const uint32_t list_size = list_sizes[list_id];
  if (inlist_id >= list_size) { return; }

  const IdxT list_offset  = list_offsets[list_id];
  using interleaved_group = Pow2<kIndexGroupSize>;
  auto group_offset       = interleaved_group::roundDown(inlist_id);
  auto ingroup_id         = interleaved_group::mod(inlist_id) * veclen;

  const T* ptr     = data + (list_offset + group_offset) * dim + ingroup_id;
  IdxT id          = indices[list_offset + inlist_id];
  ptrs_to_data[id] = (T*)ptr;
}

template <typename T, typename IdxT>
__global__ void reconstruct_batch_kernel(const IdxT* vector_ids,
                                         const T** ptrs_to_data,
                                         uint32_t dim,
                                         uint32_t veclen,
                                         IdxT n_rows,
                                         T* reconstr)
{
  const IdxT i = IdxT(blockDim.x) * IdxT(blockIdx.x) + threadIdx.x;
  if (i >= n_rows) { return; }

  const IdxT vector_id = vector_ids[i];
  const T* src         = ptrs_to_data[vector_id];
  if (!src) { return; }

  reconstr += i * dim;
  for (uint32_t l = 0; l < dim; l += veclen) {
    for (uint32_t j = 0; j < veclen; j++) {
      reconstr[l + j] = src[l * kIndexGroupSize + j];
    }
  }
}

template <typename T, typename IdxT>
void reconstruct_batch(raft::device_resources const& handle,
                       const index<T, IdxT>& index,
                       const device_mdspan<const IdxT, extent_1d<IdxT>, row_major>& vector_ids,
                       const device_mdspan<T, extent_2d<IdxT>, row_major>& vector_out)
{
  thrust::device_ptr<const IdxT> indices_ptr =
    thrust::device_pointer_cast(index.indices().data_handle());
  IdxT max_indice = *thrust::max_element(
    handle.get_thrust_policy(), indices_ptr, indices_ptr + index.indices().extent(0));

  rmm::device_uvector<T*> ptrs_to_data(max_indice + 1, handle.get_stream());
  RAFT_CUDA_TRY(
    cudaMemsetAsync(ptrs_to_data.data(), 0, ptrs_to_data.size() * sizeof(T*), handle.get_stream()));

  thrust::device_ptr<const uint32_t> list_sizes_ptr =
    thrust::device_pointer_cast(index.list_sizes().data_handle());
  uint32_t max_list_size = *thrust::max_element(
    handle.get_thrust_policy(), list_sizes_ptr, list_sizes_ptr + index.list_sizes().extent(0));

  const dim3 block_dim1(16, 16);
  const dim3 grid_dim1(raft::ceildiv<size_t>(index.n_lists(), block_dim1.x),
                       raft::ceildiv<size_t>(max_list_size, block_dim1.y));
  get_data_ptr_kernel<<<grid_dim1, block_dim1, 0, handle.get_stream()>>>(
    index.list_sizes().data_handle(),
    index.list_offsets().data_handle(),
    index.data().data_handle(),
    index.indices().data_handle(),
    index.dim(),
    index.veclen(),
    index.n_lists(),
    ptrs_to_data.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  auto n_reconstruction = vector_ids.extent(0);
  const dim3 block_dim2(256);
  const dim3 grid_dim2(raft::ceildiv<size_t>(n_reconstruction, block_dim2.x));
  reconstruct_batch_kernel<<<grid_dim2, block_dim2, 0, handle.get_stream()>>>(
    vector_ids.data_handle(),
    (const T**)ptrs_to_data.data(),
    index.dim(),
    index.veclen(),
    n_reconstruction,
    vector_out.data_handle());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
}

}  // namespace raft::spatial::knn::ivf_flat::detail
