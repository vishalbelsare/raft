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

#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/cuda_stream_pool.hpp>

#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <iostream>
#include <raft/core/device_resources.hpp>
#include <raft/distance/distance.cuh>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/init.cuh>
#include <raft/neighbors/detail/faiss_select/DistanceUtils.h>
#include <raft/neighbors/detail/faiss_select/Select.cuh>
#include <raft/neighbors/detail/knn_merge_parts.cuh>
#include <raft/neighbors/detail/selection_faiss.cuh>
#include <raft/spatial/knn/detail/fused_l2_knn.cuh>
#include <raft/spatial/knn/detail/haversine_distance.cuh>
#include <set>
#include <thrust/iterator/transform_iterator.h>

namespace raft::neighbors::detail {
using namespace raft::spatial::knn::detail;
using namespace raft::spatial::knn;

/**
 * Calculates brute force knn, using a fixed memory budget
 * by tiling over both the rows and columns of pairwise_distances
 */
template <typename ElementType = float, typename IndexType = int64_t>
void tiled_brute_force_knn(const raft::device_resources& handle,
                           const ElementType* search,  // size (m ,d)
                           const ElementType* index,   // size (n ,d)
                           size_t m,
                           size_t n,
                           size_t d,
                           int k,
                           ElementType* distances,  // size (m, k)
                           IndexType* indices,      // size (m, k)
                           raft::distance::DistanceType metric,
                           float metric_arg         = 0.0,
                           size_t max_row_tile_size = 0,
                           size_t max_col_tile_size = 0)
{
  // Figure out the number of rows/cols to tile for
  size_t tile_rows   = 0;
  size_t tile_cols   = 0;
  auto stream        = handle.get_stream();
  auto device_memory = handle.get_workspace_resource();
  auto total_mem     = device_memory->get_mem_info(stream).second;
  faiss_select::chooseTileSize(m, n, d, sizeof(ElementType), total_mem, tile_rows, tile_cols);

  // for unittesting, its convenient to be able to put a max size on the tiles
  // so we can test the tiling logic without having to use huge inputs.
  if (max_row_tile_size && (tile_rows > max_row_tile_size)) { tile_rows = max_row_tile_size; }
  if (max_col_tile_size && (tile_cols > max_col_tile_size)) { tile_cols = max_col_tile_size; }

  // tile_cols must be at least k items
  tile_cols = std::max(tile_cols, static_cast<size_t>(k));

  // stores pairwise distances for the current tile
  rmm::device_uvector<ElementType> temp_distances(tile_rows * tile_cols, stream);

  // calculate norms for L2 expanded distances - this lets us avoid calculating
  // norms repeatedly per-tile, and just do once for the entire input
  auto pairwise_metric = metric;
  rmm::device_uvector<ElementType> search_norms(0, stream);
  rmm::device_uvector<ElementType> index_norms(0, stream);
  if (metric == raft::distance::DistanceType::L2Expanded ||
      metric == raft::distance::DistanceType::L2SqrtExpanded) {
    search_norms.resize(m, stream);
    index_norms.resize(n, stream);
    raft::linalg::rowNorm(
      search_norms.data(), search, d, m, raft::linalg::NormType::L2Norm, true, stream);
    raft::linalg::rowNorm(
      index_norms.data(), index, d, n, raft::linalg::NormType::L2Norm, true, stream);
    pairwise_metric = raft::distance::DistanceType::InnerProduct;
  }

  // if we're tiling over columns, we need additional buffers for temporary output
  // distances/indices
  size_t num_col_tiles = raft::ceildiv(n, tile_cols);
  size_t temp_out_cols = k * num_col_tiles;

  // the final column tile could have less than 'k' items in it
  // in which case the number of columns here is too high in the temp output.
  // adjust if necessary
  auto last_col_tile_size = n % tile_cols;
  if (last_col_tile_size && (last_col_tile_size < static_cast<size_t>(k))) {
    temp_out_cols -= k - last_col_tile_size;
  }

  // if we have less than k items in the index, we should fill out the result
  // to indicate that we are missing items (and match behaviour in faiss)
  if (n < static_cast<size_t>(k)) {
    raft::matrix::fill(handle,
                       raft::make_device_matrix_view(distances, m, static_cast<size_t>(k)),
                       std::numeric_limits<ElementType>::lowest());

    if constexpr (std::is_signed_v<IndexType>) {
      raft::matrix::fill(
        handle, raft::make_device_matrix_view(indices, m, static_cast<size_t>(k)), IndexType{-1});
    }
  }

  rmm::device_uvector<ElementType> temp_out_distances(tile_rows * temp_out_cols, stream);
  rmm::device_uvector<IndexType> temp_out_indices(tile_rows * temp_out_cols, stream);

  bool select_min = raft::distance::is_min_close(metric);

  for (size_t i = 0; i < m; i += tile_rows) {
    size_t current_query_size = std::min(tile_rows, m - i);

    for (size_t j = 0; j < n; j += tile_cols) {
      size_t current_centroid_size = std::min(tile_cols, n - j);
      size_t current_k             = std::min(current_centroid_size, static_cast<size_t>(k));

      // calculate the top-k elements for the current tile, by calculating the
      // full pairwise distance for the tile - and then selecting the top-k from that
      // note: we're using a int32 IndexType here on purpose in order to
      // use the pairwise_distance specializations. Since the tile size will ensure
      // that the total memory is < 1GB per tile, this will not cause any issues
      distance::pairwise_distance<ElementType, int>(handle,
                                                    search + i * d,
                                                    index + j * d,
                                                    temp_distances.data(),
                                                    current_query_size,
                                                    current_centroid_size,
                                                    d,
                                                    pairwise_metric,
                                                    true,
                                                    metric_arg);
      if (metric == raft::distance::DistanceType::L2Expanded ||
          metric == raft::distance::DistanceType::L2SqrtExpanded) {
        auto row_norms = search_norms.data() + i;
        auto col_norms = index_norms.data() + j;
        auto dist      = temp_distances.data();

        raft::linalg::map_offset(
          handle,
          raft::make_device_vector_view(dist, current_query_size * current_centroid_size),
          [=] __device__(IndexType i) {
            IndexType row = i / current_centroid_size, col = i % current_centroid_size;

            auto val = row_norms[row] + col_norms[col] - 2.0 * dist[i];

            // due to numerical instability (especially around self-distance)
            // the distances here could be slightly negative, which will
            // cause NaN values in the subsequent sqrt. Clamp to 0
            val = val * (val >= 0.0001);
            if (metric == raft::distance::DistanceType::L2SqrtExpanded) { val = sqrt(val); }
            return val;
          });
      }

      select_k<IndexType, ElementType>(temp_distances.data(),
                                       nullptr,
                                       current_query_size,
                                       current_centroid_size,
                                       distances + i * k,
                                       indices + i * k,
                                       select_min,
                                       current_k,
                                       stream);

      // if we're tiling over columns, we need to do a couple things to fix up
      // the output of select_k
      // 1. The column id's in the output are relative to the tile, so we need
      // to adjust the column ids by adding the column the tile starts at (j)
      // 2. select_k writes out output in a row-major format, which means we
      // can't just concat the output of all the tiles and do a select_k on the
      // concatenation.
      // Fix both of these problems in a single pass here
      if (tile_cols != n) {
        const ElementType* in_distances = distances + i * k;
        const IndexType* in_indices     = indices + i * k;
        ElementType* out_distances      = temp_out_distances.data();
        IndexType* out_indices          = temp_out_indices.data();

        auto count = thrust::make_counting_iterator<IndexType>(0);
        thrust::for_each(handle.get_thrust_policy(),
                         count,
                         count + current_query_size * current_k,
                         [=] __device__(IndexType i) {
                           IndexType row = i / current_k, col = i % current_k;
                           IndexType out_index = row * temp_out_cols + j * k / tile_cols + col;

                           out_distances[out_index] = in_distances[i];
                           out_indices[out_index]   = in_indices[i] + j;
                         });
      }
    }

    if (tile_cols != n) {
      // select the actual top-k items here from the temporary output
      select_k<IndexType, ElementType>(temp_out_distances.data(),
                                       temp_out_indices.data(),
                                       current_query_size,
                                       temp_out_cols,
                                       distances + i * k,
                                       indices + i * k,
                                       select_min,
                                       k,
                                       stream);
    }
  }
}

/**
 * Search the kNN for the k-nearest neighbors of a set of query vectors
 * @param[in] input vector of device device memory array pointers to search
 * @param[in] sizes vector of memory sizes for each device array pointer in input
 * @param[in] D number of cols in input and search_items
 * @param[in] search_items set of vectors to query for neighbors
 * @param[in] n        number of items in search_items
 * @param[out] res_I    pointer to device memory for returning k nearest indices
 * @param[out] res_D    pointer to device memory for returning k nearest distances
 * @param[in] k        number of neighbors to query
 * @param[in] userStream the main cuda stream to use
 * @param[in] internalStreams optional when n_params > 0, the index partitions can be
 *        queried in parallel using these streams. Note that n_int_streams also
 *        has to be > 0 for these to be used and their cardinality does not need
 *        to correspond to n_parts.
 * @param[in] n_int_streams size of internalStreams. When this is <= 0, only the
 *        user stream will be used.
 * @param[in] rowMajorIndex are the index arrays in row-major layout?
 * @param[in] rowMajorQuery are the query array in row-major layout?
 * @param[in] translations translation ids for indices when index rows represent
 *        non-contiguous partitions
 * @param[in] metric corresponds to the raft::distance::DistanceType enum (default is L2Expanded)
 * @param[in] metricArg metric argument to use. Corresponds to the p arg for lp norm
 */
template <typename IntType = int, typename IdxType = std::int64_t, typename value_t = float>
void brute_force_knn_impl(
  raft::device_resources const& handle,
  std::vector<value_t*>& input,
  std::vector<IntType>& sizes,
  IntType D,
  value_t* search_items,
  IntType n,
  IdxType* res_I,
  value_t* res_D,
  IntType k,
  bool rowMajorIndex                  = true,
  bool rowMajorQuery                  = true,
  std::vector<IdxType>* translations  = nullptr,
  raft::distance::DistanceType metric = raft::distance::DistanceType::L2Expanded,
  float metricArg                     = 0)
{
  auto userStream = handle.get_stream();

  ASSERT(input.size() == sizes.size(), "input and sizes vectors should be the same size");

  std::vector<IdxType>* id_ranges;
  if (translations == nullptr) {
    // If we don't have explicit translations
    // for offsets of the indices, build them
    // from the local partitions
    id_ranges       = new std::vector<IdxType>();
    IdxType total_n = 0;
    for (size_t i = 0; i < input.size(); i++) {
      id_ranges->push_back(total_n);
      total_n += sizes[i];
    }
  } else {
    // otherwise, use the given translations
    id_ranges = translations;
  }

  // perform preprocessing
  std::unique_ptr<MetricProcessor<value_t>> query_metric_processor =
    create_processor<value_t>(metric, n, D, k, rowMajorQuery, userStream);
  query_metric_processor->preprocess(search_items);

  std::vector<std::unique_ptr<MetricProcessor<value_t>>> metric_processors(input.size());
  for (size_t i = 0; i < input.size(); i++) {
    metric_processors[i] =
      create_processor<value_t>(metric, sizes[i], D, k, rowMajorQuery, userStream);
    metric_processors[i]->preprocess(input[i]);
  }

  int device;
  RAFT_CUDA_TRY(cudaGetDevice(&device));

  rmm::device_uvector<IdxType> trans(id_ranges->size(), userStream);
  raft::update_device(trans.data(), id_ranges->data(), id_ranges->size(), userStream);

  rmm::device_uvector<value_t> all_D(0, userStream);
  rmm::device_uvector<IdxType> all_I(0, userStream);

  value_t* out_D = res_D;
  IdxType* out_I = res_I;

  if (input.size() > 1) {
    all_D.resize(input.size() * k * n, userStream);
    all_I.resize(input.size() * k * n, userStream);

    out_D = all_D.data();
    out_I = all_I.data();
  }

  // currently we don't support col_major inside tiled_brute_force_knn, because
  // of limitattions of the pairwise_distance API:
  // 1) paiwise_distance takes a single 'isRowMajor' parameter - and we have
  // multiple options here (like rowMajorQuery/rowMajorIndex)
  // 2) because of tiling, we need to be able to set a custom stride in the PW
  // api, which isn't supported
  // Instead, transpose the input matrices if they are passed as col-major.
  auto search = search_items;
  rmm::device_uvector<value_t> search_row_major(0, userStream);
  if (!rowMajorQuery) {
    search_row_major.resize(n * D, userStream);
    raft::linalg::transpose(handle, search, search_row_major.data(), n, D, userStream);
    search = search_row_major.data();
  }

  // transpose into a temporary buffer if necessary
  rmm::device_uvector<value_t> index_row_major(0, userStream);
  if (!rowMajorIndex) {
    size_t total_size = 0;
    for (auto size : sizes) {
      total_size += size;
    }
    index_row_major.resize(total_size * D, userStream);
  }

  // Make other streams from pool wait on main stream
  handle.wait_stream_pool_on_stream();

  size_t total_rows_processed = 0;
  for (size_t i = 0; i < input.size(); i++) {
    value_t* out_d_ptr = out_D + (i * k * n);
    IdxType* out_i_ptr = out_I + (i * k * n);

    auto stream = handle.get_next_usable_stream(i);

    if (k <= 64 && rowMajorQuery == rowMajorIndex && rowMajorQuery == true &&
        (metric == raft::distance::DistanceType::L2Unexpanded ||
         metric == raft::distance::DistanceType::L2SqrtUnexpanded ||
         metric == raft::distance::DistanceType::L2Expanded ||
         metric == raft::distance::DistanceType::L2SqrtExpanded)) {
      fusedL2Knn(D,
                 out_i_ptr,
                 out_d_ptr,
                 input[i],
                 search_items,
                 sizes[i],
                 n,
                 k,
                 rowMajorIndex,
                 rowMajorQuery,
                 stream,
                 metric);

      // Perform necessary post-processing
      if (metric == raft::distance::DistanceType::L2SqrtExpanded ||
          metric == raft::distance::DistanceType::L2SqrtUnexpanded ||
          metric == raft::distance::DistanceType::LpUnexpanded) {
        float p = 0.5;  // standard l2
        if (metric == raft::distance::DistanceType::LpUnexpanded) p = 1.0 / metricArg;
        raft::linalg::unaryOp<float>(
          res_D,
          res_D,
          n * k,
          [p] __device__(float input) { return powf(fabsf(input), p); },
          stream);
      }
    } else {
      switch (metric) {
        case raft::distance::DistanceType::Haversine:
          ASSERT(D == 2,
                 "Haversine distance requires 2 dimensions "
                 "(latitude / longitude).");

          haversine_knn(out_i_ptr, out_d_ptr, input[i], search_items, sizes[i], n, k, stream);
          break;
        default:
          // Create a new handle with the current stream from the stream pool
          raft::device_resources stream_pool_handle(handle);
          raft::resource::set_cuda_stream(stream_pool_handle, stream);

          auto index = input[i];
          if (!rowMajorIndex) {
            index = index_row_major.data() + total_rows_processed * D;
            total_rows_processed += sizes[i];
            raft::linalg::transpose(handle, input[i], index, sizes[i], D, stream);
          }

          // cosine/correlation are handled by metric processor, use IP distance
          // for brute force knn call.
          auto tiled_metric = metric;
          if (metric == raft::distance::DistanceType::CosineExpanded ||
              metric == raft::distance::DistanceType::CorrelationExpanded) {
            tiled_metric = raft::distance::DistanceType::InnerProduct;
          }

          tiled_brute_force_knn<value_t, IdxType>(stream_pool_handle,
                                                  search,
                                                  index,
                                                  n,
                                                  sizes[i],
                                                  D,
                                                  k,
                                                  out_d_ptr,
                                                  out_i_ptr,
                                                  tiled_metric,
                                                  metricArg);
          break;
      }
    }

    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  // Sync internal streams if used. We don't need to
  // sync the user stream because we'll already have
  // fully serial execution.
  handle.sync_stream_pool();

  if (input.size() > 1 || translations != nullptr) {
    // This is necessary for proper index translations. If there are
    // no translations or partitions to combine, it can be skipped.
    knn_merge_parts(out_D, out_I, res_D, res_I, n, input.size(), k, userStream, trans.data());
  }

  query_metric_processor->revert(search_items);
  query_metric_processor->postprocess(out_D);
  for (size_t i = 0; i < input.size(); i++) {
    metric_processors[i]->revert(input[i]);
  }

  if (translations == nullptr) delete id_ranges;
};

}  // namespace raft::neighbors::detail
