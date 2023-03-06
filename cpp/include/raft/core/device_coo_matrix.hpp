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

#include <raft/core/coo_matrix.hpp>
#include <raft/core/device_container_policy.hpp>
#include <raft/core/device_span.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/sparse_types.hpp>

namespace raft {

template <typename ElementType,
          typename RowType,
          typename ColType,
          typename NZType,
          template <typename T> typename ContainerPolicy = device_uvector_policy,
          SparsityType sparsity_type                     = SparsityType::OWNING>
using device_coo_matrix =
  coo_matrix<ElementType, RowType, ColType, NZType, true, ContainerPolicy, sparsity_type>;

/**
 * Specialization for a coo matrix view which uses device memory
 */
template <typename ElementType, typename RowType, typename ColType, typename NZType>
using device_coo_matrix_view = coo_matrix_view<ElementType, RowType, ColType, NZType, true>;

/**
 * Specialization for a sparsity-owning coo matrix which uses device memory
 */
template <typename ElementType,
          typename RowType,
          typename ColType,
          typename NZType,
          template <typename T> typename ContainerPolicy = device_uvector_policy>
using device_sparsity_owning_coo_matrix =
  coo_matrix<ElementType, RowType, ColType, NZType, true, ContainerPolicy>;

template <typename ElementType,
          typename RowType,
          typename ColType,
          typename NZType,
          template <typename T> typename ContainerPolicy = device_uvector_policy>
using device_sparsity_preserving_coo_matrix = coo_matrix<ElementType,
                                                         RowType,
                                                         ColType,
                                                         NZType,
                                                         true,
                                                         ContainerPolicy,
                                                         SparsityType::PRESERVING>;

/**
 * Specialization for a sparsity-owning coordinate structure which uses device memory
 */
template <typename RowType,
          typename ColType,
          typename NZType,
          template <typename T> typename ContainerPolicy = device_uvector_policy>
using device_coordinate_structure =
  coordinate_structure<RowType, ColType, NZType, true, ContainerPolicy>;

/**
 * Specialization for a sparsity-preserving coordinate structure view which uses device memory
 */
template <typename RowType, typename ColType, typename NZType>
using device_coordinate_structure_view = coordinate_structure_view<RowType, ColType, NZType, true>;

/**
 * Create a sparsity-owning sparse matrix in the coordinate format. sparsity-owning means that
 * all of the underlying vectors (data, indptr, indices) are owned by the coo_matrix instance. If
 * not known up front, the sparsity can be ignored in this factory function and `resize()` invoked
 * on the instance once the sparsity is known.
 *
 * @code{.cpp}
 * #include <raft/core/device_resources.hpp>
 * #include <raft/core/device_coo_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 *
 * raft::device_resources handle;
 * coo_matrix = raft::make_device_coo_matrix(handle, n_rows, n_cols);
 * ...
 * // compute expected sparsity
 * ...
 * int nnz = 5000;
 * coo_matrix.initialize_sparsity(nnz);
 * @endcode
 *
 * @tparam ElementType
 * @tparam RowType
 * @tparam ColType
 * @tparam NZType
 * @param[in] handle a raft handle for managing expensive device resources
 * @param[in] n_rows total number of rows in the matrix
 * @param[in] n_cols total number of columns in the matrix
 * @param[in] nnz number of non-zeros in the matrix if known [optional]
 * @return a sparsity-owning sparse matrix in coordinate (coo) format
 */
template <typename ElementType, typename RowType, typename ColType, typename NZType>
auto make_device_coo_matrix(raft::resources const& handle,
                            RowType n_rows,
                            ColType n_cols,
                            NZType nnz = 0)
{
  return device_sparsity_owning_coo_matrix<ElementType, RowType, ColType, NZType>(
    handle, n_rows, n_cols, nnz);
}

/**
 * Create a sparsity-preserving sparse matrix in the coordinate format. sparsity-preserving means
 * that a view of the coo sparsity is supplied, allowing the values in the sparsity to change but
 * not the sparsity itself. The csr_matrix instance does not own the sparsity, the sparsity must
 * be known up front, and cannot be resized later.
 *
 * @code{.cpp}
 * #include <raft/core/device_resources.hpp>
 * #include <raft/core/device_coo_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 *
 * raft::device_resources handle;
 * coo_structure = raft::make_device_coordinate_structure(handle, n_rows, n_cols);
 * ...
 * // compute expected sparsity
 * ...
 * coo_structure.initialize_sparsity(nnz);
 * coo_matrix = raft::make_device_coo_matrix(handle, coo_structure.view());
 * @endcode
 *
 * @tparam ElementType
 * @tparam RowType
 * @tparam ColType
 * @tparam NZType
 * @param[in] handle raft handle for managing expensive device resources
 * @param[in] structure_ a sparsity-preserving coordinate structural view
 * @return a sparsity-preserving sparse matrix in coordinate (coo) format
 */
template <typename ElementType, typename RowType, typename ColType, typename NZType>
auto make_device_coo_matrix(raft::resources const& handle,
                            device_coordinate_structure_view<RowType, ColType, NZType> structure_)
{
  return device_sparsity_preserving_coo_matrix<ElementType, RowType, ColType, NZType>(
    handle, std::make_shared(structure_));
}

/**
 * Create a non-owning sparse matrix view in the coordinate format. This is sparsity-preserving,
 * meaning that the underlying sparsity is known and cannot be changed. Use the sparsity-owning
 * coo_matrix if sparsity needs to be mutable.
 *
 * @code{.cpp}
 * #include <raft/core/device_resources.hpp>
 * #include <raft/core/device_coo_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 * int nnz = 5000;
 *
 * // The following pointer is assumed to reference device memory for a size of nnz
 * float* d_elm_ptr = ...;
 *
 * raft::device_resources handle;
 * coo_structure = raft::make_device_coordinate_structure(handle, n_rows, n_cols, nnz);
 * coo_matrix_view = raft::make_device_coo_matrix_view(handle, d_elm_ptr, coo_structure.view());
 * @endcode
 *
 * @tparam ElementType
 * @tparam RowType
 * @tparam ColType
 * @tparam NZType
 * @param[in] ptr a pointer to array of nonzero matrix elements on device (size nnz)
 * @param[in] structure_ a sparsity-preserving coordinate structural view
 * @return a sparsity-preserving sparse matrix in coordinate (coo) format
 */
template <typename ElementType, typename RowType, typename ColType, typename NZType>
auto make_device_coo_matrix_view(
  ElementType* ptr, device_coordinate_structure_view<RowType, ColType, NZType> structure_)
{
  return device_coo_matrix_view<ElementType, RowType, ColType, NZType>(
    raft::device_span<ElementType>(ptr, structure_.get_nnz()), std::make_shared(structure_));
}

/**
 * Create a non-owning sparse matrix view in the coordinate format. This is sparsity-preserving,
 * meaning that the underlying sparsity is known and cannot be changed. Use the sparsity-owning
 * coo_matrix if sparsity needs to be mutable.
 *
 * @code{.cpp}
 * #include <raft/core/device_resources.hpp>
 * #include <raft/core/device_span.hpp>
 * #include <raft/core/device_coo_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 * int nnz = 5000;
 *
 * // The following span is assumed to be of size nnz
 * raft::device_span<float> d_elm_ptr;
 *
 * raft::device_resources handle;
 * coo_structure = raft::make_device_coordinate_structure(handle, n_rows, n_cols, nnz);
 * coo_matrix_view = raft::make_device_coo_matrix_view(handle, d_elm_ptr, coo_structure.view());
 * @endcode
 *
 * @tparam ElementType
 * @tparam RowType
 * @tparam ColType
 * @tparam NZType
 * @param[in] elements a device span containing nonzero matrix elements (size nnz)
 * @param[in] structure_ a sparsity-preserving coordinate structural view
 * @return
 */
template <typename ElementType, typename RowType, typename ColType, typename NZType>
auto make_device_coo_matrix_view(
  raft::device_span<ElementType> elements,
  device_coordinate_structure_view<RowType, ColType, NZType> structure_)
{
  RAFT_EXPECTS(elements.size() == structure_.get_nnz(),
               "Size of elements must be equal to the nnz from the structure");
  return device_coo_matrix_view<ElementType, RowType, ColType, NZType>(
    elements, std::make_shared(structure_));
}

/**
 * Create a sparsity-owning coordinate structure object. If not known up front, this object can be
 * resized() once the sparsity (number of non-zeros) is known, postponing the allocation of the
 * underlying data arrays.
 *
 * @code{.cpp}
 * #include <raft/core/device_resources.hpp>
 * #include <raft/core/device_coo_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 * int nnz = 5000;
 *
 * raft::device_resources handle;
 * coo_structure = raft::make_device_coordinate_structure(handle, n_rows, n_cols, nnz);
 *  * ...
 * // compute expected sparsity
 * ...
 * coo_structure.initialize_sparsity(nnz);
 * @endcode
 *
 * @tparam RowType
 * @tparam ColType
 * @tparam NZType
 * @param[in] handle raft handle for managing expensive resources on device
 * @param[in] n_rows total number of rows
 * @param[in] n_cols total number of cols
 * @param[in] nnz number of non-zeros
 * @return a sparsity-owning coordinate structure instance
 */
template <typename RowType, typename ColType, typename NZType>
auto make_device_coordinate_structure(raft::resources const& handle,
                                      RowType n_rows,
                                      ColType n_cols,
                                      NZType nnz = 0)
{
  return device_coordinate_structure<RowType, ColType, NZType>(handle, n_rows, n_cols, nnz);
}

/**
 * Create a non-owning sparsity-preserved coordinate structure view. Sparsity-preserving means that
 * the underlying sparsity is known and cannot be changed. Use the sparsity-owning version if the
 * sparsity is not known up front.
 *
 * @code{.cpp}
 * #include <raft/core/device_resources.hpp>
 * #include <raft/core/device_coo_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 * int nnz = 5000;
 *
 * // The following pointers are assumed to reference device memory of size nnz
 * int *rows = ...;
 * int *cols = ...;
 *
 * raft::device_resources handle;
 * coo_structure = raft::make_device_coordinate_structure_view(handle, rows, cols, n_rows, n_cols,
 * nnz);
 * @endcode
 *
 * @tparam RowType
 * @tparam ColType
 * @tparam NZType
 * @param[in] rows pointer to row indices array on device (size nnz)
 * @param[in] cols pointer to column indices array on device (size nnz)
 * @param[in] n_rows total number of rows
 * @param[in] n_cols total number of columns
 * @param[in] nnz number of non-zeros
 * @return a sparsity-preserving coordinate structural view
 */
template <typename RowType, typename ColType, typename NZType>
auto make_device_coo_structure_view(
  RowType* rows, ColType* cols, RowType n_rows, ColType n_cols, NZType nnz)
{
  return device_coordinate_structure_view<RowType, ColType, NZType>(
    raft::device_span<RowType>(rows, nnz), raft::device_span<ColType>(cols, nnz), n_rows, n_cols);
}

/**
 * Create a non-owning sparsity-preserved coordinate structure view. Sparsity-preserving means that
 * the underlying sparsity is known and cannot be changed. Use the sparsity-owning version if the
 * sparsity is not known up front.
 *
 * @code{.cpp}
 * #include <raft/core/device_resources.hpp>
 * #include <raft/core/device_coo_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 * int nnz = 5000;
 *
 * // The following device spans are assumed to be of size nnz
 * raft::device_span<int> rows;
 * raft::device_span<int> cols;
 *
 * raft::device_resources handle;
 * coo_structure = raft::make_device_coordinate_structure_view(handle, rows, cols, n_rows, n_cols);
 * @endcode
 *
 * @tparam RowType
 * @tparam ColType
 * @tparam NZType
 * @param[in] rows a device span containing row indices (size nnz)
 * @param[in] cols a device span containing column indices (size nnz)
 * @param[in] n_rows total number of rows
 * @param[in] n_cols total number of columns
 * @return a sparsity-preserving coordinate structural view
 */
template <typename RowType, typename ColType, typename NZType>
auto make_device_coo_structure_view(raft::device_span<RowType> rows,
                                    raft::device_span<ColType> cols,
                                    RowType n_rows,
                                    ColType n_cols)
{
  return device_coordinate_structure_view<RowType, ColType, NZType>(rows, cols, n_rows, n_cols);
}

};  // namespace raft