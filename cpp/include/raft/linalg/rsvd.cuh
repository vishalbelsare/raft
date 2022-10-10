/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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
#ifndef __RSVD_H
#define __RSVD_H

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/linalg/detail/rsvd.cuh>

namespace raft {
namespace linalg {

/**
 * @brief randomized singular value decomposition (RSVD)
 * @param handle:   raft handle
 * @param in:       input matrix
 *                  [dim = n_rows * n_cols] 
 * @param n_rows:   number rows of input matrix
 * @param n_cols:   number columns of input matrix
 * @param k:        Rank of the k-SVD decomposition of matrix in. Number of singular values to be computed.
 *                  The rank is less than min(m,n). 
 * @param p:        Oversampling. The size of the subspace will be (k + p). (k+p) is less than min(m,n).
 *                  (Recommanded to be at least 2*k)
 * @param niters:   Number of iteration of power method.
 * @param S:        array of singular values of input matrix.
 *                  [dim = min(n_rows, n_cols)] 
 * @param U:        left singular values of input matrix.
 *                  [dim = n_rows * n_rows] if gen_U
 *                  [dim = min(n_rows,n_cols) * n_rows] else
 * @param V:        right singular values of input matrix.
 *                  [dim = n_cols * n_cols] if gen_V
 *                  [dim = min(n_rows,n_cols) * n_cols] else
 * @param trans_V:  Transpose V back ?
 * @param gen_U:    left vector needs to be generated or not?
 * @param gen_V:    right vector needs to be generated or not?
 * @param rowMajor: Is the data row major?
 */
template <typename math_t>
void randomizedSVD(const raft::handle_t& handle,
                   math_t* in,
                   std::size_t n_rows,
                   std::size_t n_cols,
                   std::size_t k,
                   std::size_t p,
                   std::size_t niters,
                   math_t* S,
                   math_t* U,
                   math_t* V,
                   bool trans_V,
                   bool gen_U,
                   bool gen_V,
                   bool rowMajor=false)
{
  detail::randomizedSVD<math_t>(handle, in, n_rows, n_cols, k, p, niters, S, U,
    V, trans_V, gen_U, gen_V);
}


/**
 * @brief randomized singular value decomposition (RSVD)
 * @param handle:  raft handle
 * @param in:      input matrix
 *                 [dim = n_rows * n_cols] 
 * @param n_rows:  number rows of input matrix
 * @param n_cols:  number columns of input matrix
 * @param k:       Rank of the k-SVD decomposition of matrix in. Number of singular values to be computed.
 *                 The rank is less than min(m,n). 
 * @param p:       Oversampling. The size of the subspace will be (k + p). (k+p) is less than min(m,n).
 *                 (Recommanded to be at least 2*k)
 * @param niters:  Number of iteration of power method. (2 is recommanded)
 * @param S:       array of singular values of input matrix.
 *                 [dim = min(n_rows, n_cols)] 
 * @param U:       left singular values of input matrix.
 *                 [dim = n_rows * n_rows] if gen_U
 *                 [dim = min(n_rows,n_cols) * n_rows] else
 * @param V:       right singular values of input matrix.
 *                 [dim = n_cols * n_cols] if gen_V
 *                 [dim = min(n_rows,n_cols) * n_cols] else
 * @param trans_V: Transpose V back ?
 * @param gen_U:   left vector needs to be generated or not?
 * @param gen_V:   right vector needs to be generated or not?
 */
template <typename math_t, typename IdxType, typename LayoutPolicy, typename AccessorPolicy>
void randomizedSVD(const raft::handle_t& handle,
                   raft::mdspan<const math_t, raft::matrix_extent<IdxType>, LayoutPolicy, AccessorPolicy> in,
                   std::size_t k,
                   std::size_t p,
                   std::size_t niters,
                   raft::mdspan<math_t, raft::vector_extent<IdxType>> S,
                   raft::mdspan<math_t, raft::matrix_extent<IdxType>> U,
                   raft::mdspan<math_t, raft::matrix_extent<IdxType>> V,
                   bool trans_V,
                   bool gen_U,
                   bool gen_V)
{
  detail::randomizedSVD<math_t>(handle, in, in.extent(0), in.extent(1), k, p, niters, S.data(), U.data(),
    V.data(), trans_V, gen_U, gen_V, std::is_same_v<LayoutPolicy, raft::row_major>);
}

/**
 * @brief randomized singular value decomposition (RSVD) on the column major
 * float type input matrix (Jacobi-based), by specifying no. of PCs and
 * upsamples directly
 * @param handle: raft handle
 * @param M: input matrix
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param S_vec: singular values of input matrix
 * @param U: left singular values of input matrix
 * @param V: right singular values of input matrix
 * @param k: no. of singular values to be computed
 * @param p: no. of upsamples
 * @param use_bbt: whether use eigen decomposition in computation or not
 * @param gen_left_vec: left vector needs to be generated or not?
 * @param gen_right_vec: right vector needs to be generated or not?
 * @param use_jacobi: whether to jacobi solver for decomposition
 * @param tol: tolerance for Jacobi-based solvers
 * @param max_sweeps: maximum number of sweeps for Jacobi-based solvers
 * @param stream cuda stream
 */
template <typename math_t>
void rsvdFixedRank(const raft::handle_t& handle,
                   math_t* M,
                   int n_rows,
                   int n_cols,
                   math_t* S_vec,
                   math_t* U,
                   math_t* V,
                   int k,
                   int p,
                   bool use_bbt,
                   bool gen_left_vec,
                   bool gen_right_vec,
                   bool use_jacobi,
                   math_t tol,
                   int max_sweeps,
                   cudaStream_t stream)
{
  detail::rsvdFixedRank(handle,
                        M,
                        n_rows,
                        n_cols,
                        S_vec,
                        U,
                        V,
                        k,
                        p,
                        use_bbt,
                        gen_left_vec,
                        gen_right_vec,
                        use_jacobi,
                        tol,
                        max_sweeps,
                        stream);
}

/**
 * @brief randomized singular value decomposition (RSVD) on the column major
 * float type input matrix (Jacobi-based), by specifying the PC and upsampling
 * ratio
 * @param handle: raft handle
 * @param M: input matrix
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param S_vec: singular values of input matrix
 * @param U: left singular values of input matrix
 * @param V: right singular values of input matrix
 * @param PC_perc: percentage of singular values to be computed
 * @param UpS_perc: upsampling percentage
 * @param use_bbt: whether use eigen decomposition in computation or not
 * @param gen_left_vec: left vector needs to be generated or not?
 * @param gen_right_vec: right vector needs to be generated or not?
 * @param use_jacobi: whether to jacobi solver for decomposition
 * @param tol: tolerance for Jacobi-based solvers
 * @param max_sweeps: maximum number of sweeps for Jacobi-based solvers
 * @param stream cuda stream
 */
template <typename math_t>
void rsvdPerc(const raft::handle_t& handle,
              math_t* M,
              int n_rows,
              int n_cols,
              math_t* S_vec,
              math_t* U,
              math_t* V,
              math_t PC_perc,
              math_t UpS_perc,
              bool use_bbt,
              bool gen_left_vec,
              bool gen_right_vec,
              bool use_jacobi,
              math_t tol,
              int max_sweeps,
              cudaStream_t stream)
{
  detail::rsvdPerc(handle,
                   M,
                   n_rows,
                   n_cols,
                   S_vec,
                   U,
                   V,
                   PC_perc,
                   UpS_perc,
                   use_bbt,
                   gen_left_vec,
                   gen_right_vec,
                   use_jacobi,
                   tol,
                   max_sweeps,
                   stream);
}

/**
 * @defgroup rsvd Randomized Singular Value Decomposition
 * @{
 */

/**
 * @brief randomized singular value decomposition (RSVD) on a column major
 * rectangular matrix using QR decomposition, by specifying no. of PCs and
 * upsamples directly
 * @param[in] handle raft::handle_t
 * @param[in] M input raft::device_matrix_view with layout raft::col_major of shape (M, N)
 * @param[out] S_vec singular values raft::device_vector_view of shape (K)
 * @param[in] p no. of upsamples
 * @param[out] U optional left singular values of raft::device_matrix_view with layout
 * raft::col_major
 * @param[out] V optional right singular values of raft::device_matrix_view with layout
 * raft::col_major
 */
template <typename ValueType, typename IndexType>
void rsvd_fixed_rank(
  const raft::handle_t& handle,
  raft::device_matrix_view<const ValueType, IndexType, raft::col_major> M,
  raft::device_vector_view<ValueType, IndexType> S_vec,
  IndexType p,
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U = std::nullopt,
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V = std::nullopt)
{
  if (U) {
    RAFT_EXPECTS(M.extent(0) == U.value().extent(0), "Number of rows in M should be equal to U");
    RAFT_EXPECTS(S_vec.extent(0) == U.value().extent(1),
                 "Number of columns in U should be equal to length of S");
  }
  if (V) {
    RAFT_EXPECTS(M.extent(1) == V.value().extent(1), "Number of columns in M should be equal to V");
    RAFT_EXPECTS(S_vec.extent(0) == V.value().extent(0),
                 "Number of rows in V should be equal to length of S");
  }

  rsvdFixedRank(handle,
                const_cast<ValueType*>(M.data_handle()),
                M.extent(0),
                M.extent(1),
                S_vec.data_handle(),
                U.value().data_handle(),
                V.value().data_handle(),
                S_vec.extent(0),
                p,
                false,
                U.has_value(),
                V.has_value(),
                false,
                static_cast<ValueType>(0),
                0,
                handle.get_stream());
}

/**
 * @brief Overload of `rsvd_fixed_rank` to help the
 *   compiler find the above overload, in case users pass in
 *   `std::nullopt` for one or both of the optional arguments.
 *
 * Please see above for documentation of `rsvd_fixed_rank`.
 */
template <typename ValueType, typename IndexType, typename UType, typename VType>
void rsvd_fixed_rank(const raft::handle_t& handle,
                     raft::device_matrix_view<const ValueType, IndexType, raft::col_major> M,
                     raft::device_vector_view<ValueType, IndexType> S_vec,
                     IndexType p,
                     ValueType tol,
                     int max_sweeps,
                     UType&& U,
                     VType&& V)
{
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U_optional =
    std::forward<UType>(U);
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V_optional =
    std::forward<VType>(V);

  rsvd_fixed_rank(handle, M, S_vec, p, tol, max_sweeps, U_optional, V_optional);
}

/**
 * @brief randomized singular value decomposition (RSVD) on a column major
 * rectangular matrix using symmetric Eigen decomposition, by specifying no. of PCs and
 * upsamples directly. The rectangular input matrix is made square and symmetric using B @ B^T
 * @param[in] handle raft::handle_t
 * @param[in] M input raft::device_matrix_view with layout raft::col_major of shape (M, N)
 * @param[out] S_vec singular values raft::device_vector_view of shape (K)
 * @param[in] p no. of upsamples
 * @param[out] U optional left singular values of raft::device_matrix_view with layout
 * raft::col_major
 * @param[out] V optional right singular values of raft::device_matrix_view with layout
 * raft::col_major
 */
template <typename ValueType, typename IndexType>
void rsvd_fixed_rank_symmetric(
  const raft::handle_t& handle,
  raft::device_matrix_view<const ValueType, IndexType, raft::col_major> M,
  raft::device_vector_view<ValueType, IndexType> S_vec,
  IndexType p,
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U = std::nullopt,
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V = std::nullopt)
{
  if (U) {
    RAFT_EXPECTS(M.extent(0) == U.value().extent(0), "Number of rows in M should be equal to U");
    RAFT_EXPECTS(S_vec.extent(0) == U.value().extent(1),
                 "Number of columns in U should be equal to length of S");
  }
  if (V) {
    RAFT_EXPECTS(M.extent(1) == V.value().extent(1), "Number of columns in M should be equal to V");
    RAFT_EXPECTS(S_vec.extent(0) == V.value().extent(0),
                 "Number of rows in V should be equal to length of S");
  }

  rsvdFixedRank(handle,
                const_cast<ValueType*>(M.data_handle()),
                M.extent(0),
                M.extent(1),
                S_vec.data_handle(),
                U.value().data_handle(),
                V.value().data_handle(),
                S_vec.extent(0),
                p,
                true,
                U.has_value(),
                V.has_value(),
                false,
                static_cast<ValueType>(0),
                0,
                handle.get_stream());
}

/**
 * @brief Overload of `rsvd_fixed_rank_symmetric` to help the
 *   compiler find the above overload, in case users pass in
 *   `std::nullopt` for one or both of the optional arguments.
 *
 * Please see above for documentation of `rsvd_fixed_rank_symmetric`.
 */
template <typename ValueType, typename IndexType, typename UType, typename VType>
void rsvd_fixed_rank_symmetric(
  const raft::handle_t& handle,
  raft::device_matrix_view<const ValueType, IndexType, raft::col_major> M,
  raft::device_vector_view<ValueType, IndexType> S_vec,
  IndexType p,
  ValueType tol,
  int max_sweeps,
  UType&& U,
  VType&& V)
{
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U_optional =
    std::forward<UType>(U);
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V_optional =
    std::forward<VType>(V);

  rsvd_fixed_rank_symmetric(handle, M, S_vec, p, tol, max_sweeps, U_optional, V_optional);
}

/**
 * @brief randomized singular value decomposition (RSVD) on a column major
 * rectangular matrix using Jacobi method, by specifying no. of PCs and
 * upsamples directly
 * @param[in] handle raft::handle_t
 * @param[in] M input raft::device_matrix_view with layout raft::col_major of shape (M, N)
 * @param[out] S_vec singular values raft::device_vector_view of shape (K)
 * @param[in] p no. of upsamples
 * @param[in] tol tolerance for Jacobi-based solvers
 * @param[in] max_sweeps maximum number of sweeps for Jacobi-based solvers
 * @param[out] U optional left singular values of raft::device_matrix_view with layout
 * raft::col_major
 * @param[out] V optional right singular values of raft::device_matrix_view with layout
 * raft::col_major
 */
template <typename ValueType, typename IndexType>
void rsvd_fixed_rank_jacobi(
  const raft::handle_t& handle,
  raft::device_matrix_view<const ValueType, IndexType, raft::col_major> M,
  raft::device_vector_view<ValueType, IndexType> S_vec,
  IndexType p,
  ValueType tol,
  int max_sweeps,
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U = std::nullopt,
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V = std::nullopt)
{
  if (U) {
    RAFT_EXPECTS(M.extent(0) == U.value().extent(0), "Number of rows in M should be equal to U");
    RAFT_EXPECTS(S_vec.extent(0) == U.value().extent(1),
                 "Number of columns in U should be equal to length of S");
  }
  if (V) {
    RAFT_EXPECTS(M.extent(1) == V.value().extent(1), "Number of columns in M should be equal to V");
    RAFT_EXPECTS(S_vec.extent(0) == V.value().extent(0),
                 "Number of rows in V should be equal to length of S");
  }

  rsvdFixedRank(handle,
                const_cast<ValueType*>(M.data_handle()),
                M.extent(0),
                M.extent(1),
                S_vec.data_handle(),
                U.value().data_handle(),
                V.value().data_handle(),
                S_vec.extent(0),
                p,
                false,
                U.has_value(),
                V.has_value(),
                true,
                tol,
                max_sweeps,
                handle.get_stream());
}

/**
 * @brief Overload of `rsvd_fixed_rank_jacobi` to help the
 *   compiler find the above overload, in case users pass in
 *   `std::nullopt` for one or both of the optional arguments.
 *
 * Please see above for documentation of `rsvd_fixed_rank_jacobi`.
 */
template <typename ValueType, typename IndexType, typename UType, typename VType>
void rsvd_fixed_rank_jacobi(const raft::handle_t& handle,
                            raft::device_matrix_view<const ValueType, IndexType, raft::col_major> M,
                            raft::device_vector_view<ValueType, IndexType> S_vec,
                            IndexType p,
                            ValueType tol,
                            int max_sweeps,
                            UType&& U,
                            VType&& V)
{
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U_optional =
    std::forward<UType>(U);
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V_optional =
    std::forward<VType>(V);

  rsvd_fixed_rank_sjacobi(handle, M, S_vec, p, tol, max_sweeps, U_optional, V_optional);
}

/**
 * @brief randomized singular value decomposition (RSVD) on a column major
 * rectangular matrix using Jacobi method, by specifying no. of PCs and
 * upsamples directly. The rectangular input matrix is made square and symmetric using B @ B^T
 * @param[in] handle raft::handle_t
 * @param[in] M input raft::device_matrix_view with layout raft::col_major of shape (M, N)
 * @param[out] S_vec singular values raft::device_vector_view of shape (K)
 * @param[in] p no. of upsamples
 * @param[in] tol tolerance for Jacobi-based solvers
 * @param[in] max_sweeps maximum number of sweeps for Jacobi-based solvers
 * @param[out] U optional left singular values of raft::device_matrix_view with layout
 * raft::col_major
 * @param[out] V optional right singular values of raft::device_matrix_view with layout
 * raft::col_major
 */
template <typename ValueType, typename IndexType>
void rsvd_fixed_rank_symmetric_jacobi(
  const raft::handle_t& handle,
  raft::device_matrix_view<const ValueType, IndexType, raft::col_major> M,
  raft::device_vector_view<ValueType, IndexType> S_vec,
  IndexType p,
  ValueType tol,
  int max_sweeps,
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U = std::nullopt,
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V = std::nullopt)
{
  if (U) {
    RAFT_EXPECTS(M.extent(0) == U.value().extent(0), "Number of rows in M should be equal to U");
    RAFT_EXPECTS(S_vec.extent(0) == U.value().extent(1),
                 "Number of columns in U should be equal to length of S");
  }
  if (V) {
    RAFT_EXPECTS(M.extent(1) == V.value().extent(1), "Number of columns in M should be equal to V");
    RAFT_EXPECTS(S_vec.extent(0) == V.value().extent(0),
                 "Number of rows in V should be equal to length of S");
  }

  rsvdFixedRank(handle,
                const_cast<ValueType*>(M.data_handle()),
                M.extent(0),
                M.extent(1),
                S_vec.data_handle(),
                U.value().data_handle(),
                V.value().data_handle(),
                S_vec.extent(0),
                p,
                true,
                U.has_value(),
                V.has_value(),
                true,
                tol,
                max_sweeps,
                handle.get_stream());
}

/**
 * @brief Overload of `rsvd_fixed_rank_symmetric_jacobi` to help the
 *   compiler find the above overload, in case users pass in
 *   `std::nullopt` for one or both of the optional arguments.
 *
 * Please see above for documentation of `rsvd_fixed_rank_symmetric_jacobi`.
 */
template <typename ValueType, typename IndexType, typename UType, typename VType>
void rsvd_fixed_rank_symmetric_jacobi(
  const raft::handle_t& handle,
  raft::device_matrix_view<const ValueType, IndexType, raft::col_major> M,
  raft::device_vector_view<ValueType, IndexType> S_vec,
  IndexType p,
  ValueType tol,
  int max_sweeps,
  UType&& U,
  VType&& V)
{
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U_optional =
    std::forward<UType>(U);
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V_optional =
    std::forward<VType>(V);

  rsvd_fixed_rank_symmetric_jacobi(handle, M, S_vec, p, tol, max_sweeps, U_optional, V_optional);
}

/**
 * @brief randomized singular value decomposition (RSVD) on a column major
 * rectangular matrix using QR decomposition, by specifying the PC and upsampling
 * ratio
 * @param[in] handle raft::handle_t
 * @param[in] M input raft::device_matrix_view with layout raft::col_major of shape (M, N)
 * @param[out] S_vec singular values raft::device_vector_view of shape (K)
 * @param[in] PC_perc percentage of singular values to be computed
 * @param[in] UpS_perc upsampling percentage
 * @param[out] U optional left singular values of raft::device_matrix_view with layout
 * raft::col_major
 * @param[out] V optional right singular values of raft::device_matrix_view with layout
 * raft::col_major
 */
template <typename ValueType, typename IndexType>
void rsvd_perc(
  const raft::handle_t& handle,
  raft::device_matrix_view<const ValueType, IndexType, raft::col_major> M,
  raft::device_vector_view<ValueType, IndexType> S_vec,
  ValueType PC_perc,
  ValueType UpS_perc,
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U = std::nullopt,
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V = std::nullopt)
{
  if (U) {
    RAFT_EXPECTS(M.extent(0) == U.value().extent(0), "Number of rows in M should be equal to U");
    RAFT_EXPECTS(S_vec.extent(0) == U.value().extent(1),
                 "Number of columns in U should be equal to length of S");
  }
  if (V) {
    RAFT_EXPECTS(M.extent(1) == V.value().extent(1), "Number of columns in M should be equal to V");
    RAFT_EXPECTS(S_vec.extent(0) == V.value().extent(0),
                 "Number of rows in V should be equal to length of S");
  }

  rsvdPerc(handle,
           const_cast<ValueType*>(M.data_handle()),
           M.extent(0),
           M.extent(1),
           S_vec.data_handle(),
           U.value().data_handle(),
           V.value().data_handle(),
           PC_perc,
           UpS_perc,
           false,
           U.has_value(),
           V.has_value(),
           false,
           static_cast<ValueType>(0),
           0,
           handle.get_stream());
}

/**
 * @brief Overload of `rsvd_perc` to help the
 *   compiler find the above overload, in case users pass in
 *   `std::nullopt` for one or both of the optional arguments.
 *
 * Please see above for documentation of `rsvd_perc`.
 */
template <typename ValueType, typename IndexType, typename UType, typename VType>
void rsvd_perc(const raft::handle_t& handle,
               raft::device_matrix_view<const ValueType, IndexType, raft::col_major> M,
               raft::device_vector_view<ValueType, IndexType> S_vec,
               ValueType PC_perc,
               ValueType UpS_perc,
               ValueType tol,
               int max_sweeps,
               UType&& U,
               VType&& V)
{
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U_optional =
    std::forward<UType>(U);
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V_optional =
    std::forward<VType>(V);

  rsvd_perc(handle, M, S_vec, PC_perc, UpS_perc, tol, max_sweeps, U_optional, V_optional);
}

/**
 * @brief randomized singular value decomposition (RSVD) on a column major
 * rectangular matrix using symmetric Eigen decomposition, by specifying the PC and upsampling
 * ratio. The rectangular input matrix is made square and symmetric using B @ B^T
 * @param[in] handle raft::handle_t
 * @param[in] M input raft::device_matrix_view with layout raft::col_major of shape (M, N)
 * @param[out] S_vec singular values raft::device_vector_view of shape (K)
 * @param[in] PC_perc percentage of singular values to be computed
 * @param[in] UpS_perc upsampling percentage
 * @param[out] U optional left singular values of raft::device_matrix_view with layout
 * raft::col_major
 * @param[out] V optional right singular values of raft::device_matrix_view with layout
 * raft::col_major
 */
template <typename ValueType, typename IndexType>
void rsvd_perc_symmetric(
  const raft::handle_t& handle,
  raft::device_matrix_view<const ValueType, IndexType, raft::col_major> M,
  raft::device_vector_view<ValueType, IndexType> S_vec,
  ValueType PC_perc,
  ValueType UpS_perc,
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U = std::nullopt,
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V = std::nullopt)
{
  if (U) {
    RAFT_EXPECTS(M.extent(0) == U.value().extent(0), "Number of rows in M should be equal to U");
    RAFT_EXPECTS(S_vec.extent(0) == U.value().extent(1),
                 "Number of columns in U should be equal to length of S");
  }
  if (V) {
    RAFT_EXPECTS(M.extent(1) == V.value().extent(1), "Number of columns in M should be equal to V");
    RAFT_EXPECTS(S_vec.extent(0) == V.value().extent(0),
                 "Number of rows in V should be equal to length of S");
  }

  rsvdPerc(handle,
           const_cast<ValueType*>(M.data_handle()),
           M.extent(0),
           M.extent(1),
           S_vec.data_handle(),
           U.value().data_handle(),
           V.value().data_handle(),
           PC_perc,
           UpS_perc,
           true,
           U.has_value(),
           V.has_value(),
           false,
           static_cast<ValueType>(0),
           0,
           handle.get_stream());
}

/**
 * @brief Overload of `rsvd_perc_symmetric` to help the
 *   compiler find the above overload, in case users pass in
 *   `std::nullopt` for one or both of the optional arguments.
 *
 * Please see above for documentation of `rsvd_perc_symmetric`.
 */
template <typename ValueType, typename IndexType, typename UType, typename VType>
void rsvd_perc_symmetric(const raft::handle_t& handle,
                         raft::device_matrix_view<const ValueType, IndexType, raft::col_major> M,
                         raft::device_vector_view<ValueType, IndexType> S_vec,
                         ValueType PC_perc,
                         ValueType UpS_perc,
                         ValueType tol,
                         int max_sweeps,
                         UType&& U,
                         VType&& V)
{
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U_optional =
    std::forward<UType>(U);
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V_optional =
    std::forward<VType>(V);

  rsvd_perc_symmetric(handle, M, S_vec, PC_perc, UpS_perc, tol, max_sweeps, U_optional, V_optional);
}

/**
 * @brief randomized singular value decomposition (RSVD) on a column major
 * rectangular matrix using Jacobi method, by specifying the PC and upsampling
 * ratio
 * @param[in] handle raft::handle_t
 * @param[in] M input raft::device_matrix_view with layout raft::col_major of shape (M, N)
 * @param[out] S_vec singular values raft::device_vector_view of shape (K)
 * @param[in] PC_perc percentage of singular values to be computed
 * @param[in] UpS_perc upsampling percentage
 * @param[in] tol tolerance for Jacobi-based solvers
 * @param[in] max_sweeps maximum number of sweeps for Jacobi-based solvers
 * @param[out] U optional left singular values of raft::device_matrix_view with layout
 * raft::col_major
 * @param[out] V optional right singular values of raft::device_matrix_view with layout
 * raft::col_major
 */
template <typename ValueType, typename IndexType>
void rsvd_perc_jacobi(
  const raft::handle_t& handle,
  raft::device_matrix_view<const ValueType, IndexType, raft::col_major> M,
  raft::device_vector_view<ValueType, IndexType> S_vec,
  ValueType PC_perc,
  ValueType UpS_perc,
  ValueType tol,
  int max_sweeps,
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U = std::nullopt,
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V = std::nullopt)
{
  if (U) {
    RAFT_EXPECTS(M.extent(0) == U.value().extent(0), "Number of rows in M should be equal to U");
    RAFT_EXPECTS(S_vec.extent(0) == U.value().extent(1),
                 "Number of columns in U should be equal to length of S");
  }
  if (V) {
    RAFT_EXPECTS(M.extent(1) == V.value().extent(1), "Number of columns in M should be equal to V");
    RAFT_EXPECTS(S_vec.extent(0) == V.value().extent(0),
                 "Number of rows in V should be equal to length of S");
  }

  rsvdPerc(handle,
           const_cast<ValueType*>(M.data_handle()),
           M.extent(0),
           M.extent(1),
           S_vec.data_handle(),
           U.value().data_handle(),
           V.value().data_handle(),
           PC_perc,
           UpS_perc,
           false,
           U.has_value(),
           V.has_value(),
           true,
           tol,
           max_sweeps,
           handle.get_stream());
}

/**
 * @brief Overload of `rsvd_perc_jacobi` to help the
 *   compiler find the above overload, in case users pass in
 *   `std::nullopt` for one or both of the optional arguments.
 *
 * Please see above for documentation of `rsvd_perc_jacobi`.
 */
template <typename ValueType, typename IndexType, typename UType, typename VType>
void rsvd_perc_jacobi(const raft::handle_t& handle,
                      raft::device_matrix_view<const ValueType, IndexType, raft::col_major> M,
                      raft::device_vector_view<ValueType, IndexType> S_vec,
                      ValueType PC_perc,
                      ValueType UpS_perc,
                      ValueType tol,
                      int max_sweeps,
                      UType&& U,
                      VType&& V)
{
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U_optional =
    std::forward<UType>(U);
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V_optional =
    std::forward<VType>(V);

  rsvd_perc_jacobi(handle, M, S_vec, PC_perc, UpS_perc, tol, max_sweeps, U_optional, V_optional);
}

/**
 * @brief randomized singular value decomposition (RSVD) on a column major
 * rectangular matrix using Jacobi method, by specifying the PC and upsampling
 * ratio. The rectangular input matrix is made square and symmetric using B @ B^T
 * @param[in] handle raft::handle_t
 * @param[in] M input raft::device_matrix_view with layout raft::col_major of shape (M, N)
 * @param[out] S_vec singular values raft::device_vector_view of shape (K)
 * @param[in] PC_perc percentage of singular values to be computed
 * @param[in] UpS_perc upsampling percentage
 * @param[in] tol tolerance for Jacobi-based solvers
 * @param[in] max_sweeps maximum number of sweeps for Jacobi-based solvers
 * @param[out] U optional left singular values of raft::device_matrix_view with layout
 * raft::col_major
 * @param[out] V optional right singular values of raft::device_matrix_view with layout
 * raft::col_major
 */
template <typename ValueType, typename IndexType>
void rsvd_perc_symmetric_jacobi(
  const raft::handle_t& handle,
  raft::device_matrix_view<const ValueType, IndexType, raft::col_major> M,
  raft::device_vector_view<ValueType, IndexType> S_vec,
  ValueType PC_perc,
  ValueType UpS_perc,
  ValueType tol,
  int max_sweeps,
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U = std::nullopt,
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V = std::nullopt)
{
  if (U) {
    RAFT_EXPECTS(M.extent(0) == U.value().extent(0), "Number of rows in M should be equal to U");
    RAFT_EXPECTS(S_vec.extent(0) == U.value().extent(1),
                 "Number of columns in U should be equal to length of S");
  }
  if (V) {
    RAFT_EXPECTS(M.extent(1) == V.value().extent(1), "Number of columns in M should be equal to V");
    RAFT_EXPECTS(S_vec.extent(0) == V.value().extent(0),
                 "Number of rows in V should be equal to length of S");
  }

  rsvdPerc(handle,
           const_cast<ValueType*>(M.data_handle()),
           M.extent(0),
           M.extent(1),
           S_vec.data_handle(),
           U.value().data_handle(),
           V.value().data_handle(),
           PC_perc,
           UpS_perc,
           true,
           U.has_value(),
           V.has_value(),
           true,
           tol,
           max_sweeps,
           handle.get_stream());
}

/**
 * @brief Overload of `rsvd_perc_symmetric_jacobi` to help the
 *   compiler find the above overload, in case users pass in
 *   `std::nullopt` for one or both of the optional arguments.
 *
 * Please see above for documentation of `rsvd_perc_symmetric_jacobi`.
 */
template <typename ValueType, typename IndexType, typename UType, typename VType>
void rsvd_perc_symmetric_jacobi(
  const raft::handle_t& handle,
  raft::device_matrix_view<const ValueType, IndexType, raft::col_major> M,
  raft::device_vector_view<ValueType, IndexType> S_vec,
  ValueType PC_perc,
  ValueType UpS_perc,
  ValueType tol,
  int max_sweeps,
  UType&& U,
  VType&& V)
{
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U_optional =
    std::forward<UType>(U);
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V_optional =
    std::forward<VType>(V);

  rsvd_perc_symmetric_jacobi(
    handle, M, S_vec, PC_perc, UpS_perc, tol, max_sweeps, U_optional, V_optional);
}

/** @} */  // end of group rsvd

};  // end namespace linalg
};  // end namespace raft

#endif