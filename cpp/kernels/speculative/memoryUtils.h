#pragma once

#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>

__device__ __forceinline__ int4 reduceMaxInt4(int4 a, int4 b)
{
    return a.x > b.x ? a : b;
}

template <typename T, typename TIndex>
__inline__ __host__ __device__ T constexpr flat_index2(TIndex const& index_0, TIndex const& index_1, T const& dim_1)
{
    assert(index_1 < dim_1);
    return index_0 * dim_1 + index_1;
}

template <typename T, typename TIndex>
__inline__ __host__ __device__ T constexpr flat_index3(
    TIndex const& index_0, TIndex const& index_1, TIndex const& index_2, T const& dim_1, T const& dim_2)
{
    assert(index_2 < dim_2);
    return flat_index2(index_0, index_1, dim_1) * dim_2 + index_2;
}

template <typename T, typename TIndex>
__inline__ __host__ __device__ T constexpr flat_index4(TIndex const& index_0, TIndex const& index_1,
    TIndex const& index_2, TIndex const& index_3, T const& dim_1, T const& dim_2, T const& dim_3)
{
    assert(index_3 < dim_3);
    return flat_index3(index_0, index_1, index_2, dim_1, dim_2) * dim_3 + index_3;
}

template <typename T, typename TIndex>
__inline__ __host__ __device__ T constexpr flat_index5(TIndex const& index_0, TIndex const& index_1,
    TIndex const& index_2, TIndex const& index_3, TIndex const& index_4, T const& dim_1, T const& dim_2, T const& dim_3,
    T const& dim_4)
{
    assert(index_4 < dim_4);
    return flat_index4(index_0, index_1, index_2, index_3, dim_1, dim_2, dim_3) * dim_4 + index_4;
}

template <typename T>
__inline__ __host__ __device__ T constexpr divUp(T m, T n)
{
    return (m + n - 1) / n;
}
