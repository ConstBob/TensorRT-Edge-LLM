#pragma once

#include <cfloat>
#include <cuda_fp16.h>
#include <type_traits>

static float constexpr HALF_FLT_MAX = 65504.F;

template <typename T>
struct TopK_2
{
    int p = -1;
    T u = -((std::is_same<T, half>::value) ? HALF_FLT_MAX : FLT_MAX);

    __device__ __forceinline__ void insert(T elem, int elem_id)
    {
        if (elem > u)
        {
            u = elem;
            p = elem_id;
        }
    }

    __device__ __forceinline__ void init()
    {
        u = -((std::is_same<T, half>::value) ? HALF_FLT_MAX : FLT_MAX);
        p = -1;
    }
};

template <typename T>
__device__ __forceinline__ TopK_2<T> reduce_topk_op_2(TopK_2<T> const& a, TopK_2<T> const& b)
{
    return a.u > b.u ? a : b;
}
