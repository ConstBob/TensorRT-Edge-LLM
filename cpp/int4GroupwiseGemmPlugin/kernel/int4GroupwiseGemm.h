#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

void gemv_forward_cuda_new(half* in_feats, int8_t* kernel, half* scaling_factors, half* out_feats, int m, int n, int k,
    int group_size, cudaStream_t stream);

void gemm_forward_cuda_new(half* in_feats, int8_t* kernel, half* scaling_factors, half* out_feats, int m, int n, int k,
    int group_size, cudaStream_t stream);
