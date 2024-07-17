/***************************************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR 
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND 
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE 
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <fused_multihead_attention_api.h>
#include <fmha/hopper/tma_types.h>
#include <fmha/attention_mask_type.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <float.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string>

using Launch_params = bert::Fused_multihead_attention_launch_params;
using Attention_mask_type = fmha::Attention_mask_type;

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_softmax_fp32(void *dst, const void *src, const void *mask, 
                      int s_inner, int s_outer, int b, int h, int warps_n, bool has_alibi);

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_softmax_e4m3(void *dst, const void *src, const void *mask, 
                      int s_inner, int s_outer, int b, int h, float scale_softmax, int warps_n, bool has_alibi);

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_softmax_fp16(void *dst, const void *src, const void *mask, 
                      int s_inner, int s_outer, int b, int h, int warps_n, bool has_alibi);

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_softmax_bf16(void *dst, const void *src, const void *mask, 
                      int s_inner, int s_outer, int b, int h, int warps_n, bool has_alibi);

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_softmax_int8(void *dst,
                       const void *src,
                       const void *mask,
                       int s_inner,
                       int s_outer,
                       int b,
                       int h,
                       float scale_i2f,
                       float scale_f2i,
                      int warps_n,
                      bool has_alibi);

////////////////////////////////////////////////////////////////////////////////////////////////////


void run_conversion_int32_to_int8(void *dst,
                                  const void *src,
                                  int s,
                                  int b,
                                  int h,
                                  int d,
                                  float scale);

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_conversion_fp32_to_fp16(void *dst, const void *src, int s, int b, int h, int d);

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_conversion_fp32_to_bf16(void *dst, const void *src, int s, int b, int h, int d);

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_conversion_fp32_to_e4m3(void *dst, const void *src, int s, int b, int h, int d, float scale_o);

////////////////////////////////////////////////////////////////////////////////////////////////////

void ground_truth(RefBMM & bmm1,
                  RefBMM & bmm2,
                  const Data_type data_type,
                  const Data_type acc_type,
                  const float scale_bmm1,
                  const float scale_softmax,
                  const float scale_bmm2,
                  void *qkv_d,
                  void *vt_d,
                  void *mask_d,
                  void *p_d,
                  void *s_d,
                  void *tmp_d,
                  void *o_d,
                  const size_t b,
                  const size_t s,
                  const size_t h,
                  const size_t d,
                  const int runs,
                  const int warps_m,
                  const int warps_n,
                  const bool has_alibi
		 ) {

    cudaStream_t stream = 0;
    // The stride between rows of the QKV matrix.
    size_t qkv_stride = get_size_in_bytes(d, data_type);

    // 1st GEMMd.
    uint32_t alpha, beta = 0u;

    for( int ii = 0; ii < runs; ++ii ) {

        // If we run the INT8 kernel, defer the scaling of P to softmax.
        set_alpha(alpha, data_type == DATA_TYPE_INT8 ? 1.f : scale_bmm1, acc_type);

        // P = Q x K'
        bmm1(static_cast<char *>(qkv_d) + 0 * qkv_stride,
             static_cast<char *>(qkv_d) + 1 * qkv_stride,
             p_d,
             &alpha,
             &beta,
             stream
             );

        // Softmax. 
        if( data_type == DATA_TYPE_FP16 && acc_type ==  DATA_TYPE_FP16 ) {
            run_softmax_fp16(s_d, p_d, mask_d, s, s, b, h, warps_n, has_alibi);
        } else if( data_type == DATA_TYPE_BF16 && acc_type ==  DATA_TYPE_FP32 ) {
            run_softmax_bf16(s_d, p_d, mask_d, s, s, b, h, warps_n, has_alibi);
        } else if( data_type == DATA_TYPE_FP16 && acc_type ==  DATA_TYPE_FP32 ) {
            run_softmax_fp32(s_d, p_d, mask_d, s, s, b, h, warps_n, has_alibi);
        } else if( data_type == DATA_TYPE_E4M3 && acc_type ==  DATA_TYPE_FP32 ) {
            run_softmax_e4m3(s_d, p_d, mask_d, s, s, b, h, scale_softmax, warps_n, has_alibi);
        } else if( data_type == DATA_TYPE_INT8 && acc_type == DATA_TYPE_INT32 ) {
            run_softmax_int8(s_d, p_d, mask_d, s, s, b, h, scale_bmm1, scale_softmax, warps_n, has_alibi);
        } else {
            assert(false && "Reference Softmax: Unsupported type config");
        }

        // 2nd GEMM.
        set_alpha(alpha, 1.f, acc_type);

        void *out_d = o_d;  

        // We may have to do a final conversion.
        if( data_type != acc_type ) {
            out_d = tmp_d;
        }

        // O = S x V
        bmm2(static_cast<char *>(s_d),
             static_cast<char *>(vt_d),//static_cast<char *>(qkv_d) + 2 * qkv_stride, 
             out_d,
             &alpha,
             &beta,
             stream
            );

        // Conversion to output type.
        if( data_type == DATA_TYPE_FP16 && acc_type ==  DATA_TYPE_FP16 ) {
            // Noop.
        } else if( data_type == DATA_TYPE_FP16 && acc_type ==  DATA_TYPE_FP32 ) {
            run_conversion_fp32_to_fp16(o_d, out_d, s, b, h, d);
        } else if( data_type == DATA_TYPE_BF16 && acc_type ==  DATA_TYPE_FP32 ) {
            run_conversion_fp32_to_bf16(o_d, out_d, s, b, h, d);
        } else if( data_type == DATA_TYPE_E4M3 && acc_type ==  DATA_TYPE_FP32 ) {
            run_conversion_fp32_to_e4m3(o_d, out_d, s, b, h, d, scale_bmm2);
        } else if( data_type == DATA_TYPE_INT8 && acc_type == DATA_TYPE_INT32 ) {
            // quantize output in second step
            run_conversion_int32_to_int8(o_d, out_d, s, b, h, d, scale_bmm2);
        }

    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void set_params(bert::Fused_multihead_attention_params_v1 & params,
                              // types
                              Data_type data_type,
                              Data_type acc_type,
                              // sizes
                              const size_t b,
                              const size_t s,
                              const size_t h,
                              const size_t d,
                              const size_t packed_mask_stride,
                              // device pointers
                              void *qkv_d,
                              void *packed_mask_d,
                              void *o_d,
                              void *p_d,
                              void *s_d,
                              // scale factors
                              const float scale_bmm1,
                              const float scale_softmax,
                              const float scale_bmm2,
                              // flags
                              const bool has_alibi) {
    memset(&params, 0, sizeof(params));

    // Set the pointers.
    params.qkv_ptr = qkv_d;
    params.qkv_stride_in_bytes = get_size_in_bytes(b * h * 3 * d, data_type);
    // params.qkv_stride_in_bytes = get_size_in_bytes(h * 3 * d, data_type);
    params.packed_mask_ptr = packed_mask_d;
    // params.packed_mask_stride_in_bytes = mmas_m * threads_per_cta * sizeof(uint32_t);
    params.packed_mask_stride_in_bytes = packed_mask_stride * sizeof(uint32_t);
    params.o_ptr = o_d;
    params.o_stride_in_bytes = get_size_in_bytes(b * h * d, data_type);
    params.has_alibi = has_alibi;
    params.alibi_params = fmha::AlibiParams(h);

#if defined(STORE_P)
    params.p_ptr = p_d;
    params.p_stride_in_bytes = get_size_in_bytes(b * h * s, acc_type);
#endif  // defined(STORE_P)

#if defined(STORE_S)
    params.s_ptr = s_d;
    params.s_stride_in_bytes = get_size_in_bytes(b * h * s, data_type);
#endif  // defined(STORE_S)

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.s = s;
    params.d = d;

    // Set the different scale values.
    Data_type scale_type1 =
        (data_type == DATA_TYPE_FP16) || (data_type == DATA_TYPE_BF16)
        ? acc_type
        : DATA_TYPE_FP32;
    Data_type scale_type2 =
        (data_type == DATA_TYPE_FP16) || (data_type == DATA_TYPE_BF16)
        ? data_type
        : DATA_TYPE_FP32;

    set_alpha(params.scale_bmm1, scale_bmm1, scale_type1);
    set_alpha(params.scale_softmax, scale_softmax, scale_type1);
    set_alpha(params.scale_bmm2, scale_bmm2, scale_type2);

    // Do we enable the trick to replace I2F with FP math in the 2nd GEMM?
    if( data_type == DATA_TYPE_INT8 ) {
        params.enable_i2f_trick = -double(1 << 22) * double(scale_bmm2) <= -128.f &&
                                   double(1 << 22) * double(scale_bmm2) >=  127.f;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void set_params(bert::Fused_multihead_attention_params_v2 &params,
                              // types
                              Data_type data_type,
                              Data_type acc_type,
                              // sizes
                              const size_t b,
                              const size_t s,
                              const size_t h,
                              const size_t d,
                              const size_t total,
                              const size_t max_past_s,
                              // device pointers
                              void *qkv_packed_d,
                              void *cu_seqlens_d,
                              void *o_packed_d,
                              void *p_d,
                              void *s_d,
                              // scale factors
                              const float scale_bmm1,
                              const float scale_softmax,
                              const float scale_bmm2,
                              // flags
                              const bool use_int8_scale_max,
                              const bool interleaved,
                              const bool is_s_padded,
                              const bool has_alibi,
                              // attention type,
                              const int h_kv // h_kv < h if MQA or GQA
                              ) {

    memset(&params, 0, sizeof(params));

    float scale_bmm1_log2e = scale_bmm1 * float(M_LOG2E);

    // Set the pointers.
    params.qkv_ptr = qkv_packed_d;
    // For grouped- or multi-query attention (h denotes num_q_heads; h' denotes h_kv):
    //   qkv_layout = [b, s, [q_hd, k_h'd, v_h'd]]
    //   qkv_stride = (h+2*h')d * bytes_per_elt
    // Otherwise:
    //   qkv_layout = [b, s, 3, h, d] or [b, s, h, 3, d]
    //   qkv_stride = 3hd * bytes_per_elt
    params.qkv_stride_in_bytes = get_size_in_bytes((h + 2 * h_kv) * d, data_type);
    params.o_ptr = o_packed_d;
    params.o_stride_in_bytes = get_size_in_bytes(h * d, data_type);

    if( interleaved ) {
        params.qkv_stride_in_bytes = total;
        params.o_stride_in_bytes = total;
    }

    params.cu_seqlens = static_cast<int *>(cu_seqlens_d);

#if defined(STORE_P)
    params.p_ptr = p_d;
    params.p_stride_in_bytes = get_size_in_bytes(b * h * s, acc_type);
#endif  // defined(STORE_P)

#if defined(STORE_S)
    params.s_ptr = s_d;
    params.s_stride_in_bytes = get_size_in_bytes(b * h * s, data_type);
#endif  // defined(STORE_S)

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.s = s;
    params.d = d;
    params.max_past_s = max_past_s;

    // Set the different scale values.
    Data_type scale_type1 = 
        (data_type == DATA_TYPE_FP16) || (data_type == DATA_TYPE_BF16 )
        ? acc_type
        : DATA_TYPE_FP32;
    Data_type scale_softmax_type = scale_type1;
    Data_type scale_type2 =
        (data_type == DATA_TYPE_FP16) || (data_type == DATA_TYPE_BF16 )
        ? data_type
        : DATA_TYPE_FP32;
    if (data_type == DATA_TYPE_E4M3) {
        scale_type1 = acc_type;
        scale_type2 = acc_type;
    }

    set_alpha(params.scale_bmm1, scale_bmm1_log2e, DATA_TYPE_FP32);
    set_alpha(params.scale_softmax, scale_softmax, scale_softmax_type);
    set_alpha(params.scale_bmm2, scale_bmm2, scale_type2);

    // attention type, h_kv < h if MQA or GQA   
    params.h_kv = h_kv;
    params.has_alibi = has_alibi;
    params.alibi_params = fmha::AlibiParams(h);

    // Set flags
    params.is_s_padded = is_s_padded;
    params.use_int8_scale_max = use_int8_scale_max;

    // Do we enable the trick to replace I2F with FP math in the 2nd GEMM?
    if( data_type == DATA_TYPE_INT8 ) {
        params.enable_i2f_trick = -double(1 << 22) * double(scale_bmm2) <= -128.f &&
                                   double(1 << 22) * double(scale_bmm2) >=  127.f;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void determine_launch_params(Launch_params &launch_params,
                                           Data_type data_type,
                                           int sm,
                                           const size_t s,
                                           const size_t d,
                                           const Attention_mask_type attention_mask_type,
                                           const bool interleaved,
                                           const bool ignore_b1opt,
                                           const bool force_unroll,
                                           const bool use_tma,
                                           const bool force_non_flash_attention,
                                           const bool force_non_warp_specialization,
                                           const bool force_non_granular_tiling,
                                           const bool force_fp32_acc,
                                           // device props
                                           const cudaDeviceProp props
                                           ) {

    // Set launch params to choose kernels
    launch_params.ignore_b1opt          = ignore_b1opt;
    launch_params.force_unroll          = force_unroll;
    launch_params.force_fp32_acc        = force_fp32_acc;
    launch_params.interleaved           = interleaved;
    launch_params.attention_mask_type   = attention_mask_type;

    // Set SM count and L2 cache size (used to determine launch blocks/grids to maximum performance)
    launch_params.multi_processor_count = props.multiProcessorCount;
    launch_params.device_l2_cache_size = props.l2CacheSize;

    // threshold for adopting flash attention or warp_specialized kernels.
    launch_params.flash_attention =
        (data_type == DATA_TYPE_FP16 || data_type == DATA_TYPE_BF16) &&
        (s >= 16 && d >= 16) && !force_non_flash_attention;
    
    // enable warp_speialized kernels when s >= 512 on hopper
    // note that warp_speialized kernels need flash attention + tma
    launch_params.warp_specialization = 
        (data_type == DATA_TYPE_FP16 || data_type == DATA_TYPE_BF16) && sm == 90 && 
        launch_params.flash_attention && !force_non_warp_specialization;
    // warp specialization kernels on hopper need tma
    launch_params.use_tma           = use_tma || launch_params.warp_specialization;

    // use granular tiling on Ampere-style flash attention
    launch_params.use_granular_tiling = !force_non_granular_tiling && launch_params.flash_attention
        && !launch_params.warp_specialization && sm >= 80;

}

////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {

    // The device. Reset on destruction
    CudaDevice device;
    int sm = device.sm;
    cudaDeviceProp props = device.props;

    GpuTimer timer;

    // The batch size.
    size_t b = 128;
    // The number of heads.
    size_t h = 16;
    // The dimension of the Q, K and V vectors.
    size_t d = 64;
    // The length of the sequence.
    size_t s = 384;
    // The max past length of the sequence
    // only pay attention to [max(0, seq_length - max_past_length), seq_length)
    size_t max_past_s = size_t(INT_MAX);

    // The data type of the kernel.
    Data_type data_type = DATA_TYPE_FP16;
    // The type of the intermediate P matrix.
    Data_type acc_type = DATA_TYPE_FP16;

    // The scaling factors.
    float scale_bmm1 = 0.f, scale_softmax = 0.f, scale_bmm2 = 0.25f;
    // The number of runs.
    int runs = 1, warm_up_runs = 0;
    // Do we use 1s for Q, K, V.
    bool use_1s_q = false, use_1s_k = false, use_1s_v = false;
    // The range of the different inputs.
    int range_q = 5, range_k = 3, range_v = 5;
    // The scale.
    float scale_q = 0.f, scale_k = 0.f, scale_v = 0.f;
    // The threshold for dropout. By default, drop 10%.
    float dropout = 0.1f;
    // Do we skip the checks.
    bool skip_checks = false;
    // The tolerance when checking results.
    float epsilon = -1.f; // data_type == DATA_TYPE_FP16 ? 0.015f : 0.f;
    // Use causal mask / padding_mask / limited_length_causal mask
    Attention_mask_type attention_mask_type = Attention_mask_type::PADDING;
    // Use padded format for input QKV tensor & output O tensor.
    // Instead of variable lengths [total, h, 3, d]  where total = b1*s1 + b2*s2 + ... bn*sn,
    // use padded length [b, max_s, h, 3, d]         where max_s is the maximum expected seq len
    bool is_s_padded = false;

    // minimum sequence length for sampling variable seqlens
    uint32_t min_s = s;

    // run interleaved kernels and tranpose input and output accordingly
    bool interleaved = false;
    bool ignore_b1opt = false;
    bool force_unroll = false;
    // used by kernels that have different acc data types (like hmma, qmma)
    bool force_fp32_acc = false;
    bool force_non_flash_attention = false;
    // enable warp specialization kernels on sm 90
    bool force_non_warp_specialization = (sm != 90);
    bool use_int8_scale_max = false;
    bool verbose = true;

    // use granular tiling
    // supported only by Ampere-based Flash Attention at this moment
    bool force_non_granular_tiling = false;

    // set all sequence lengths to min(s, min_s)
    bool fix_s = false;

    bool v1 = false;

    // use TMA or not. ignored if not in SM90
    bool use_tma = false;

    bool has_alibi = false;

    // In multi-query or grouped-query attention (MQA/GQA), several Q heads are associated with one KV head
    bool multi_query_attention = false;
    size_t h_kv = 0;

    // Read the parameters from the command-line.
    for( int ii = 1; ii < argc; ++ii ) {
        if( !strcmp(argv[ii], "-1s") ) {
            use_1s_k = use_1s_q = use_1s_v = true;
        } else if( !strcmp(argv[ii], "-1s-k") ) {
            use_1s_k = true;
        } else if( !strcmp(argv[ii], "-1s-q") ) {
            use_1s_q = true;
        } else if( !strcmp(argv[ii], "-1s-v") ) {
            use_1s_v = true;
        } else if( !strcmp(argv[ii], "-b") && ++ii < argc ) {
            b = strtol(argv[ii], nullptr, 10);
        } else if( !strcmp(argv[ii], "-d") && ++ii < argc ) {
            d = strtol(argv[ii], nullptr, 10);
        } else if( !strcmp(argv[ii], "-dropout") && ++ii < argc ) {
            dropout = (float) strtod(argv[ii], nullptr);
        } else if( !strcmp(argv[ii], "-epsilon") && ++ii < argc ) {
            epsilon = (float) strtod(argv[ii], nullptr);
        } else if( !strcmp(argv[ii], "-h") && ++ii < argc ) {
            h = strtol(argv[ii], nullptr, 10);
        } else if( !strcmp(argv[ii], "-int8") ) {
            data_type = DATA_TYPE_INT8;
            acc_type = DATA_TYPE_INT32;
        } else if( !strcmp(argv[ii], "-fp16") ) {
            data_type = DATA_TYPE_FP16;
            acc_type = DATA_TYPE_FP16;
        } else if( !strcmp(argv[ii], "-fp16-fp32") ) {
            data_type = DATA_TYPE_FP16;
            acc_type = DATA_TYPE_FP32;
            force_fp32_acc = true;
        } else if( !strcmp(argv[ii], "-bf16") ) {
            data_type = DATA_TYPE_BF16;
            acc_type = DATA_TYPE_FP32;
            force_fp32_acc = true;
        } else if( !strcmp(argv[ii], "-e4m3") ) {
            data_type = DATA_TYPE_E4M3;
            // Technically not the acc type.
            acc_type = DATA_TYPE_FP32;
            force_fp32_acc = true;
        } else if( !strcmp(argv[ii], "-e4m3-fp16") ) { // Ada QMMA only
            data_type = DATA_TYPE_E4M3;
            // Technically not the acc type.
            acc_type = DATA_TYPE_FP16;
        } else if( !strcmp(argv[ii], "-e4m3-fp32") ) {
            data_type = DATA_TYPE_E4M3;
            // Technically not the acc type.
            acc_type = DATA_TYPE_FP32;
            force_fp32_acc = true;
        } else if( !strcmp(argv[ii], "-range-k") && ++ii < argc ) {
            range_k = strtol(argv[ii], nullptr, 10);
        } else if( !strcmp(argv[ii], "-range-q") && ++ii < argc ) {
            range_q = strtol(argv[ii], nullptr, 10);
        } else if( !strcmp(argv[ii], "-range-v") && ++ii < argc ) {
            range_v = strtol(argv[ii], nullptr, 10);
        } else if( !strcmp(argv[ii], "-runs") && ++ii < argc ) {
            runs = strtol(argv[ii], nullptr, 10);
        } else if( !strcmp(argv[ii], "-s") && ++ii < argc ) {
            s = strtol(argv[ii], nullptr, 10);
        } else if( !strcmp(argv[ii], "-max-past-seq") && ++ii < argc ) {
            max_past_s = strtol(argv[ii], nullptr, 10);
        } else if( !strcmp(argv[ii], "-scale-bmm1") && ++ii < argc ) {
            scale_bmm1 = (float)strtod(argv[ii], nullptr);
        } else if( !strcmp(argv[ii], "-scale-bmm2") && ++ii < argc ) {
            scale_bmm2 = (float)strtod(argv[ii], nullptr);
        } else if( !strcmp(argv[ii], "-scale-k") && ++ii < argc ) {
            scale_k = (float)strtod(argv[ii], nullptr);
        } else if( !strcmp(argv[ii], "-scale-softmax") && ++ii < argc ) {
            scale_softmax = (float)strtod(argv[ii], nullptr);
        } else if( !strcmp(argv[ii], "-scale-q") && ++ii < argc ) {
            scale_q = (float)strtod(argv[ii], nullptr);
        } else if( !strcmp(argv[ii], "-scale-v") && ++ii < argc ) {
            scale_v = (float)strtod(argv[ii], nullptr);
        } else if( !strcmp(argv[ii], "-skip-checks") ) {
            skip_checks = true;
        } else if( !strcmp(argv[ii], "-warm-up-runs") && ++ii < argc ) {
            warm_up_runs = strtol(argv[ii], nullptr, 10);
        } else if( !strcmp(argv[ii], "-min-s") && ++ii < argc ) {
            min_s = strtol(argv[ii], nullptr, 10);
        } else if( !strcmp(argv[ii], "-il") ) {
            interleaved = true;
        } else if( !strcmp(argv[ii], "-causal-mask") ) {
            attention_mask_type = Attention_mask_type::CAUSAL;
        } else if( !strcmp(argv[ii], "-limited-past-causal-mask") ) {
            attention_mask_type = Attention_mask_type::LIMITED_LENGTH_CAUSAL;
        } else if( !strcmp(argv[ii], "-multi-query-attention") || !strcmp(argv[ii], "-mqa") ) {
            h_kv = 1;
            multi_query_attention = true; // subset of GQA
        } else if( (!strcmp(argv[ii], "-grouped-query-attention") || !strcmp(argv[ii], "-gqa")) && ++ii < argc) {
            h_kv = strtol(argv[ii], nullptr, 10);
            multi_query_attention = true;
        } else if( !strcmp(argv[ii], "-pad-s") ) {
            is_s_padded = true;
        } else if( !strcmp(argv[ii], "-ignore-b1opt") ) {
            ignore_b1opt = true;
        } else if( !strcmp(argv[ii], "-force-unroll") ) {
            force_unroll = true;
        } else if( !strcmp(argv[ii], "-force-non-flash-attention") ) {
            force_non_flash_attention = true;
            force_non_warp_specialization = true;
        } else if( !strcmp(argv[ii], "-force-flash-attention") ) {
            fprintf(stderr, "Deprecation warning: -force-flash-attention is no longer valid; use "
                    "-force-non-flash-attention instead, as Flash Attention is enabled by default.\n");
        } else if( !strcmp(argv[ii], "-force-non-warp-specialization") ) {
            force_non_warp_specialization = true;
        } else if( !strcmp(argv[ii], "-force-non-granular-tiling") || !strcmp(argv[ii], "-force-non-tiled") ) {
            force_non_granular_tiling = true;
        } else if( !strcmp(argv[ii], "-fix-s") ) {
            fix_s = true;
        } else if( !strcmp(argv[ii], "-scale-max") ) {
            use_int8_scale_max = true;
        } else if( !strcmp(argv[ii], "-v") && ++ii < argc ) {
            int v = strtol(argv[ii], nullptr, 10);
            verbose = v != 0;
        } else if( !strcmp(argv[ii], "-v1") ) {
            v1 = true;
        } else if( !strcmp(argv[ii], "-use-tma") ) {
            use_tma = true;
            // flash attention + tma + non_warp_specialized kernels are not supported
            // use non_flash_attention + tma + non_warp_specialized instead
            if( force_non_warp_specialization ) {
                force_non_flash_attention = true;
            }
        } else if( !strcmp(argv[ii], "-alibi")) {
            has_alibi = true;
        } else {
            fprintf(stderr, "Unrecognized option: %s. Aborting!\n", argv[ii]);
            return -1;
        }
    }
    // Sanitize
    min_s = std::min<uint32_t>(s, min_s);
    h_kv = multi_query_attention ? h_kv : h;

    // Past limited attention (only pay attention to max-past-s long previous tokens).
    if( max_past_s < s ) {
        attention_mask_type = Attention_mask_type::LIMITED_LENGTH_CAUSAL;
    }

    // Set the norm.
    if( scale_bmm1 == 0.f ) {
        scale_bmm1 = 1.f / sqrtf((float)d);
    }

    // Force the softmax scale to 1.f for the FP16 kernel.
    if( data_type == DATA_TYPE_FP16 ) {
        scale_softmax = 1.f;
    } else if( data_type == DATA_TYPE_INT8 && scale_softmax == 0.f ) {
        scale_softmax = std::max(512.f, (float) s);
    }

    // Define the scaling factor for the different inputs.
    if( scale_q == 0.f ) {
        scale_q = 1.f;
    }
    if( scale_k == 0.f ) {
        scale_k = 1.f;
    }
    if( scale_v == 0.f ) {
        // BF16 here just for debug.
        scale_v = (data_type == DATA_TYPE_FP16  || data_type == DATA_TYPE_BF16) ? 0.125f : 1.f;
    }
    if(has_alibi && attention_mask_type == Attention_mask_type::PADDING) {
        attention_mask_type = Attention_mask_type::CAUSAL;
    }

    // BF16 only support FP32 acc_type.
    if ( data_type == DATA_TYPE_BF16 && acc_type != DATA_TYPE_FP32 ) {
        fprintf(stderr, "Only FP32 accumulation is supported for BF16 I/O\n");
        exit(1);
    }

    // Set the tolerance if not already set by the user.
    if( epsilon < 0.f ) {
        switch( data_type ) {
        case DATA_TYPE_FP16:
            epsilon = 0.015f;
            break;
        case DATA_TYPE_BF16:
            epsilon = 0.025f;
            break;
        default:
            epsilon = 0.f;
        }
    }

    // Debug info -- only in verbose mode.
    if( verbose ) {
        // Running the following command.
        printf("Command.......: %s", argv[0]);
        for( int ii = 1; ii < argc; ++ii ) {
            printf(" %s", argv[ii]);
        }
        printf("\n");

        // Device info.
        printf("Device........: %s\n", props.name);
        printf("Arch.(sm).....: %d\n", sm);
        printf("#.of.SMs......: %d\n", props.multiProcessorCount);

        // Problem info.
        printf("Batch ........: %lu\n", b);
        printf("Heads ........: %lu\n", h);
        printf("Dimension ....: %lu\n", d);
        printf("Seq length ...: %lu\n", s);
        printf("Warm-up runs .: %d\n", warm_up_runs);
        printf("Runs..........: %d\n\n", runs);

        // The scaling factors for the 3 operations.
        printf("Scale bmm1 ...: %.6f\n", scale_bmm1);
        printf("Scale softmax.: %.6f\n", scale_softmax);
        printf("Scale bmm2 ...: %.6f\n", scale_bmm2);
        printf("\n");
    }

    // determine the launch params to select kernels
    Launch_params launch_params;
    determine_launch_params(launch_params,
                            data_type,
                            sm,
                            s,
                            d,
                            attention_mask_type,
                            interleaved,
                            ignore_b1opt,
                            force_unroll,
                            use_tma,
                            force_non_flash_attention,
                            force_non_warp_specialization,
                            force_non_granular_tiling,
                            force_fp32_acc,
                            props);

    // The Q, K and V matrices are packed into one big matrix of size S x B x H x 3 x D.
    const size_t qkv_size = s * b * h * 3 * d;
    // Allocate on the host.
    float *qkv_h = (float *)malloc(qkv_size * sizeof(float));
    // The size in bytes.
    const size_t qkv_size_in_bytes = get_size_in_bytes(qkv_size, data_type);
    // Allocate on the device.
    void *qkv_sbh3d_d = nullptr, *qkv_bsh3d_d = nullptr;
    FMHA_CHECK_CUDA(cudaMalloc(&qkv_sbh3d_d, qkv_size_in_bytes));
    FMHA_CHECK_CUDA(cudaMalloc(&qkv_bsh3d_d, qkv_size_in_bytes));

    // The mask for dropout.
    const size_t mask_size = s * b * s;
    // Allocate on the host.
    float *mask_h = (float *)malloc(mask_size * sizeof(float));
    // The size in bytes.
    const size_t mask_size_in_bytes = get_size_in_bytes(mask_size, DATA_TYPE_INT8);
    // Allocate on the device.
    void *mask_d = nullptr;
    FMHA_CHECK_CUDA(cudaMalloc(&mask_d, mask_size_in_bytes));

    // The decomposition of threads and warps for BMM1.
    size_t warps_m, warps_n, warps_k;
    std::tie(warps_m, warps_n, warps_k) = get_warps(launch_params,
                                                    sm,
                                                    data_type,
                                                    s,
                                                    b,
                                                    d,
                                                    v1 ? 1 : 2);

    // print launch configuration
    printf("v1=%d il=%d s=%lu b=%lu h=%lu/%lu d=%lu dtype=%s, flash_attn=%s, warp_spec=%s, mask=%s, alibi=%s, attn=%s, wm=%lu wn=%lu\n",
            v1,
            interleaved,
            s,
            b,
            h, h_kv,
            d,
            data_type_to_name(data_type).c_str(),
            launch_params.flash_attention ?
                (launch_params.use_granular_tiling ? "true_tiled" : "true") : "false",
            launch_params.warp_specialization ? "true" : "false",
            attention_mask_type == Attention_mask_type::PADDING ? "padding" :
                (attention_mask_type == Attention_mask_type::CAUSAL ? "causal" : "limited_length_causal"),
            has_alibi ? "true" : "false",
            h_kv == 1 ? "mqa" : (h_kv == h ? "mha" : "gqa"),
            warps_m,
            warps_n);

    // For multi-CTA cases, determine the size of the CTA wave.
    int heads_per_wave, ctas_per_head;
    get_grid_size(heads_per_wave,
                  ctas_per_head,
                  sm,
                  data_type,
                  b,
                  s,
                  h,
                  d,
                  false, // disable multi-cta kernels by default
                  v1 ? 1 : 2);

    // The number of threads per CTA.
    const size_t threads_per_cta = warps_m * warps_n * warps_k * 32;
    // The number of mmas in the M dimension. We use one uint32_t per MMA in the M dimension.
    const size_t mmas_m = (s + 16 * warps_m - 1) / (16 * warps_m);
    // The number of mmas in the N dimension.
    const size_t mmas_n = (s + 16 * warps_n - 1) / (16 * warps_n);
    // We do not support more than 4 MMAS in the N dimension (as each MMA needs 8 bits in the mask).
    assert(!v1 || mmas_n <= 4);
    // The packed mask for dropout (in the fused kernel). Layout is B * MMAS_M * THREADS_PER_CTA.
    const size_t packed_mask_size = b * mmas_m * threads_per_cta;
    // The size in bytes.
    const size_t packed_mask_size_in_bytes = packed_mask_size * sizeof(uint32_t);
    // Allocate on the host.
    uint32_t *packed_mask_h = (uint32_t *)malloc(packed_mask_size_in_bytes);
    // Allocate on the device.
    void *packed_mask_d = nullptr;

    // The O matrix is packed as S * B * H * D.
    const size_t o_size = s * b * h * d;
    // Allocate on the host.
    float *o_h = (float *)malloc(o_size * sizeof(float));
    // The size in bytes.
    const size_t o_size_in_bytes = get_size_in_bytes(o_size, data_type);
    // Allocate on the device.
    void *o_d = nullptr;
    FMHA_CHECK_CUDA(cudaMalloc(&o_d, o_size_in_bytes));

    // The size in bytes.
    const size_t tmp_size_in_bytes = get_size_in_bytes(o_size, acc_type);
    // Allocate on the device.
    void *tmp_d = nullptr;
    if( data_type != acc_type ) {
        FMHA_CHECK_CUDA(cudaMalloc(&tmp_d, tmp_size_in_bytes));
    }

    // Allocate the reference on the host.
    float *o_ref_h = (float *)malloc(o_size * sizeof(float));

    // The P matrix is stored as one big matrix of size S x B x H x S.
    const size_t p_size = s * b * h * s;
    // The size in bytes.
    const size_t p_size_in_bytes = get_size_in_bytes(p_size, acc_type);
    // Allocate on the device.
    void *p_d = nullptr;
    FMHA_CHECK_CUDA(cudaMalloc(&p_d, p_size_in_bytes));
    
    // Allocate the reference on the host.
    float *p_ref_h = (float *)malloc(p_size * sizeof(float));
#if defined(STORE_P)
    // Allocate on the host.
    float *p_h = (float *)malloc(p_size * sizeof(float));
#endif  // defined(STORE_P)

    // The size in bytes of the S matrix (the data type may be different from P for int8).
    const size_t s_size_in_bytes = get_size_in_bytes(p_size, data_type);
    // Allocate on the device.
    void *s_d = nullptr;
    FMHA_CHECK_CUDA(cudaMalloc(&s_d, s_size_in_bytes));

    // Allocate the reference on the host.
    float *s_ref_h = (float *)malloc(p_size * sizeof(float));

    // Allocate on the host.
    float *s_h = (float *)malloc(p_size * sizeof(float));
    // Make sure we set the seed for reproducible results.
    srand(1234UL);

    // Set the Q, K and V matrices.
    random_init("Q", qkv_h + 0 * d, d, s * b * h, 3 * d, use_1s_q, range_q, scale_q, verbose);
    random_init("K", qkv_h + 1 * d, d, s * b * h, 3 * d, use_1s_k, range_k, scale_k, verbose);
    random_init("V", qkv_h + 2 * d, d, s * b * h, 3 * d, use_1s_v, range_v, scale_v, verbose);
    // iota_init("Q", qkv_h + 0 * d, d, s * b * h, 3 * d, use_1s_q, range_q, scale_q, verbose, true, 0);
    // iota_init("K", qkv_h + 1 * d, d, s * b * h, 3 * d, use_1s_k, range_k, scale_k, verbose, true, 128);
    // iota_init("V", qkv_h + 2 * d, d, s * b * h, 3 * d, use_1s_v, range_v, scale_v, verbose, true, 256);

    // Multi-query or grouped-query attention for reference input
    if( multi_query_attention ) {
        for( size_t sbi = 0; sbi < s * b; sbi++ ) {
            for( size_t hi = 0; hi < h; hi++ ) {
                for( size_t di = 0; di < d; di++ ) {
                    // E.g., h=8, h_kv=4
                    //            hi: 0, 1, 2, 3, 4, 5, 6, 7
                    // hi_kv_scatter: 0, 0, 2, 2, 4, 4, 6, 6
                    const int h_per_group = h / h_kv;
                    const int hi_kv_scatter = (hi / h_per_group) * h_per_group;
                    size_t src_offset = sbi * h * 3 * d + hi_kv_scatter * 3 * d + di; // [sbi, hi_kv_scatter, 0, di]
                    size_t dst_offset = sbi * h * 3 * d + hi * 3 * d + di;            // [sbi, hi, 0, di]

                    // make sure all heads of kv in a group share the same d
                    qkv_h[dst_offset + 1 * d] = qkv_h[src_offset + 1 * d]; // qkv[sbi, hi, 1, di] = qkv[sbi, hi_kv_scatter, 1, di]
                    qkv_h[dst_offset + 2 * d] = qkv_h[src_offset + 2 * d]; // qkv[sbi, hi, 2, di] = qkv[sbi, hi_kv_scatter, 2, di]
                }
            }
        }
    }


    //   WAR fOR MISSING CUBLAS FP8 NN SUPPORT.
    //   Transpose V, so that we can do a TN BMM2, i.e. O = S x V'  instead of O = S x V.
    float *vt_h = (float *)malloc(o_size * sizeof(float));
    void *vt_d = nullptr;
    FMHA_CHECK_CUDA(cudaMalloc(&vt_d, o_size_in_bytes));
    for( size_t it = 0; it < o_size; it++ ) {
        // vt is B x H x D x S
        size_t si =    it % s;
        size_t di =   (it / s) % d;
        size_t hi =  ((it / s) / d) % h;
        size_t bi = (((it / s) / d) / h) % b;
        // qkv is S x B x H x 3 x D
        size_t qkv_idx = si * b  * h  * 3 * d 
                            + bi * h  * 3 * d
                                 + hi * 3 * d
                                      + 2 * d // index V here
                                          + di;
        vt_h[it] = qkv_h[qkv_idx];
    }
    FMHA_CHECK_CUDA(cuda_memcpy_h2d(vt_d, vt_h, o_size, data_type));

    // // DEBUG.
    // float sum = 0.f;
    // for( size_t si = 0; si < s; ++si ) {
    //   float v = qkv_h[si*b*h*3*d + 2*d];
    //   printf("V[%3d]=%8.3f\n", si, v);
    //   sum += v;
    // }
    // printf("Sum of V = %8.3f\n", sum);
    // // END OF DEBUG.

    // Copy from the host to the device.
    FMHA_CHECK_CUDA(cuda_memcpy_h2d(qkv_sbh3d_d, qkv_h, qkv_size, data_type));

    // Create the buffer of mask.
    // if(verbose) {printf("Init .........: mask\n"); }
    // random_init_with_zeroes_or_ones(mask_h, b*s, false, 1.f - dropout, verbose);

    std::vector<uint32_t> seqlens(b, 0);  // randomly draw a batch of sequence lengths >= min_s
    std::transform(seqlens.begin(), seqlens.end(), seqlens.begin(), [=](const uint32_t) {
        if( fix_s ) {
            return std::min(uint32_t(s), min_s);
        }
        if( s == min_s ) {
            return min_s;
        }
        uint32_t s_ = s - min_s + 1;
        uint32_t ret = min_s + (rand() % s_);
        assert(ret <= s);
        return ret;
    });

    // Compute the prefix sum of the sequence lenghts.
    std::vector<int> cu_seqlens(b + 1, 0);
    for( int it = 0; it < b; it++ ) {
        cu_seqlens[it + 1] = cu_seqlens[it] + seqlens[it];
    }
    int total = cu_seqlens.back();
    seqlens.emplace_back(total);

    // transfer to device
    void *cu_seqlens_d;
    FMHA_CHECK_CUDA(cudaMalloc(&cu_seqlens_d, sizeof(int) * cu_seqlens.size()));
    FMHA_CHECK_CUDA(cudaMemcpy(cu_seqlens_d,
                               cu_seqlens.data(),
                               sizeof(int) * cu_seqlens.size(),
                               cudaMemcpyHostToDevice));

    size_t qkv_packed_size = cu_seqlens.back() * h * d * 3;
    size_t qkv_packed_size_in_bytes = get_size_in_bytes(qkv_packed_size, data_type);
    void *qkv_packed_d = nullptr;
    FMHA_CHECK_CUDA(cudaMalloc(&qkv_packed_d, qkv_packed_size_in_bytes));

    // Specify device buffers for multi-query attention or grouped-query attention
    // TODO: Use the same buffer for all cases, and allow to set name to aid tracing/debugging
    // e.g.,
    //   Buffer<float> qkv_buf(size);
    //   if( packed ) { qkv_buf.set_name("QKV_packed[total, h, 3, d]"); }
    //   else { qkv_buf.set_name("QKV_padded[b, s, h, 3, d]"); }
    //   qkv_buf.copy_to_device();
    //   float *qkv_buf_d = qkv_buf.get_device_buf();
    // Or, more aggressively, use torch::Tensor from PyTorch ATen
    size_t mqa_qkv_packed_size = cu_seqlens.back() * (h + 2 * h_kv) * d;
    size_t mqa_qkv_packed_size_in_bytes = get_size_in_bytes(mqa_qkv_packed_size, data_type);
    size_t mqa_qkv_size = b * s * (h + 2 * h_kv) * d; // original padded tensor
    size_t mqa_qkv_size_in_bytes = get_size_in_bytes(mqa_qkv_size, data_type);
    void *mqa_qkv_packed_d = nullptr;
    void *mqa_qkv_d = nullptr;
    FMHA_CHECK_CUDA(cudaMalloc(&mqa_qkv_packed_d, mqa_qkv_packed_size_in_bytes));
    FMHA_CHECK_CUDA(cudaMalloc(&mqa_qkv_d, mqa_qkv_size_in_bytes));

    const size_t o_packed_size = cu_seqlens.back() * h * d;
    // Allocate on the host.
    float *o_packed_h = (float *)malloc(o_packed_size * sizeof(float));
    void *o_packed_d = nullptr;

    size_t o_packed_size_in_bytes = get_size_in_bytes(o_packed_size, data_type);
    FMHA_CHECK_CUDA(cudaMalloc(&o_packed_d, o_packed_size_in_bytes));

    // qkv_packed_h is TotalH3D
    std::vector<float> qkv_packed_h(qkv_packed_size);
    extract_and_transpose_input<float>(qkv_packed_h.data(), qkv_h, seqlens, s, b, h, d, 3, false);
    if( interleaved ) {
        x_vec32(true, qkv_packed_h.data(), h, total, 3);
    }

    // qkv_h is SBH3D
    // qkv_bsh3d_h is BSH3D
    std::vector<float> qkv_bsh3d_h(qkv_size);
    extract_and_transpose_input<float>(qkv_bsh3d_h.data(), qkv_h, seqlens, s, b, h, d, 3, is_s_padded);
    if( interleaved ) {
        x_vec32(true, qkv_bsh3d_h.data(), h, b * h, 3);
    }

    std::vector<float> mqa_qkv_packed_h(mqa_qkv_packed_size);
    std::vector<float> mqa_qkv_h(mqa_qkv_size);
    // from qkv[s, h, 3, d] to mqa_qkv[s, h + 2*h_kv, d]
    // where
    //  Q is qkv[s, h, 0, d],
    //  K is qkv[s, h, 1, d],
    //  V is qkv[s, h, 2, d]
    // and
    //  MQA_Q is mqa_qkv[s, h, [       0 :          h - 1], d],
    //  MQA_K is mqa_qkv[s, h, [       h :   h + h_kv - 1], d],
    //  MQA_V is mqa_qkv[s, h, [h + h_kv : h + 2*h_kv - 1], d]
    for( size_t si = 0; si < cu_seqlens.back(); si++ ) {
        for( size_t hi = 0; hi < h; hi++ ) {
            for( size_t di = 0; di < d; di++ ) {
                // Q: [si, hi, di] <- [si, hi, 0, di]
                mqa_qkv_packed_h[si * (h + 2 * h_kv) * d + hi * d + di] =
                    qkv_packed_h[si * h * 3 * d + hi * 3 * d + 0 * d + di];
                if( hi < h_kv ) {
                    // E.g., h=8, h_kv=4
                    //     src kv id: 0, 0, 1, 1, 2, 2, 3, 3
                    //            hi: 0, 1, 2, 3, 4, 5, 6, 7
                    // hi_kv_scatter: 0, 2, 4, 6, x, x, x, x
                    const int h_per_group = h / h_kv;
                    const int hi_kv_scatter = hi * h_per_group;
                    // K: [si, h + hi, di] <- [si, hi_kv_scatter, 1, di]
                    mqa_qkv_packed_h[si * (h + 2 * h_kv) * d + (h + hi) * d + di] =
                        qkv_packed_h[si * 3 * h * d + hi_kv_scatter * 3 * d + 1 * d + di];
                    // V: [si, h + h_kv + hi, di] <- [si, hi_kv_scatter, 2, di]
                    mqa_qkv_packed_h[si * (h + 2 * h_kv) * d + (h + h_kv + hi) * d + di] =
                        qkv_packed_h[si * 3 * h * d + hi_kv_scatter * 3 * d + 2 * d + di];
                }
            }
        }
    }

    // from qkv_bsh3d_h[b, s, h, 3, d] to mqa_qkv[b, s, h + 2*h_kv, d]
    for( size_t bi = 0; bi < b; bi++ ) {
        int actual_s = seqlens[bi];
        for( size_t si = 0; si < actual_s; si++ ) {
            for( size_t hi = 0; hi < h; hi++ ) {
                for( size_t di = 0; di < d; di++ ) {
                    mqa_qkv_h[bi * s * (h + 2 * h_kv) * d + si * (h + 2 * h_kv) * d + hi * d + di] =
                        qkv_bsh3d_h[bi * s * h * 3 * d + si * h * 3 * d + hi * 3 * d + 0 * d + di];
                    if( hi < h_kv ) {
                        // E.g., h=8, h_kv=4
                        //     src kv id: 0, 0, 1, 1, 2, 2, 3, 3
                        //            hi: 0, 1, 2, 3, 4, 5, 6, 7
                        // hi_kv_scatter: 0, 2, 4, 6, x, x, x, x
                        const int h_per_group = h / h_kv;
                        const int hi_kv_scatter = hi * h_per_group;
                        mqa_qkv_h[bi * s * (h + 2 * h_kv) * d + si * (h + 2 * h_kv) * d +
                                  (h + hi) * d + di] =
                            qkv_bsh3d_h[bi * s * h * 3 * d + si * h * 3 * d +
                                        hi_kv_scatter * 3 * d + 1 * d + di];
                        mqa_qkv_h[bi * s * (h + 2 * h_kv) * d + si * (h + 2 * h_kv) * d +
                                  (h + h_kv + hi) * d + di] =
                            qkv_bsh3d_h[bi * s * h * 3 * d + si * h * 3 * d +
                                        hi_kv_scatter * 3 * d + 2 * d + di];
                    }
                }
            }
        }
    }

    // if( verbose ) {
    //     print_tensor(qkv_packed_h.data() + 0 * d, d, total * h, 3 * d, "Packed Q[bs, h, d]");
    //     print_tensor(qkv_packed_h.data() + 1 * d, d, total * h, 3 * d, "Packed K[bs, h, d]");
    //     print_tensor(qkv_packed_h.data() + 2 * d, d, total * h, 3 * d, "Packed V[bs, h, d]");

    //     print_tensor(mqa_qkv_packed_h.data() + 0 * d,            h * d,    total, (h + 2 * h_kv) * d, "Packed MQA Q[bs, h*d]");
    //     print_tensor(mqa_qkv_packed_h.data() + h * d,            h_kv * d, total, (h + 2 * h_kv) * d, "Packed MQA K[bs, h_kv*d]");
    //     print_tensor(mqa_qkv_packed_h.data() + h * d + h_kv * d, h_kv * d, total, (h + 2 * h_kv) * d, "Packed MQA V[bs, h_kv*d]");

    //     print_tensor(qkv_bsh3d_h.data() + 0 * d, d, b * h * s, 3 * d, "Padded Q[b, s, h, d]");
    //     print_tensor(qkv_bsh3d_h.data() + 1 * d, d, b * h * s, 3 * d, "Padded K[b, s, h, d]");
    //     print_tensor(qkv_bsh3d_h.data() + 2 * d, d, b * h * s, 3 * d, "Padded V[b, s, h, d]");

    //     print_tensor(mqa_qkv_h.data() + 0 * d,            h * d,    b * s, (h + 2 * h_kv) * d, "Padded MQA Q[b, s, h*d]");
    //     print_tensor(mqa_qkv_h.data() + h * d,            h_kv * d, b * s, (h + 2 * h_kv) * d, "Padded MQA K[b, s, h_kv*d]");
    //     print_tensor(mqa_qkv_h.data() + h * d + h_kv * d, h_kv * d, b * s, (h + 2 * h_kv) * d, "Padded MQA V[b, s, h_kv*d]");
    // }

    // Copy packed, padded, mqa packed, mqa padded data buffers
    // TODO: use the same buffer for all cases
    FMHA_CHECK_CUDA(
        cuda_memcpy_h2d(qkv_packed_d, qkv_packed_h.data(), qkv_packed_size, data_type));
    FMHA_CHECK_CUDA(
        cuda_memcpy_h2d(mqa_qkv_packed_d, mqa_qkv_packed_h.data(), mqa_qkv_packed_size, data_type));
    FMHA_CHECK_CUDA(cuda_memcpy_h2d(mqa_qkv_d, mqa_qkv_h.data(), mqa_qkv_size, data_type));
    FMHA_CHECK_CUDA(cuda_memcpy_h2d(qkv_bsh3d_d, qkv_bsh3d_h.data(), qkv_size, data_type));

    for( size_t so = 0; so < s; ++so ) { // s_q
        for( size_t bi = 0; bi < b; ++bi ) {
            int actual_seqlen = seqlens[bi];
            for( size_t si = 0; si < s; ++si ) { // s_kv
                // Are both the query and the key inside the sequence?
                bool valid = (si < actual_seqlen) && (so < actual_seqlen);
                if( attention_mask_type == Attention_mask_type::CAUSAL ||
                    attention_mask_type == Attention_mask_type::LIMITED_LENGTH_CAUSAL ) {
                    valid = valid && (so >= si);
                }
                if( attention_mask_type == Attention_mask_type::LIMITED_LENGTH_CAUSAL ) {
                    valid = valid && (si >= std::max(int(so - max_past_s), 0));
                }
                // The mask is stored as floats.
                mask_h[so * b * s + bi * s + si] = valid ? 1.f : 0.f; // mask dims [s_q, b, s_kv]
            }
        }
    }

    if( verbose ) {
        printf("Seqence lengths (first 10 batches): ");
        for( int bi = 0; bi < seqlens.size() && bi < 10; bi++ ) {
            printf("%d, ", seqlens[bi]);
        }
        printf("\n");
    }

    if( v1 ) {
        assert(!interleaved && "Interleaved not supported in v1");
        assert(mmas_n <= 4 && "Not supported");

        FMHA_CHECK_CUDA(cudaMalloc(&packed_mask_d, packed_mask_size_in_bytes));
        if( sm == 70 ) {
            pack_mask_sm70(packed_mask_h, 
                           mask_h, 
                           s, 
                           b, 
                           mmas_m, 
                           mmas_n, 
                           warps_m, 
                           warps_n, 
                           threads_per_cta);
        } else {
            pack_mask(packed_mask_h, 
                      mask_h, 
                      s, 
                      b, 
                      mmas_m, 
                      mmas_n, 
                      warps_m, 
                      warps_n, 
                      threads_per_cta);
        }

        // Copy the packed mask to the device.
        FMHA_CHECK_CUDA(cudaMemcpy(
            packed_mask_d, packed_mask_h, packed_mask_size_in_bytes, cudaMemcpyHostToDevice));
    }

    // Copy the mask to the device.
    FMHA_CHECK_CUDA(cuda_memcpy_h2d(mask_d, mask_h, mask_size, DATA_TYPE_INT8));

    // non-owning pointer to the IO buffer
    void *qkv_d_view = nullptr;
    void *o_d_view = nullptr;
    int o_view_size = 0;
    if( is_s_padded ) {
        qkv_d_view = multi_query_attention ? mqa_qkv_d : qkv_bsh3d_d;
        o_d_view = o_d;
        o_view_size = o_size;
    } else {
        qkv_d_view = multi_query_attention ? mqa_qkv_packed_d : qkv_packed_d;
        o_d_view = o_packed_d;
        o_view_size = o_packed_size;
    }

    // Set the params.
    bert::Fused_multihead_attention_params_v1 params_v1;
    set_params(params_v1,
               data_type,
               acc_type,
               b,
               s,
               h,
               d,
               mmas_m * threads_per_cta,
               qkv_sbh3d_d,
               packed_mask_d,
               o_d,
               p_d,
               s_d,
               scale_bmm1,
               scale_softmax,
               scale_bmm2,
               has_alibi);

    bert::Fused_multihead_attention_params_v2 params_v2;
    set_params(params_v2,
               data_type,
               acc_type,
               b,
               s,
               h,
               d,
               total,
               max_past_s,
               qkv_d_view,
               cu_seqlens_d,
               o_d_view,
               p_d,
               s_d,
               scale_bmm1,
               scale_softmax,
               scale_bmm2,
               use_int8_scale_max,
               interleaved,
               is_s_padded,
               has_alibi,
               h_kv);

    // seqlens is needed to set TMA desc on the host. 
    launch_params.seqlens = seqlens.data();

    // Allocate barriers and locks.
    void *counters_d = nullptr;
    if( ctas_per_head > 1 ) {
        size_t sz = heads_per_wave * sizeof(int);
        FMHA_CHECK_CUDA(cudaMalloc((void**) &counters_d, 3*sz));
    }

    // Allocate scratch storage for softmax.
    void *max_scratch_d = nullptr, *sum_scratch_d = nullptr;
    if( ctas_per_head > 1 ) {
        size_t sz = heads_per_wave * ctas_per_head * threads_per_cta * sizeof(float);
        FMHA_CHECK_CUDA(cudaMalloc((void**) &max_scratch_d, sz));
        FMHA_CHECK_CUDA(cudaMalloc((void**) &sum_scratch_d, sz));
    }

    // Allocate temporary storage for the parallel reduction.
    void *o_scratch_d = nullptr;
    if( ctas_per_head > 1 && data_type != DATA_TYPE_FP16 ) {
        size_t sz = heads_per_wave * threads_per_cta * MAX_STGS_PER_LOOP * sizeof(uint4);
        FMHA_CHECK_CUDA(cudaMalloc((void**) &o_scratch_d, sz));
    }

    // The number of heads computed per wave.
    params_v1.heads_per_wave = heads_per_wave;
    params_v2.heads_per_wave = heads_per_wave;

    // Barriers for the global sync in the multi-CTA kernel(s).
    params_v1.counters     = (int*) counters_d + 0*heads_per_wave;
    params_v2.counters     = (int*) counters_d + 0*heads_per_wave;
    params_v1.max_barriers = (int*) counters_d + 0*heads_per_wave;
    params_v2.max_barriers = (int*) counters_d + 0*heads_per_wave;
    params_v1.sum_barriers = (int*) counters_d + 1*heads_per_wave;
    params_v2.sum_barriers = (int*) counters_d + 1*heads_per_wave;
    params_v1.locks        = (int*) counters_d + 2*heads_per_wave;
    params_v2.locks        = (int*) counters_d + 2*heads_per_wave;

    // Scratch storage for softmax.
    params_v1.max_scratch_ptr = (float*) max_scratch_d;
    params_v2.max_scratch_ptr = (float*) max_scratch_d;
    params_v1.sum_scratch_ptr = (float*) sum_scratch_d;
    params_v2.sum_scratch_ptr = (float*) sum_scratch_d;

    // Scratch storage for output.
    params_v1.o_scratch_ptr = (int*) o_scratch_d;
    params_v2.o_scratch_ptr = (int*) o_scratch_d;

#if defined(DEBUG_HAS_PRINT_BUFFER)
    auto & params = params_v2;
    constexpr size_t bytes = 32 * 1024;
    void *print_ptr = nullptr;
    FMHA_CHECK_CUDA(cudaMalloc(&params.print_ptr, bytes));
    std::vector<float> print_buffer(bytes / sizeof(float));
#endif

    // Run a few warm-up kernels.
    for( int ii = 0; ii < warm_up_runs; ++ii ) {
        if( v1 ) {
            run_fmha_v1(params_v1, launch_params, data_type, sm, 0);
        } else {
            run_fmha_v2(params_v2, 
                        launch_params, 
                        data_type, 
                        sm,
                        0);
        }
    }
    FMHA_CHECK_CUDA(cudaPeekAtLastError());

    float non_fused_elapsed = INFINITY;
    if( !skip_checks ) {
        // Run cuBLAS.
        
        RefBMM bmm1(data_type_to_cuda(data_type), // a
                    data_type_to_cuda(data_type), // b
                    data_type_to_cuda(acc_type), // d
                    data_type_to_cublas(acc_type), //compute
                    data_type_to_cuda(acc_type), // scale
                    false, // Q
                    true, // K'
                    s, // m
                    s, // n 
                    d, // k
                    b * h * 3 * d, // ld Q
                    b * h * 3 * d, // ld K
                    b * h * s, // ld P
                    3 * d, // stride Q
                    3 * d, // stride K
                    s, // stride P
                    b * h // batch count
                   );

        /*
        RefBMM bmm2(data_type_to_cuda(data_type), // a
                    data_type_to_cuda(data_type), // b
                    data_type_to_cuda(acc_type), // d
                    data_type_to_cublas(acc_type), //compute
                    data_type_to_cuda(acc_type), // scale
                    false, // S
                    false, // V
                    s, // m
                    d, // n 
                    s, // k
                    b * h * s, // ld S
                    b * h * 3 * d, // ld V
                    b * h * d, // ld O
                    s, // stride S
                    3 * d, // stride V
                    d, // stride O
                    b * h // batch count
                   );
        */

        // WAR fOR MISSING CUBLAS FP8 NN SUPPORT.
        // Transpose V, so that we can do a TN BMM2, i.e. O = S x V'  instead of O = S x V.
        RefBMM bmm2(data_type_to_cuda(data_type), // a
                    data_type_to_cuda(data_type), // b
                    data_type_to_cuda(acc_type), // d
                    data_type_to_cublas(acc_type), //compute
                    data_type_to_cuda(acc_type), // scale
                    false, // S
                    true, // V'
                    s, // m
                    d, // n 
                    s, // k
                    b * h * s, // ld S
                    s, // ld V
                    b * h * d, // ld O
                    s, // stride S
                    s * d, // stride V
                    d, // stride O
                    b * h // batch count
                   );
 
        timer.start();
        ground_truth(bmm1,
                     bmm2,
                     data_type,
                     acc_type,
                     scale_bmm1,
                     scale_softmax,
                     scale_bmm2,
                     qkv_sbh3d_d,
                     vt_d, // WAR pass in V'
                     mask_d,
                     p_d,
                     s_d,
                     tmp_d,
                     o_d,
                     b,
                     s,
                     h,
                     d,
                     runs,
                     warps_m,
                     warps_n,
                     has_alibi);
        timer.stop();
        FMHA_CHECK_CUDA(cudaPeekAtLastError());
        FMHA_CHECK_CUDA(cudaDeviceSynchronize());
        non_fused_elapsed = timer.millis();
        
#if defined(STORE_P)
        FMHA_CHECK_CUDA(cuda_memcpy_d2h(p_ref_h, p_d, p_size, acc_type));
#endif  // defined(STORE_P)

#if defined(STORE_S)
        FMHA_CHECK_CUDA(cuda_memcpy_d2h(s_ref_h, s_d, p_size, data_type));
#endif  // defined(STORE_S)

        // Read the results.
        FMHA_CHECK_CUDA(cuda_memcpy_d2h(o_ref_h, o_d, o_size, data_type));
    }

    // Fill-in p/s/o with garbage data.
    // WAR: if sequence is padded, we zero-fill the output buffer as kernel will not write to the
    // padded area, and the host expects to check the padded area
    FMHA_CHECK_CUDA(cudaMemset(p_d, 0xdc, p_size_in_bytes));
    FMHA_CHECK_CUDA(cudaMemset(s_d, 0xdc, s_size_in_bytes));
    FMHA_CHECK_CUDA(cudaMemset(o_d, 0x00, o_size_in_bytes));

    // Run the kernel.
    timer.start();
    for( int ii = 0; ii < runs; ++ii ) {
        if( v1 ) {
            run_fmha_v1(params_v1, launch_params, data_type, sm, 0);
        } else {
            run_fmha_v2(params_v2, launch_params, data_type, sm, 0);
        }
    }
    timer.stop();
    FMHA_CHECK_CUDA(cudaPeekAtLastError());

    FMHA_CHECK_CUDA(cudaDeviceSynchronize());
    float fused_elapsed = timer.millis();

#if defined(STORE_P)
    FMHA_CHECK_CUDA(cuda_memcpy_d2h(p_h, p_d, p_size, acc_type));
    printf("\nChecking .....: P = norm * K^T * Q\n");

    // DEBUG.
    printf("seqlens[0]=%d\n", seqlens[0]);
    // END OF DEBUG.

    // Clear the invalid region of P.
    set_mat<float>(p_ref_h, seqlens, s, b, h, s, 0.f, true);
    set_mat<float>(p_h,     seqlens, s, b, h, s, 0.f, true);

    // Do the check.
    check_results(p_h, p_ref_h, s, s * b * h, s, 0.f, true, true);
#endif  // defined(STORE_P)


#if defined(STORE_S)
    FMHA_CHECK_CUDA(cuda_memcpy_d2h(s_h, s_d, p_size, data_type));
    printf("\nChecking .....: S = softmax(P)\n");
#if defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)
    float softmax_epsilon = data_type == DATA_TYPE_FP16 ? 1e-3f : 0.f;
#else
    float softmax_epsilon = 1.e-3f;
#endif  // defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)

    // Clear the invalid region of S.
    set_mat<float>(s_ref_h, seqlens, s, b, h, s, 0.f);
    set_mat<float>(s_h,     seqlens, s, b, h, s, 0.f);

    // Do the check.
    check_results(s_h, s_ref_h, s, s * b * h, s, softmax_epsilon, true, true);
#endif  // defined(STORE_S)

    // Check the final results.
    int status = -1;
    if( skip_checks ) {
        status = 0;
        printf("\n");
        print_results(true, false);
    } else {
        if( v1 ) {
            FMHA_CHECK_CUDA(cuda_memcpy_d2h(o_h, o_d, o_size, data_type));
            status = check_results(o_h, o_ref_h, d, s * b * h, d, epsilon, verbose, true);
        } else {
            std::vector<float> o_ref_trans_h(o_size);

            FMHA_CHECK_CUDA(cuda_memcpy_d2h(o_h, o_d_view, o_view_size, data_type));

            if( interleaved ) {
                // revert batch-interleaved format: 3 x h/32 x total x d x 32 => total x
                // h x 3 x d
                x_vec32(false, o_h, h, is_s_padded ? b * h : total, 1);
            }

            extract_and_transpose_output<float>(
                o_ref_trans_h.data(), o_ref_h, seqlens, s, b, h, d, is_s_padded);

            if( verbose ) {
                printf("\nChecking .....: O = V * S\n");
            }
            status = check_results(
                o_h, o_ref_trans_h.data(),
                d,
                is_s_padded ? s * b * h : cu_seqlens.back() * h,
                d,
                epsilon, verbose, true);
            // if( verbose ) {
            //     print_tensor(o_ref_trans_h.data(), d, b * h * s, d, "Host O[b, s, h, d]");
            //     print_tensor(o_h, d, is_s_padded ? s * b * h : total * h, d, "Device O[b, s, h, d]");
            // }
        }
        if( status != 0 ) {  // if there was an error, print the config of the run
            printf("v1=%d il=%d s=%lu b=%lu h=%lu d=%lu dtype=%s\n",
                    v1,
                    interleaved,
                    s,
                    b,
                    h,
                    d,
                    data_type_to_name(data_type).c_str()
                    );
        }

        if( !verbose ) {  // this just prints the SUCCESS/ERROR line
            print_results(true, true, status == 0);
        }
    }

    // accounts for tensor core flops only; excludes flops spent in softmax
    size_t total_flops = 0;
    // remove last seqlen(total_seqlen)
    seqlens.pop_back();
    for (auto& s_ : seqlens) {
        total_flops += 2ull * h * (s_ * s_ * d + s_ * d * s_); // 1st BMM + 2nd BMM
    }
    size_t total_bytes = o_packed_size_in_bytes + qkv_packed_size_in_bytes;
    if( verbose ) {
        // Runtimes.
        printf("\n");
        if( !skip_checks ) {
            printf("Non-fused time: %.6f ms\n", non_fused_elapsed / float(runs));
        }
        printf("Fused time ...: %.6f us\n", fused_elapsed * 1000 / float(runs));
        printf("Tensor core ..: %.2f Tflop/s\n", total_flops / (fused_elapsed / float(runs) / 1e-9));
        printf("Bandwidth ....: %.2f GB/s\n", total_bytes / (fused_elapsed / float(runs) / 1e-6));
        if( !skip_checks ) {
            printf("Ratio ........: %.2fx\n", non_fused_elapsed / fused_elapsed);
        }
    } else {
        printf("Elapsed ......: %.6f us (%.2fx), %.2f Tflop/s, %.2f GB/s\n",
               fused_elapsed * 1000 / float(runs),
               non_fused_elapsed / fused_elapsed,
               total_flops / (fused_elapsed / float(runs) / 1e-9),
               total_bytes / (fused_elapsed / float(runs) / 1e-6));
    }
#if defined(DEBUG_HAS_PRINT_BUFFER)
    FMHA_CHECK_CUDA(cuda_memcpy_d2h(print_buffer.data(), params.print_ptr, print_buffer.size(), DATA_TYPE_FP32));

    printf("\n====================\n");
    for( int it = 0; it < 16; it++ ) {
        printf("% .4f ", print_buffer[it] );
    }
    printf("\n====================\n");

    FMHA_CHECK_CUDA(cudaFree(params.print_ptr));

#endif

    // Release memory.
    FMHA_CHECK_CUDA(cudaFree(qkv_sbh3d_d));
    FMHA_CHECK_CUDA(cudaFree(qkv_packed_d));
    FMHA_CHECK_CUDA(cudaFree(mqa_qkv_d));
    FMHA_CHECK_CUDA(cudaFree(mqa_qkv_packed_d));
    FMHA_CHECK_CUDA(cudaFree(qkv_bsh3d_d));
    FMHA_CHECK_CUDA(cudaFree(mask_d));
    FMHA_CHECK_CUDA(cudaFree(packed_mask_d));
    FMHA_CHECK_CUDA(cudaFree(p_d));
    FMHA_CHECK_CUDA(cudaFree(s_d));
    FMHA_CHECK_CUDA(cudaFree(o_d));
    FMHA_CHECK_CUDA(cudaFree(tmp_d));
    FMHA_CHECK_CUDA(cudaFree(cu_seqlens_d));
    FMHA_CHECK_CUDA(cudaFree(max_scratch_d));
    FMHA_CHECK_CUDA(cudaFree(sum_scratch_d));
    FMHA_CHECK_CUDA(cudaFree(o_scratch_d));
    FMHA_CHECK_CUDA(cudaFree(counters_d));

    free(qkv_h);
    free(mask_h);
    free(packed_mask_h);
    free(s_h);
    free(o_h);
    free(o_ref_h);
    
    free(p_ref_h);
#if defined(STORE_P)
    free(p_h);
#endif  // defined(STORE_P)
    free(s_ref_h);

    return status;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
