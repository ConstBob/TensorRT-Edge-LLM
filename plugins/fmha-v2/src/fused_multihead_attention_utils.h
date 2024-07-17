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
#pragma once

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <fmha/numeric_types.h>
#include <string>
#include <cfloat>

////////////////////////////////////////////////////////////////////////////////////////////////////

#define FMHA_CHECK_CUDA( call )                                                                    \
    do {                                                                                           \
        cudaError_t status_ = call;                                                                \
        if( status_ != cudaSuccess ) {                                                             \
            fprintf( stderr,                                                                       \
                     "CUDA error (%s:%d): %s\n",                                                   \
                     __FILE__,                                                                     \
                     __LINE__,                                                                     \
                     cudaGetErrorString( status_ ) );                                              \
            exit( 1 );                                                                             \
        }                                                                                          \
    } while( 0 )

////////////////////////////////////////////////////////////////////////////////////////////////////

static const char *_cudaGetErrorEnum(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "<unknown>";
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#define FMHA_CHECK_CUBLAS(call)                                                                                                \
    do {                                                                                                                       \
        cublasStatus_t status_ = call;                                                                                         \
        if( status_ != CUBLAS_STATUS_SUCCESS ) {                                                                               \
            fprintf(stderr, "CUBLAS error %d (%s:%d): %s\n", (int)status_, __FILE__, __LINE__, _cudaGetErrorEnum(status_));    \
            exit(1);                                                                                                           \
        }                                                                                                                      \
    } while( 0 )

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void random_init( const char *name,
                                float *dst,
                                size_t m,
                                size_t n,
                                size_t ld,
                                bool use_1s,
                                int range,
                                float scale,
                                bool verbose ) {
    if( verbose ) {
        printf( "Init .........: %s\n", name );
        printf( "Use 1s .......: %s\n", use_1s ? "true" : "false" );
        printf( "Address ......: 0x%016lx\n", (size_t)dst );
        printf( "Range ........: %d\n", range );
        printf( "Scale ........: %.3f\n", scale );
        printf( "Values .......: " );
    }
    for( size_t ni = 0; ni < n; ++ni ) {
        for( size_t mi = 0; mi < m; ++mi ) {
            float x = 1.f;
            if( !use_1s ) {
                x = (float)( rand() % range - range / 2 ) * scale;
            }
            if( verbose && ni * m + mi < 8 ) {
                printf( "%.3f ", x );
            }
            dst[ni * ld + mi] = x;
        }
    }
    if( verbose ) {
        printf( "...\n\n" );
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Sequentially increasing values in either m or n dimension
static inline void iota_init( const char *name,
                              float *dst,
                              size_t m,
                              size_t n,
                              size_t ld,
                              bool use_1s,
                              int range,   // ignore
                              float scale, // ignore
                              bool verbose,
                              bool iota_m = true ) {
    if( verbose ) {
        printf( "Init .........: %s\n", name );
        printf( "Use 1s .......: %s\n", use_1s ? "true" : "false" );
        printf( "Address ......: 0x%016lx\n", (size_t)dst );
        printf( "Values .......: \n" );
    }
    for( size_t ni = 0; ni < n; ++ni ) {
        if( verbose && ni < 32 ) printf("ni %zd: ", ni);
        for( size_t mi = 0; mi < m; ++mi ) {
            float x = iota_m ? mi : ni;
            x = use_1s ? 1.f : x;
            if( verbose && ni < 32 && mi < 16 ) {
                printf( "%2.0f ",  x);
            }
            dst[ni * ld + mi] = x;
        }
        if( verbose && ni < 32 ) printf("\n");
    }
    if( verbose ) {
        printf( "...\n\n" );
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void
random_init_with_zeroes_or_ones( float *dst, size_t n, bool use_1s, float prob_1s, bool verbose ) {
    if( verbose ) {
        printf( "Use 1s .......: %s\n", use_1s ? "true" : "false" );
        printf( "Address ......: 0x%016lx\n", (size_t)dst );
        printf( "Prob 1s ......: %.3f\n", prob_1s );
        printf( "Values .......: " );
    }
    for( size_t ni = 0; ni < n; ++ni ) {
        float x = 1.f;
        if( !use_1s ) {
            x = ( (double)rand() / (double)RAND_MAX ) < (double)prob_1s ? 1.f : 0.f;
        }
        if( verbose && ni < 8 ) {
            printf( "%.3f ", x );
        }
        dst[ni] = x;
    }
    if( verbose ) {
        printf( "...\n" );
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

enum Data_type { 
    DATA_TYPE_FP16=0,
    DATA_TYPE_FP32=1, 
    DATA_TYPE_INT32=2, 
    DATA_TYPE_INT8=3,
    DATA_TYPE_BF16=4,
    DATA_TYPE_E4M3=5,
    DATA_TYPE_E5M2 =6
};

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline size_t get_size_in_bytes( size_t n, Data_type dtype ) {
    switch( dtype ) {
    case DATA_TYPE_FP32:
        return n * 4;
    case DATA_TYPE_FP16:
        return n * 2;
    case DATA_TYPE_INT32:
        return n * 4;
    case DATA_TYPE_INT8:
        return n;
    case DATA_TYPE_BF16:
        return n * 2;
    case DATA_TYPE_E4M3:
        return n;
    case DATA_TYPE_E5M2:
        return n;
    default:
        assert( false );
        return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void set_alpha( uint32_t &alpha, float norm, Data_type dtype ) {
    if( dtype == DATA_TYPE_FP16 ) {
        half x = __float2half_rn( norm );
        uint16_t h = reinterpret_cast<const uint16_t &>( x );
        ushort2 h2 = { h, h };
        alpha = reinterpret_cast<const uint32_t &>( h2 );
    } else if( dtype == DATA_TYPE_FP32 ) {
        alpha = reinterpret_cast<const uint32_t &>( norm );
    } else if( dtype == DATA_TYPE_INT32 ) {
        int32_t inorm = static_cast<int32_t>( norm );
        alpha = reinterpret_cast<const uint32_t &>( inorm );
    } else if( dtype == DATA_TYPE_BF16 ) {
        //TODO HACK!! BF16 Outputs are computed in FP32 for FP8.
        //This is because cublas does not allow current FP32 output.
        // alpha = reinterpret_cast<const uint32_t &>( norm );
        __nv_bfloat16 x = __float2bfloat16( norm );
        uint16_t h = reinterpret_cast<const uint16_t &>( x );
        ushort2 h2 = { h, h };
        alpha = reinterpret_cast<const uint32_t &>( h2 );
    } else {
        assert( false );
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dst, typename Src>
static inline void expand_and_transpose_input( void *dst_,
                                               void *src_,
                                               const std::vector<uint32_t> &seqlens,
                                               const int s,
                                               const int b,
                                               const int h,
                                               const int d ) {
    // input comes in sxbxhx3xd
    // output will be b tensors of size s_ixhx3xd
    Dst *dst = static_cast<Dst *>( dst_ );
    Src *src = static_cast<Src *>( src_ );
    for( int bi = 0; bi < b; bi++ ) {
        int s_ = seqlens[bi];
        for( int si = 0; si < s_; si++ ) {
            for( int hi = 0; hi < h; hi++ ) {
                for( int ti = 0; ti < 3; ti++ ) {
                    for( int di = 0; di < d; di++ ) {
                        int out_idx =
                            si * b * h * 3 * d + bi * h * 3 * d + hi * 3 * d + ti * d + di;
                        dst[out_idx] = *src;
                        *src++;
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
static inline void extract_and_transpose_input( void *dst_,
                                                void *src_,
                                                const std::vector<uint32_t> &seqlens,
                                                const int s,
                                                const int b,
                                                const int h,
                                                const int d,
                                                const int t,
                                                const bool s_padded = false ) {
    // input comes in sxbxhxtxd
    // output will be b tensors of size s_ixhxtxd
    T *dst = static_cast<T *>( dst_ );
    T *src = static_cast<T *>( src_ );
    for( int bi = 0; bi < b; bi++ ) {
        int s_ = s_padded ? s : seqlens[bi];
        for( int si = 0; si < s_; si++ ) {
            for( int hi = 0; hi < h; hi++ ) {
                for( int ti = 0; ti < t; ti++ ) {
                    for( int di = 0; di < d; di++ ) {
                        int in_idx = si * b * h * t * d + bi * h * t * d + hi * t * d + ti * d + di;
                        dst[0] = src[in_idx];
                        dst++;
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T >
static inline void set_mat(void *mat_,
                           const std::vector<uint32_t> &seqlens,
                           const int s,
                           const int b,
                           const int h,
                           const int d,
                           const T val,
                           bool inner = false) {
    assert((int64_t) (s*b*h*d) == (int64_t) s*b*h*d);

    // s x b x h x d
    T *mat = static_cast<T *>( mat_ );
    for( int si = 0; si < s; si++ ) {
        for( int bi = 0; bi < b; bi++ ) {
            for( int hi = 0; hi < h; hi++ ) {
                for( int di = 0; di < d; di++ ) {
                    int idx = si * b * h * d + bi * h * d + hi * d + di;
                    if( si >= seqlens[bi] || (inner && di >= seqlens[bi]) ) {
                        mat[idx] = val;
                    }
                }
            }
        }
    }
}

template< typename T >
static inline void set_mat(void *mat_,
                           const std::vector<uint32_t> &seqlens_q,
                           const std::vector<uint32_t> &seqlens_kv,
                           const int s,
                           const int b,
                           const int h,
                           const int d,
                           const T val,
                           bool inner = false) {
    assert((int64_t) (s*b*h*d) == (int64_t) s*b*h*d);

    // s x b x h x d
    T *mat = static_cast<T *>( mat_ );
    for( int si = 0; si < s; si++ ) {
        for( int bi = 0; bi < b; bi++ ) {
            for( int hi = 0; hi < h; hi++ ) {
                for( int di = 0; di < d; di++ ) {
                    int idx = si * b * h * d + bi * h * d + hi * d + di;
                    if( si >= seqlens_q[bi] || (inner && di >= seqlens_kv[bi]) ) {
                        mat[idx] = val;
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
static inline void extract_and_transpose_output( void *dst_,
                                                 void *src_,
                                                 const std::vector<uint32_t> &seqlens,
                                                 const int s,
                                                 const int b,
                                                 const int h,
                                                 const int d,
                                                 const bool s_padded = false ) {
    T *dst = static_cast<T *>( dst_ );
    T *src = static_cast<T *>( src_ );
    for( int bi = 0; bi < b; bi++ ) {
        int s_ = s_padded ? s : seqlens[bi];
        for( int si = 0; si < s_; si++ ) {
            for( int hi = 0; hi < h; hi++ ) {
                for( int di = 0; di < d; di++ ) {
                    int in_idx = si * b * h * d + bi * h * d + hi * d + di;
                    *dst = src[in_idx];
                    dst++;
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
static inline void expand_and_transpose_output( void *dst_,
                                                void *src_,
                                                const std::vector<uint32_t> &seqlens,
                                                const int s,
                                                const int b,
                                                const int h,
                                                const int d ) {
    T *dst = static_cast<T *>( dst_ );
    T *src = static_cast<T *>( src_ );
    for( int bi = 0; bi < b; bi++ ) {
        int s_ = seqlens[bi];
        for( int si = 0; si < s_; si++ ) {
            for( int hi = 0; hi < h; hi++ ) {
                for( int di = 0; di < d; di++ ) {
                    int out_idx = si * b * h * d + bi * h * d + hi * d + di;
                    dst[out_idx] = *src;
                    src++;
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
int eval( T *ref,
          T *test,
          const std::vector<uint32_t> &seqlens,
          const int B,
          const int S,
          const int N,
          const int H,
          bool verbose,
          const int print_n = 0 ) {

    size_t errors = 0;
    for( int b = 0; b < B; b++ ) {
        int actual_seqlen = seqlens[b];
        for( int s = 0; s < actual_seqlen; s++ ) {
            for( int n = 0; n < N; n++ ) {
                for( int h = 0; h < H; h++ ) {
                    // int it = b * S * N * H + n * H + s * N * H + h;
                    int it = s * B * N * H + n * H + b * N * H + h;

                    int x = ref[it];
                    int y = test[it];
                    if( errors < print_n && x != y ) {
                        printf( "%6d: %d, %d [%d,%d,%d,%d]\n", it, x, y, b, s, n, h );
                    }
                    errors += ( x != y );
                }
            }
        }
    }

    if( verbose ) {
        printf("Size .........: %d\n", B * S * N * H); 
        printf("Errors .......: %lu\n", errors);
    }
    return errors > 0 ? 1 : 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void x_vec32( const bool to, T *src_dst, int h, int total, int mats, int d = 64 ) {
    // to=true:  total x h x mats x d => mats x h x (d/32) x total x 32
    // to=false: mats x h x (d/32) x total x 32 => total x h x mats x d
    std::vector<T> tmp( total * h * mats * d );

    int slices = d / 32;
    for( int si = 0; si < total; si++ ) {
        for( int hi = 0; hi < h; hi++ ) {
            for( int mi = 0; mi < mats; mi++ ) {
                for( int di = 0; di < d; di++ ) {
                    int slice = di / 32;
                    int ii = di % 32;
                    int src_idx = si * h * mats * d + hi * mats * d + mi * d + di;
                    int dst_idx = mi * h * slices * total * 32 + hi * slices * total * 32 +
                                  slice * total * 32 + si * 32 + ii;
                    if( to ) {
                        tmp[dst_idx] = src_dst[src_idx];
                    } else {  // from
                        tmp[src_idx] = src_dst[dst_idx];
                    }
                }
            }
        }
    }
    for( auto x : tmp ) {
        *src_dst = x;
        src_dst++;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

struct CudaDevice {

    CudaDevice() {
        FMHA_CHECK_CUDA( cudaGetDeviceProperties( &props, 0 ) );
        sm = props.major * 10 + props.minor;
    }

    ~CudaDevice() {
        FMHA_CHECK_CUDA( cudaDeviceReset() );
    }

    cudaDeviceProp props;
    int sm;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct GpuTimer {

    GpuTimer() {
        FMHA_CHECK_CUDA( cudaEventCreate( &begin ) );
        FMHA_CHECK_CUDA( cudaEventCreate( &end ) );
    }

    ~GpuTimer() {
        FMHA_CHECK_CUDA( cudaEventDestroy( begin ) );
        FMHA_CHECK_CUDA( cudaEventDestroy( end ) );
    }

    inline void start() {
        FMHA_CHECK_CUDA( cudaEventRecord( begin ) );
    }
    inline void stop() {
        FMHA_CHECK_CUDA( cudaEventRecord( end ) );
    }

    inline float millis() {
        float ms = 0;
        FMHA_CHECK_CUDA( cudaEventElapsedTime( &ms, begin, end ) );
        return ms;
    }

    cudaEvent_t begin, end;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> struct dvec {
    T *data;
    size_t size;
    size_t bytes;

    dvec( size_t sz ) : size( sz ), bytes( sizeof( T ) * sz ) {
        FMHA_CHECK_CUDA( cudaMalloc( &data, bytes ) );
    }

    ~dvec() {
        FMHA_CHECK_CUDA( cudaFree( data ) );
    }

    template <typename U> U *get() {
        return reinterpret_cast<U *>( data );
    }

    void fill( const T val ) {
        FMHA_CHECK_CUDA( cudaMemset( data, val, bytes ) );
    }

    void zeros() {
        fill( T( 0 ) );
    }

    void h2d( const std::vector<T> &v ) {
        assert( v.size() == size );
        FMHA_CHECK_CUDA( cudaMemcpy( data, v.data(), bytes, cudaMemcpyHostToDevice ) );
    }

    void d2h( std::vector<T> &v ) {
        assert( v.size() == size );
        FMHA_CHECK_CUDA( cudaMemcpy( v.data(), data, bytes, cudaMemcpyDeviceToHost ) );
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void pack_mask(uint32_t *packed_mask_h,
                             const float *mask_h,
                             const size_t s,
                             const size_t b,
                             const size_t mmas_m,
                             const size_t mmas_n,
                             const size_t warps_m,
                             const size_t warps_n,
                             const size_t threads_per_cta) {

    // Generate the packed mask. We use a gather approach.
    for( size_t bi = 0; bi < b; ++bi ) {
        for( size_t mi = 0; mi < mmas_m; ++mi ) {
            for( size_t tidx = 0; tidx < threads_per_cta; ++tidx ) {

                // The warp position.
                size_t warp = tidx / 32;
                size_t lane = tidx % 32;

                // The warp index.
                size_t warp_m = warp % warps_m;
                size_t warp_n = warp / warps_m;

                // The row/col of the 1st element for that MMA.
                size_t row = warp_m * 16 + lane / 4;
                size_t col = warp_n * 16 + lane % 4 * 2;

                // The offset to the 1st element computed by that thread in the mask.
                size_t offset = ( mi * warps_m * 16 + row ) * b * s + bi * s + col;

                // The mask for each row of MMAs.
                uint32_t mask = 0u;

                // Iterate over the items.
                for( size_t ni = 0; ni < mmas_n; ++ni, offset += 16 * warps_n ) {

                    mask |= ( mask_h[offset + 0 * b * s + 0] == 1.f ? 1u : 0u ) << ( 8 * ni + 0 );
                    mask |= ( mask_h[offset + 0 * b * s + 1] == 1.f ? 1u : 0u ) << ( 8 * ni + 1 );
                    mask |= ( mask_h[offset + 8 * b * s + 0] == 1.f ? 1u : 0u ) << ( 8 * ni + 2 );
                    mask |= ( mask_h[offset + 8 * b * s + 1] == 1.f ? 1u : 0u ) << ( 8 * ni + 3 );
                    mask |= ( mask_h[offset + 0 * b * s + 8] == 1.f ? 1u : 0u ) << ( 8 * ni + 4 );
                    mask |= ( mask_h[offset + 0 * b * s + 9] == 1.f ? 1u : 0u ) << ( 8 * ni + 5 );
                    mask |= ( mask_h[offset + 8 * b * s + 8] == 1.f ? 1u : 0u ) << ( 8 * ni + 6 );
                    mask |= ( mask_h[offset + 8 * b * s + 9] == 1.f ? 1u : 0u ) << ( 8 * ni + 7 );
                }

                // Store the mask.
                packed_mask_h[bi * mmas_m * threads_per_cta + mi * threads_per_cta + tidx] = mask;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void pack_mask_sm70(uint32_t *packed_mask_h,
                                  const float *mask_h,
                                  const size_t s,
                                  const size_t b,
                                  const size_t mmas_m,
                                  const size_t mmas_n,
                                  const size_t warps_m,
                                  const size_t warps_n,
                                  const size_t threads_per_cta) {

    // Generate the packed mask. We use a gather approach.
    for( size_t bi = 0; bi < b; ++bi ) {
        for( size_t mi = 0; mi < mmas_m; ++mi ) {
            for( size_t tidx = 0; tidx < threads_per_cta; ++tidx ) {

                // The warp position.
                size_t warp = tidx / 32;
                size_t lane = tidx % 32;

                // The warp index.
                size_t warp_m = warp % warps_m;
                size_t warp_n = warp / warps_m;

                // The row/col of the 1st element for that MMA.
                size_t row = warp_m * 16 + (lane & 0x10) / 2 + (lane & 0x07);
                size_t col = warp_n * 16 + (lane & 0x08) / 2;

                // The offset to the 1st element computed by that thread in the mask.
                size_t offset = (mi * warps_m * 16 + row) * b * s + bi * s + col;

                // The mask for each row of MMAs.
                uint32_t mask = 0u;

                // Iterate over the items.
                for( size_t ni = 0; ni < mmas_n; ++ni, offset += 16 * warps_n ) {

                    mask |= (mask_h[offset +  0] == 1.f ? 1u : 0u) << (8 * ni + 0);
                    mask |= (mask_h[offset +  1] == 1.f ? 1u : 0u) << (8 * ni + 1);
                    mask |= (mask_h[offset +  2] == 1.f ? 1u : 0u) << (8 * ni + 2);
                    mask |= (mask_h[offset +  3] == 1.f ? 1u : 0u) << (8 * ni + 3);
                    mask |= (mask_h[offset +  8] == 1.f ? 1u : 0u) << (8 * ni + 4);
                    mask |= (mask_h[offset +  9] == 1.f ? 1u : 0u) << (8 * ni + 5);
                    mask |= (mask_h[offset + 10] == 1.f ? 1u : 0u) << (8 * ni + 6);
                    mask |= (mask_h[offset + 11] == 1.f ? 1u : 0u) << (8 * ni + 7);
                }

                // Store the mask.
                packed_mask_h[bi * mmas_m * threads_per_cta + mi * threads_per_cta + tidx] = mask;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

struct RefBMM {

    RefBMM(cudaDataType_t type_A,
           cudaDataType_t type_B,
           cudaDataType_t type_D,
           cublasComputeType_t type_compute,
           cudaDataType_t type_scale,  // type for alpha/beta
           bool A_transposed,
           bool B_transposed,
           size_t m,
           size_t n,
           size_t k,
           size_t ldA = 0,
           size_t ldB = 0,
           size_t ldD = 0,
           size_t strideA = 0,
           size_t strideB = 0,
           size_t strideD = 0,
           int batch_count = 1,
           size_t wsSizeBytes = 4 * 1024 * 1024) : workspaceSizeBytes(wsSizeBytes) {

        // Compute C_i = A_i x B_i, where matrices are row-major.
        // as C_i' = B_i' x A_i' where matrices are col-major
        // We swap them when calling matmul.

        auto op_A = CUBLAS_OP_N;
        size_t rows_A = m;
        size_t cols_A = k;
        if(A_transposed) {
            std::swap(rows_A, cols_A);
            op_A = CUBLAS_OP_T;
        }
        if( ldA == 0 ) { // Default leading dim.
            ldA = cols_A;
        }
        if(strideA == 0) { // Default batch stride.
            strideA = rows_A * cols_A;
        }

        auto op_B = CUBLAS_OP_N;
        size_t rows_B = k;
        size_t cols_B = n;
        if(B_transposed) {
            std::swap(rows_B, cols_B);
            op_B = CUBLAS_OP_T;
        }
        if( ldB == 0 ) { // Default leading dim.
            ldB = cols_B;
        }
        if(strideB == 0) { // Default batch stride.
            strideB = rows_B * cols_B;
        }

        size_t rows_D = m;
        size_t cols_D = n;
        if( ldD == 0 ) { // Default leading dim.
            ldD = cols_D;
        }
        if(strideD == 0) { // Default batch stride.
            strideD = rows_D * cols_D;
        }

        FMHA_CHECK_CUDA(cudaMalloc(&workspace, wsSizeBytes));

        FMHA_CHECK_CUBLAS(cublasLtCreate(&ltHandle));
        FMHA_CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmul_desc, type_compute, type_scale));

        FMHA_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_B, sizeof(op_B)));
        FMHA_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_A, sizeof(op_A)));

        // Need to swap rows <=> cols.
        FMHA_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, type_A, cols_A, rows_A, ldA));
        FMHA_CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        FMHA_CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));

#if 0
        printf("A: [%lu x %lu]:%lu\n", rows_A, cols_A, ldA);
        printf("B: [%lu x %lu]:%lu\n", rows_B, cols_B, ldB);
        printf("D: [%lu x %lu]:%lu\n", rows_D, cols_D, ldD);

        printf("A: [%lu x %lu]:%lu\n", cols_B, rows_B, ldB);
        printf("B: [%lu x %lu]:%lu\n", cols_A, rows_A, ldA);
        printf("D: [%lu x %lu]:%lu\n", cols_D, rows_D, ldD);
#endif

        // Need to swap rows <=> cols.
        FMHA_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, type_B, cols_B, rows_B, ldB));
        FMHA_CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        FMHA_CHECK_CUBLAS(
            cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));

        // Need to swap rows <=> cols.
        FMHA_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Ddesc, type_D, cols_D, rows_D, ldD));
        FMHA_CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(Ddesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        FMHA_CHECK_CUBLAS(
            cublasLtMatrixLayoutSetAttribute(Ddesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideD, sizeof(strideD)));
    }

    ~RefBMM() {
        FMHA_CHECK_CUBLAS(cublasLtDestroy(ltHandle));
        FMHA_CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmul_desc));
        FMHA_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Adesc));
        FMHA_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Bdesc));
        FMHA_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Ddesc));
        FMHA_CHECK_CUDA(cudaFree(workspace));
    }

    void operator()(const void * A, const void * B, void * D, const void * alpha, const void * beta, cudaStream_t stream) {

        cublasStatus_t status = cublasLtMatmul(ltHandle,
                                               matmul_desc,
                                               alpha,
                                               B,
                                               Bdesc,
                                               A,
                                               Adesc,
                                               beta,
                                               D,
                                               Ddesc,
                                               D,
                                               Ddesc,
                                               nullptr,  // &algo,
                                               workspace,
                                               workspaceSizeBytes,
                                               stream);
        FMHA_CHECK_CUBLAS(status);
    }

    cublasLtHandle_t ltHandle;
    void *workspace;
    size_t workspaceSizeBytes;

    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t Adesc;
    cublasLtMatrixLayout_t Bdesc;
    cublasLtMatrixLayout_t Ddesc;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void print_results(bool with_colors, bool enabled, bool success = false) {
    // The opening tag.
    char beg[16];
    if( with_colors && enabled && success ) {  // Succeeded -> green
        strcpy(beg, "\033[0;32m");
    } else if( with_colors && enabled ) {  // Failed -> red
        strcpy(beg, "\033[0;31m");
    } else if( with_colors ) {  // Disabled -> yellow
        strcpy(beg, "\033[0;33m");
    }

    // The message.
    char msg[16];
    if( enabled && success ) {
        strcpy(msg, "SUCCESS");
    } else if( enabled ) {
        strcpy(msg, "FAILED");
    } else {
        strcpy(msg, "DISABLED");
    }

    // The closing tag.
    char end[16];
    if( with_colors ) {
        strcpy(end, "\033[0m");
    }

    // Print the results.
    if( with_colors ) {
        printf("Checks........: %s%s%s\n", beg, msg, end);
    } else {
        printf("Checks........: %s\n", msg);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static void print_tensor(const float *buffer,
                         const size_t m,
                         const size_t n,
                         const size_t ld_ = 0,
                         const std::string &str = "") {
    printf("Buffer %s:\n", str.c_str());
    size_t ld = ld_ == 0 ? m : ld_;
    for( size_t ni = 0; ni < n; ni++ ) {
        printf("ni %ld: ", ni);
        for( size_t mi = 0; mi < m; mi++ ) {
            // The offset.
            size_t ii = (size_t)ni * ld + mi;
            printf(" %2.3f,", buffer[ii]);
        }
        printf("\n");
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline int check_results(const float *out,
                                const float *ref,
                                size_t m,
                                size_t n,
                                size_t ld,
                                float epsilon,
                                bool verbose,
                                bool with_colors) {
    int failed = 0, infs = 0;
    float min_val = +FLT_MAX, max_val = -FLT_MAX, min_err = +FLT_MAX, max_err = -FLT_MAX;
    double avg_val = 0.0, sqr_val = 0.0, avg_err = 0.0, sqr_err = 0.0;
    double inv_mn = 1.0 / (double)m / (double)n;
    for( size_t ni = 0; ni < n; ++ni ) {
        for( size_t mi = 0; mi < m; ++mi ) {
            // The offset.
            size_t ii = (size_t)ni * ld + mi;

            // The elements.
            float a = out[ii];
            float b = ref[ii];

            // Compute the error.
            float den = fabsf(a) + fabsf(b);
            float err = den <= epsilon ? fabsf(a - b) : fabsf(a - b) / den;

            // Min/max values.
            min_val = fminf(a, min_val);
            max_val = fmaxf(a, max_val);
            min_err = fminf(err, min_err);
            max_err = fmaxf(err, max_err);

            // Sums to compute the average value.
            avg_val += (double) a * inv_mn;
            sqr_val += (double) a * a * inv_mn;
            avg_err += (double) err * inv_mn;
            sqr_err += (double) err * err * inv_mn;

            // Does it fail?
            if( isnan(a) || isnan(b) || err > epsilon ) {
                if( failed < 8 ) {
                    printf("\tInvalid result for ni=%lu (on %lu) mi=%lu (on %lu) ii=%lu:\n", 
                            ni, 
                            n,
                            mi, 
                            m,
                            ii);
                    printf("\t    Found...: 0x%08x (%10.6f)\n", *(const int *)&out[ii], a);
                    printf("\t    Expected: 0x%08x (%10.6f)\n", *(const int *)&ref[ii], b);
                    printf("\t    Error...: %10.6f\n", err);
                }
                failed++;
            }
            infs += !isfinite(a);
            infs += !isfinite(b);
        }
    }

    double std_val = sqrtf(sqr_val - avg_val * avg_val);
    double std_err = sqrtf(sqr_err - avg_err * avg_err);

    if( verbose ) {
        printf("Epsilon.......: %.8f\n", epsilon);
        printf("Tested........: %lu\n", m * n);
        printf("Failed........: %d\n", failed);
        printf("Values........: Min=%12.6f, Max=%12.6f, Avg=%10.6lf, Std=%10.6lf\n",
                min_val,
                max_val,
                avg_val,
                std_val);
        printf("Error.........: Min=%12.6f, Max=%12.6f, Avg=%10.6lf, Std=%10.6lf\n",
                min_err,
                max_err,
                avg_err,
                std_err);
        printf("Epsilon.......: %.6f\n", epsilon);
        printf("Infs..........: %d\n", infs);
        print_results(with_colors, true, !failed);
    }
    return failed ? 1 : 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_to_float_(float *dst, const uint16_t *src, size_t n) {
    for( size_t ii = 0; ii < n; ++ii ) {
        dst[ii] = __half2float(reinterpret_cast<const __half *>(src)[ii]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_to_float_(float *dst, const int32_t *src, size_t n) {
    for( size_t ii = 0; ii < n; ++ii ) {
        dst[ii] = (float)src[ii];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_to_float_(float *dst, const int8_t *src, size_t n) {
    for( size_t ii = 0; ii < n; ++ii ) {
        dst[ii] = (float)(int32_t)src[ii];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_to_float_(float *dst, const fmha::bf16_t *src, size_t n) {
    for( size_t ii = 0; ii < n; ++ii ) {
        dst[ii] = __bfloat162float(src[ii]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_to_float_(float *dst, const fmha::e4m3_t *src, size_t n) {
    for( size_t ii = 0; ii < n; ++ii ) {
        dst[ii] = float(src[ii]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_to_float(float *dst, const void *src, size_t n, Data_type dtype) {
    switch( dtype ) {
    case DATA_TYPE_FP32:
        memcpy(dst, src, n * sizeof(float));
        break;
    case DATA_TYPE_FP16:
        convert_to_float_(dst, reinterpret_cast<const uint16_t *>(src), n);
        break;
    case DATA_TYPE_INT32:
        convert_to_float_(dst, reinterpret_cast<const int32_t *>(src), n);
        break;
    case DATA_TYPE_INT8:
        convert_to_float_(dst, reinterpret_cast<const int8_t *>(src), n);
        break;
    case DATA_TYPE_BF16:
        convert_to_float_(dst, reinterpret_cast<const fmha::bf16_t *>(src), n);
        break;
    case DATA_TYPE_E4M3:
        convert_to_float_(dst, reinterpret_cast<const fmha::e4m3_t *>(src), n);
        break;
    default:
        assert(false);  // Not implemented!
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_from_float_(uint16_t *dst, const float *src, size_t n) {
    for( size_t ii = 0; ii < n; ++ii ) {
        reinterpret_cast<__half *>(dst)[ii] = __float2half_rn(src[ii]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_from_float_(fmha::bf16_t *dst, const float *src, size_t n) {
    for( size_t ii = 0; ii < n; ++ii ) {
        dst[ii] = __float2bfloat16(src[ii]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_from_float_(int32_t *dst, const float *src, size_t n) {
    for( size_t ii = 0; ii < n; ++ii ) {
        dst[ii] = (int32_t)src[ii];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_from_float_(int8_t *dst, const float *src, size_t n) {
    for( size_t ii = 0; ii < n; ++ii ) {
        float x = src[ii];
        dst[ii] = (int8_t)(int32_t)(x < -128.f ? -128.f : (x > 127.f ? 127.f : x));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_from_float_(fmha::e4m3_t *dst, const float *src, size_t n) {
    for( size_t ii = 0; ii < n; ++ii ) {
        dst[ii] = fmha::e4m3_t(src[ii]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_from_float(void *dst, const float *src, size_t n, Data_type dtype) {
    switch( dtype ) {
    case DATA_TYPE_FP32:
        memcpy(dst, src, n * sizeof(float));
        break;
    case DATA_TYPE_FP16:
        convert_from_float_(reinterpret_cast<uint16_t *>(dst), src, n);
        break;
    case DATA_TYPE_BF16:
        convert_from_float_(reinterpret_cast<fmha::bf16_t *>(dst), src, n);
        break;
    case DATA_TYPE_INT32:
        convert_from_float_(reinterpret_cast<int32_t *>(dst), src, n);
        break;
    case DATA_TYPE_INT8:
        convert_from_float_(reinterpret_cast<int8_t *>(dst), src, n);
        break;
    case DATA_TYPE_E4M3:
        convert_from_float_(reinterpret_cast<fmha::e4m3_t *>(dst), src, n);
        break;
    default:
        assert(false);  // Not implemented!
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline cudaError cuda_memcpy_d2h(float *dst, const void *src, size_t n, Data_type dtype) {
    size_t sz = get_size_in_bytes(n, dtype);
    void *tmp = malloc(sz);
    cudaError err = cudaMemcpy(tmp, src, sz, cudaMemcpyDeviceToHost);
    convert_to_float(dst, tmp, n, dtype);
    free(tmp);
    return err;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline cudaError cuda_memcpy_h2d(void *dst, const float *src, size_t n, Data_type dtype) {
    size_t sz = get_size_in_bytes(n, dtype);
    void *tmp = malloc(sz);
    convert_from_float(tmp, src, n, dtype);
    cudaError err = cudaMemcpy(dst, tmp, sz, cudaMemcpyHostToDevice);
    free(tmp);
    return err;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline cudaDataType_t data_type_to_cuda(Data_type dtype) {
    switch(dtype ) {
    case DATA_TYPE_FP32:
        return CUDA_R_32F;
    case DATA_TYPE_FP16:
        return CUDA_R_16F;
    case DATA_TYPE_INT32:
        return CUDA_R_32I;
    case DATA_TYPE_INT8:
        return CUDA_R_8I;
    case DATA_TYPE_BF16:
        return CUDA_R_16BF;
#if FMHA_CUDA_SUPPORTS_FP8
    case DATA_TYPE_E4M3:
        return CUDA_R_8F_E4M3;
    case DATA_TYPE_E5M2:
        return CUDA_R_8F_E5M2;
#endif
    default:
        assert(false);
        return CUDA_R_32F;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline std::string data_type_to_name(Data_type dtype) {
    switch(dtype ) {
    case DATA_TYPE_FP32:
        return "FP32";
    case DATA_TYPE_FP16:
        return "FP16";
    case DATA_TYPE_INT32:
        return "INT32";
    case DATA_TYPE_INT8:
        return "INT8";
    case DATA_TYPE_BF16:
        return "BF16";
#if FMHA_CUDA_SUPPORTS_FP8
    case DATA_TYPE_E4M3:
        return "FP8_E4M3";
    case DATA_TYPE_E5M2:
        return "FP8_E5M2";
#endif
    default:
        assert(false);
        return "";
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline cublasComputeType_t data_type_to_cublas(Data_type dtype) {
    switch(dtype ) {
    case DATA_TYPE_FP32:
        return CUBLAS_COMPUTE_32F;
    case DATA_TYPE_FP16:
        return CUBLAS_COMPUTE_16F;
    case DATA_TYPE_INT32:
        return CUBLAS_COMPUTE_32I;
    //case DATA_TYPE_BF16:
    //    //TODO HACK!! 
    //    return CUBLAS_COMPUTE_32F;
    default:
        assert(false);
        return CUBLAS_COMPUTE_32F;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
