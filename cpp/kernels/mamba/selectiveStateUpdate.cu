/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * This file contains code derived from FlashInfer (https://github.com/flashinfer-ai/flashinfer)
 * Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)
 * Licensed under the Apache License, Version 2.0.
 *
 * Modifications by NVIDIA:
 * - Ported simple selective state update kernel for TensorRT Edge-LLM
 * - Added explicit stride-based memory access for padded layouts
 * - Renamed namespace from flashinfer::mamba to mamba_ssm
 * - Added BFloat16 template instantiation
 * - Replaced FLASHINFER_CHECK with direct std::runtime_error throws
 */

#include "selectiveStateUpdate.h"

#include "common.cuh"
#include "conversion.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mamba_ssm
{

// =============================================================================
// Internal parameter struct (not exposed in the public header)
// =============================================================================
struct SelectiveStateUpdateParams
{
    uint32_t batch{}, nheads{}, dim{}, dstate{}, ngroups{}, state_cache_size{};
    int32_t pad_slot_id{-1};

    // Batch strides
    int64_t x_stride_batch{};
    int64_t dt_stride_batch{};
    int64_t B_stride_batch{};
    int64_t C_stride_batch{};
    int64_t out_stride_batch{};
    int64_t z_stride_batch{};
    int64_t state_stride_batch{};

    // Head strides (for padded layouts, set to padded dimension size)
    int64_t x_stride_head{};
    int64_t z_stride_head{};
    int64_t out_stride_head{};
    int64_t state_stride_head{};
    int64_t state_stride_dim{};

    // Sequence strides (prefill kernel only)
    int64_t x_stride_seq{};
    int64_t dt_stride_seq{};
    int64_t B_stride_seq{};
    int64_t C_stride_seq{};
    int64_t out_stride_seq{};
    int32_t seq_len{1};

    void* __restrict__ state{nullptr};
    void* __restrict__ x{nullptr};
    void* __restrict__ dt{nullptr};
    void* __restrict__ dt_bias{nullptr};
    void* __restrict__ A{nullptr};
    void* __restrict__ B{nullptr};
    void* __restrict__ C{nullptr};
    void* __restrict__ D{nullptr};
    void* __restrict__ z{nullptr};
    void* __restrict__ output{nullptr};
    void* __restrict__ state_batch_indices{nullptr};

    bool dt_softplus{false};
    bool update_state{true};
};

inline void setContiguousStrides(SelectiveStateUpdateParams& params)
{
    params.x_stride_head = params.dim;
    params.z_stride_head = params.dim;
    params.out_stride_head = params.dim;
    params.state_stride_dim = params.dstate;
    params.state_stride_head = params.dim * params.dstate;
    params.x_stride_batch = params.nheads * params.x_stride_head;
    params.z_stride_batch = params.nheads * params.z_stride_head;
    params.out_stride_batch = params.nheads * params.out_stride_head;
    params.state_stride_batch = params.nheads * params.state_stride_head;
    params.dt_stride_batch = params.nheads;
    params.B_stride_batch = params.ngroups * params.dstate;
    params.C_stride_batch = params.ngroups * params.dstate;
}

using namespace conversion;

// =============================================================================
// Shared memory structure for simple kernel
// =============================================================================
template <typename input_t, int dim, int dstate>
struct SharedStorageSimple
{
    alignas(alignof(PackedAligned<input_t>)) input_t x[dim];
    alignas(alignof(PackedAligned<input_t>)) input_t z[dim];
    alignas(alignof(PackedAligned<input_t>)) input_t B[dstate];
    alignas(alignof(PackedAligned<input_t>)) input_t C[dstate];
    float out[dim];
};

// =============================================================================
// Simple selective state update kernel (works on all GPU architectures)
// =============================================================================
template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, typename stateIndex_t, int DIM,
    int DSTATE, int numWarps>
__global__ void selective_state_update_kernel_simple(SelectiveStateUpdateParams params)
{
    auto* __restrict__ output = reinterpret_cast<input_t*>(params.output);
    auto* __restrict__ state = reinterpret_cast<state_t*>(params.state);

    auto const* __restrict__ x = reinterpret_cast<input_t const*>(params.x);
    auto const* __restrict__ dt = reinterpret_cast<weight_t const*>(params.dt);
    auto const* __restrict__ A = reinterpret_cast<matrixA_t const*>(params.A);
    auto const* __restrict__ B = reinterpret_cast<input_t const*>(params.B);
    auto const* __restrict__ C = reinterpret_cast<input_t const*>(params.C);
    auto const* __restrict__ D = reinterpret_cast<weight_t const*>(params.D);
    auto const* __restrict__ dt_bias = reinterpret_cast<weight_t const*>(params.dt_bias);
    auto const* __restrict__ z = reinterpret_cast<input_t const*>(params.z);
    auto const* __restrict__ state_batch_indices = reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);
    bool const dt_softplus = params.dt_softplus;

    int const nheads = params.nheads;
    int const ngroups = params.ngroups;

    constexpr auto rowsPerWarp = (DIM + numWarps - 1) / numWarps;

    auto const batch = blockIdx.x;
    auto const head = blockIdx.y;
    auto const group = head / (nheads / ngroups);
    auto lane = threadIdx.x % kWARP_SIZE;
    auto warp = threadIdx.y;

    auto const state_batch = (state_batch_indices) ? state_batch_indices[batch] : batch;
    state += state_batch * params.state_stride_batch + head * params.state_stride_head;

    __shared__ SharedStorageSimple<input_t, DIM, DSTATE> sram;

    static constexpr auto stateLoadSize = getVectorLoadSizeForFullUtilization<state_t, DSTATE>();
    using load_state_t = PackedAligned<state_t, stateLoadSize>;
    using load_input_t = PackedAligned<input_t>;

    auto const A_value = toFloat(A[head]);

    auto dt_value = toFloat(dt[batch * params.dt_stride_batch + head]);
    if (dt_bias)
        dt_value += toFloat(dt_bias[head]);
    if (dt_softplus)
    {
        dt_value = thresholded_softplus(dt_value);
    }

    auto const dA = __expf(A_value * dt_value);

    auto d_value = D ? toFloat(D[head]) : 0.f;

    // Load x and B (warp 0)
    if (warp == 0)
    {
        for (auto d = lane * load_input_t::count; d < DIM; d += kWARP_SIZE * load_input_t::count)
        {
            auto* dst = reinterpret_cast<load_input_t*>(&sram.x[d]);
            *dst = *reinterpret_cast<load_input_t const*>(
                &x[batch * params.x_stride_batch + head * params.x_stride_head + d]);
        }
        for (auto i = lane * load_input_t::count; i < DSTATE; i += kWARP_SIZE * load_input_t::count)
        {
            auto* dst = reinterpret_cast<load_input_t*>(&sram.B[i]);
            *dst = *reinterpret_cast<load_input_t const*>(&B[batch * params.B_stride_batch + group * DSTATE + i]);
        }
    }
    // Load z and C (warp 1)
    else if (warp == 1)
    {
        for (auto d = lane * load_input_t::count; d < DIM; d += kWARP_SIZE * load_input_t::count)
        {
            auto* dst = reinterpret_cast<load_input_t*>(&sram.z[d]);
            *dst = z ? *reinterpret_cast<load_input_t const*>(
                           &z[batch * params.z_stride_batch + head * params.z_stride_head + d])
                     : make_zeros<load_input_t>();
        }
        for (auto i = lane * load_input_t::count; i < DSTATE; i += kWARP_SIZE * load_input_t::count)
        {
            auto* dst = reinterpret_cast<load_input_t*>(&sram.C[i]);
            *dst = *reinterpret_cast<load_input_t const*>(&C[batch * params.C_stride_batch + group * DSTATE + i]);
        }
    }
    __syncthreads();

    // Main computation loop: each warp processes a subset of dim rows
    for (auto _d = warp * rowsPerWarp; _d < (warp + 1) * rowsPerWarp; _d++)
    {
        auto d = _d;
        if (d >= DIM)
            break;

        float x_value = toFloat(sram.x[_d]);
        // D*x is a scalar contribution added once per dim row; only lane 0 seeds it
        // so after warpReduceSum it is included exactly once in the final output.
        float out_value = (lane == 0) ? (d_value * x_value) : 0.0f;

        // Process state dimension
        for (int i = lane * load_state_t::count; i < DSTATE; i += kWARP_SIZE * load_state_t::count)
        {
            auto rState = make_zeros<load_state_t>();
            if (state_batch != params.pad_slot_id)
                rState = *reinterpret_cast<load_state_t*>(&state[d * params.state_stride_dim + i]);

            for (int ii = 0; ii < load_state_t::count; ii++)
            {
                auto state_value = toFloat(rState.val[ii]);
                auto B_value = toFloat(sram.B[i + ii]);
                auto C_value = toFloat(sram.C[i + ii]);

                auto const dB = B_value * dt_value;
                auto const new_state = state_value * dA + dB * x_value;

                convertAndStore(&rState.val[ii], new_state);

                out_value += new_state * C_value;
            }
            if (params.update_state && state_batch != params.pad_slot_id)
                *reinterpret_cast<load_state_t*>(&state[d * params.state_stride_dim + i]) = rState;
        }

        // Warp reduce the output value
        out_value = warpReduceSum(out_value);
        if (lane == 0)
        {
            sram.out[_d] = out_value;
        }
    }

    __syncthreads();

    // Write output with optional SiLU gating
    for (int l = lane; l < rowsPerWarp; l += kWARP_SIZE)
    {
        auto d = warp * rowsPerWarp + l;
        if (d < DIM)
        {
            auto out_value = sram.out[d];
            if (z)
            {
                float z_value = toFloat(sram.z[d]);
                float sig_z = __fdividef(1.f, (1.f + __expf(0.f - z_value)));
                float silu_z = z_value * sig_z;
                out_value *= silu_z;
            }
            convertAndStore(&output[batch * params.out_stride_batch + head * params.out_stride_head + d], out_value);
        }
    }
}

// =============================================================================
// Prefill kernel: full-sequence SSM scan with fp32 state kept in registers
// =============================================================================
//
// Unlike selective_state_update_kernel_simple (which is launched once per token
// from a host-side loop), this kernel processes the entire token sequence inside
// a single CUDA kernel.  The running SSM state is accumulated as float in
// registers across all time steps and written to global memory only once at the
// end, eliminating the fp16 quantisation round-trip that the host loop incurs on
// every token.
//
// Loop nesting:  (d-outer) → (t-middle) → (dstate-inner)
// Each lane holds DSTATE/kWARP_SIZE = 4 float registers for its state slice.
// No shared memory is used, so no __syncthreads() is required.
template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, typename stateIndex_t, int DIM,
    int DSTATE, int numWarps>
__global__ void selective_state_update_prefill_kernel_simple(SelectiveStateUpdateParams params)
{
    // Ceil division: e.g. DSTATE=80, kWARP_SIZE=32 → dstatePerLane=3.
    // Last few lanes may own elements with i >= DSTATE; those are guarded below.
    constexpr int dstatePerLane = (DSTATE + kWARP_SIZE - 1) / kWARP_SIZE;

    auto* __restrict__ output = reinterpret_cast<input_t*>(params.output);
    auto* __restrict__ state = reinterpret_cast<state_t*>(params.state);

    auto const* __restrict__ x = reinterpret_cast<input_t const*>(params.x);
    auto const* __restrict__ dt = reinterpret_cast<weight_t const*>(params.dt);
    auto const* __restrict__ A = reinterpret_cast<matrixA_t const*>(params.A);
    auto const* __restrict__ B = reinterpret_cast<input_t const*>(params.B);
    auto const* __restrict__ C = reinterpret_cast<input_t const*>(params.C);
    auto const* __restrict__ D = reinterpret_cast<weight_t const*>(params.D);
    auto const* __restrict__ dt_bias = reinterpret_cast<weight_t const*>(params.dt_bias);

    int const nheads = params.nheads;
    int const ngroups = params.ngroups;

    auto const batch = blockIdx.x;
    auto const head = blockIdx.y;
    auto const group = head / (nheads / ngroups);
    auto const lane = threadIdx.x % kWARP_SIZE;
    auto const warp = threadIdx.y;

    state += batch * params.state_stride_batch + head * params.state_stride_head;

    auto const A_value = toFloat(A[head]);
    auto const d_value = D ? toFloat(D[head]) : 0.f;
    auto const seqLen = params.seq_len;

    constexpr auto rowsPerWarp = (DIM + numWarps - 1) / numWarps;

    // Process each dim row assigned to this warp.
    for (int _d = warp * rowsPerWarp; _d < (warp + 1) * rowsPerWarp; _d++)
    {
        if (_d >= DIM)
            break;

        // ----------------------------------------------------------------
        // Load initial SSM state into fp32 registers (no quantisation here).
        // ----------------------------------------------------------------
        float runState[dstatePerLane];
        bool const validSlot = (params.pad_slot_id < 0 || batch != static_cast<uint32_t>(params.pad_slot_id));
#pragma unroll
        for (int ii = 0; ii < dstatePerLane; ++ii)
        {
            int const i = lane * dstatePerLane + ii;
            runState[ii] = (validSlot && i < DSTATE) ? toFloat(state[_d * params.state_stride_dim + i]) : 0.f;
        }

        // ----------------------------------------------------------------
        // Scan over the token sequence, state stays fp32 in registers.
        // ----------------------------------------------------------------
        for (int32_t t = 0; t < seqLen; ++t)
        {
            // dt[batch, t, head]
            float dt_val = toFloat(dt[batch * params.dt_stride_batch + t * params.dt_stride_seq + head]);
            if (dt_bias)
                dt_val += toFloat(dt_bias[head]);
            if (params.dt_softplus)
                dt_val = thresholded_softplus(dt_val);
            float const dA = __expf(A_value * dt_val);

            // x[batch, t, head, _d]
            float const x_val = toFloat(
                x[batch * params.x_stride_batch + t * params.x_stride_seq + head * params.x_stride_head + _d]);

            // D * x contribution: only lane 0 adds it to avoid double-counting
            // in the warp reduce that follows.
            float out_val = (lane == 0) ? d_value * x_val : 0.f;

            // State update and output accumulation (over this lane's dstate slice).
#pragma unroll
            for (int ii = 0; ii < dstatePerLane; ++ii)
            {
                int const i = lane * dstatePerLane + ii;
                if (i < DSTATE)
                {
                    float const B_val
                        = toFloat(B[batch * params.B_stride_batch + t * params.B_stride_seq + group * DSTATE + i]);
                    float const C_val
                        = toFloat(C[batch * params.C_stride_batch + t * params.C_stride_seq + group * DSTATE + i]);
                    // Key: runState stays float — no fp16 quantisation between tokens.
                    runState[ii] = runState[ii] * dA + B_val * dt_val * x_val;
                    out_val += runState[ii] * C_val;
                }
            }

            // Warp reduce: sum contributions across lanes (each handles a dstate slice).
            out_val = warpReduceSum(out_val);

            // Lane 0 writes the output token to global memory.
            if (lane == 0)
            {
                convertAndStore(&output[batch * params.out_stride_batch + t * params.out_stride_seq
                                    + head * params.out_stride_head + _d],
                    out_val);
            }
        }

        // ----------------------------------------------------------------
        // Write final state to global memory — ONE quantisation per sequence.
        // ----------------------------------------------------------------
        if (params.update_state && validSlot)
        {
#pragma unroll
            for (int ii = 0; ii < dstatePerLane; ++ii)
            {
                int const i = lane * dstatePerLane + ii;
                if (i < DSTATE)
                    convertAndStore(&state[_d * params.state_stride_dim + i], runState[ii]);
            }
        }
    }
}

// =============================================================================
// Kernel launcher functors (at namespace scope for nvcc compatibility)
// =============================================================================
template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, typename stateIndex_t>
struct SsmKernelLauncher
{
    SelectiveStateUpdateParams& params;
    cudaStream_t stream;

    template <int DIM, int DSTATE>
    void operator()()
    {
        constexpr auto stateLoadSize = getVectorLoadSizeForFullUtilization<state_t, DSTATE>();
        using load_state_t = PackedAligned<state_t, stateLoadSize>;

        auto const stateAlign = std::to_string(sizeof(load_state_t));
        if (reinterpret_cast<uintptr_t>(params.state) % sizeof(load_state_t) != 0)
            throw std::runtime_error("state pointer must be aligned to " + stateAlign + " bytes");
        if ((params.dim * params.dstate * sizeof(state_t)) % sizeof(load_state_t) != 0)
            throw std::runtime_error("state head stride must be aligned to " + stateAlign + " bytes");

        constexpr int numWarps = 4;
        dim3 block(kWARP_SIZE, numWarps);
        dim3 grid(params.batch, params.nheads);
        selective_state_update_kernel_simple<input_t, weight_t, matrixA_t, state_t, stateIndex_t, DIM, DSTATE, numWarps>
            <<<grid, block, 0, stream>>>(params);
    }
};

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, typename stateIndex_t>
struct SsmPrefillKernelLauncher
{
    SelectiveStateUpdateParams& params;
    cudaStream_t stream;

    template <int DIM, int DSTATE>
    void operator()()
    {
        constexpr int numWarps = 4;
        dim3 block(kWARP_SIZE, numWarps);
        dim3 grid(params.batch, params.nheads);
        selective_state_update_prefill_kernel_simple<input_t, weight_t, matrixA_t, state_t, stateIndex_t, DIM, DSTATE,
            numWarps><<<grid, block, 0, stream>>>(params);
    }
};

// =============================================================================
// Internal kernel dispatch (params-based) — not part of the public API
// =============================================================================
template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, typename stateIndex_t>
static void invokeSelectiveStateUpdateImpl(SelectiveStateUpdateParams& params, cudaStream_t stream)
{
    check_ptr_alignment_input_vars<input_t>(params);

    SsmKernelLauncher<input_t, weight_t, matrixA_t, state_t, stateIndex_t> launcher{params, stream};
    dispatchDimDstate(params, AllowedDims{}, AllowedDstates{}, launcher);
}

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, typename stateIndex_t>
static void invokeSelectiveStateUpdatePrefillImpl(SelectiveStateUpdateParams& params, cudaStream_t stream)
{
    check_ptr_alignment_input_vars<input_t>(params);

    SsmPrefillKernelLauncher<input_t, weight_t, matrixA_t, state_t, stateIndex_t> launcher{params, stream};
    dispatchDimDstate(params, AllowedDims{}, AllowedDstates{}, launcher);
}

// =============================================================================
// Public tensor-based dispatchers
// =============================================================================

// Helper: fill the fields that are common to both decode and prefill
static void fillCommonParams(
    SsmUpdateTensors const& tensors, bool dt_softplus, SelectiveStateUpdateParams& params, int headDim)
{
    // Dimensions are read from the state tensor [batch, nheads, dim, dstate]
    params.batch = static_cast<uint32_t>(tensors.state->getShape()[0]);
    params.nheads = static_cast<uint32_t>(tensors.state->getShape()[1]);
    params.dim = static_cast<uint32_t>(tensors.state->getShape()[2]);
    params.dstate = static_cast<uint32_t>(tensors.state->getShape()[3]);
    // B is [batch, (seq_len,) ngroups, dstate]; ngroups is at dim (ndims-2)
    auto const bndims = tensors.B->getShape().getNumDims();
    params.ngroups = static_cast<uint32_t>(tensors.B->getShape()[bndims - 2]);
    params.dt_softplus = dt_softplus;
    params.update_state = true;

    // State strides: [batch, nheads, dim, dstate]
    params.state_stride_batch = tensors.state->getStride(0);
    params.state_stride_head = tensors.state->getStride(1);
    params.state_stride_dim = tensors.state->getStride(2);

    // Pointers
    params.x = const_cast<void*>(tensors.x->rawPointer());
    params.A = const_cast<void*>(tensors.A->rawPointer());
    params.B = const_cast<void*>(tensors.B->rawPointer());
    params.C = const_cast<void*>(tensors.C->rawPointer());
    params.dt = const_cast<void*>(tensors.dt->rawPointer());
    params.dt_bias = tensors.dt_bias ? const_cast<void*>(tensors.dt_bias->rawPointer()) : nullptr;
    params.D = tensors.D ? const_cast<void*>(tensors.D->rawPointer()) : nullptr;
    params.z = tensors.z ? const_cast<void*>(tensors.z->rawPointer()) : nullptr;
    params.state = tensors.state->rawPointer();
    params.output = tensors.output->rawPointer();
}

/*!
 * \brief Decode dispatcher: x is 3D [batch, nheads, dim].
 *
 * All strides are read directly from the tensor layout, so non-contiguous
 * buffers (e.g. padded heads) are handled automatically.
 */
template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, typename stateIndex_t>
void invokeSelectiveStateUpdate(SsmUpdateTensors const& tensors, bool dt_softplus, cudaStream_t stream)
{
    SelectiveStateUpdateParams params{};
    fillCommonParams(tensors, dt_softplus, params, /*headDim=*/0);

    // x: [batch, nheads, dim]
    params.x_stride_batch = tensors.x->getStride(0);
    params.x_stride_head = tensors.x->getStride(1);
    params.dt_stride_batch = tensors.dt->getStride(0);
    params.B_stride_batch = tensors.B->getStride(0);
    params.C_stride_batch = tensors.C->getStride(0);
    params.out_stride_batch = tensors.output->getStride(0);
    params.out_stride_head = tensors.output->getStride(1);
    if (tensors.z)
    {
        params.z_stride_batch = tensors.z->getStride(0);
        params.z_stride_head = tensors.z->getStride(1);
    }

    invokeSelectiveStateUpdateImpl<input_t, weight_t, matrixA_t, state_t, stateIndex_t>(params, stream);
}

/*!
 * \brief Prefill dispatcher: x is 4D [batch, seq_len, nheads, dim].
 *
 * Per-token strides (seq dimension) are read from the tensor layout.
 */
template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, typename stateIndex_t>
void invokeSelectiveStateUpdatePrefill(SsmUpdateTensors const& tensors, bool dt_softplus, cudaStream_t stream)
{
    SelectiveStateUpdateParams params{};
    fillCommonParams(tensors, dt_softplus, params, /*headDim=*/0);

    // x: [batch, seq_len, nheads, dim]
    params.seq_len = static_cast<int32_t>(tensors.x->getShape()[1]);
    params.x_stride_batch = tensors.x->getStride(0);
    params.x_stride_seq = tensors.x->getStride(1);
    params.x_stride_head = tensors.x->getStride(2);
    params.dt_stride_batch = tensors.dt->getStride(0);
    params.dt_stride_seq = tensors.dt->getStride(1);
    params.B_stride_batch = tensors.B->getStride(0);
    params.B_stride_seq = tensors.B->getStride(1);
    params.C_stride_batch = tensors.C->getStride(0);
    params.C_stride_seq = tensors.C->getStride(1);
    params.out_stride_batch = tensors.output->getStride(0);
    params.out_stride_seq = tensors.output->getStride(1);
    params.out_stride_head = tensors.output->getStride(2);
    if (tensors.z)
    {
        params.z_stride_batch = tensors.z->getStride(0);
        params.z_stride_head = tensors.z->getStride(2);
    }

    invokeSelectiveStateUpdatePrefillImpl<input_t, weight_t, matrixA_t, state_t, stateIndex_t>(params, stream);
}

// =============================================================================
// Explicit template instantiations
// =============================================================================
// FP16 main precision. A is always FP32. Accumulation inside the kernel is FP32.
// Two weight_t variants: half (plugin path) and float (test/debug path).
// TODO: Use Tensor class to dispatch the kernels based on dtype of the input tensors.
template void invokeSelectiveStateUpdate<half, half, float, half, int32_t>(
    SsmUpdateTensors const& tensors, bool dt_softplus, cudaStream_t stream);

template void invokeSelectiveStateUpdate<half, float, float, half, int32_t>(
    SsmUpdateTensors const& tensors, bool dt_softplus, cudaStream_t stream);

template void invokeSelectiveStateUpdatePrefill<half, half, float, half, int32_t>(
    SsmUpdateTensors const& tensors, bool dt_softplus, cudaStream_t stream);

template void invokeSelectiveStateUpdatePrefill<half, float, float, half, int32_t>(
    SsmUpdateTensors const& tensors, bool dt_softplus, cudaStream_t stream);

} // namespace mamba_ssm
