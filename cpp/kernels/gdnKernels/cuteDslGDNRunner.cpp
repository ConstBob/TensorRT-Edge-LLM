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

#ifdef CUTE_DSL_GDN_ENABLED

#include "cuteDslGDNRunner.h"

#include "common/logger.h"

#include <cmath>
#include <mutex>

namespace trt_edgellm
{

gdn_decode_Kernel_Module_t CuteDslGDNRunner::sDecodeModule = {};
gdn_prefill_Kernel_Module_t CuteDslGDNRunner::sPrefillModule = {};
bool CuteDslGDNRunner::sLoaded = false;

static std::mutex sGDNMutex;

#define SET_4D_TENSOR(tensor, data_ptr, dim0, dim1, dim2, dim3)                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        (tensor).data = (data_ptr);                                                                                    \
        (tensor).dynamic_shapes[0] = (dim0);                                                                           \
        (tensor).dynamic_shapes[1] = (dim1);                                                                           \
        (tensor).dynamic_shapes[2] = (dim2);                                                                           \
        (tensor).dynamic_shapes[3] = (dim3);                                                                           \
        (tensor).dynamic_strides[0] = static_cast<int64_t>(dim1) * (dim2) * (dim3);                                    \
        (tensor).dynamic_strides[1] = static_cast<int64_t>(dim2) * (dim3);                                             \
        (tensor).dynamic_strides[2] = static_cast<int64_t>(dim3);                                                      \
    } while (0)

#define SET_3D_TENSOR(tensor, data_ptr, dim0, dim1, dim2)                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        (tensor).data = (data_ptr);                                                                                    \
        (tensor).dynamic_shapes[0] = (dim0);                                                                           \
        (tensor).dynamic_shapes[1] = (dim1);                                                                           \
        (tensor).dynamic_shapes[2] = (dim2);                                                                           \
        (tensor).dynamic_strides[0] = static_cast<int64_t>(dim1) * (dim2);                                             \
        (tensor).dynamic_strides[1] = static_cast<int64_t>(dim2);                                                      \
    } while (0)

#define SET_1D_TENSOR(tensor, data_ptr, dim0)                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        (tensor).data = (data_ptr);                                                                                    \
        (tensor).dynamic_shapes[0] = (dim0);                                                                           \
    } while (0)

bool CuteDslGDNRunner::canImplement(int32_t kDim, int32_t vDim, int32_t smVersion)
{
    return (smVersion >= 80) && (kDim == 128) && (vDim == 128);
}

bool CuteDslGDNRunner::loadKernelModules()
{
    std::lock_guard<std::mutex> lock(sGDNMutex);
    if (sLoaded)
    {
        return true;
    }
    try
    {
        gdn_decode_Kernel_Module_Load(&sDecodeModule);
        gdn_prefill_Kernel_Module_Load(&sPrefillModule);
        sLoaded = true;
        LOG_DEBUG("CuTe DSL GDN kernel modules (decode + prefill) loaded");
        return true;
    }
    catch (...)
    {
        LOG_ERROR("Failed to load CuTe DSL GDN kernel modules");
        return false;
    }
}

void CuteDslGDNRunner::unloadKernelModules()
{
    std::lock_guard<std::mutex> lock(sGDNMutex);
    if (sLoaded)
    {
        gdn_decode_Kernel_Module_Unload(&sDecodeModule);
        gdn_prefill_Kernel_Module_Unload(&sPrefillModule);
        sLoaded = false;
    }
}

int CuteDslGDNRunner::run(GDNParams const& params, cudaStream_t stream)
{
    return (params.seq_len == 1) ? runDecode(params, stream) : runPrefill(params, stream);
}

int CuteDslGDNRunner::runDecode(GDNParams const& params, cudaStream_t stream)
{
    if (!sLoaded)
    {
        LOG_ERROR("CuTe DSL GDN decode kernel module not loaded.");
        return -1;
    }
    int32_t const n = params.n;
    int32_t const h = params.h;
    int32_t const hv = params.hv;
    int32_t const k = params.k_dim;
    int32_t const v = params.v_dim;

    if (params.seq_len != 1)
    {
        LOG_ERROR("CuTe DSL GDN decode requires seq_len == 1, got %d", params.seq_len);
        return -1;
    }

    gdn_decode_Tensor_q_t qTensor{};
    SET_4D_TENSOR(qTensor, params.q, n, 1, h, k);

    gdn_decode_Tensor_k_t kTensor{};
    SET_4D_TENSOR(kTensor, params.k, n, 1, h, k);

    gdn_decode_Tensor_v_t vTensor{};
    SET_4D_TENSOR(vTensor, params.v, n, 1, hv, v);

    gdn_decode_Tensor_a_t aTensor{};
    SET_3D_TENSOR(aTensor, params.a, n, 1, hv);

    gdn_decode_Tensor_b_t bTensor{};
    SET_3D_TENSOR(bTensor, params.b, n, 1, hv);

    gdn_decode_Tensor_A_log_t A_logTensor{};
    SET_1D_TENSOR(A_logTensor, params.A_log, hv);

    gdn_decode_Tensor_dt_bias_t dt_biasTensor{};
    SET_1D_TENSOR(dt_biasTensor, params.dt_bias, hv);

    gdn_decode_Tensor_h0_source_t h0_sourceTensor{};
    h0_sourceTensor.data = params.h0_source;
    h0_sourceTensor.dynamic_shapes[0] = n;
    h0_sourceTensor.dynamic_shapes[1] = params.hv;
    h0_sourceTensor.dynamic_strides[0] = static_cast<int64_t>(params.hv) * params.k_dim * params.v_dim;

    gdn_decode_Tensor_context_lengths_t contextLengthsTensor{};
    SET_1D_TENSOR(contextLengthsTensor, params.context_lengths, n);

    gdn_decode_Tensor_o_t oTensor{};
    SET_4D_TENSOR(oTensor, params.o, n, 1, hv, v);

    cute_dsl_gdn_decode_wrapper(&sDecodeModule, &qTensor, &kTensor, &vTensor, &aTensor, &bTensor, &A_logTensor,
        &dt_biasTensor, &h0_sourceTensor, &contextLengthsTensor, &oTensor, stream);

    return 0;
}

int CuteDslGDNRunner::runPrefill(GDNParams const& params, cudaStream_t stream)
{
    if (!sLoaded)
    {
        LOG_ERROR("CuTe DSL GDN prefill kernel module not loaded.");
        return -1;
    }
    int32_t const n = params.n;
    int32_t const seq_len = params.seq_len;
    int32_t const h = params.h;
    int32_t const hv = params.hv;
    int32_t const k = params.k_dim;
    int32_t const v = params.v_dim;

    gdn_prefill_Tensor_q_t qTensor{};
    SET_4D_TENSOR(qTensor, params.q, n, seq_len, h, k);

    gdn_prefill_Tensor_k_t kTensor{};
    SET_4D_TENSOR(kTensor, params.k, n, seq_len, h, k);

    gdn_prefill_Tensor_v_t vTensor{};
    SET_4D_TENSOR(vTensor, params.v, n, seq_len, hv, v);

    gdn_prefill_Tensor_a_t aTensor{};
    SET_3D_TENSOR(aTensor, params.a, n, seq_len, hv);

    gdn_prefill_Tensor_b_t bTensor{};
    SET_3D_TENSOR(bTensor, params.b, n, seq_len, hv);

    gdn_prefill_Tensor_A_log_t A_logTensor{};
    SET_1D_TENSOR(A_logTensor, params.A_log, hv);

    gdn_prefill_Tensor_dt_bias_t dt_biasTensor{};
    SET_1D_TENSOR(dt_biasTensor, params.dt_bias, hv);

    gdn_prefill_Tensor_h0_source_t h0_sourceTensor{};
    h0_sourceTensor.data = params.h0_source;
    h0_sourceTensor.dynamic_shapes[0] = n;
    h0_sourceTensor.dynamic_shapes[1] = params.hv;
    h0_sourceTensor.dynamic_strides[0] = static_cast<int64_t>(params.hv) * params.k_dim * params.v_dim;

    gdn_prefill_Tensor_context_lengths_t contextLengthsTensor{};
    SET_1D_TENSOR(contextLengthsTensor, params.context_lengths, n);

    gdn_prefill_Tensor_o_t oTensor{};
    SET_4D_TENSOR(oTensor, params.o, n, seq_len, hv, v);

    cute_dsl_gdn_prefill_wrapper(&sPrefillModule, &qTensor, &kTensor, &vTensor, &aTensor, &bTensor, &A_logTensor,
        &dt_biasTensor, &h0_sourceTensor, &contextLengthsTensor, &oTensor, seq_len, stream);

    return 0;
}

} // namespace trt_edgellm

#endif // CUTE_DSL_GDN_ENABLED
