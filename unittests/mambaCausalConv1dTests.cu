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

#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <vector>

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/mamba/causalConv1d.h"
#include "testUtils.h"

using namespace trt_edgellm;
using namespace nvinfer1;

namespace
{

void runCausalConv1dReference(int32_t batch, int32_t seqLen, int32_t dim, int32_t width, int32_t padding,
    std::vector<half> const& x, std::vector<half> const& weight, std::vector<half> const& bias,
    std::vector<half>& outRef)
{
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t s = 0; s < seqLen; ++s)
        {
            int32_t const inBase = s - padding;
            for (int32_t d = 0; d < dim; ++d)
            {
                float acc = __half2float(bias[d]);
                for (int32_t k = 0; k < width; ++k)
                {
                    int32_t const inPos = inBase + k;
                    if (inPos >= 0 && inPos < seqLen)
                    {
                        int64_t const xIdx
                            = static_cast<int64_t>(b) * seqLen * dim + static_cast<int64_t>(inPos) * dim + d;
                        int64_t const wIdx = static_cast<int64_t>(d) * width + k;
                        acc += __half2float(x[xIdx]) * __half2float(weight[wIdx]);
                    }
                }
                int64_t const outIdx = static_cast<int64_t>(b) * seqLen * dim + static_cast<int64_t>(s) * dim + d;
                outRef[outIdx] = __float2half(acc);
            }
        }
    }
}

void runCausalConv1dTest(int32_t batch, int32_t seqLen, int32_t dim, int32_t width)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5F, 0.5F);

    std::vector<half> xHost(batch * seqLen * dim);
    std::vector<half> weightHost(dim * width);
    std::vector<half> biasHost(dim);
    std::vector<half> outputRef(batch * seqLen * dim, __float2half(0.F));

    for (auto& v : xHost)
    {
        v = __float2half(dist(rng));
    }
    for (auto& v : weightHost)
    {
        v = __float2half(dist(rng));
    }
    for (auto& v : biasHost)
    {
        v = __float2half(dist(rng));
    }

    runCausalConv1dReference(batch, seqLen, dim, width, width - 1, xHost, weightHost, biasHost, outputRef);

    auto xDevice = rt::Tensor({batch, seqLen, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto weightDevice = rt::Tensor({dim, 1, width}, rt::DeviceType::kGPU, DataType::kHALF);
    auto biasDevice = rt::Tensor({dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto outputDevice = rt::Tensor({batch, seqLen, dim}, rt::DeviceType::kGPU, DataType::kHALF);

    CUDA_CHECK(cudaMemcpy(xDevice.rawPointer(), xHost.data(), xHost.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        weightDevice.rawPointer(), weightHost.data(), weightHost.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(biasDevice.rawPointer(), biasHost.data(), biasHost.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(outputDevice.rawPointer(), 0, outputDevice.getMemoryCapacity()));

    mamba_ssm::CausalConv1dTensors tensors{&xDevice, &weightDevice, &biasDevice, &outputDevice};
    mamba_ssm::invokeCausalConv1d<half>(tensors, 1, width - 1, 1, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> outputHost(outputRef.size());
    CUDA_CHECK(cudaMemcpy(
        outputHost.data(), outputDevice.rawPointer(), outputHost.size() * sizeof(half), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < outputRef.size(); ++i)
    {
        EXPECT_TRUE(isclose(outputHost[i], outputRef[i], 1e-3F, 1e-3F))
            << "Output mismatch at index " << i << ": got " << __half2float(outputHost[i]) << ", expected "
            << __half2float(outputRef[i]);
    }
}

TEST(MambaCausalConv1d, Width2)
{
    runCausalConv1dTest(2, 16, 128, 2);
}

TEST(MambaCausalConv1d, Width3)
{
    runCausalConv1dTest(2, 23, 128, 3);
}

TEST(MambaCausalConv1d, Width4)
{
    runCausalConv1dTest(2, 31, 256, 4);
}

} // namespace
