/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
#include "../contextFMHARunner.h"
#include "refAttention.h"
#include "testWrapper.h"

#include <algorithm>
#include <cuda.h>
#include <fstream>
#include <iostream>

inline void check(bool condition, std::string errorMsg)
{
    if (!condition)
    {
        throw std::runtime_error(errorMsg);
    }
}

template <typename T>
class ManagedMemBuf
{
public:
    ManagedMemBuf(size_t nbElems)
        : mSize{nbElems}
    {
        if (nbElems != 0)
        {
            void* p;
            checkCuda(cudaMallocManaged(&p, sizeof(T) * nbElems));
            mData.reset(reinterpret_cast<T*>(p));
        }
    }
    T* get() const
    {
        return mData.get();
    }
    size_t size() const
    {
        return mSize;
    }
    void prefetch(int dstDevice, cudaStream_t stream = nullptr) const
    {
        checkCuda(cudaMemPrefetchAsync(get(), sizeof(T) * size(), dstDevice, stream));
    }
    T& operator[](size_t i) const
    {
        return mData[i];
    };

private:
    struct CudaDeleter
    {
        void operator()(void* p) const
        {
            cudaFree(p);
        }
    };
    std::unique_ptr<T[], CudaDeleter> mData;
    size_t mSize;
};

void loadDataFromFile(std::string name, half* dataPtr, int32_t nbData)
{
    std::ifstream file(name, std::ios::binary);
    for (int32_t i = 0; i < nbData; ++i)
    {
        float value;
        file.read(reinterpret_cast<char*>(&value), sizeof(float));
        dataPtr[i] = __float2half_rn(value);
    }
    file.close();
}

void loadDataFromFile(std::string name, std::vector<float>& dataVec, int32_t nbData)
{
    std::ifstream file(name, std::ios::binary);
    for (int32_t i = 0; i < nbData; ++i)
    {
        float value;
        file.read(reinterpret_cast<char*>(&value), sizeof(float));
        dataVec[i] = value;
    }
    file.close();
}

void loadDataFromFile(std::string name, std::vector<half>& dataVec, int32_t nbData)
{
    std::ifstream file(name, std::ios::binary);
    for (int32_t i = 0; i < nbData; ++i)
    {
        float value;
        file.read(reinterpret_cast<char*>(&value), sizeof(float));
        dataVec[i] = __float2half_rn(value);
    }
    file.close();
}

// Run flash-attention GQA test.
template <int32_t NB_K_HEADS, int32_t MAX_SEQ_LEN>
void runFMHATest(int32_t batchSize, int32_t seqLen, bool testPerf, bool refCheck)
{
    constexpr uint32_t nbKHeads = NB_K_HEADS;
    constexpr uint32_t nbVHeads = NB_K_HEADS;
    constexpr uint32_t nbQHeads = NB_K_HEADS * headGrpSize;
    constexpr uint32_t nbQKVHeads = nbQHeads + nbKHeads + nbVHeads;

    checkCuda(cudaFree(nullptr));
    int device;
    checkCuda(cudaGetDevice(&device));
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, device));

    // Prepare QKV host/device buffer for MQA.
    // QKV for padded sequence length has shape [B, MAX_SEQ_LEN, (Hq + 2Hkv), D]
    // Output result has shape [B, MAX_SEQ_LEN, Hq, D]
    auto qkvData = ManagedMemBuf<IOHead[MAX_SEQ_LEN][nbQKVHeads]>(batchSize);
    auto outdata = ManagedMemBuf<IOHead[MAX_SEQ_LEN][nbQHeads]>(batchSize);
    auto const seqLenList = ManagedMemBuf<int32_t>(batchSize + 1);
    size_t const totalQKVElems = validElemsPerHead * (nbKHeads + nbVHeads + nbQHeads) * MAX_SEQ_LEN * batchSize;
    size_t const totalVElems = validElemsPerHead * (nbKHeads) *MAX_SEQ_LEN * batchSize;
    size_t const totalOutElems = validElemsPerHead * nbQHeads * MAX_SEQ_LEN * batchSize;

    printf("Total QKV:%ld, and total out: %ld elements.\n", totalQKVElems, totalOutElems);
    InputElem const qkvFillVal = InputElem(0.1f);
    InputElem const outFillVal = InputElem(NAN);
    std::fill_n(qkvData[0][0][0].data, totalQKVElems, qkvFillVal);
    std::fill_n(outdata[0][0][0].data, totalOutElems, outFillVal);

    // fmha kernel requires to provide the prefix-sum of batch of sequence length
    seqLenList[0] = 0;
    seqLenList[1] = seqLen;

    loadDataFromFile("../tests/fmha-io/qkv128_128.bin", qkvData[0][0][0].data, totalQKVElems);

    cudaStream_t const stream = nullptr;
    auto prefetchToDevice = [&](int dev) {
        qkvData.prefetch(dev, stream);
        outdata.prefetch(dev, stream);
        seqLenList.prefetch(dev, stream);
    };
    prefetchToDevice(device);
    checkCuda(cudaStreamSynchronize(stream));

    drivellm::ContextFMHARunner runner(
        nvinfer1::DataType::kHALF, batchSize, MAX_SEQ_LEN, nbQHeads, nbKHeads, validElemsPerHead, 86);
    Fused_multihead_attention_params_v2 params;
    params.clear();
    runner.setupParams(params);
    bool status = runner.prepareToRun();
    check(status != 0, "Error in fetch kernel list.");

    params.qkv_ptr = &(qkvData[0][0][0]);
    params.o_ptr = &(outdata[0][0][0]);
    params.cu_q_seqlens = &(seqLenList[0]);

    for (int i = 0; i < 1; ++i)
    {
        runner.dispatchFMHAKernel(params, stream);
        checkCuda(cudaStreamSynchronize(stream));
    }

    checkCuda(cudaStreamSynchronize(stream));
    checkCuda(cudaGetLastError());
    if (refCheck)
    {
        std::vector<float> refDataVec(totalOutElems);
        loadDataFromFile("../tests/fmha-io/out128_128.bin", refDataVec, totalOutElems);
        bool pass{true};

        for (int32_t req = 0; req < batchSize; req++)
        {
            for (int32_t s = 0; s < MAX_SEQ_LEN; s++)
            {
                for (int32_t q = 0; q < nbQHeads; q++)
                {
                    for (int32_t i = 0; i < validElemsPerHead; i++)
                    {
                        float data = float(outdata[req][s][q][i]);
                        int32_t refIdx = req * (MAX_SEQ_LEN * nbQHeads * validElemsPerHead)
                            + s * (nbQHeads * validElemsPerHead) + q * validElemsPerHead + i;
                        float refData = refDataVec[refIdx];
                        if (std::abs(data - refData) > 1e-3)
                        {
                            printf("At %d %d %d %d, data: %f refdata: %f. \n", req, s, q, i, data, refData);
                            pass = false;
                            check(pass, "Expect output match with saved data.");
                        }
                    }
                }
            }
        }

        check(pass, "Expect output match with saved data.");
    }
}

TEST_CASE(sanity, fmha_llama_V3_8b_128)
{
    runFMHATest<8, 128>(1, 128, true, true);
}