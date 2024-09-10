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

#include "../decoderXQARunner.h"
#include "refAttention.h"
#include "xqa/cubin/xqa_kernel_cubin.h"
#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <random>

#include <cuda.h>

// x: horizontal stacking for cta horizontal tile size
// y: vertical stacking for cta vertical tile size
// z: must be 2 for warp specialization.
constexpr uint3 ctaShapeInWarps = {4, 1, 2};
constexpr uint32_t nbValidRows = headGrpSize * beamWidth;
constexpr uint2 warpTile = {64, roundUp(nbValidRows, 16U)};

constexpr uint2 ctaTile = {warpTile.x * ctaShapeInWarps.x, // if .x is greater than headSize, then gemm1 uses split-K
    warpTile.y* ctaShapeInWarps.y};

inline constexpr uint32_t warpSize = 32;

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

struct KVCache
{
    GMemKVCacheHead* data;            // shape: KVCacheHead[batchSize][beamWidth][2][nbKHeads][capacity]
    SeqLenDataType const* seqLenList; // shape: [batchSize][beamWidth] (for compatibility)
    uint32_t capacity;
};

struct XQALaunchParam
{
    uint32_t numKHeads;
    IOHead* output;
    IOHead const* qVecs;
    KVCache kvCache;
    uint32_t batchSize;
    float const* kvScale = nullptr;
    uint32_t* semaphores = nullptr;
    void* scratch = nullptr;
};

template <typename D, typename S>
void save(char const* file, S const* src, size_t size)
{
    std::ofstream fout{file, std::ios::trunc};
    for (size_t i = 0; i < size; i++)
    {
        D data{src[i]};
        fout.write((char const*) &data, sizeof(D));
    }
    fout.close();
}

template <int32_t nbKHeads>
void runTest(
    int32_t batchSize, int32_t seqLen, bool testPerf, bool refCheck, bool verbose = false, bool saveData = false)
{
    constexpr int32_t nbVHeads = nbKHeads;
    constexpr int32_t nbQHeads = nbKHeads * headGrpSize;

    checkCuda(cudaFree(nullptr));
    int device;
    checkCuda(cudaGetDevice(&device));
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, device));
    if (verbose)
    {
        printf("SM count: %d\n", prop.multiProcessorCount);
        if (!refCheck && (batchSize * nbKHeads) % prop.multiProcessorCount != 0)
        {
            printf("Tail effect will impact performance.\n");
        }
    }
    bool const useQGMMA = [&]() -> bool {
        if (std::getenv("XQA_USE_QGMMA"))
        {
            return std::stoi(std::getenv("XQA_USE_QGMMA")) != 0;
        }
        return (beamWidth == 1 && std::is_same_v<CacheElem, __nv_fp8_e4m3> && prop.major == 9 && prop.minor == 0);
    }();
    if (batchSize == 0)
    {
        uint32_t const ctaPerWave = (uint32_t) prop.multiProcessorCount
            * (useQGMMA && headElems * roundUp(headGrpSize * beamWidth, 8U) <= 128 * 16 ? 3 : 2);
        batchSize = exactDiv(std::lcm(ctaPerWave, nbKHeads), nbKHeads);
    }
    if (seqLen == 0)
    {
        seqLen = (16U << 20) / gmemCacheHeadBytes; // 32MB per K+V head.
    }
    int32_t ctxLen = seqLen;
    float const kScale = cacheElemSize == 2 ? 1.f : 1 / 4.f;
    float const vScale = kScale;
    float const qkScale = sqrtf(1.f / validElemsPerHead) * kScale;
    size_t const histLen = seqLen;
    if (verbose)
    {
        printf("batchSize=%u, nbKHeads=%u, seqLen=%u, histLen=%lu\n", batchSize, nbKHeads, seqLen, histLen);
    }

    int32_t const maxSeqLen = seqLen;
    int32_t const totalNbCacheHeads = (nbKHeads + nbVHeads) * maxSeqLen * beamWidth * batchSize;
    size_t const totalNbCacheElems = validElemsPerHead * size_t(totalNbCacheHeads);
    size_t const qElems = validElemsPerHead * nbQHeads * beamWidth * batchSize;
    size_t const outElems = validElemsPerHead * nbQHeads * beamWidth * batchSize;
    size_t const cacheBytes = cacheElemSize * totalNbCacheElems;
    size_t const inputBytes = inputElemSize * qElems;
    size_t const outputBytes = inputElemSize * outElems;
    size_t const seqLenListBytes = sizeof(uint32_t) * beamWidth * batchSize;
    size_t const ctxLenListBytes = sizeof(uint32_t) * beamWidth * batchSize;

    size_t const cacheIndirBytes = beamWidth == 1 ? 0 : sizeof(uint32_t) * maxSeqLen * beamWidth * batchSize;
    size_t const totalBytes
        = cacheBytes + inputBytes + outputBytes + seqLenListBytes + ctxLenListBytes + cacheIndirBytes;
    size_t const nbSemaphores = 128u << 10;
    auto const semaphores = ManagedMemBuf<uint32_t>(nbSemaphores);
    size_t const scratchSize = (256u << 20);
    auto const scratchBuf = ManagedMemBuf<std::byte>(scratchSize);
    auto const kvCacheScale = ManagedMemBuf<float>(1);
    kvCacheScale[0] = kScale;
    cudaEvent_t tic, toc;
    checkCuda(cudaEventCreate(&tic));
    checkCuda(cudaEventCreate(&toc));
    std::unique_ptr<CUevent_st, cudaError (*)(cudaEvent_t)> const ticEv{tic, &cudaEventDestroy};
    std::unique_ptr<CUevent_st, cudaError (*)(cudaEvent_t)> const tocEv{toc, &cudaEventDestroy};

    auto const cacheHeads = ManagedMemBuf<GMemCacheHead>(totalNbCacheHeads);
    auto const qHeads = ManagedMemBuf<IOHead[beamWidth][nbQHeads]>(batchSize);
    auto const output = ManagedMemBuf<IOHead[beamWidth][nbQHeads]>(batchSize);
    auto const seqLenList = ManagedMemBuf<uint32_t[beamWidth]>(batchSize);
    auto const ctxLenList = ManagedMemBuf<uint32_t[beamWidth]>(batchSize);

    std::fill_n(&seqLenList[0][0], beamWidth * batchSize, seqLen);
    std::fill_n(&ctxLenList[0][0], beamWidth * batchSize, ctxLen);
    if (verbose)
    {
        printf("cacheHeads= %p q= %p output= %p\n", cacheHeads.get(), qHeads.get(), output.get());
        printf("cacheBytes= %lu  qByte= %lu  outbytes= %lu  totalBytes= %lu\n", cacheElemSize * totalNbCacheElems,
            inputElemSize * qElems, inputElemSize * outElems, totalBytes);
        printf("generating input data\n");
    }
    uint64_t seed = std::getenv("SEED") ? std::stoi(std::getenv("SEED")) : 0;
    std::mt19937_64 rng{seed};
    auto const cacheIndir = ManagedMemBuf<uint32_t>(beamWidth == 1 ? 0 : batchSize * beamWidth * maxSeqLen);
    if (beamWidth > 1)
    {
        std::uniform_int_distribution<uint32_t> cacheIndirDist(0, beamWidth - 1);
        for (uint32_t req = 0; req < batchSize; req++)
        {
            for (uint32_t b = 0; b < beamWidth; b++)
            {
                auto indices = cacheIndir.get() + maxSeqLen * (b + req * beamWidth);
                std::fill_n(indices, ctxLen, 0);
                std::generate_n(indices + ctxLen, seqLen - ctxLen, [&]() { return cacheIndirDist(rng); });
                std::fill_n(indices + seqLen, maxSeqLen - seqLen, ~0U);
            }
        }
    }
    bool const zeroInput = !refCheck && std::getenv("XQA_ZERO_FILL") && std::stoi(std::getenv("XQA_ZERO_FILL"));
    if (!zeroInput)
    {
        std::normal_distribution<float> dist{0.f, 1.f};
        auto genCacheElem = [&]() {
#if CACHE_ELEM_ENUM == 0
            return InputElem(dist(rng));
#elif CACHE_ELEM_ENUM == 1
            return static_cast<int8_t>(std::clamp<float>(std::round(dist(rng) / kScale), -127, 127));
#elif CACHE_ELEM_ENUM == 2
            return __nv_fp8_e4m3{dist(rng) / kScale};
#endif
        };
        std::generate_n(cacheHeads[0].data, totalNbCacheElems, genCacheElem);
        std::generate_n(qHeads[0][0][0].data, qElems, [&] { return InputElem(genCacheElem()); });
        std::fill_n(output[0][0][0].data, outElems, InputElem(NAN));
    }
    else
    {
#if CACHE_ELEM_ENUM == 0
        InputElem const cacheFillVal = InputElem(0.01f);
#elif CACHE_ELEM_ENUM == 1
        int8_t const cacheFillVal = 1;
#elif CACHE_ELEM_ENUM == 2
        __nv_fp8_e4m3 const cacheFillVal{0.01f};
#endif
        std::fill_n(&cacheHeads[0][0], totalNbCacheElems, cacheFillVal);
        std::fill_n(qHeads[0][0][0].data, qElems, InputElem(0.01f));
        std::fill_n(output[0][0][0].data, outElems, InputElem(NAN));
    }
    if (verbose)
    {
        printf("migrating data to gpu\n");
    }
    cudaStream_t const stream = nullptr;
    auto prefetchToDevice = [&](int dev) {
        semaphores.prefetch(dev, stream);
        scratchBuf.prefetch(dev, stream);
        kvCacheScale.prefetch(dev, stream);
        cacheHeads.prefetch(dev, stream);
        qHeads.prefetch(dev, stream);
        output.prefetch(dev, stream);
        seqLenList.prefetch(dev, stream);
        ctxLenList.prefetch(dev, stream);
    };
    prefetchToDevice(device);
    checkCuda(cudaMemsetAsync(semaphores.get(), 0, 4 * nbSemaphores, stream));
    checkCuda(cudaStreamSynchronize(stream));

    auto const scratch = reinterpret_cast<void*>(roundUp<uintptr_t>(reinterpret_cast<uintptr_t>(scratchBuf.get()),
        (useQGMMA ? ioHeadBytes : paddedInputHeadBytes) * headGrpSize * beamWidth)); // 8 is sufficent for qgmma kernel.

    // CUmodule cuModule;
    // CUfunction kernelFunction;
    // checkCu(cuModuleLoadData(&cuModule, cubinData));
    // checkCu(cuModuleGetFunction(&kernelFunction, cuModule, "kernel_mha"));

    // uint32_t smemSize;
    // uint32_t* deviceSmemSize{nullptr};
    // size_t dataSize{0};
    // checkCu(cuModuleGetGlobal(reinterpret_cast<CUdeviceptr*>(&deviceSmemSize), &dataSize, cuModule, "smemSize"));
    // checkCuda(cudaMemcpy(&smemSize, deviceSmemSize, dataSize, cudaMemcpyDeviceToHost));

    // if (smemSize >= 46 * 1024)
    // {
    //     checkCu(cuFuncSetAttribute(kernelFunction, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smemSize));
    // }

    // KVCache kvCache{cacheHeads.get(), &seqLenList[0][0], maxSeqLen};
    // XQALaunchParam xqaParams{nbKHeads, &output[0][0][0], &qHeads[0][0][0], kvCache, batchSize, kvCacheScale.get(),
    // semaphores.get(), scratch}; void* kernelParams[] = {&xqaParams.numKHeads, &xqaParams.output, &xqaParams.qVecs,
    // &xqaParams.kvCache, &xqaParams.batchSize,
    //     &xqaParams.kvScale, &xqaParams.semaphores, &xqaParams.scratch, nullptr};

    // uint32_t const nbSubSeqPerSeq = [&]()->uint32_t {
    //     return std::min<uint32_t>(std::max<uint32_t>(1U, prop.multiProcessorCount / (batchSize * nbKHeads)),
    //     divUp(maxSeqLen, ctaTile.x));
    // }();

    // dim3 const dimGrid{nbSubSeqPerSeq, nbKHeads, batchSize};
    // dim3 const dimCta{warpSize * ctaShapeInWarps.x, ctaShapeInWarps.y, ctaShapeInWarps.z};

    // auto runKernel = [&](){
    //     checkCu(cuLaunchKernel(kernelFunction, dimGrid.x, dimGrid.y, dimGrid.z, dimCta.x, dimCta.y, dimCta.z,
    //     smemSize, stream, kernelParams, nullptr)); checkCuda(cudaGetLastError());
    // };

    drivellm::DecoderXQARunner runner(nvinfer1::DataType::kHALF, batchSize, nbQHeads, nbVHeads, 128, 86);
    drivellm::XQALaunchParams params = runner.initXQAParams();
    runner.prepareToRun();

    params.output = &(output[0][0][0]);
    params.qInputPtr = &(qHeads[0][0][0]);
    params.kvCache.data = &(cacheHeads[0]);
    params.kvCache.sequence_lengths = reinterpret_cast<int32_t const*>(&(seqLenList[0][0]));
    params.kvCache.capacity = maxSeqLen;

    auto runKernel = [&]() {
        runner.dispatchXQAKernel(params, stream);
        checkCuda(cudaGetLastError());
    };

    checkCuda(cudaGetLastError());

    if (testPerf)
    {
        if (verbose)
        {
            printf("warming up\n");
        }

        for (int32_t i = 0; i < 20; i++)
        {
            runKernel();
        }
        if (verbose)
        {
            printf("testing\n");
        }
    }
    checkCuda(cudaEventRecord(tic, stream));
    int32_t const nbIters = 100;
    for (int32_t i = 0; i < nbIters; i++)
    {
        runKernel();
    }
    checkCuda(cudaEventRecord(toc, stream));
    prefetchToDevice(cudaCpuDeviceId);
    checkCuda(cudaStreamSynchronize(stream));
    if (testPerf)
    {
        float ms;
        checkCuda(cudaEventElapsedTime(&ms, tic, toc));
        ms /= nbIters;
        float const bandwidth = 2.f * prop.memoryBusWidth * prop.memoryClockRate * 1000 / 8;

        size_t nbLoadedCacheTokens = seqLen * beamWidth * batchSize;
        size_t const totalNbCacheLoadBytes = gmemCacheHeadBytes * (nbKHeads + nbVHeads) * nbLoadedCacheTokens;
        float const totalTraffic = totalNbCacheLoadBytes + inputElemSize * (outElems + outElems);
        float const solTime = totalTraffic / bandwidth * 1E3f;
        float const solRatio = solTime / ms;
        if (verbose)
        {
            printf("done\n");
            printf("time: %f ms\n", ms);
            printf("mem bus width = %d\nmem clock rate = %d\n", prop.memoryBusWidth, prop.memoryClockRate);
            printf("bandwidth = %e\n", (float) bandwidth);
            printf("traffic=%e\n", (float) totalTraffic);
        }
        printf("solRatio: %f%% (%f ms)\n", solRatio * 100, ms);
    }
    if (refCheck)
    {
        if (saveData)
        {
            save<float>("kv.bin", &cacheHeads[0][0], validElemsPerHead * cacheHeads.size());
            save<float>("q.bin", &qHeads[0][0][0][0], validElemsPerHead * nbQHeads * beamWidth * batchSize);
        }

        std::vector<std::array<std::array<Vec<float, validElemsPerHead>, nbQHeads>, beamWidth>> outputF32(batchSize);
#pragma omp for
        for (uint32_t req = 0; req < batchSize; req++)
        {
            for (uint32_t b = 0; b < beamWidth; b++)
            {
                for (uint32_t q = 0; q < nbQHeads; q++)
                {
                    for (uint32_t i = 0; i < validElemsPerHead; i++)
                    {
                        outputF32[req][b][q][i] = float(output[req][b][q][i]);
                    }
                }
            }
        }
        std::ofstream fout_refOutput;
        if (saveData)
        {
            save<float>("out.bin", &outputF32[0][0][0][0], validElemsPerHead * nbQHeads * beamWidth * batchSize);
            fout_refOutput = std::ofstream("ref_cpp.bin", std::ios::binary | std::ios::trunc);
        }

        constexpr float kE4M3_MAX = 448.F;
        float const xScale = useQGMMA ? 1 / kE4M3_MAX : 1.f;
        float maxErr = 0.F;
        float const allowedErr = (useQGMMA ? 0.15f : 0.005f);
        for (uint32_t req = 0; req < batchSize; req++)
        {
            for (uint32_t b = 0; b < beamWidth; b++)
            {
                for (uint32_t idxKHead = 0; idxKHead < nbKHeads; idxKHead++)
                {
                    auto const kv
                        = reinterpret_cast<GMemCacheHead(*)[beamWidth][2][nbKHeads][maxSeqLen]>(cacheHeads.get());
                    CacheSeq const kCacheSeq{kv[req][b][0][idxKHead]};
                    CacheSeq const vCacheSeq{kv[req][b][1][idxKHead]};

                    Eigen::Matrix<float, headGrpSize, validElemsPerHead, Eigen::RowMajor> refOutput;
                    if (useQGMMA)
                    {
                        refOutput = refFlashAttention<CacheElem, 64>(&qHeads[req][b][headGrpSize * idxKHead], kCacheSeq,
                            vCacheSeq, seqLen, kvCacheScale[0], xScale);
                        // refOutput = refAttention<CacheElem>(&qHeads[req][b][headGrpSize * idxKHead], kCacheSeq,
                        // vCacheSeq, seqLen, kvCacheScale[0], xScale);
                    }
                    else
                    {
                        // refOutput = refFlashAttention<InputElem, 64>(&qHeads[req][b][headGrpSize * idxKHead],
                        // kCacheSeq, vCacheSeq, seqLen, kvCacheScale[0], xScale);
                        refOutput = refAttention<InputElem>(&qHeads[req][b][headGrpSize * idxKHead], kCacheSeq,
                            vCacheSeq, seqLen, kvCacheScale[0], xScale);
                    }
                    if (saveData)
                    {
                        fout_refOutput.write((char const*) refOutput.data(), sizeof(refOutput[0]) * refOutput.size());
                    }
                    for (uint32_t i = 0; i < headGrpSize; i++)
                    {
                        for (uint32_t j = 0; j < validElemsPerHead; j++)
                        {
                            float const val = outputF32[req][b][headGrpSize * idxKHead + i][j];
                            float const ref = refOutput(i, j);
                            float const err = std::abs(val - ref);
                            EXPECT_TRUE(std::isfinite(err));
                            maxErr = std::max(maxErr, err);
                            EXPECT_NEAR(val, ref, allowedErr);
                        }
                    }
                }
            }
        }
        if (saveData)
        {
            fout_refOutput.close();
        }

        if (verbose)
        {
            printf("max abs error: %f\n", maxErr);
        }
        EXPECT_LE(maxErr, allowedErr);
    }
}

TEST(sanity, gqa_llama_V3_8b_128)
{
    runTest<8>(1, 960, true, true, true);
}