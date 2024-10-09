
#include "../pluginUtils.h"
#include "../utilKernels.h"
#include "testWrapper.h"

#include <algorithm>
#include <cmath>
#include <random>

// Use 1 / 10000 for RoPE base frequency
// This value is configurable
constexpr float kROPE_BASE_FREQUENCY = 10000.f;
constexpr float kROPE_SCALE = 1.0f;
constexpr PositionEmbeddingType kROPE_TYPE = PositionEmbeddingType::kROPE_ORIGINAL;
constexpr RopeInitType kROPE_INIT_TYPE = RopeInitType::kDEFAULT;

float2 applyRopeTransformation(float2 src, int32_t seqIdx, int32_t tIdx, int32_t embed_dim)
{
    float const mTheta = seqIdx * kROPE_SCALE / pow(kROPE_BASE_FREQUENCY, 2 * tIdx / (float) embed_dim);
    float2 coeff{cos(mTheta), sin(mTheta)};
    // [cos  -sin] [x]
    // [sin   cos] [y]
    return {src.x * coeff.x - src.y * coeff.y, src.x * coeff.y + src.y * coeff.x};
}

bool checkDataPair(float2 left, float2 right, float abs_diff)
{
    float diffx = std::abs(left.x - right.x);
    float diffy = std::abs(left.y - right.y);

    return diffx < abs_diff && diffy < abs_diff;
}

template <int32_t NB_K_HEADS, int32_t MAX_SEQ_LEN>
void runRopeTestContext(int32_t batchSize, int32_t sequenceLen)
{
    // Context QKV host data
    constexpr int32_t nbKHeads = NB_K_HEADS;
    constexpr int32_t nbVHeads = NB_K_HEADS;
    constexpr int32_t nbQHeads = NB_K_HEADS * 4;
    constexpr int32_t sizePerHead = 128;

    // We use 2 * MAX_SEQ_LEN as KVcache capacity
    constexpr int32_t kvcacheCapacity = MAX_SEQ_LEN * 2;

    checkCuda(cudaFree(nullptr));
    int device;
    checkCuda(cudaGetDevice(&device));
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, device));

    size_t const totalQKVElems = sizePerHead * (nbKHeads + nbVHeads + nbQHeads) * MAX_SEQ_LEN * batchSize;
    size_t const totalKVCacheElems = sizePerHead * (nbKHeads + nbVHeads) * kvcacheCapacity * batchSize;

    // Here we use plain buffer to store data to minic input from TRT
    std::vector<float> inputQKVData(totalQKVElems);
    std::mt19937_64 rng{0};
    std::normal_distribution<float> dist{0.f, 2.f};
    std::generate_n(inputQKVData.begin(), totalQKVElems, [&]() {
        // Convert to half first then convert back to avoid trunction error.
        return __half2float(__float2half_rn(dist(rng)));
    });

    // Prepare device buffer to contain the result
    half* qkv_device_ptr = nullptr;
    half* kvcache_ptr = nullptr;

    checkCuda(cudaMalloc(reinterpret_cast<void**>(&qkv_device_ptr), sizeof(half) * totalQKVElems));
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&kvcache_ptr), sizeof(half) * totalKVCacheElems));
    checkCuda(cudaMemset(qkv_device_ptr, 0, sizeof(half) * totalQKVElems));
    checkCuda(cudaMemset(kvcache_ptr, 0, sizeof(half) * totalKVCacheElems));

    {
        std::vector<half> temp(totalQKVElems);
        for (int32_t i = 0; i < totalQKVElems; ++i)
        {
            temp[i] = __float2half_rn(inputQKVData[i]);
        }
        checkCuda(cudaMemcpy(qkv_device_ptr, temp.data(), sizeof(half) * totalQKVElems, cudaMemcpyHostToDevice));
    }

    // Prepare sequence length buffer
    int32_t* seqlen_device_ptr = nullptr;
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&seqlen_device_ptr), sizeof(int32_t) * batchSize));
    checkCuda(cudaMemset(seqlen_device_ptr, 0, sizeof(int32_t) * batchSize));
    {
        std::vector<int32_t> temp(batchSize, sequenceLen);
        checkCuda(cudaMemcpy(seqlen_device_ptr, temp.data(), sizeof(int32_t) * batchSize, cudaMemcpyHostToDevice));
    }

    cudaStream_t const stream = nullptr;
    invokeContextApplyRopeUpdateKVFP16(qkv_device_ptr, nullptr, kvcache_ptr, seqlen_device_ptr, nbQHeads, nbKHeads,
        sizePerHead, kvcacheCapacity, MAX_SEQ_LEN, kROPE_TYPE, kROPE_BASE_FREQUENCY, kROPE_SCALE, kROPE_INIT_TYPE,
        MAX_SEQ_LEN * batchSize, stream);

    checkCuda(cudaStreamSynchronize(stream));
    checkCuda(cudaGetLastError());

    // Check output data contents, based on the nature of rope, we will compare the data pair by pair.
    std::vector<float> kvCacheHost(totalKVCacheElems);
    std::vector<float> qkvUpdatedData(totalQKVElems);

    {
        std::vector<half> temp_kvCacheHost(totalKVCacheElems);
        std::vector<half> temp_qkvUpdatedData(totalQKVElems);
        checkCuda(cudaMemcpy(
            temp_qkvUpdatedData.data(), qkv_device_ptr, sizeof(half) * totalQKVElems, cudaMemcpyDeviceToHost));
        checkCuda(
            cudaMemcpy(temp_kvCacheHost.data(), kvcache_ptr, sizeof(half) * totalKVCacheElems, cudaMemcpyDeviceToHost));

        std::transform(temp_kvCacheHost.begin(), temp_kvCacheHost.end(), kvCacheHost.begin(),
            [](half value) { return __half2float(value); });
        std::transform(temp_qkvUpdatedData.begin(), temp_qkvUpdatedData.end(), qkvUpdatedData.begin(),
            [](half value) { return __half2float(value); });
    }

    for (int32_t b = 0; b < batchSize; ++b)
    {
        for (int32_t s = 0; s < sequenceLen; ++s)
        {
            for (int32_t h = 0; h < nbQHeads; ++h)
            {
                for (int32_t i = 0; i < sizePerHead / 2; ++i)
                {
                    if (h % 4 == 0)
                    {
                        int32_t const h_kv = h / 4;
                        // First check V, which doesn't require any rope transformation.
                        int32_t srcOffsetV = b * (sizePerHead * (nbKHeads + nbVHeads + nbQHeads) * MAX_SEQ_LEN)
                            + s * (sizePerHead * (nbKHeads + nbVHeads + nbQHeads)) + (nbQHeads + nbKHeads) * sizePerHead
                            + h_kv * sizePerHead + i * 2;
                        float2 vSrcPair{inputQKVData[srcOffsetV], inputQKVData[srcOffsetV + 1]};
                        float2 vDstPair{qkvUpdatedData[srcOffsetV], qkvUpdatedData[srcOffsetV + 1]};

                        int32_t kvCacheOffsetV = b * (sizePerHead * (nbKHeads + nbVHeads) * kvcacheCapacity)
                            + nbKHeads * (sizePerHead * kvcacheCapacity) + h_kv * (sizePerHead * kvcacheCapacity)
                            + s * sizePerHead + i * 2;
                        float2 vKvPair{kvCacheHost[kvCacheOffsetV], kvCacheHost[kvCacheOffsetV + 1]};

                        bool status = checkDataPair(vSrcPair, vKvPair, 1e-8);
                        if (!status)
                        {
                            printf("At %d %d %d %d. The value of src : %f %f .\n", b, s, h, i, vSrcPair.x, vSrcPair.y);
                            printf("At %d %d %d %d. The value of dst : %f %f .\n", b, s, h, i, vKvPair.x, vKvPair.y);
                            check(checkDataPair(vSrcPair, vKvPair, 1e-8), "Check v of kcache");
                        }

                        // --------------------------------------check v completes ------------------------------------
                        // Then check Value of K, which apply rope transformation
                        int32_t srcOffsetK = b * (sizePerHead * (nbKHeads + nbVHeads + nbQHeads) * MAX_SEQ_LEN)
                            + s * (sizePerHead * (nbKHeads + nbVHeads + nbQHeads)) + nbQHeads * sizePerHead
                            + h_kv * sizePerHead + i * 2;
                        float2 kSrcPair{inputQKVData[srcOffsetK], inputQKVData[srcOffsetK + 1]};
                        float2 kDstPair{qkvUpdatedData[srcOffsetK], qkvUpdatedData[srcOffsetK + 1]};
                        float2 kRopePair = applyRopeTransformation(kSrcPair, s, i, sizePerHead);

                        status = checkDataPair(kDstPair, kRopePair, 5e-3);
                        if (!status)
                        {
                            printf(
                                "At %d %d %d %d. The value of src QKV : %f %f .\n", b, s, h, i, kSrcPair.x, kSrcPair.y);
                            printf(
                                "At %d %d %d %d. The value of dst QKV : %f %f .\n", b, s, h, i, kDstPair.x, kDstPair.y);
                            printf("At %d %d %d %d. The value of reference : %f %f .\n", b, s, h, i, kRopePair.x,
                                kRopePair.y);
                            check(checkDataPair(kDstPair, kRopePair, 5e-3), "Check k of qkv");
                        }

                        int32_t kvCacheOffsetK = b * (sizePerHead * (nbKHeads + nbVHeads) * kvcacheCapacity)
                            + h_kv * (sizePerHead * kvcacheCapacity) + s * sizePerHead + i * 2;
                        float2 kKvPair{kvCacheHost[kvCacheOffsetK], kvCacheHost[kvCacheOffsetK + 1]};
                        status = checkDataPair(kDstPair, kKvPair, 1e-8);
                        if (!status)
                        {
                            printf("At %d %d %d %d. The value of dst : %f %f .\n", b, s, h, i, kDstPair.x, kDstPair.y);
                            printf(
                                "At %d %d %d %d. The value of kvcache : %f %f .\n", b, s, h, i, kKvPair.x, kKvPair.y);
                            check(checkDataPair(vSrcPair, vKvPair, 1e-8), "Check v of kcache");
                        }
                        // --------------------------------------check k completes ------------------------------------
                    }

                    // Check Value of Q
                    int32_t srcOffsetQ = b * (sizePerHead * (nbKHeads + nbVHeads + nbQHeads) * MAX_SEQ_LEN)
                        + s * (sizePerHead * (nbKHeads + nbVHeads + nbQHeads)) + h * sizePerHead + i * 2;
                    float2 qSrcPair{inputQKVData[srcOffsetQ], inputQKVData[srcOffsetQ + 1]};
                    float2 qDstPair{qkvUpdatedData[srcOffsetQ], qkvUpdatedData[srcOffsetQ + 1]};
                    float2 qRopePair = applyRopeTransformation(qSrcPair, s, i, sizePerHead);
                    bool status = checkDataPair(qDstPair, qRopePair, 5e-3);
                    if (!status)
                    {
                        printf("At %d %d %d %d. The value of src QKV : %f %f .\n", b, s, h, i, qSrcPair.x, qSrcPair.y);
                        printf("At %d %d %d %d. The value of dst QKV : %f %f .\n", b, s, h, i, qDstPair.x, qDstPair.y);
                        printf(
                            "At %d %d %d %d. The value of reference : %f %f .\n", b, s, h, i, qRopePair.x, qRopePair.y);
                        check(checkDataPair(qDstPair, qRopePair, 5e-3), "Check q of qkv");
                    }
                }
            }
        }
    }
}

template <int32_t NB_K_HEADS, int32_t MAX_SEQ_LEN>
void runRopeTestGeneration(int32_t batchSize, int32_t sequenceLen)
{
    // Context QKV host data
    constexpr int32_t nbKHeads = NB_K_HEADS;
    constexpr int32_t nbVHeads = NB_K_HEADS;
    constexpr int32_t nbQHeads = NB_K_HEADS * 4;
    constexpr int32_t sizePerHead = 128;

    // We use 2 * MAX_SEQ_LEN as KVcache capacity
    constexpr int32_t kvcacheCapacity = MAX_SEQ_LEN * 2;

    checkCuda(cudaFree(nullptr));
    int device;
    checkCuda(cudaGetDevice(&device));
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, device));

    size_t const totalQKVElems = sizePerHead * (nbKHeads + nbVHeads + nbQHeads) * 1 * batchSize;
    size_t const totalKVCacheElems = sizePerHead * (nbKHeads + nbVHeads) * kvcacheCapacity * batchSize;
    size_t const totalQElems = sizePerHead * (nbQHeads) *1 * batchSize;

    // Here we use plain buffer to store data to minic input from TRT
    std::vector<float> inputQKVData(totalQKVElems);
    std::vector<float> inputKVcacheData(totalKVCacheElems, 0.f);

    std::mt19937_64 rng{0};
    std::normal_distribution<float> dist{0.f, 2.f};
    std::generate_n(inputQKVData.begin(), totalQKVElems, [&]() {
        // Convert to half first then convert back to avoid trunction error.
        return __half2float(__float2half_rn(dist(rng)));
    });

    // Prepare kvcache data till sequenceId - 1
    for (int b = 0; b < batchSize; ++b)
    {
        for (int h = 0; h < (nbKHeads + nbVHeads); ++h)
        {
            for (int s = 0; s < sequenceLen - 1; ++s)
            {
                for (int i = 0; i < sizePerHead; ++i)
                {
                    int offset = b * (nbKHeads + nbVHeads) * kvcacheCapacity * sizePerHead
                        + h * kvcacheCapacity * sizePerHead + s * sizePerHead + i;
                    inputKVcacheData[offset] = __half2float(__float2half_rn(dist(rng)));
                }
            }
        }
    }
    // Prepare device buffer to contain the result
    half* qkv_device_ptr = nullptr;
    half* kvcache_ptr = nullptr;
    half* q_ptr = nullptr;
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&qkv_device_ptr), sizeof(half) * totalQKVElems));
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&kvcache_ptr), sizeof(half) * totalKVCacheElems));
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&q_ptr), sizeof(half) * totalQElems));
    checkCuda(cudaMemset(qkv_device_ptr, 0, sizeof(half) * totalQKVElems));
    checkCuda(cudaMemset(kvcache_ptr, 0, sizeof(half) * totalKVCacheElems));
    checkCuda(cudaMemset(q_ptr, 0, sizeof(half) * totalQElems));

    {
        std::vector<half> temp(totalQKVElems);
        for (int32_t i = 0; i < totalQKVElems; ++i)
        {
            temp[i] = __float2half_rn(inputQKVData[i]);
        }
        checkCuda(cudaMemcpy(qkv_device_ptr, temp.data(), sizeof(half) * totalQKVElems, cudaMemcpyHostToDevice));
    }
    {
        std::vector<half> temp(totalKVCacheElems);
        for (int32_t i = 0; i < totalKVCacheElems; ++i)
        {
            temp[i] = __float2half_rn(inputKVcacheData[i]);
        }
        checkCuda(cudaMemcpy(kvcache_ptr, temp.data(), sizeof(half) * totalKVCacheElems, cudaMemcpyHostToDevice));
    }

    // Prepare sequence length buffer
    int32_t* seqlen_device_ptr = nullptr;
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&seqlen_device_ptr), sizeof(int32_t) * batchSize));
    checkCuda(cudaMemset(seqlen_device_ptr, 0, sizeof(int32_t) * batchSize));
    {
        std::vector<int32_t> temp(batchSize, sequenceLen);
        checkCuda(cudaMemcpy(seqlen_device_ptr, temp.data(), sizeof(int32_t) * batchSize, cudaMemcpyHostToDevice));
    }

    cudaStream_t const stream = nullptr;
    invokeGenerationApplyRopeUpdateKVFP16(qkv_device_ptr, q_ptr, kvcache_ptr, seqlen_device_ptr, nbQHeads, nbKHeads,
        sizePerHead, kvcacheCapacity, MAX_SEQ_LEN, kROPE_TYPE, kROPE_BASE_FREQUENCY, kROPE_SCALE, kROPE_INIT_TYPE,
        batchSize, stream);

    // Check output data contents, based on the nature of rope, we will compare the data pair by pair.
    std::vector<float> kvCacheHost(totalKVCacheElems);
    std::vector<float> qUpdatedData(totalQElems);
    {
        std::vector<half> temp_kvCacheHost(totalKVCacheElems);
        std::vector<half> temp_qUpdatedData(totalQElems);
        checkCuda(cudaMemcpy(temp_qUpdatedData.data(), q_ptr, sizeof(half) * totalQElems, cudaMemcpyDeviceToHost));
        checkCuda(
            cudaMemcpy(temp_kvCacheHost.data(), kvcache_ptr, sizeof(half) * totalKVCacheElems, cudaMemcpyDeviceToHost));

        std::transform(temp_kvCacheHost.begin(), temp_kvCacheHost.end(), kvCacheHost.begin(),
            [](half value) { return __half2float(value); });
        std::transform(temp_qUpdatedData.begin(), temp_qUpdatedData.end(), qUpdatedData.begin(),
            [](half value) { return __half2float(value); });
    }

    // check Q data has been placed and computed correctly
    for (int b = 0; b < batchSize; ++b)
    {
        for (int h = 0; h < nbQHeads; ++h)
        {
            for (int i = 0; i < sizePerHead / 2; ++i)
            {
                int srcOffsetQ = b * (nbQHeads + nbKHeads + nbVHeads) * sizePerHead + h * sizePerHead + i * 2;
                int dstOffsetQ = b * nbQHeads * sizePerHead + h * sizePerHead + i * 2;
                float2 qSrcPair{inputQKVData[srcOffsetQ], inputQKVData[srcOffsetQ + 1]};
                float2 qDstPair{qUpdatedData[dstOffsetQ], qUpdatedData[dstOffsetQ + 1]};
                float2 qRopePair = applyRopeTransformation(qSrcPair, sequenceLen - 1, i, sizePerHead);
                bool status = checkDataPair(qDstPair, qRopePair, 5e-3);
                if (!status)
                {
                    printf("At %d %d %d. The value of src QKV : %f %f .\n", b, h, i, qSrcPair.x, qSrcPair.y);
                    printf("At %d %d %d. The value of dst Q : %f %f .\n", b, h, i, qDstPair.x, qDstPair.y);
                    printf("At %d %d %d. The value of rope : %f %f .\n", b, h, i, qRopePair.x, qRopePair.y);
                    check(checkDataPair(qDstPair, qRopePair, 5e-3), "Check q of qkv");
                }
            }
        }
    }

    // Check K and V data have been written to KVcache
    for (int b = 0; b < batchSize; ++b)
    {
        for (int h_kv = 0; h_kv < nbKHeads; ++h_kv)
        {
            for (int i = 0; i < sizePerHead / 2; ++i)
            {
                // First check V which doesn't require rope
                int srcOffsetV = b * (nbQHeads + nbKHeads + nbVHeads) * sizePerHead
                    + (nbQHeads + nbKHeads) * sizePerHead + h_kv * sizePerHead + i * 2;
                int kvCacheOffsetV = b * (nbKHeads + nbVHeads) * kvcacheCapacity * sizePerHead
                    + nbKHeads * kvcacheCapacity * sizePerHead + h_kv * kvcacheCapacity * sizePerHead
                    + (sequenceLen - 1) * sizePerHead + i * 2;
                float2 vSrcPair{inputQKVData[srcOffsetV], inputQKVData[srcOffsetV + 1]};
                float2 vDstPair{kvCacheHost[kvCacheOffsetV], kvCacheHost[kvCacheOffsetV + 1]};
                bool status = checkDataPair(vSrcPair, vDstPair, 1e-8);
                if (!status)
                {
                    printf("At %d %d %d. The value of src QKV : %f %f .\n", b, h_kv, i, vSrcPair.x, vSrcPair.y);
                    printf("At %d %d %d. The value of dst KV-cache : %f %f .\n", b, h_kv, i, vDstPair.x, vDstPair.y);
                    check(checkDataPair(vSrcPair, vDstPair, 1e-8), "Check v of kvcache");
                }

                // Then check K which requires rope
                int srcOffsetK = b * (nbQHeads + nbKHeads + nbVHeads) * sizePerHead + nbQHeads * sizePerHead
                    + h_kv * sizePerHead + i * 2;
                int kvCacheOffsetK = b * (nbKHeads + nbVHeads) * kvcacheCapacity * sizePerHead
                    + h_kv * kvcacheCapacity * sizePerHead + (sequenceLen - 1) * sizePerHead + i * 2;
                float2 kSrcPair{inputQKVData[srcOffsetK], inputQKVData[srcOffsetK + 1]};
                float2 kDstPair{kvCacheHost[kvCacheOffsetK], kvCacheHost[kvCacheOffsetK + 1]};
                float2 kRopePair = applyRopeTransformation(kSrcPair, sequenceLen - 1, i, sizePerHead);
                status = checkDataPair(kRopePair, kDstPair, 5e-3);
                if (!status)
                {
                    printf("At %d %d %d. The value of src QKV : %f %f .\n", b, h_kv, i, kSrcPair.x, kSrcPair.y);
                    printf("At %d %d %d. The dst value of KVcache : %f %f .\n", b, h_kv, i, kDstPair.x, kDstPair.y);
                    printf("At %d %d %d. The value rope result : %f %f .\n", b, h_kv, i, kRopePair.x, kRopePair.y);
                    check(checkDataPair(kRopePair, kDstPair, 5e-3), "Check k of kvcache");
                }
            }
        }
    }
}

TEST_CASE(sanity, rope_kv_context)
{
    // Check 128 max len with 64 context len
    runRopeTestContext<8, 128>(1, 64);
}

TEST_CASE(sanity, rope_kv_context_multi_batch)
{
    // Check 128 max len with 64 context len under batch 2
    runRopeTestContext<8, 128>(2, 64);
}

TEST_CASE(sanity, rope_kv_generation)
{
    // Check 128 max input-len with 200 context len
    runRopeTestGeneration<8, 128>(1, 200);
}

TEST_CASE(sanity, rope_kv_generation_multi_batch)
{
    // Check 128 max input-len with 200 context len under batch 2
    runRopeTestGeneration<8, 128>(2, 200);
}