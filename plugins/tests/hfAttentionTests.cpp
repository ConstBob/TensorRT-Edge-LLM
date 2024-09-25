#include <gtest/gtest.h>

#include "../contextFMHARunner.h"
#include "../pluginUtils.h"
#include "../utilKernels.h"

#include <fstream>

namespace
{

// The sequence length is 8 when dump the reference I/O tensor from huggingface's LLAMA3 8B model.
constexpr int32_t kBATCH_SIZE = 1;
constexpr int32_t kSEQUENCE_LENGTH = 8;
constexpr int32_t kNUM_Q_HEADS = 32;
constexpr int32_t kNUM_K_HEADS = 8;
constexpr int32_t kNUM_V_HEADS = 8;
constexpr int32_t kDIM_HEAD = 128;

// We use a fix number for max number that KV-cache can contain.
constexpr int32_t kINPUT_LENGTH_PADDED = 128;
constexpr int32_t kKV_CACHE_CAPACITY = 256;

int32_t const nbQdata = kSEQUENCE_LENGTH * kNUM_Q_HEADS * kDIM_HEAD;
int32_t const nbKData = kSEQUENCE_LENGTH * kNUM_K_HEADS * kDIM_HEAD;
int32_t const nbVData = kSEQUENCE_LENGTH * kNUM_V_HEADS * kDIM_HEAD;
int32_t const nbQKVData = nbQdata + nbKData + nbVData;
int32_t const nbTotalKVCacheData = kKV_CACHE_CAPACITY * (kNUM_K_HEADS + kNUM_V_HEADS) * kDIM_HEAD;
int32_t const nvTotalQKVPaddedData = kINPUT_LENGTH_PADDED * (kNUM_Q_HEADS + kNUM_Q_HEADS + kNUM_V_HEADS) * kDIM_HEAD;
int32_t const nbTotalAttentionPaddedData = kINPUT_LENGTH_PADDED * kNUM_Q_HEADS * kDIM_HEAD;

// The default rope frequency and scale used by HF llama3 8B model
constexpr float kROPE_BASE_FREQUENCY = 500000.f;
constexpr float kROPE_SCALE = 1.0f;

// Huggingface use rotate-half rope which is different from original Meta implementation.
constexpr PositionEmbeddingType kROPE_TYPE = PositionEmbeddingType::kROPE_ROTATE_HALF;
constexpr RopeInitType kROPE_INIT_TYPE = RopeInitType::kLLAMA3;

void loadDataFromFile(std::string name, std::vector<half>& dataVec, int32_t nbData)
{
    std::ifstream file(name, std::ios::binary);
    check(file.is_open(), "Verify the data file has been correctly opened");
    for (int32_t i = 0; i < nbData; ++i)
    {
        half value;
        file.read(reinterpret_cast<char*>(&value), sizeof(half));
        dataVec[i] = value;
    }
    file.close();
}

// Concat seperate QKV tensor of shape [1, S, H_{q, k, v}, D] into [1, S, Hq+Hk+Hv, D]
std::vector<half> concatQKVData(
    std::vector<half> const& qTensor, std::vector<half> const& kTensor, std::vector<half> const& vTensor)
{
    std::vector<half> concatData;
    auto appendSequenceOfData = [&](std::vector<half> const& source, int32_t baseIdx, int32_t nbHeads) {
        int32_t const nbDataAppend = nbHeads * kDIM_HEAD;
        concatData.insert(concatData.end(), source.begin() + baseIdx, source.begin() + baseIdx + nbDataAppend);
    };

    for (int32_t sId = 0; sId < kSEQUENCE_LENGTH; ++sId)
    {
        int32_t const qOffset = sId * kNUM_Q_HEADS * kDIM_HEAD;
        appendSequenceOfData(qTensor, qOffset, kNUM_Q_HEADS);

        int32_t const kvOffset = sId * kNUM_K_HEADS * kDIM_HEAD;
        appendSequenceOfData(kTensor, kvOffset, kNUM_K_HEADS);
        appendSequenceOfData(vTensor, kvOffset, kNUM_V_HEADS);
    }

    return concatData;
}
} // namespace

void testRope()
{
    std::vector<half> inputQData(nbQdata);
    std::vector<half> inputKData(nbKData);
    std::vector<half> inputVData(nbVData);

    loadDataFromFile("../tests/hf-tensor/qtensor.bin", inputQData, nbQdata);
    loadDataFromFile("../tests/hf-tensor/ktensor.bin", inputKData, nbKData);
    loadDataFromFile("../tests/hf-tensor/vtensor.bin", inputVData, nbVData);

    // Prepare device buffers
    half* qkv_device_ptr = nullptr;
    half* kvcache_ptr = nullptr;
    int32_t* seqlen_device_ptr = nullptr;

    checkCuda(cudaMalloc(reinterpret_cast<void**>(&qkv_device_ptr), sizeof(half) * nvTotalQKVPaddedData));
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&kvcache_ptr), sizeof(half) * nbTotalKVCacheData));
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&seqlen_device_ptr), sizeof(int32_t) * 2));
    checkCuda(cudaMemset(qkv_device_ptr, 0, sizeof(half) * nbTotalKVCacheData));
    checkCuda(cudaMemset(kvcache_ptr, 0, sizeof(half) * nbTotalKVCacheData));
    checkCuda(cudaMemset(seqlen_device_ptr, 0, sizeof(int32_t) * 2));

    std::vector<half> concatQKV = concatQKVData(inputQData, inputKData, inputVData);
    check(concatQKV.size() == nbQKVData, "Check number of QKV data is consistent");
    checkCuda(cudaMemcpy(qkv_device_ptr, concatQKV.data(), sizeof(half) * nbQKVData, cudaMemcpyHostToDevice));

    std::vector<int32_t> temp{kSEQUENCE_LENGTH};
    checkCuda(cudaMemcpy(seqlen_device_ptr, temp.data(), sizeof(int32_t) * 1, cudaMemcpyHostToDevice));

    cudaStream_t const stream = nullptr;
    invokeContextApplyRopeUpdateKVFP16(qkv_device_ptr, nullptr, kvcache_ptr, seqlen_device_ptr, kNUM_Q_HEADS,
        kNUM_K_HEADS, kDIM_HEAD, kKV_CACHE_CAPACITY, kINPUT_LENGTH_PADDED, kROPE_TYPE, kROPE_BASE_FREQUENCY, 
        kROPE_SCALE, kROPE_INIT_TYPE, kINPUT_LENGTH_PADDED, stream);
    checkCuda(cudaStreamSynchronize(stream));
    checkCuda(cudaGetLastError());

    // Load QKV tensor after the rope kernel.
    std::vector<half> qkvUpdatedData(nbQKVData);
    checkCuda(cudaMemcpy(qkvUpdatedData.data(), qkv_device_ptr, sizeof(half) * nbQKVData, cudaMemcpyDeviceToHost));

    // Read reference data dumped from huggingface as reference.
    std::vector<half> refQData(nbQdata);
    std::vector<half> refKData(nbKData);

    loadDataFromFile("../tests/hf-tensor/post-rope-qtensor.bin", refQData, nbQdata);
    loadDataFromFile("../tests/hf-tensor/post-rope-ktensor.bin", refKData, nbKData);

    // V data should be unchaned.
    std::vector<half> refQKV = concatQKVData(refQData, refKData, inputVData);
    double totalDiff = 0;
    for (int32_t i = 0; i < nbQKVData; ++i)
    {
        float const val = __half2float(qkvUpdatedData[i]);
        float const refVal = __half2float(refQKV[i]);
        float const diff = std::abs(val - refVal);
        totalDiff += diff;

        if (diff >= 4e-3)
        {
            float const srcData0 = __half2float(concatQKV[i]);
            float const srcData1 = __half2float(concatQKV[i - 64]);
            printf("At %d index. value is %f refValue is %f.\n", i, val, refVal);
            printf("Source data are %f %f.\n", srcData0, srcData1);
            check(false, "Mismatch data in rope computation.");
        }
    }

    float avgDiff = totalDiff / (nbQdata + nbKData);
    printf("Average difference of QK rope data is %f .\n", avgDiff);
}

void test_fmha()
{
    std::vector<half> inputQData(nbQdata);
    std::vector<half> inputKData(nbKData);
    std::vector<half> inputVData(nbVData);

    loadDataFromFile("../tests/hf-tensor/post-rope-qtensor.bin", inputQData, nbQdata);
    loadDataFromFile("../tests/hf-tensor/post-rope-ktensor.bin", inputKData, nbKData);
    loadDataFromFile("../tests/hf-tensor/vtensor.bin", inputVData, nbVData);

    // Prepare device buffers
    half* qkv_device_ptr = nullptr;
    int32_t* seqlen_device_ptr = nullptr;
    half* attention_ptr = nullptr;

    checkCuda(cudaMalloc(reinterpret_cast<void**>(&qkv_device_ptr), sizeof(half) * nvTotalQKVPaddedData));
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&attention_ptr), sizeof(half) * nbTotalAttentionPaddedData));
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&seqlen_device_ptr), sizeof(int32_t) * 2));
    checkCuda(cudaMemset(qkv_device_ptr, 0, sizeof(half) * nbTotalKVCacheData));
    checkCuda(cudaMemset(attention_ptr, 0, sizeof(half) * nbTotalAttentionPaddedData));
    checkCuda(cudaMemset(seqlen_device_ptr, 0, sizeof(int32_t) * 2));

    std::vector<half> concatQKV = concatQKVData(inputQData, inputKData, inputVData);
    check(concatQKV.size() == nbQKVData, "Check number of QKV data is consistent");
    checkCuda(cudaMemcpy(qkv_device_ptr, concatQKV.data(), sizeof(half) * nbQKVData, cudaMemcpyHostToDevice));

    std::vector<int32_t> temp{0, kSEQUENCE_LENGTH};
    checkCuda(cudaMemcpy(seqlen_device_ptr, temp.data(), sizeof(int32_t) * 2, cudaMemcpyHostToDevice));

    drivellm::ContextFMHARunner runner(
        nvinfer1::DataType::kHALF, kBATCH_SIZE, kINPUT_LENGTH_PADDED, kNUM_Q_HEADS, kNUM_K_HEADS, kDIM_HEAD, 86);

    Fused_multihead_attention_params_v2 params{};
    params.clear();
    runner.setupParams(params);

    params.qkv_ptr = qkv_device_ptr;
    params.cu_q_seqlens = seqlen_device_ptr;
    params.o_ptr = attention_ptr;

    cudaStream_t const stream = nullptr;
    runner.dispatchFMHAKernel(params, stream);

    checkCuda(cudaStreamSynchronize(stream));
    checkCuda(cudaGetLastError());

    std::vector<half> attentionResult(nbQdata);
    checkCuda(cudaMemcpy(attentionResult.data(), attention_ptr, sizeof(half) * nbQdata, cudaMemcpyDeviceToHost));

    std::vector<half> refAttentionResult(nbQdata);
    loadDataFromFile("../tests/hf-tensor/attention_output.bin", refAttentionResult, nbQdata);

    double totalDiff = 0;
    for (int32_t i = 0; i < nbQdata; ++i)
    {
        float const val = __half2float(attentionResult[i]);
        float const refVal = __half2float(refAttentionResult[i]);
        float const diff = std::abs(val - refVal);
        totalDiff += diff;

        if (diff >= 5e-3)
        {
            printf("At %d index. value is %f refValue is %f.\n", i, val, refVal);
            check(false, "Mismatch data in rope computation.");
        }
    }

    float avgDiff = totalDiff / nbQdata;
    printf("Average difference of attention result data is %f .\n", avgDiff);
}

TEST(hf_test, rope_kv_context)
{
    testRope();
}

TEST(hf_test, fmha_context)
{
    test_fmha();
}