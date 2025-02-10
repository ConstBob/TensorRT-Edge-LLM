/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
 
#include "vit_runner.h"
#include <cmath>
#include <random>
#include <tuple>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

bool Qwen2ViTRunner::setup(
    std::filesystem::path const& fp, cudaStream_t& stream, int batchSize, int minPixels, int maxPixels)
{
    try
    {
        mStream = stream;
        mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
        StreamReader _sr(fp);
        mVisualEngine = std::unique_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(_sr));
        mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mVisualEngine->createExecutionContext());
        mContext->setOptimizationProfileAsync(0, mStream);
        mBatchSize = batchSize;
        mConfig.minPixels = minPixels;
        mConfig.maxPixels = maxPixels;

        isSetup = true;
    }
    catch (std::exception const& e)
    {
        isSetup = false;
        LOG_ERROR(e.what());
        return false;
    }
    return true;
}

void Qwen2ViTRunner::allocateBuffer()
{
    // Allocate buffers and set shapes
    int64_t inputDim = mContext->getTensorShape("input").d[1];
    void* inputDevice;
    CUDA_CHECK(cudaMalloc(&inputDevice, (mHW * inputDim) * sizeof(half)));
    mDeviceBuffer["input"] = inputDevice;
    mContext->setTensorAddress("input", inputDevice);
    mContext->setInputShape("input", {2, {mHW, inputDim}});

    void* attentionMaskDevice;
    CUDA_CHECK(cudaMalloc(&attentionMaskDevice, (mHW * mHW) * sizeof(half)));
    mDeviceBuffer["attention_mask"] = attentionMaskDevice;
    mContext->setTensorAddress("attention_mask", attentionMaskDevice);
    mContext->setInputShape("attention_mask", {3, {1, mHW, mHW}});

    int64_t posEmbDim = mContext->getTensorShape("rotary_pos_emb").d[1];
    void* rotaryPosEmbDevice;
    CUDA_CHECK(cudaMalloc(&rotaryPosEmbDevice, (mHW * posEmbDim) * sizeof(float)));
    mDeviceBuffer["rotary_pos_emb"] = rotaryPosEmbDevice;
    mContext->setTensorAddress("rotary_pos_emb", rotaryPosEmbDevice);
    mContext->setInputShape("rotary_pos_emb", {2, {mHW, posEmbDim}});

    nvinfer1::Dims outputShape = mContext->getTensorShape("output");
    nvinfer1::DataType outType = mVisualEngine->getTensorDataType("output");
    int64_t nToken = outputShape.d[0];
    int64_t hiddenDim = outputShape.d[1];
    void* outputDevice;
    CUDA_CHECK(cudaMalloc(&outputDevice, (nToken * hiddenDim) * sizeof(half)));
    mDeviceBuffer["output"] = outputDevice;
    mContext->setTensorAddress("output", outputDevice);

    void* mropeRotaryCosSinDevice;
    int64_t mropeRotaryCosSinSize = mBatchSize * mConfig.maxPositionEmbeddings * mConfig.rotaryEmbedDim * sizeof(float);
    CUDA_CHECK(cudaMalloc(&mropeRotaryCosSinDevice, mropeRotaryCosSinSize));
    mDeviceBuffer["mropeRotaryCosSin"] = mropeRotaryCosSinDevice;
    void* mropePositionDeltasDevice;
    CUDA_CHECK(cudaMalloc(&mropePositionDeltasDevice, mBatchSize * sizeof(int64_t)));
    mDeviceBuffer["mropePositionDeltas"] = mropePositionDeltasDevice;
}

void Qwen2ViTRunner::freeBuffer()
{
    for (auto deviceMem : mDeviceBuffer)
    {
        cudaFree(deviceMem.second);
    }
    mDeviceBuffer.clear();
}

void Qwen2ViTRunner::initRotaryEmbedding(
    int numPos, int dim, float theta, std::vector<std::vector<float>>& sinusoidInp, float scale)
{
    std::vector<float> invFreq;
    for (int i = 0; i < dim; i += 2)
    {
        float value = scale / pow(theta, (static_cast<float>(i) / dim));
        invFreq.emplace_back(value);
    }

    for (int i = 0; i < numPos; ++i)
    {
        for (int j = 0; j < (dim / 2); ++j)
        {
            sinusoidInp[i][j] = i * invFreq[j];
        }
    }

    return;
}

void Qwen2ViTRunner::preprocessImage(unsigned char* image, int const& width, int const& height, int const& channels,
    std::vector<half>& patches, std::vector<std::vector<int64_t>>& grids, int64_t& totalSeqLength)
{
    // resize
    auto [resizedHeight, resizedWidth] = smartResize(height / 2, width / 2, mConfig.patchSize * mConfig.mergeSize);
    unsigned char* resizedImage = (unsigned char*) malloc(resizedHeight * resizedWidth * channels);
    stbir_resize_uint8_linear(
        image, width, height, 0, resizedImage, resizedWidth, resizedHeight, 0, stbir_pixel_layout::STBIR_RGB);

    std::vector<int64_t> curGrid{1, (resizedHeight / mConfig.patchSize), (resizedWidth / mConfig.patchSize)};
    grids.emplace_back(curGrid);
    totalSeqLength += (resizedHeight / mConfig.patchSize) * (resizedWidth / mConfig.patchSize);

    int curSize = mConfig.temporalPatchSize * resizedHeight * resizedWidth * channels;
    std::vector<half> curPatch(curSize);

    // Normalize and store to patches. Reorder dimensions according to:
    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L299
    for (int gridH = 0; gridH < curGrid[1] / mConfig.mergeSize; ++gridH)
    {
        for (int gridW = 0; gridW < curGrid[2] / mConfig.mergeSize; ++gridW)
        {
            for (int mergeH = 0; mergeH < mConfig.mergeSize; ++mergeH)
            {
                for (int mergeW = 0; mergeW < mConfig.mergeSize; ++mergeW)
                {
                    for (int c = 0; c < channels; ++c)
                    {
                        for (int patchH = 0; patchH < mConfig.patchSize; ++patchH)
                        {
                            for (int patchW = 0; patchW < mConfig.patchSize; ++patchW)
                            {

                                // src dimensions: (H, W, C) => (gridH, mergeSize, patchSize, gridW, mergeSize,
                                // patchSize, C)
                                int originalH = gridH * mConfig.mergeSize * mConfig.patchSize
                                    + mergeH * mConfig.patchSize + patchH;
                                int originalW = gridW * mConfig.mergeSize * mConfig.patchSize
                                    + mergeW * mConfig.patchSize + patchW;
                                unsigned char value
                                    = resizedImage[originalH * resizedWidth * channels + originalW * channels + c];
                                half normalized
                                    = __double2half((value / 255.0 - mConfig.imageMean[c]) / mConfig.imageStd[c]);

                                // duplicate pixels to temporalPatchSize
                                // dst dimensions: (gridH, gridW, mergeSize, mergeSize) x (channels,
                                // temporalPatchSize, patchSize, patchSize)
                                for (int t = 0; t < mConfig.temporalPatchSize; ++t)
                                {
                                    int dstHW = gridH * curGrid[2] * mConfig.mergeSize
                                        + gridW * mConfig.mergeSize * mConfig.mergeSize + mergeH * mConfig.mergeSize
                                        + mergeW;
                                    int dstDim = c * mConfig.temporalPatchSize * mConfig.patchSize * mConfig.patchSize
                                        + t * mConfig.patchSize * mConfig.patchSize + patchH * mConfig.patchSize
                                        + patchW;
                                    curPatch[dstHW * channels * mConfig.temporalPatchSize * mConfig.patchSize
                                            * mConfig.patchSize
                                        + dstDim]
                                        = normalized;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    patches.insert(patches.end(), curPatch.begin(), curPatch.end());
}

void Qwen2ViTRunner::computeRotaryPosEmb(
    std::vector<std::vector<int64_t>> const& grids, std::vector<float>& rotaryPosEmb)
{
    int64_t maxGridSize{0};
    for (auto const& grid : grids)
    {
        maxGridSize = std::max(maxGridSize, std::max(grid[1], grid[2]));
    }

    int dim = mConfig.embedDim / mConfig.numHeads / 2;
    std::vector<std::vector<float>> rotaryPosEmbFull(maxGridSize, std::vector<float>(dim / 2));
    initRotaryEmbedding(maxGridSize, dim, 10000.0f, rotaryPosEmbFull);

    std::vector<int> posIds;
    for (auto const& grid : grids)
    {
        int64_t T = grid[0], H = grid[1], W = grid[2];
        std::vector<int> curPosIds(T * H * W * 2);

        for (int i = 0; i < H; ++i)
        {
            for (int j = 0; j < W; ++j)
            {
                // (H, W) => (H / mergeSize, mergeSize, W / mergeSize, mergeSize)
                // => (H / mergeSize, W / mergeSize, mergeSize, mergeSize)
                int dstHW = (i / mConfig.mergeSize) * W * mConfig.mergeSize
                    + (j / mConfig.mergeSize) * mConfig.mergeSize * mConfig.mergeSize
                    + (i % mConfig.mergeSize) * mConfig.mergeSize + (j % mConfig.mergeSize);

                // duplicate for T
                for (int t = 0; t < T; ++t)
                {
                    int baseIdx = t * H * W * 2 + dstHW * 2;
                    curPosIds[baseIdx] = i;
                    curPosIds[baseIdx + 1] = j;
                }
            }
        }

        posIds.insert(posIds.end(), curPosIds.begin(), curPosIds.end());
    }

    rotaryPosEmb.resize(posIds.size() * (dim / 2));
    for (int i = 0; i < posIds.size(); ++i)
    {
        auto emb = rotaryPosEmbFull[posIds[i]];
        std::copy(emb.begin(), emb.end(), rotaryPosEmb.begin() + i * (dim / 2));
    }
}

std::tuple<int, int> Qwen2ViTRunner::smartResize(int const height, int const width, int const maxRatio)
{
    // According to https://github.com/QwenLM/Qwen2-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
    auto roundByFactor
        = [](int value, int factor) -> int { return std::round(static_cast<double>(value) / factor) * factor; };
    auto floorByFactor
        = [](int value, int factor) -> int { return std::floor(static_cast<double>(value) / factor) * factor; };
    auto ceilByFactor
        = [](int value, int factor) -> int { return std::ceil(static_cast<double>(value) / factor) * factor; };

    if (std::max(height, width) / std::min(height, width) > maxRatio)
    {
        throw std::invalid_argument("absolute aspect ratio must be smaller than " + std::to_string(maxRatio) + ", got "
            + std::to_string(std::max(height, width) / std::min(height, width)));
    }

    int factor = mConfig.patchSize * mConfig.mergeSize;
    int hBar = std::max(factor, roundByFactor(height, factor));
    int wBar = std::max(factor, roundByFactor(width, factor));

    if (hBar * wBar > mConfig.maxPixels)
    {
        double beta = std::sqrt(static_cast<double>(height * width) / mConfig.maxPixels);
        hBar = floorByFactor(static_cast<int>(height / beta), factor);
        wBar = floorByFactor(static_cast<int>(width / beta), factor);
    }
    else if (hBar * wBar < mConfig.minPixels)
    {
        double beta = std::sqrt(static_cast<double>(mConfig.minPixels) / (height * width));
        hBar = ceilByFactor(static_cast<int>(height * beta), factor);
        wBar = ceilByFactor(static_cast<int>(width * beta), factor);
    }

    return {hBar, wBar};
}

void Qwen2ViTRunner::visualPreprocess(std::vector<unsigned char*> const& imageBuffers,
    std::vector<std::vector<int>> const& imageSizes, std::vector<half>& patches, std::vector<half>& attentionMask,
    std::vector<float>& rotaryPosEmb, std::vector<std::vector<int64_t>>& grids)
{
    int64_t totalSeqLength = 0;
    for (int i = 0; i < imageBuffers.size(); ++i)
    {
        preprocessImage(
            imageBuffers[i], imageSizes[i][0], imageSizes[i][1], imageSizes[i][2], patches, grids, totalSeqLength);
    }

    attentionMask.resize(totalSeqLength * totalSeqLength, -CUDART_MAX_NORMAL_FP16);
    int start = 0;
    for (auto const grid : grids)
    {
        int64_t len = grid[1] * grid[2];
        for (int t = 0; t < grid[0]; ++t)
        {
            for (int i = start; i < start + len; ++i)
            {
                for (int j = start; j < start + len; ++j)
                {
                    attentionMask[i * totalSeqLength + j] = CUDART_ZERO_FP16;
                }
            }
            start += len;
        }
    }

    computeRotaryPosEmb(grids, rotaryPosEmb);

    mHW = totalSeqLength;
    return;
}

void Qwen2ViTRunner::getRopeIdx(std::vector<std::vector<int64_t>> const& batchInputIds,
    std::vector<std::vector<int64_t>> const& imageGridTHWs, std::vector<int64_t>& mropePositionIds,
    std::vector<int64_t>& mropePositionDeltas, int64_t maxPositionEmbeddings, int64_t visionStartTokenId,
    int64_t spacialMergeSize)
{
    int totalImageIdx = 0;

    for (auto inputIds : batchInputIds)
    {
        std::vector<std::vector<int64_t>> positionIds(3);

        auto start = inputIds.begin();
        auto end = inputIds.end();
        auto it = inputIds.begin();
        int startIdx = 0;

        while ((it = std::find(start, end, visionStartTokenId)) != end)
        {
            // Text part
            int textLen = it + 1 - start;
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < textLen; ++j)
                {
                    positionIds[i].emplace_back(j + startIdx);
                }
            }

            // Visual part
            int64_t T = imageGridTHWs[totalImageIdx][0];
            int64_t H = imageGridTHWs[totalImageIdx][1] / spacialMergeSize;
            int64_t W = imageGridTHWs[totalImageIdx][2] / spacialMergeSize;
            ++totalImageIdx;

            for (int t = 0; t < T; ++t)
            {
                for (int h = 0; h < H; ++h)
                {
                    for (int w = 0; w < W; ++w)
                    {
                        positionIds[0].emplace_back(t + textLen + startIdx);
                        positionIds[1].emplace_back(h + textLen + startIdx);
                        positionIds[2].emplace_back(w + textLen + startIdx);
                    }
                }
            }

            start = it + 1 + T * H * W;
            startIdx += std::max(T, std::max(H, W)) + textLen;
        }

        // Remaining text part
        if (start < end)
        {
            int textLen = end - start;
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < textLen; ++j)
                {
                    positionIds[i].emplace_back(j + startIdx);
                }
            }

            startIdx += textLen;
        }

        // Pad to maxPositionEmbeddings
        for (int i = 0; i < 3; ++i)
        {
            positionIds[i].resize(maxPositionEmbeddings);
            mropePositionIds.insert(mropePositionIds.end(), positionIds[i].begin(), positionIds[i].end());
        }

        mropePositionDeltas.emplace_back(startIdx - inputIds.size());
    }
}

void Qwen2ViTRunner::generateMropeParams(
    std::vector<std::vector<int64_t>> const& batchInputIds, std::vector<std::vector<int64_t>> const& visualGridTHWs)
{
    std::vector<float> mropeRotaryCosSin;
    std::vector<int64_t> mropePositionDeltas;

    std::vector<int64_t> mropePositionIds; // (bs, 3, maxPositionEmbeddings)
    getRopeIdx(batchInputIds, visualGridTHWs, mropePositionIds, mropePositionDeltas, mConfig.maxPositionEmbeddings);

    std::vector<std::vector<float>> sinusoidInp(
        mConfig.maxPositionEmbeddings, std::vector<float>(mConfig.rotaryEmbedDim / 2));
    initRotaryEmbedding(mConfig.maxPositionEmbeddings, mConfig.rotaryEmbedDim, mConfig.theta, sinusoidInp);

    std::vector<std::vector<float>> cosOri(
        mConfig.maxPositionEmbeddings, std::vector<float>(mConfig.rotaryEmbedDim / 2));
    std::vector<std::vector<float>> sinOri(
        mConfig.maxPositionEmbeddings, std::vector<float>(mConfig.rotaryEmbedDim / 2));
    for (int i = 0; i < mConfig.maxPositionEmbeddings; ++i)
    {
        for (int j = 0; j < (mConfig.rotaryEmbedDim / 2); ++j)
        {
            cosOri[i][j] = cos(sinusoidInp[i][j]);
            sinOri[i][j] = sin(sinusoidInp[i][j]);
        }
    }

    std::vector<int> mRopeSections{0, 16, 40, 64}; // cumsum of {16, 24, 24}
    int64_t mropeRotaryCosSinSize = mBatchSize * mConfig.maxPositionEmbeddings * mConfig.rotaryEmbedDim;
    mropeRotaryCosSin.resize(mropeRotaryCosSinSize);

    for (int b = 0; b < mBatchSize; ++b)
    {
        for (int sec = 0; sec < 3; ++sec)
        {
            for (int i = 0; i < mConfig.maxPositionEmbeddings; ++i)
            {
                int pos
                    = mropePositionIds[b * 3 * mConfig.maxPositionEmbeddings + sec * mConfig.maxPositionEmbeddings + i];
                for (int j = mRopeSections[sec]; j < mRopeSections[sec + 1]; ++j)
                {
                    int dstIdx = b * mConfig.maxPositionEmbeddings * mConfig.rotaryEmbedDim + i * mConfig.rotaryEmbedDim
                        + j * 2;
                    mropeRotaryCosSin[dstIdx] = cosOri[pos][j];
                    mropeRotaryCosSin[dstIdx + 1] = sinOri[pos][j];
                }
            }
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["mropeRotaryCosSin"], mropeRotaryCosSin.data(),
        mropeRotaryCosSinSize * sizeof(float), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["mropePositionDeltas"], mropePositionDeltas.data(),
        mBatchSize * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));
}

std::string Qwen2ViTRunner::applyChatTemplate(std::string const& inputString, int const& numImage,
    std::vector<std::vector<int64_t>> const& visualGridTHWs, int& totalImageIdx, int64_t imageMergeSize,
    bool addGenerationPrompt)
{
    // System prefix
    std::string prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n";

    // Images
    for (int i = 0; i < numImage; ++i)
    {
        auto grid = visualGridTHWs[totalImageIdx++];
        int imagePadLen = grid[0] * grid[1] * grid[2] / (imageMergeSize * imageMergeSize);

        prompt += "<|vision_start|>";
        for (int j = 0; j < imagePadLen; ++j)
        {
            prompt += "<|image_pad|>";
        }

        prompt += "<|vision_end|>";
    }

    prompt += inputString;
    prompt += "<|im_end|>\n";

    if (addGenerationPrompt)
    {
        prompt += "<|im_start|>assistant\n";
    }

    return prompt;
}

void Qwen2ViTRunner::textPreprocess(std::vector<std::string> const& inputStrings, std::vector<int> const& numImages,
    std::vector<std::vector<int64_t>> const& visualGridTHWs, Tokenizer* tokenizer, std::vector<int64_t>& inputIds,
    std::vector<int32_t>& contextLengths, int const maxContextLength, int const vocabSize)
{
    std::vector<std::vector<int64_t>> batchInputIds;
    int totalImageIdx = 0;
    int value = vocabSize;
    for (int i = 0; i < inputStrings.size(); ++i)
    {
        std::string prompt = applyChatTemplate(inputStrings[i], numImages[i], visualGridTHWs, totalImageIdx);
        std::vector<int64_t> inputIds = tokenizer->encode(prompt);
        // replace vis tokens
        for (int j = 0; j < inputIds.size(); ++j)
        {
            // <|vision_pad|>, <|image_pad|>, <|video_pad|>
            if (inputIds[j] == 151654 || inputIds[j] == 151655 || inputIds[j] == 151656)
            {
                inputIds[j] = value;
                ++value;
            }
        }

        batchInputIds.emplace_back(inputIds);
    }

    generateMropeParams(batchInputIds, visualGridTHWs);

    // Pad to maxContextLength
    int64_t padId = tokenizer->getPadId();
    for (int i = 0; i < batchInputIds.size(); ++i)
    {
        int32_t inputSize = static_cast<int32_t>(batchInputIds[i].size());
        if (inputSize > maxContextLength)
        {
            LOG_WARNING("Input length > maxContextLength. The last tokens will be truncated.");
        }
        contextLengths.emplace_back(std::min(inputSize, maxContextLength));
        batchInputIds[i].resize(maxContextLength, padId);
        inputIds.insert(inputIds.end(), batchInputIds[i].begin(), batchInputIds[i].end());
    }
}

std::vector<EngineInputDesc> Qwen2ViTRunner::getExtraLLMInputs()
{
    std::vector<EngineInputDesc> extraInputs;
    nvinfer1::Dims imageEmbedsDims = mContext->getTensorShape("output");
    int64_t imageHiddenSize = imageEmbedsDims.d[1];
    extraInputs.emplace_back(EngineInputDesc{
        "image_embeds", mDeviceBuffer["output"], imageEmbedsDims, {2, {1, imageHiddenSize}}});

    nvinfer1::Dims cosSinDims = {2, {mBatchSize, mConfig.maxPositionEmbeddings * mConfig.rotaryEmbedDim}};
    extraInputs.emplace_back(EngineInputDesc{
        "mrope_rotary_cos_sin", mDeviceBuffer["mropeRotaryCosSin"], cosSinDims, cosSinDims});

    nvinfer1::Dims deltasDim = {2, {mBatchSize, 1}};
    extraInputs.emplace_back(EngineInputDesc{
        "mrope_position_deltas", mDeviceBuffer["mropePositionDeltas"], deltasDim, deltasDim});

    return extraInputs;
}

void Qwen2ViTRunner::visualInfer(
    std::vector<half> const& input, std::vector<half> const& attentionMask, std::vector<float> const& rotaryPosEmb)
{
    int64_t inputDim = mContext->getTensorShape("input").d[1];
    int64_t posEmbDim = mContext->getTensorShape("rotary_pos_emb").d[1];

    CUDA_CHECK(cudaMemcpyAsync(
        mDeviceBuffer["input"], input.data(), (mHW * inputDim) * sizeof(half), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["attention_mask"], attentionMask.data(), (mHW * mHW) * sizeof(half),
        cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["rotary_pos_emb"], rotaryPosEmb.data(), (mHW * posEmbDim) * sizeof(float),
        cudaMemcpyHostToDevice, mStream));

    mContext->enqueueV3(mStream);

    CUDA_CHECK(cudaStreamSynchronize(mStream));
}

void Qwen2ViTRunner::initRandomInputs(std::vector<half>& visualInput, std::vector<half>& visualAttentionMask,
    std::vector<float>& visualRotaryPosEmb, std::vector<int64_t>& inputIds, int const textTokenLength,
    int const imageTokenLength, int const maxContextLength, int const vocabSize)
{
    std::random_device dev;
    std::mt19937 rng(dev());

    // Init visual inputs
    // In Qwen2-VL, HW is always 4ximageTokens because it equals to spatial_merge_size ** 2.
    mHW = 4 * imageTokenLength;

    int64_t inputDim = mContext->getTensorShape("input").d[1];
    visualInput.resize(mHW * inputDim);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::generate(visualInput.begin(), visualInput.end(), [&rng, &dist]() { return __float2half(dist(rng)); });

    int64_t posEmbDim = mContext->getTensorShape("rotary_pos_emb").d[1];
    visualRotaryPosEmb.resize(mHW * posEmbDim);
    std::generate(visualRotaryPosEmb.begin(), visualRotaryPosEmb.end(), [&rng, &dist]() { return dist(rng); });

    visualAttentionMask.resize(mHW * mHW, CUDART_ZERO_FP16);

    allocateBuffer();

    // Init input ids
    std::uniform_int_distribution<std::mt19937::result_type> intDist(0, 10000);
    int value = vocabSize;
    for (int i = 0; i < mBatchSize; ++i)
    {
        auto beginIter = inputIds.begin() + i * maxContextLength;
        std::generate(
            beginIter, beginIter + textTokenLength + imageTokenLength, [&rng, &intDist]() { return intDist(rng); });
        // Replace image tokens at the beginning of each batch
        for (int j = 0; j < imageTokenLength; ++j)
        {
            *(beginIter + j) = value;
            ++value;
        }
    }

    // Init mrope params
    int64_t mropeRotaryCosSinSize = mBatchSize * mConfig.maxPositionEmbeddings * mConfig.rotaryEmbedDim;
    std::vector<float> mropeRotaryCosSin(mropeRotaryCosSinSize);
    std::generate(mropeRotaryCosSin.begin(), mropeRotaryCosSin.end(), [&rng, &dist]() { return dist(rng); });
    std::vector<int64_t> mropePositionDeltas(mBatchSize, 0);

    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["mropeRotaryCosSin"], mropeRotaryCosSin.data(),
        mropeRotaryCosSinSize * sizeof(float), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer["mropePositionDeltas"], mropePositionDeltas.data(),
        mBatchSize * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));
}