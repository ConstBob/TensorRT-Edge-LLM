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

bool InternVLViTRunner::setup(std::filesystem::path const& fp, cudaStream_t& stream, int llmBatchSize)
{
    try
    {
        mStream = stream;
        mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
        char const* disableMmapLoad = std::getenv("DISABLE_MMAP_LOAD");
        if (disableMmapLoad != nullptr)
        {
            StreamReader _sr(fp);
            mVisualEngine = std::unique_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(_sr));
        }
        else
        {
            auto mmapReader = std::make_unique<MmapReader>(fp);
            mVisualEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
                mRuntime->deserializeCudaEngine(mmapReader->getData(), mmapReader->getSize()));
        }

        mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mVisualEngine->createExecutionContext());
        mContext->setOptimizationProfileAsync(0, mStream);
        validateAndFillConfig(llmBatchSize);
        allocateBuffer();
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

void InternVLViTRunner::validateAndFillConfig(int llmBatchSize)
{
    mConfig.llmBatchSize = llmBatchSize;
    nvinfer1::Dims inputShape = mVisualEngine->getProfileShape("input", 0, nvinfer1::OptProfileSelector::kMAX);
    mConfig.maxHW = inputShape.d[0];
    mConfig.inputDim = mContext->getTensorShape("input").d[1];
    mConfig.hiddenDim = mVisualEngine->getTensorShape("output").d[1];
    // Set vocabSize for InternVL3
    mConfig.vocabSize = 151674;
}

void InternVLViTRunner::allocateBuffer()
{
    // Allocate buffers and set shapes
    void* inputDevice;
    CUDA_CHECK(cudaMalloc(&inputDevice, (mConfig.maxHW * mConfig.inputDim) * sizeof(half)));
    mDeviceBuffer["input"] = inputDevice;
    mContext->setTensorAddress("input", inputDevice);

    void* outputDevice;
    // In InternVL3, VIT output is downsampled by 4x, so output size is maxHW/4
    CUDA_CHECK(cudaMalloc(&outputDevice, (mConfig.maxHW / 4 * mConfig.hiddenDim) * sizeof(half)));
    mDeviceBuffer["output"] = outputDevice;
    mContext->setTensorAddress("output", outputDevice);
}

int InternVLViTRunner::setInputShape()
{
    std::vector<std::tuple<std::string, nvinfer1::Dims>> shapes = {{"input", {2, {mConfig.curHW, mConfig.inputDim}}}};

    for (auto const& [tensorName, dims] : shapes)
    {
        if (!mContext->setInputShape(tensorName.c_str(), dims))
        {
            LOG_ERROR("Failed to set input shape for tensor: %s", tensorName.c_str());
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}

void InternVLViTRunner::preprocessImage(unsigned char* image, unsigned char* thumbnailImage, int const& width,
    int const& height, int const& channels, std::vector<half>& patches, int64_t& totalSeqLength, bool useThumbnail,
    std::vector<int64_t>& imageTokenLengths)
{
    int64_t curSeqLength = (height / mConfig.patchSize) * (width / mConfig.patchSize);
    int curSize = height * width * channels;

    if (useThumbnail)
    {
        curSeqLength += 1024; // 448x448x3 size of thumbnail image has (448/patchSize)^2 = 1024 where patchSize=14
    }

    imageTokenLengths.push_back(curSeqLength / 4); // Image token length here is seq/4 because of the downsampling
    totalSeqLength += curSeqLength;

    std::vector<half> curPatch(curSize);
    std::vector<half> thumbnailPatch(mConfig.blockImageSize * mConfig.blockImageSize * 3);

    // Normalize and store to blocks of 448x448. Reorder dimensions according to:
    for (int gridH = 0; gridH < height / mConfig.blockImageSize; ++gridH)
    {
        for (int gridW = 0; gridW < width / mConfig.blockImageSize; ++gridW)
        {
            for (int c = 0; c < channels; ++c)
            {
                for (int mergeH = 0; mergeH < mConfig.blockImageSize / mConfig.patchSize; ++mergeH)
                {
                    for (int mergeW = 0; mergeW < mConfig.blockImageSize / mConfig.patchSize; ++mergeW)
                    {
                        for (int patchH = 0; patchH < mConfig.patchSize; ++patchH)
                        {
                            for (int patchW = 0; patchW < mConfig.patchSize; ++patchW)
                            {

                                // src dimensions: (H, W, C) => (gridH, blockSize, patchSize, gridW, blockSize,
                                // patchSize, C)
                                int originalH = gridH * mConfig.blockImageSize + mergeH * mConfig.patchSize + patchH;
                                int originalW = gridW * mConfig.blockImageSize + mergeW * mConfig.patchSize + patchW;

                                unsigned char value = image[originalH * width * channels + originalW * channels + c];
                                half normalized
                                    = __double2half((value / 255.0 - mConfig.imageMean[c]) / mConfig.imageStd[c]);

                                // dst dimensions: (gridH, gridW, channels) x (blockSize/patchSize, blockSize/patchSize,
                                // patchSize, patchSize)
                                int dstHW = gridH * (width / mConfig.blockImageSize) * channels + gridW * channels + c;
                                int dstDim = mergeH * mConfig.blockImageSize * mConfig.patchSize
                                    + patchH * mConfig.blockImageSize + mergeW * mConfig.patchSize + patchW;
                                curPatch[dstHW * mConfig.blockImageSize * mConfig.blockImageSize + dstDim] = normalized;
                            }
                        }
                    }
                }
            }
        }
    }

    if (useThumbnail)
    {
        for (int c = 0; c < channels; ++c)
        {
            for (int mergeH = 0; mergeH < mConfig.blockImageSize / mConfig.patchSize; ++mergeH)
            {
                for (int mergeW = 0; mergeW < mConfig.blockImageSize / mConfig.patchSize; ++mergeW)
                {
                    for (int patchH = 0; patchH < mConfig.patchSize; ++patchH)
                    {
                        for (int patchW = 0; patchW < mConfig.patchSize; ++patchW)
                        {
                            int originalH = mergeH * mConfig.patchSize + patchH;
                            int originalW = mergeW * mConfig.patchSize + patchW;
                            unsigned char value = thumbnailImage[originalH * mConfig.blockImageSize * channels
                                + originalW * channels + c];
                            half normalized
                                = __double2half((value / 255.0 - mConfig.imageMean[c]) / mConfig.imageStd[c]);

                            // dst dimensions: (1, channels) x (blockSize/patchSize, blockSize/patchSize,
                            // patchSize, patchSize)
                            int dstHW = c;
                            int dstDim = mergeH * mConfig.blockImageSize * mConfig.patchSize
                                + patchH * mConfig.blockImageSize + mergeW * mConfig.patchSize + patchW;
                            thumbnailPatch[dstHW * mConfig.blockImageSize * mConfig.blockImageSize + dstDim]
                                = normalized;
                        }
                    }
                }
            }
        }
    }
    patches.insert(patches.end(), curPatch.begin(), curPatch.end());
    if (useThumbnail)
    {
        patches.insert(patches.end(), thumbnailPatch.begin(), thumbnailPatch.end());
    }
}

std::tuple<int, int> InternVLViTRunner::adjustImageSize(
    int const height, int const width, std::vector<std::pair<int, int>> const& targetRatios)
{
    int64_t imageSize = mConfig.blockImageSize;

    double aspect_ratio = static_cast<double>(width) / height;
    double best_ratio_diff = HUGE_VAL;
    std::pair<int, int> best_ratio = {1, 1};
    int area = width * height;

    for (auto const& ratio : targetRatios)
    {
        double target_aspect_ratio = static_cast<double>(ratio.first) / ratio.second;
        double ratio_diff = std::abs(aspect_ratio - target_aspect_ratio);

        if (ratio_diff < best_ratio_diff)
        {
            best_ratio_diff = ratio_diff;
            best_ratio = ratio;
        }
        else if (ratio_diff == best_ratio_diff)
        {
            if (area > 0.5 * imageSize * imageSize * ratio.first * ratio.second)
            {
                best_ratio = ratio;
            }
        }
    }
    // return (height, width)
    return {best_ratio.second * imageSize, best_ratio.first * imageSize};
}

void InternVLViTRunner::visualPreprocess(std::vector<unsigned char*> const& imageBuffers,
    std::vector<unsigned char*> const& thumbnailImageBuffers, std::vector<std::vector<int>> const& imageSizes,
    std::vector<half>& patches, std::vector<int64_t>& imageTokenLengths, bool useThumbnail)
{
    int64_t totalSeqLength = 0;
    for (size_t i = 0; i < imageBuffers.size(); ++i)
    {
        preprocessImage(imageBuffers[i], thumbnailImageBuffers[i], imageSizes[i][0], imageSizes[i][1], imageSizes[i][2],
            patches, totalSeqLength, useThumbnail, imageTokenLengths);
    }

    if (totalSeqLength > mConfig.maxHW)
    {
        LOG_ERROR("Image tokens number exceeds the maximum limitation of VIT engine.");
        return;
    }

    mConfig.curHW = totalSeqLength;
    return;
}

std::string InternVLViTRunner::applyChatTemplate(std::string const& inputString, int const& numImages,
    std::vector<int64_t> const& imageTokenLengths, int& totalImageIdx, bool addGenerationPrompt)
{
    // System prefix
    std::string prompt
        = "<|im_start|>"
          "system\n你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语"
          "言模型。<|im_end|>\n<|im_start|>user\n";

    // Images
    for (int i = 0; i < numImages; ++i)
    {
        int imagePadLen = imageTokenLengths[totalImageIdx++];

        prompt += "<img>";
        for (int j = 0; j < imagePadLen; ++j)
        {
            prompt += "<IMG_CONTEXT>";
        }

        prompt += "</img>\n";
    }

    prompt += inputString;
    prompt += "<|im_end|>\n";

    if (addGenerationPrompt)
    {
        prompt += "<|im_start|>assistant\n";
    }
    return prompt;
}

void InternVLViTRunner::textPreprocess(std::vector<std::string> const& inputStrings, std::vector<int> const& numImages,
    std::vector<int64_t> const& imageTokenLengths, Tokenizer* tokenizer, std::vector<int64_t>& inputIds,
    std::vector<int32_t>& contextLengths, int const maxContextLength)
{
    std::vector<std::vector<int64_t>> batchInputIds;
    int totalImageIdx = 0;
    int value = mConfig.vocabSize;
    for (size_t i = 0; i < inputStrings.size(); ++i)
    {
        std::string prompt = applyChatTemplate(inputStrings[i], numImages[i], imageTokenLengths, totalImageIdx);
        std::vector<int64_t> ids = tokenizer->encode(prompt);
        // replace vis tokens
        for (size_t j = 0; j < ids.size(); ++j)
        {
            // <IMG_CONTEXT>
            if (ids[j] == 151667)
            {
                ids[j] = value;
                ++value;
            }
        }
        batchInputIds.emplace_back(ids);
    }

    // Pad to maxContextLength
    int64_t padId = tokenizer->getPadId();
    for (size_t i = 0; i < batchInputIds.size(); ++i)
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

std::vector<EngineInputDesc> InternVLViTRunner::getExtraLLMInputs()
{
    std::vector<EngineInputDesc> extraInputs;

    extraInputs.emplace_back(EngineInputDesc{"image_embeds", mDeviceBuffer["output"], mDeviceBuffer["output"],
        {2, {mConfig.maxHW / 4, mConfig.hiddenDim}}, {2, {1, mConfig.hiddenDim}}});

    return extraInputs;
}

void InternVLViTRunner::internVLViTInfer(std::vector<half> const& input)
{
    if (setInputShape() != EXIT_SUCCESS)
    {
        LOG_ERROR("Failed to set input shapes. Aborting inference.");
        return;
    }

    CUDA_CHECK(cudaMemcpyAsync(
        mDeviceBuffer["input"], input.data(), input.size() * sizeof(half), cudaMemcpyHostToDevice, mStream));

    mContext->enqueueV3(mStream);

    CUDA_CHECK(cudaStreamSynchronize(mStream));
}

void InternVLViTRunner::initRandomInputs(std::vector<half>& visualInput, std::vector<int64_t>& inputIds,
    int const textTokenLength, int const imageTokenLength, int const maxContextLength)
{
    std::random_device dev;
    std::mt19937 rng(dev());

    // Init visual inputs
    // In InternVL3, curHW is 4*imageTokenLength due to downsampling ratio of 0.5 (factor = (1/0.5)^2 = 4)
    mConfig.curHW = 4 * imageTokenLength;
    if (mConfig.curHW > mConfig.maxHW)
    {
        LOG_ERROR("Image tokens number exceeds the maximum limitation of VIT engine.");
        return;
    }

    int64_t inputDim = mContext->getTensorShape("input").d[1];
    visualInput.resize(mConfig.curHW * inputDim);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::generate(visualInput.begin(), visualInput.end(), [&rng, &dist]() { return __float2half(dist(rng)); });

    // Init input ids
    std::uniform_int_distribution<std::mt19937::result_type> intDist(0, 10000);
    int value = mConfig.vocabSize;
    for (int i = 0; i < mConfig.llmBatchSize; ++i)
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
}