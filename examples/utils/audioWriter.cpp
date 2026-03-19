/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "audioWriter.h"
#include "common/logger.h"
#include "common/stringUtils.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>
#include <fstream>
#include <vector>

using namespace trt_edgellm;

// WAV file header structure
struct WavHeader
{
    // RIFF chunk descriptor
    char riffTag[4];   // "RIFF"
    uint32_t fileSize; // File size - 8
    char waveTag[4];   // "WAVE"

    // fmt sub-chunk
    char fmtTag[4];         // "fmt "
    uint32_t fmtSize;       // Size of fmt chunk (16 for PCM)
    uint16_t audioFormat;   // Audio format (1 = PCM)
    uint16_t numChannels;   // Number of channels
    uint32_t sampleRate;    // Sample rate
    uint32_t byteRate;      // = sampleRate * numChannels * bitsPerSample/8
    uint16_t blockAlign;    // = numChannels * bitsPerSample/8
    uint16_t bitsPerSample; // Bits per sample

    // data sub-chunk
    char dataTag[4];   // "data"
    uint32_t dataSize; // = numSamples * numChannels * bitsPerSample/8
};

bool saveAudioToWav(std::string const& filepath, trt_edgellm::rt::audioUtils::AudioData const& audio)
{
    if (!audio.waveform || audio.waveform->isEmpty())
    {
        LOG_WARNING("Empty audio waveform, skipping WAV file creation");
        return false;
    }

    if (audio.numChannels != 1)
    {
        LOG_ERROR("Only mono audio (1 channel) is supported, got %d channels", audio.numChannels);
        return false;
    }

    try
    {
        // Convert FP32 or FP16 samples [-1.0, 1.0] to int16 [-32768, 32767]
        int64_t const numDims = audio.waveform->getShape().getNumDims();
        int64_t numSamples = audio.waveform->getShape()[numDims - 1];
        std::vector<int16_t> pcmData;
        pcmData.reserve(numSamples);

        bool const isFP32 = (audio.waveform->getDataType() == nvinfer1::DataType::kFLOAT);
        float const* fp32Data = isFP32 ? static_cast<float const*>(audio.waveform->rawPointer()) : nullptr;
        __half const* fp16Data = !isFP32 ? static_cast<__half const*>(audio.waveform->rawPointer()) : nullptr;
        for (int64_t i = 0; i < numSamples; ++i)
        {
            float sample = isFP32 ? fp32Data[i] : __half2float(fp16Data[i]);
            // Clamp to [-1.0, 1.0] range
            sample = std::max(-1.0f, std::min(1.0f, sample));
            // Convert to int16
            int32_t intSample = static_cast<int32_t>(sample * 32767.0f);
            pcmData.push_back(static_cast<int16_t>(intSample));
        }

        // Prepare WAV header
        WavHeader header;
        std::memcpy(header.riffTag, "RIFF", 4);
        std::memcpy(header.waveTag, "WAVE", 4);
        std::memcpy(header.fmtTag, "fmt ", 4);
        std::memcpy(header.dataTag, "data", 4);

        header.audioFormat = 1; // PCM
        header.numChannels = static_cast<uint16_t>(audio.numChannels);
        header.sampleRate = static_cast<uint32_t>(audio.sampleRate);
        header.bitsPerSample = 16;
        header.byteRate = header.sampleRate * header.numChannels * header.bitsPerSample / 8;
        header.blockAlign = header.numChannels * header.bitsPerSample / 8;
        header.fmtSize = 16;
        header.dataSize = static_cast<uint32_t>(pcmData.size() * sizeof(int16_t));
        header.fileSize = 36 + header.dataSize; // 36 = header size without riffTag and fileSize

        // Write to file
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open())
        {
            LOG_ERROR("Failed to open output file: %s", filepath.c_str());
            return false;
        }

        // Write header
        file.write(reinterpret_cast<char const*>(&header), sizeof(WavHeader));

        // Write PCM data
        file.write(reinterpret_cast<char const*>(pcmData.data()), pcmData.size() * sizeof(int16_t));

        file.close();

        if (!file.good())
        {
            LOG_ERROR("Error writing to file: %s", filepath.c_str());
            return false;
        }

        return true;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Exception while saving WAV file: %s", e.what());
        return false;
    }
}
