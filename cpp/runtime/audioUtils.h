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

#pragma once

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace audioUtils
{

//! Audio data container
//! For input: provide melSpectrogramPath for pre-computed Mel-spectrogram
//! For output: contains generated audio waveform data, sampleRate, and numChannels
struct AudioData
{
    // For audio input: pre-computed Mel-spectrogram
    std::string melSpectrogramPath;   //!< Path to pre-computed Mel-spectrogram file (.npy or .raw)
    std::string melSpectrogramFormat; //!< Format of the mel-spectrogram file: "npy" or "raw"

    // For audio output: generated waveform
    std::shared_ptr<Tensor> waveform; //!< Waveform samples [1, numSamples], FP16, range [-1, 1], CPU
    int32_t sampleRate{24000};        //!< Sample rate in Hz
    int32_t numChannels{1};           //!< Number of audio channels (typically 1 for mono)

    // For audio output: codebook codes (if waveform generation is not available)
    std::vector<std::vector<int32_t>> codebookCodes; //!< RVQ codebook codes [numCodebooks][seqLen]
    bool hasWaveform{false};                         //!< True if waveform contains valid data
};

} // namespace audioUtils
} // namespace rt
} // namespace trt_edgellm
