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

#include "audioUtils.h"

#include "audioLoader.h"

namespace trt_edgellm
{
namespace rt
{
namespace audioUtils
{

bool loadAudioDataFromBytes(uint8_t const* bytes, size_t size, int32_t targetSampleRate, AudioData& out)
{
    auto pcm = std::make_shared<audio::AudioPCM>();
    if (!audio::loadAudioBytes(bytes, size, targetSampleRate, *pcm))
    {
        return false;
    }
    out.pcm = std::move(pcm);
    out.sampleRate = targetSampleRate;
    return true;
}

bool loadAudioDataFromFile(std::filesystem::path const& path, int32_t targetSampleRate, AudioData& out)
{
    auto pcm = std::make_shared<audio::AudioPCM>();
    if (!audio::loadAudioFile(path, targetSampleRate, *pcm))
    {
        return false;
    }
    out.pcm = std::move(pcm);
    out.sampleRate = targetSampleRate;
    return true;
}

} // namespace audioUtils
} // namespace rt
} // namespace trt_edgellm
