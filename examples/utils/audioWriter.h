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

#include "runtime/audioUtils.h"
#include <string>

/*!
 * @brief Save audio data to WAV file
 *
 * Saves audio waveform to a standard WAV file format.
 * Supports mono audio with 16-bit PCM encoding.
 *
 * @param filepath Output file path (should end with .wav)
 * @param audio Audio data containing samples and metadata
 * @return True if save succeeded, false otherwise
 */
bool saveAudioToWav(std::string const& filepath, trt_edgellm::rt::audioUtils::AudioData const& audio);
