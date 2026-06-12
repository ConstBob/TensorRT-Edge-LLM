/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>

namespace trt_edgellm
{

/*! \brief Return the RoPE cos/sin cache width for one attention head.
 *
 *  `headDim` is the full per-head Q/K dimension. `rotaryDim` is the width of
 *  the runtime `rope_rotary_cos_sin` cache consumed by the attention plugin.
 *  For the usual partial-RoPE representation, it is also the number of head
 *  channels that receive RoPE values; channels outside `rotaryDim` bypass
 *  rotation in the attention kernel.
 *
 *  `rope_scaling` is the normalized runtime/HuggingFace object that selects the
 *  RoPE variant and optional scaling parameters. It is not Gemma-specific:
 *  Gemma4 sliding/full RoPE configs, LongRoPE, MRoPE, dynamic scaling, and
 *  proportional RoPE all use this common field shape after export
 *  normalization.
 *
 *  Most RoPE variants use `partial_rotary_factor` to shrink `rotaryDim` to the
 *  rotated prefix of the head. Proportional RoPE is the exception: it keeps a
 *  full-head cache (`rotaryDim == headDim`) and consumes
 *  `partial_rotary_factor` in `collectRopeConfig()` as the fraction of full-head
 *  angle slots that receive non-identity cos/sin values. The inactive slots are
 *  materialized as identity values (`cos=1`, `sin=0`). The attention plugin
 *  does not need a Gemma-specific path for this case because it reads the cache
 *  width from the binding shape and already accepts `rotaryDim <= headDim`.
 */
inline int64_t getRotaryDim(nlohmann::json const& configJson, int64_t headDim)
{
    if (configJson.contains("rope_scaling") && configJson["rope_scaling"].is_object())
    {
        nlohmann::json const& ropeScaling = configJson["rope_scaling"];
        std::string const ropeType = ropeScaling.value("type", ropeScaling.value("rope_type", std::string{}));
        if (ropeType == "proportional")
        {
            return headDim;
        }
    }
    float const partialRotaryFactor = configJson.value("partial_rotary_factor", 1.0F);
    return static_cast<int64_t>(static_cast<float>(headDim) * partialRotaryFactor);
}

} // namespace trt_edgellm
