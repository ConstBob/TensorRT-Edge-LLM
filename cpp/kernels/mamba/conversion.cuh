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
 *
 * This file contains code derived from FlashInfer (https://github.com/flashinfer-ai/flashinfer)
 * Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)
 * Licensed under the Apache License, Version 2.0.
 *
 * Modifications by NVIDIA:
 * - Extracted type conversion utilities for Mamba kernel
 * - Made BFloat16 conversions unconditionally available
 */

#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace mamba_ssm::conversion
{

inline __device__ float toFloat(float f)
{
    return f;
}

inline __device__ float toFloat(__half h)
{
    return __half2float(h);
}

inline __device__ float toFloat(__nv_bfloat16 val)
{
    return __bfloat162float(val);
}

inline __device__ void convertAndStore(float* output, float input)
{
    *output = input;
}

inline __device__ void convertAndStore(__half* output, float input)
{
    *output = __float2half(input);
}

inline __device__ void convertAndStore(__nv_bfloat16* output, float input)
{
    *output = __float2bfloat16(input);
}

} // namespace mamba_ssm::conversion
