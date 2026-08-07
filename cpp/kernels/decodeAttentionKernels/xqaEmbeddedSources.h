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
#include <utility>
#include <vector>

namespace trt_edgellm
{

//! \brief Embedded XQA kernel source files for NVRTC compilation.
//!
//! All source files (XQA kernels, project headers, CUDA toolkit headers) are embedded
//! at build time so that NVRTC compilation does not depend on any filesystem paths.
struct XQAEmbeddedSources
{
    char const* mainSource; //!< Content of mha.cu (the NVRTC entry point)
    //! Virtual headers passed to nvrtcCreateProgram: {includeName, content}
    std::vector<std::pair<char const*, char const*>> headers;
};

//! \brief Get the embedded XQA source files.
//! \return Reference to a singleton containing all embedded source content.
XQAEmbeddedSources const& getXQAEmbeddedSources();

} // namespace trt_edgellm
