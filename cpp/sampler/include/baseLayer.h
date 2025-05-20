/*
 * Copyright 2024 The TensorRT-LLM Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "decodingParams.h"
#include <cstdint>
#include <memory>
#include <utility>

class BaseLayer
{
public:
    using SizeType32 = std::int32_t;
    using TokenIdType = std::int32_t;
    using BufferConstPtr = void const*;
    using BufferPtr = void*;

    BaseLayer(DecoderDomain const& decoderDomain)
        : mDecoderDomain(decoderDomain)
    {
    }

    virtual ~BaseLayer() = default;

    //! @returns cuda stream associated with layer
    [[nodiscard]] cudaStream_t getStream() const noexcept
    {
        return nullptr;
    }

    //! @returns workspace needed for this layer in bytes
    [[nodiscard]] virtual size_t getWorkspaceSize() const noexcept
    {
        return 0;
    };

    // clang-format off
    //! \brief Virtual function to setup internal states of the layer with sampling params
    //! specified in setupParams for the entries specified by batchSlots.
    //! It updates data for new requests in internal tensors inplace.
    //! Thus, it must be called only once for new requests.
    //!
    //! \param batchSize current batch size configured in the system
    //! \param beamWidth current beam width configured in the system
    //! \param batchSlots input buffer [maxBatchSize], address map of the new requests, in pinned memory
    //! \param setupParams shared pointer to params inherited from BaseSetupParams
    // clang-format on
    virtual void setup(std::int32_t batchSize, std::int32_t beamWidth, BufferConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& setupParams)
        = 0;

    // clang-format off
    //! \brief Virtual function to execute layer async on GPU.
    //! There must be no stream synchronization inside this function.
    //!
    //! \param outputs shared pointer to params inherited from BaseDecodingOutputs
    //! \param inputs shared pointer to params inherited from BaseForwardParams
    // clang-format on
    virtual void forwardAsync(
        std::shared_ptr<BaseDecodingOutputs> const& outputs, std::shared_ptr<BaseDecodingInputs> const& inputs)
        = 0;

    // clang-format off
    //! \brief Virtual function to execute layer synchronously on CPU / GPU.
    //! It is allowed (but not necassary) to synchronize on stream inside this function.
    //! It is targeted mainly for prototyping.
    //!
    //! \param outputs shared pointer to params inherited from BaseDecodingOutputs
    //! \param inputs shared pointer to params inherited from BaseForwardParams
    // clang-format on
    virtual void forwardSync(
        [[maybe_unused]] std::shared_ptr<BaseDecodingOutputs> const& outputs,
        [[maybe_unused]] std::shared_ptr<BaseDecodingInputs> const& inputs)
    {
    }

protected:
    // Domain in which token decoding is computed
    DecoderDomain mDecoderDomain;
};