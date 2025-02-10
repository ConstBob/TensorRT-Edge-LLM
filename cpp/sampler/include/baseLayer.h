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
        std::shared_ptr<BaseDecodingOutputs> const& outputs, std::shared_ptr<BaseDecodingInputs> const& inputs)
    {
    }

protected:
    // Domain in which token decoding is computed
    DecoderDomain mDecoderDomain;
};