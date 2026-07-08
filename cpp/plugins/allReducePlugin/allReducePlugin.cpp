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

#include "allReducePlugin.h"

#include "common/logger.h"
#include "common/tensor.h"
// {$edge-llm-internal-release begin}
#include "kernels/multiDeviceKernels/shmAllReduce.h"
// {$edge-llm-internal-release end}
#include "plugins/utils/pluginUtils.h"

#include <atomic>
#include <cstdint>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kALL_REDUCE_PLUGIN_VERSION{"1"};
constexpr char const* kALL_REDUCE_PLUGIN_NAME{"AllReducePlugin"};

// Input/output indices
constexpr int32_t kIN_TENSOR_IDX{0};
constexpr int32_t kOUT_TENSOR_IDX{0};

// NCCL data type and op constants
constexpr int32_t kNcclFloat16 = 6;
constexpr int32_t kNcclFloat32 = 7;
constexpr int32_t kNcclBfloat16 = 10;
constexpr int32_t kNcclSum = 0;
constexpr int32_t kNcclSuccess = 0;

// Per-device NCCL state for single-process multi-GPU TP.
// Each GPU device gets its own NCCL communicator handle.
using NcclAllReduceFn = int (*)(void const*, void*, size_t, int, int, void*, cudaStream_t);
static std::unordered_map<int, void*> gNcclCommMap; //!< cudaDevice -> ncclComm_t
static NcclAllReduceFn gNcclAllReduceFn = nullptr;
static std::mutex gCommStateMutex;

// {$edge-llm-internal-release begin}
constexpr int32_t kShmWorldSize{2};

// SHM AllReduce state registered by the runtime before plugin execution.
static kernels::ShmAllReduceState* gShmState = nullptr;
static std::unordered_map<int, int> gDeviceRankMap; //!< cudaDevice -> TP rank

bool isValidShmRank(int rank) noexcept
{
    return rank >= 0 && rank < kShmWorldSize;
}

bool isValidCudaDeviceId(int deviceId) noexcept
{
    if (deviceId < 0)
    {
        return false;
    }

    int deviceCount = 0;
    cudaError_t const err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        LOG_ERROR(
            "Failed to query CUDA device count while registering SHM AllReduce state: %s", cudaGetErrorString(err));
        return false;
    }
    return deviceId < deviceCount;
}

// {$edge-llm-internal-release end}
} // namespace

void registerNcclCommForAllReducePlugin(int deviceId, void* ncclComm, void* ncclAllReduceFunc) noexcept
{
    std::lock_guard<std::mutex> lock(gCommStateMutex);
    gNcclCommMap[deviceId] = ncclComm;
    gNcclAllReduceFn = reinterpret_cast<NcclAllReduceFn>(ncclAllReduceFunc);
}

// {$edge-llm-internal-release begin}
void registerShmAllReduceForPlugin(void* state, int deviceId, int rank) noexcept
{
    auto* shmState = static_cast<kernels::ShmAllReduceState*>(state);
    if (shmState == nullptr)
    {
        LOG_ERROR("Cannot register SHM AllReduce state: state is null.");
        return;
    }
    if (!isValidShmRank(rank))
    {
        LOG_ERROR("Cannot register SHM AllReduce state: rank must be 0 or 1, got %d.", rank);
        return;
    }
    if (!isValidCudaDeviceId(deviceId))
    {
        LOG_ERROR("Cannot register SHM AllReduce state: invalid CUDA device id %d.", deviceId);
        return;
    }
    if (shmState->tpSize != kShmWorldSize)
    {
        LOG_ERROR("Cannot register SHM AllReduce state: tpSize must be %d, got %d.", kShmWorldSize, shmState->tpSize);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(gCommStateMutex);
        gShmState = shmState;
        gDeviceRankMap[deviceId] = rank;
    }

    LOG_INFO("SHM AllReduce registered for device %d (rank %d), maxElements=%ld, allReduceElementThreshold=%ld",
        deviceId, rank, static_cast<long>(shmState->maxElements),
        static_cast<long>(shmState->allReduceElementThreshold));
}

// {$edge-llm-internal-release end}
// Static class fields initialization
PluginFieldCollection AllReducePluginCreator::mFieldCollection{};
std::vector<PluginField> AllReducePluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(AllReducePluginCreator);

// ========================== AllReducePlugin Implementation ==========================

AllReducePlugin::AllReducePlugin(std::string const& name, int32_t tpSize)
    : mLayerName(name)
    , mTpSize(tpSize)
{
    LOG_DEBUG("AllReducePlugin created: name=%s, tpSize=%d", name.c_str(), tpSize);
}

AllReducePlugin::AllReducePlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    if (fc == nullptr)
    {
        throw std::invalid_argument("AllReducePlugin requires plugin fields");
    }
    auto tpSize = parsePluginScalarField<int32_t>("tp_size", fc);
    if (!tpSize.has_value())
    {
        throw std::invalid_argument("AllReducePlugin requires 'tp_size' field");
    }
    mTpSize = tpSize.value();
    LOG_DEBUG("AllReducePlugin deserialized from fields: name=%s, tpSize=%d", name.c_str(), mTpSize);
}

AllReducePlugin::~AllReducePlugin() {}

IPluginCapability* AllReducePlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    try
    {
        if (type == PluginCapabilityType::kBUILD)
        {
            return static_cast<IPluginV3OneBuild*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME)
        {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        return static_cast<IPluginV3OneCore*>(this);
    }
    catch (std::exception const& e)
    {
        return nullptr;
    }
}

IPluginV3* AllReducePlugin::clone() noexcept
{
    try
    {
        auto* plugin = new AllReducePlugin(mLayerName, mTpSize);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("AllReducePlugin clone failed: %s", e.what());
        return nullptr;
    }
}

int32_t AllReducePlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t AllReducePlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        if (nbOutputs != 1 || nbInputs != 1)
        {
            return -1;
        }
        outputTypes[kOUT_TENSOR_IDX] = inputTypes[kIN_TENSOR_IDX];
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

int32_t AllReducePlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* /* shapeInputs */,
    int32_t /* nbShapeInputs */, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& /* exprBuilder */) noexcept
{
    try
    {
        if (nbInputs != 1 || nbOutputs != 1)
        {
            return -1;
        }
        outputs[kOUT_TENSOR_IDX] = inputs[kIN_TENSOR_IDX];
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

bool AllReducePlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (nbInputs != 1 || nbOutputs != 1 || pos >= nbInputs + nbOutputs)
    {
        return false;
    }

    auto const& desc = inOut[pos].desc;
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }

    if (desc.type != DataType::kHALF && desc.type != DataType::kFLOAT && desc.type != DataType::kBF16)
    {
        return false;
    }

    if (pos >= nbInputs)
    {
        return desc.type == inOut[kIN_TENSOR_IDX].desc.type;
    }

    return true;
}

int32_t AllReducePlugin::configurePlugin(DynamicPluginTensorDesc const* /* in */, int32_t nbInputs,
    DynamicPluginTensorDesc const* /* out */, int32_t nbOutputs) noexcept
{
    return (nbInputs == 1 && nbOutputs == 1) ? 0 : -1;
}

size_t AllReducePlugin::getWorkspaceSize(DynamicPluginTensorDesc const* /* inputs */, int32_t /* nbInputs */,
    DynamicPluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept
{
    return 0;
}

int32_t AllReducePlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /* outputDesc */,
    void const* const* inputs, void* const* outputs, void* /* workspace */, cudaStream_t stream) noexcept
{
    try
    {
        auto const& inDesc = inputDesc[kIN_TENSOR_IDX];

        // Compute total number of elements
        int64_t numElements = 1;
        for (int32_t d = 0; d < inDesc.dims.nbDims; ++d)
        {
            numElements *= inDesc.dims.d[d];
        }

        // Look up NCCL comm for the current CUDA device
        int currentDevice = -1;
        cudaError_t const deviceErr = cudaGetDevice(&currentDevice);
        if (deviceErr != cudaSuccess)
        {
            LOG_ERROR("AllReducePlugin: cudaGetDevice failed: %s", cudaGetErrorString(deviceErr));
            return -1;
        }

        void* ncclComm = nullptr;
        NcclAllReduceFn ncclAllReduceFn = nullptr;
        {
            std::lock_guard<std::mutex> lock(gCommStateMutex);
            auto commIt = gNcclCommMap.find(currentDevice);
            if (commIt != gNcclCommMap.end())
            {
                ncclComm = commIt->second;
            }
            ncclAllReduceFn = gNcclAllReduceFn;
        }

        if (mTpSize <= 1 || ncclComm == nullptr || ncclAllReduceFn == nullptr)
        {
            // No tensor parallelism or no comm for this device: just copy input to output
            size_t const typeSize = rt::utils::getTypeSize(inDesc.type);
            cudaMemcpyAsync(outputs[kOUT_TENSOR_IDX], inputs[kIN_TENSOR_IDX], numElements * typeSize,
                cudaMemcpyDeviceToDevice, stream);
            return 0;
        }

        // {$edge-llm-internal-release begin}
        kernels::ShmAllReduceState* shmState = nullptr;
        int32_t shmRank = -1;
        {
            std::lock_guard<std::mutex> lock(gCommStateMutex);
            shmState = gShmState;
            auto rankIt = gDeviceRankMap.find(currentDevice);
            if (rankIt != gDeviceRankMap.end())
            {
                shmRank = rankIt->second;
            }
        }

        // SHM AllReduce path for FP16 payloads. Payloads above the runtime
        // threshold fall back to NCCL.
        if (shmState != nullptr && inDesc.type == DataType::kHALF)
        {
            if (numElements <= shmState->allReduceElementThreshold)
            {
                if (shmRank >= 0)
                {
                    int32_t const rank = shmRank;

                    syncShmHostBarrier(shmState, rank);

                    kernels::shmAllReduceExec(
                        shmState, inputs[kIN_TENSOR_IDX], outputs[kOUT_TENSOR_IDX], numElements, rank, stream);
                    return 0;
                }
            }
        }

        // {$edge-llm-internal-release end}

        // Map TRT data type to NCCL data type
        int32_t ncclType = kNcclFloat16;
        if (inDesc.type == DataType::kFLOAT)
        {
            ncclType = kNcclFloat32;
        }
        else if (inDesc.type == DataType::kBF16)
        {
            ncclType = kNcclBfloat16;
        }

        // ---- Profiling: GPU event timing around AllReduce (zero-overhead) ----

        // Perform NCCL all-reduce using the per-device communicator
        int32_t ncclResult = ncclAllReduceFn(
            inputs[kIN_TENSOR_IDX], outputs[kOUT_TENSOR_IDX], numElements, ncclType, kNcclSum, ncclComm, stream);
        if (ncclResult != kNcclSuccess)
        {
            LOG_ERROR("AllReducePlugin: NCCL allReduce failed with error %d on device %d", ncclResult, currentDevice);
            return -1;
        }

        return 0;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("AllReducePlugin enqueue failed: %s", e.what());
        return -1;
    }
}

int32_t AllReducePlugin::onShapeChange(
    PluginTensorDesc const* /* in */, int32_t nbInputs, PluginTensorDesc const* /* out */, int32_t nbOutputs) noexcept
{
    return (nbInputs == 1 && nbOutputs == 1) ? 0 : -1;
}

IPluginV3* AllReducePlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

PluginFieldCollection const* AllReducePlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("tp_size", &mTpSize, PluginFieldType::kINT32, 1);
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

char const* AllReducePlugin::getPluginName() const noexcept
{
    return kALL_REDUCE_PLUGIN_NAME;
}

char const* AllReducePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void AllReducePlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

char const* AllReducePlugin::getPluginVersion() const noexcept
{
    return kALL_REDUCE_PLUGIN_VERSION;
}

// ========================== AllReducePluginCreator Implementation ==========================

AllReducePluginCreator::AllReducePluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("tp_size", nullptr, PluginFieldType::kINT32, 1));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* AllReducePluginCreator::getPluginName() const noexcept
{
    return kALL_REDUCE_PLUGIN_NAME;
}

PluginFieldCollection const* AllReducePluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void AllReducePluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

char const* AllReducePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* AllReducePluginCreator::getPluginVersion() const noexcept
{
    return kALL_REDUCE_PLUGIN_VERSION;
}

IPluginV3* AllReducePluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase /* phase */) noexcept
{
    try
    {
        auto* plugin = new AllReducePlugin(name, fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("AllReducePluginCreator::createPlugin failed: %s", e.what());
        return nullptr;
    }
}

// {$edge-llm-internal-release begin}
void syncShmHostBarrier(kernels::ShmAllReduceState* shmState, int rank) noexcept
{
    if (!isValidShmRank(rank) || shmState == nullptr || shmState->hostBarrierArrived == nullptr)
    {
        return;
    }
    uint64_t volatile* arrived = shmState->hostBarrierArrived;
    int const other = 1 - rank;
    uint64_t const myRound = __atomic_add_fetch(&arrived[rank], 1, __ATOMIC_ACQ_REL);
    while (static_cast<uint64_t>(__atomic_load_n(&arrived[other], __ATOMIC_ACQUIRE)) < myRound)
    {
#if defined(__aarch64__)
        __asm__ volatile("yield");
#elif defined(__x86_64__)
        __asm__ volatile("pause");
#endif
    }
}

void syncShmHostBarrier(int rank) noexcept
{
    syncShmHostBarrier(getShmAllReduceState(), rank);
}

void getShmAllReduceRegistrationForDevice(int deviceId, kernels::ShmAllReduceState** state, int32_t* rank) noexcept
{
    if (state != nullptr)
    {
        *state = nullptr;
    }
    if (rank != nullptr)
    {
        *rank = -1;
    }
    if (deviceId < 0)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(gCommStateMutex);
    if (state != nullptr)
    {
        *state = gShmState;
    }
    if (rank != nullptr)
    {
        auto it = gDeviceRankMap.find(deviceId);
        if (it != gDeviceRankMap.end())
        {
            *rank = it->second;
        }
    }
}

kernels::ShmAllReduceState* getShmAllReduceState() noexcept
{
    std::lock_guard<std::mutex> lock(gCommStateMutex);
    return gShmState;
}

int32_t getShmRankForDevice(int deviceId) noexcept
{
    if (deviceId < 0)
    {
        return -1;
    }
    std::lock_guard<std::mutex> lock(gCommStateMutex);
    auto it = gDeviceRankMap.find(deviceId);
    return (it != gDeviceRankMap.end()) ? it->second : -1;
}

// {$edge-llm-internal-release end}
void getNcclRegistrationForDevice(int deviceId, void** ncclComm, void** ncclAllReduceFunc) noexcept
{
    if (ncclComm != nullptr)
    {
        *ncclComm = nullptr;
    }
    if (ncclAllReduceFunc != nullptr)
    {
        *ncclAllReduceFunc = nullptr;
    }
    if (deviceId < 0)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(gCommStateMutex);
    auto it = gNcclCommMap.find(deviceId);
    if (ncclComm != nullptr && it != gNcclCommMap.end())
    {
        *ncclComm = it->second;
    }
    if (ncclAllReduceFunc != nullptr)
    {
        *ncclAllReduceFunc = reinterpret_cast<void*>(gNcclAllReduceFn);
    }
}

void* getNcclCommForDevice(int deviceId) noexcept
{
    std::lock_guard<std::mutex> lock(gCommStateMutex);
    auto it = gNcclCommMap.find(deviceId);
    return (it != gNcclCommMap.end()) ? it->second : nullptr;
}

void* getNcclAllReduceFunc() noexcept
{
    std::lock_guard<std::mutex> lock(gCommStateMutex);
    return reinterpret_cast<void*>(gNcclAllReduceFn);
}

} // namespace plugins
} // namespace trt_edgellm

extern "C" void edgellmRegisterNcclCommForAllReducePlugin(
    int deviceId, void* ncclComm, void* ncclAllReduceFunc) noexcept
{
    trt_edgellm::plugins::registerNcclCommForAllReducePlugin(deviceId, ncclComm, ncclAllReduceFunc);
}

// {$edge-llm-internal-release begin}
extern "C" void edgellmRegisterShmAllReduceForPlugin(void* state, int deviceId, int rank) noexcept
{
    trt_edgellm::plugins::registerShmAllReduceForPlugin(state, deviceId, rank);
}
// {$edge-llm-internal-release end}
