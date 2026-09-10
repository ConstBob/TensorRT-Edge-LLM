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

#include "runtime/pi05Policy.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/pi05Bindings.h"
#include "runtime/imageUtils.h"
#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_fp16.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace trt_edgellm
{
namespace pi05
{

namespace
{

// The quantile arrays Pi05Policy normalizes the state and unnormalizes the action
// chunk with. Absent is not an error here; the coverage check below is what rejects it.
std::vector<float> readVector(nlohmann::json const& node, char const* key)
{
    if (!node.contains(key) || node.at(key).is_null())
    {
        return {};
    }
    return node.at(key).get<std::vector<float>>();
}

NormStats readStats(nlohmann::json const& root, char const* key)
{
    if (!root.contains(key))
    {
        return {};
    }
    nlohmann::json const& node = root.at(key);
    NormStats stats;
    stats.q01 = readVector(node, "q01");
    stats.q99 = readVector(node, "q99");
    return stats;
}

//! Locate norm_stats.json anywhere under assets/. openpi nests it two levels
//! deep (``assets/<org>/<dataset>/norm_stats.json``), so the search recurses
//! rather than assuming a fixed depth.
std::filesystem::path findNormStats(std::filesystem::path const& assetsDir)
{
    std::filesystem::path const direct = assetsDir / "norm_stats.json";
    if (std::filesystem::exists(direct))
    {
        return direct;
    }
    if (!std::filesystem::is_directory(assetsDir))
    {
        return {};
    }
    std::vector<std::filesystem::path> found;
    for (auto const& entry : std::filesystem::recursive_directory_iterator(assetsDir))
    {
        if (entry.is_regular_file() && entry.path().filename() == "norm_stats.json")
        {
            found.push_back(entry.path());
        }
    }
    if (found.size() > 1)
    {
        // Several embodiments staged together: the caller must say which one,
        // since picking the wrong statistics silently mis-scales every command.
        throw std::runtime_error("Multiple norm_stats.json under " + assetsDir.string()
            + "; stage exactly one embodiment's assets for this engine directory");
    }
    return found.empty() ? std::filesystem::path{} : found.front();
}

nlohmann::json readJson(std::filesystem::path const& path)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open " + path.string());
    }
    nlohmann::json doc;
    file >> doc;
    return doc;
}

//! Enough statistics of the right kind to cover \p dims components.
bool statsUsable(NormStats const& stats, int32_t dims)
{
    return std::min(stats.q01.size(), stats.q99.size()) >= static_cast<size_t>(dims);
}

//! Python ``str.strip()`` plus the two substitutions openpi applies to a task
//! string; the underscore rule is why dataset task ids tokenize as words.
std::string cleanTask(std::string const& task)
{
    static constexpr char kWhitespace[] = " \t\n\r\f\v";
    size_t const begin = task.find_first_not_of(kWhitespace);
    if (begin == std::string::npos)
    {
        return {};
    }
    std::string cleaned = task.substr(begin, task.find_last_not_of(kWhitespace) - begin + 1);
    std::replace(cleaned.begin(), cleaned.end(), '_', ' ');
    std::replace(cleaned.begin(), cleaned.end(), '\n', ' ');
    return cleaned;
}

//! ``np.digitize(x, np.linspace(-1, 1, bins + 1)[:-1]) - 1``: the count of bin
//! edges at or below \p x, minus one. Values under the first edge yield -1, and
//! the reference writes that -1 into the prompt, so it is reproduced here.
int32_t digitize(float value, int32_t bins)
{
    double const step = 2.0 / static_cast<double>(bins);
    int32_t count = 0;
    for (int32_t i = 0; i < bins; ++i)
    {
        if (-1.0 + static_cast<double>(i) * step <= static_cast<double>(value))
        {
            ++count;
        }
    }
    return count - 1;
}

//! Camera names, tolerating an export that records only how many there are: the
//! blank entries still carry the count the compact prefix has to leave out.
std::vector<std::string> readCameraNames(nlohmann::json const& cameras, char const* key)
{
    if (!cameras.contains(key) || cameras.at(key).is_null())
    {
        return {};
    }
    // The exporter writes int(empty_cameras); the entries carry no names.
    return std::vector<std::string>(cameras.at(key).get<size_t>());
}

//! Configurations whose observation and action math this runtime implements, under
//! the names openpi gives them; an export resolves the name from the embodiment.
constexpr char const* kSupportedPolicyConfigs[] = {"pi05_libero"};

std::string supportedPolicyConfigs()
{
    std::string list;
    for (char const* name : kSupportedPolicyConfigs)
    {
        list += (list.empty() ? "" : ", ") + std::string(name);
    }
    return list;
}

//! Refuse a bundle whose math this runtime does not implement. Every adapter below --
//! the camera slots, the state discretization, the action unnormalization -- is
//! per-embodiment, so an unsupported configuration would run and emit plausible but
//! wrong commands rather than fail.
void validateConfiguration(nlohmann::json const& contract, std::string const& path)
{
    int32_t const version = contract.value("contract_version", 0);
    if (version != kContractVersion)
    {
        throw std::runtime_error("pi0.5 " + path + " was exported against contract version " + std::to_string(version)
            + " but this runtime implements " + std::to_string(kContractVersion)
            + "; re-export it with tensorrt-edgellm-export");
    }
    std::string const family = contract.value("model_family", std::string{});
    if (family != "pi05")
    {
        throw std::runtime_error("pi0.5 " + path + " declares model family '" + family + "', not pi05");
    }
    std::string const config = contract.value("policy_config", std::string{});
    if (std::find_if(std::begin(kSupportedPolicyConfigs), std::end(kSupportedPolicyConfigs),
            [&config](char const* name) { return config == name; })
        == std::end(kSupportedPolicyConfigs))
    {
        throw std::runtime_error("pi0.5 " + path + " declares policy configuration '" + config
            + "'; this runtime implements " + supportedPolicyConfigs()
            + ". Its observation and action processing is embodiment-specific, so running it here "
              "would produce plausible but wrong commands");
    }
}

} // namespace

bool Pi05Policy::available(std::string const& engineDir)
{
    return std::filesystem::exists(std::filesystem::path(engineDir) / "policy.json");
}

void Pi05Policy::loadContract(std::string const& engineDir)
{
    std::filesystem::path const root(engineDir);
    std::filesystem::path const contractPath = root / "policy.json";
    if (!std::filesystem::exists(contractPath))
    {
        throw std::runtime_error("pi0.5 policy.json not found under " + engineDir
            + "; re-export the checkpoint so the observation contract is staged with the engines");
    }
    nlohmann::json const contract = readJson(contractPath);
    validateConfiguration(contract, contractPath.string());
    nlohmann::json const& state = contract.at("state");
    nlohmann::json const& action = contract.at("action");
    mContract.policyConfig = contract.value("policy_config", std::string{});
    mContract.checkpointFingerprint = contract.value("checkpoint_fingerprint", std::string{});
    mContract.exportId = contract.value(kExportIdKey, std::string{});
    if (mContract.exportId.empty())
    {
        throw std::runtime_error("pi0.5 " + contractPath.string() + " carries no " + kExportIdKey
            + "; re-export the checkpoint so the engines staged beside it can be checked against it");
    }
    mContract.stateDim = state.at("dim").get<int32_t>();
    mContract.robotActionDim = action.at("dim").get<int32_t>();
    mContract.modelActionDim = action.value("max_dim", mContract.robotActionDim);
    mContract.actionHorizon = action.at("horizon").get<int32_t>();
    mContract.numBins = state.value("num_bins", 256);
    mContract.discreteStateInput = contract.at("discrete_state_input").get<bool>();
    mContract.stateEps = state.value("eps", 1e-8F);
    mContract.cameraNames = contract.at("cameras").at("present").get<std::vector<std::string>>();
    mContract.emptyCameras = readCameraNames(contract.at("cameras"), "empty");
    if (mContract.cameraNames.empty())
    {
        throw std::runtime_error("pi0.5 policy.json declares no camera; the prefix cannot be assembled");
    }
    if (mContract.robotActionDim > mContract.modelActionDim)
    {
        throw std::runtime_error("pi0.5 policy.json declares a robot action dim wider than the model's");
    }
    auto const resolution = contract.at("image_resolution").get<std::vector<int32_t>>();
    if (resolution.size() != 2)
    {
        throw std::runtime_error("pi0.5 policy.json image_resolution must be [height, width]");
    }
    mContract.imageHeight = resolution[0];
    mContract.imageWidth = resolution[1];
    mContract.maxTokenLen = contract.at("tokenizer").at("max_length").get<int32_t>();

    std::filesystem::path const statsPath = findNormStats(root / "assets");
    if (statsPath.empty())
    {
        throw std::runtime_error("pi0.5 norm_stats.json not found under " + (root / "assets").string()
            + "; the state cannot be normalized and actions cannot be converted to robot units without it");
    }
    nlohmann::json const doc = readJson(statsPath);
    // openpi serializes as {"norm_stats": {"<key>": {...}}}.
    nlohmann::json const& stats = doc.contains("norm_stats") ? doc.at("norm_stats") : doc;
    mStateStats = readStats(stats, "state");
    mActionStats = readStats(stats, "actions");
    if (!statsUsable(mStateStats, mContract.stateDim) || !statsUsable(mActionStats, mContract.robotActionDim))
    {
        throw std::runtime_error(
            "pi0.5 norm_stats.json has no q01/q99 covering the declared state/action dims: " + statsPath.string());
    }
    LOG_INFO("pi0.5 policy: %s, checkpoint %s", mContract.policyConfig.c_str(),
        mContract.checkpointFingerprint.empty() ? "unrecorded" : mContract.checkpointFingerprint.c_str());
    LOG_INFO("pi0.5 policy: state %dd, action %dd of %d, %zu empty camera(s), %dx%d, %s", mContract.stateDim,
        mContract.robotActionDim, mContract.modelActionDim, mContract.emptyCameras.size(), mContract.imageHeight,
        mContract.imageWidth, statsPath.string().c_str());
    for (size_t i = 0; i < mContract.cameraNames.size(); ++i)
    {
        LOG_INFO("pi0.5 camera %zu: %s", i, mContract.cameraNames[i].c_str());
    }
}

Pi05Policy::Pi05Policy(std::string const& engineDir)
    : mEngineDir(engineDir)
{
    loadContract(engineDir);
}

Pi05Policy::Pi05Policy(std::string const& engineDir, cudaStream_t stream)
    : mEngineDir(engineDir)
    , mStream(stream)
{
    loadContract(engineDir);
    mRuntime = std::make_unique<Pi05Runtime>(engineDir, stream);
    Pi05PolicyConfig const& cfg = mRuntime->getConfig();
    // The runtime only ties the components to each other, so engines rebuilt over an older
    // bundle keep that bundle's camera order, prompt contract and normalization statistics.
    if (mContract.exportId != cfg.exportId)
    {
        throw std::runtime_error("pi0.5 engines under " + engineDir + " are from export " + cfg.exportId
            + " but the policy.json staged beside them is from export " + mContract.exportId
            + "; rebuild the bundle from one export");
    }
    if (mContract.imageHeight != cfg.imageSize || mContract.imageWidth != cfg.imageSize)
    {
        throw std::runtime_error("pi0.5 policy.json image_resolution disagrees with the visual engine contract");
    }
    // max_dim is the padded width the engine emits; dim is the embodiment's own prefix of it.
    if (mContract.modelActionDim != cfg.actionDim || mContract.actionHorizon != cfg.actionHorizon)
    {
        throw std::runtime_error("pi0.5 policy.json declares a [" + std::to_string(mContract.actionHorizon) + ", "
            + std::to_string(mContract.modelActionDim) + "] action chunk but the action engine emits ["
            + std::to_string(cfg.actionHorizon) + ", " + std::to_string(cfg.actionDim) + "]");
    }
    // A request's view count is fixed by the contract, so the staging pair is sized once
    // rather than per call; the host half is pinned because the upload is asynchronous.
    std::vector<int64_t> const shape{numCameras(), 3, cfg.imageSize, cfg.imageSize};
    mPixelValues = rt::Tensor(shape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "pi05::pixelValues");
    mPixelValuesHost = rt::Tensor(shape, rt::DeviceType::kCPU, nvinfer1::DataType::kHALF, "pi05::pixelValuesHost");
}

Pi05Policy::~Pi05Policy() = default;

Pi05Runtime& Pi05Policy::runtime()
{
    if (mRuntime == nullptr)
    {
        throw std::runtime_error("pi0.5 policy was built from " + mEngineDir
            + " without engines; construct it with a stream to run inference");
    }
    return *mRuntime;
}

std::vector<size_t> Pi05Policy::orderCameraSlots(std::vector<Pi05CameraView> const& views) const
{
    validateImageCount(static_cast<int32_t>(views.size()));
    bool const anyNamed
        = std::any_of(views.begin(), views.end(), [](Pi05CameraView const& v) { return !v.name.empty(); });
    auto const kUnfilled = std::numeric_limits<size_t>::max();
    std::vector<size_t> slots(views.size(), kUnfilled);
    if (!anyNamed)
    {
        std::iota(slots.begin(), slots.end(), size_t{0});
        return slots;
    }
    for (size_t v = 0; v < views.size(); ++v)
    {
        Pi05CameraView const& view = views[v];
        if (view.name.empty())
        {
            throw std::invalid_argument(
                "pi0.5 request names some camera views but not all; name every view or none, in contract order: "
                + cameraOrderSummary(mContract.cameraNames));
        }
        auto const slot = std::find(mContract.cameraNames.begin(), mContract.cameraNames.end(), view.name);
        if (slot == mContract.cameraNames.end())
        {
            throw std::invalid_argument("pi0.5 contract has no camera named " + view.name + "; it declares "
                + cameraOrderSummary(mContract.cameraNames));
        }
        auto const index = static_cast<size_t>(std::distance(mContract.cameraNames.begin(), slot));
        if (slots[index] != kUnfilled)
        {
            throw std::invalid_argument("pi0.5 request supplies camera " + view.name + " twice");
        }
        slots[index] = v;
    }
    return slots;
}

double Pi05Policy::stageOneView(unsigned char const* rgb, int32_t srcH, int32_t srcW, size_t viewIdx)
{
    using Clock = std::chrono::steady_clock;
    auto const viewElems = static_cast<size_t>(3) * mContract.imageHeight * mContract.imageWidth;
    mPlanarView.resize(viewElems);
    auto const start = Clock::now();
    resizeWithPad(rgb, srcH, srcW, mContract.imageHeight, mContract.imageWidth, mPlanarView.data());
    auto* rows = mPixelValuesHost.dataPointer<__half>();
    for (size_t i = 0; i < viewElems; ++i)
    {
        rows[viewIdx * viewElems + i] = __float2half(mPlanarView[i]);
    }
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void Pi05Policy::stagePixelValues(std::vector<Pi05CameraView const*> const& ordered, Pi05ObservationTimes& times)
{
    using Clock = std::chrono::steady_clock;
    auto const viewElems = static_cast<size_t>(3) * mContract.imageHeight * mContract.imageWidth;
    for (size_t v = 0; v < ordered.size(); ++v)
    {
        Pi05CameraView const& view = *ordered[v];
        if (view.rgb != nullptr)
        {
            times.resizeMs += stageOneView(view.rgb, view.height, view.width, v);
            continue;
        }
        auto const decodeStart = Clock::now();
        double preprocessMs{0.0};
        int64_t srcH{0};
        int64_t srcW{0};
        {
            // Scoped so the decoded frame's pinned host buffer is freed, and charged, here:
            // allocating and pinning it is part of the cost of being handed a file path.
            rt::imageUtils::ImageData const image = rt::imageUtils::loadRgbImageFromFile(view.imagePath);
            srcH = image.height;
            srcW = image.width;
            preprocessMs = stageOneView(image.data(), static_cast<int32_t>(srcH), static_cast<int32_t>(srcW), v);
        }
        times.decodeMs += std::chrono::duration<double, std::milli>(Clock::now() - decodeStart).count() - preprocessMs;
        times.resizeMs += preprocessMs;
        LOG_INFO("Loaded %s (%ldx%ld) -> %dx%d", view.imagePath.c_str(), srcW, srcH, mContract.imageWidth,
            mContract.imageHeight);
    }
    // No sync: the upload and the engines share mStream, and the previous request
    // drained it before returning, so the pinned staging is free again.
    CUDA_CHECK(cudaMemcpyAsync(mPixelValues.rawPointer(), mPixelValuesHost.dataPointer<__half>(),
        ordered.size() * viewElems * sizeof(__half), cudaMemcpyHostToDevice, mStream));
}

std::vector<int32_t> Pi05Policy::tokenize(std::string const& prompt)
{
    if (mTokenizer == nullptr)
    {
        std::filesystem::path const dir = std::filesystem::path(mEngineDir) / "text_tokenizer";
        auto loaded = std::make_unique<tokenizer::Tokenizer>();
        if (!loaded->loadFromHF(dir))
        {
            throw std::runtime_error("Failed to load the PaliGemma tokenizer from " + dir.string());
        }
        mTokenizer = std::move(loaded);
    }
    std::vector<int32_t> ids = mTokenizer->encode(prompt, /*addBos=*/true, /*addEos=*/false);
    if (!mContract.discreteStateInput)
    {
        // openpi encodes the "\n" separately from the task text, as the start-of-answer token.
        std::vector<int32_t> const newline = mTokenizer->encode("\n", /*addBos=*/false, /*addEos=*/false);
        ids.insert(ids.end(), newline.begin(), newline.end());
    }
    if (static_cast<int32_t>(ids.size()) > mContract.maxTokenLen)
    {
        LOG_WARNING(
            "Prompt tokenizes to %zu tokens, truncating to the contract's %d", ids.size(), mContract.maxTokenLen);
        ids.resize(static_cast<size_t>(mContract.maxTokenLen));
    }
    return ids;
}

Pi05ActionChunk Pi05Policy::inferTensors(
    rt::Tensor const& pixelValues, std::vector<int32_t> const& tokenIds, int32_t batch)
{
    validatePixelValues(pixelValues);
    if (batch < 1)
    {
        throw std::invalid_argument("pi0.5 request batch must be at least 1");
    }
    Pi05Runtime& engines = runtime();
    using Clock = std::chrono::steady_clock;

    Pi05ActionChunk chunk;
    chunk.tokenIds = tokenIds;
    chunk.batch = batch;
    auto const engineStart = Clock::now();
    chunk.normalizedActions = engines.generate(pixelValues, tokenIds, batch);
    chunk.timings.engineMs = std::chrono::duration<double, std::milli>(Clock::now() - engineStart).count();
    if (chunk.normalizedActions.empty())
    {
        throw std::runtime_error("pi0.5 policy produced no actions");
    }

    Pi05PolicyConfig const& cfg = engines.getConfig();
    chunk.horizon = cfg.actionHorizon;
    chunk.modelActionDim = cfg.actionDim;
    auto const chunkElems = static_cast<size_t>(cfg.actionHorizon) * cfg.actionDim;
    chunk.robotActions = unnormalizeActions(std::vector<float>(chunk.normalizedActions.begin(),
                                                chunk.normalizedActions.begin() + static_cast<ptrdiff_t>(chunkElems)),
        cfg.actionHorizon, cfg.actionDim);
    chunk.timings.stages = engines.getStageTimes();
    chunk.timings.policyMs = std::chrono::duration<double, std::milli>(Clock::now() - engineStart).count();
    return chunk;
}

Pi05ActionChunk Pi05Policy::infer(Pi05Observation const& observation)
{
    (void) runtime(); // fail before any of the observation work when there are no engines
    using Clock = std::chrono::steady_clock;
    Pi05ObservationTimes times;
    auto const observationStart = Clock::now();
    std::vector<size_t> const slots = orderCameraSlots(observation.cameras);
    std::vector<Pi05CameraView const*> ordered(slots.size());
    for (size_t i = 0; i < slots.size(); ++i)
    {
        Pi05CameraView const& view = observation.cameras[slots[i]];
        bool const hasRgb = view.rgb != nullptr;
        if (hasRgb == !view.imagePath.empty())
        {
            throw std::invalid_argument("pi0.5 camera view " + std::to_string(i)
                + (hasRgb ? " sets both a decoded frame and an image path"
                          : " sets neither a decoded frame nor an "
                            "image path")
                + "; a view carries exactly one source");
        }
        if (hasRgb && (view.height <= 0 || view.width <= 0))
        {
            throw std::invalid_argument("pi0.5 camera view " + std::to_string(i)
                + " supplies a decoded frame with a non-positive height or width; both are needed to resize it");
        }
        ordered[i] = &view;
    }
    stagePixelValues(ordered, times);
    auto const restStart = Clock::now();
    std::string prompt = buildPrompt(observation.task, observation.state);
    std::vector<int32_t> const tokenIds = tokenize(prompt);
    auto const observationEnd = Clock::now();
    times.restMs = std::chrono::duration<double, std::milli>(observationEnd - restStart).count();
    times.totalMs = std::chrono::duration<double, std::milli>(observationEnd - observationStart).count();

    Pi05ActionChunk chunk = inferTensors(mPixelValues, tokenIds, observation.batch);
    chunk.prompt = std::move(prompt);
    chunk.timings.observation = times;
    // Widen inferTensors' window to the front end this call also paid for.
    chunk.timings.policyMs = std::chrono::duration<double, std::milli>(Clock::now() - observationStart).count();
    return chunk;
}

void Pi05Policy::validateImageCount(int32_t images) const
{
    if (images != numCameras())
    {
        throw std::invalid_argument("pi0.5 request supplies " + std::to_string(images)
            + " image(s) but the contract declares " + std::to_string(numCameras())
            + " camera(s), in order: " + cameraOrderSummary(mContract.cameraNames));
    }
}

void Pi05Policy::validateViewCount(int32_t views) const
{
    int32_t const withEmpty = numCameras() + static_cast<int32_t>(mContract.emptyCameras.size());
    if (views != numCameras() && views != withEmpty)
    {
        throw std::invalid_argument("pi0.5 request has " + std::to_string(views) + " view(s); the contract admits "
            + std::to_string(numCameras()) + " (compact) or " + std::to_string(withEmpty)
            + " (with the empty cameras), in order: " + cameraOrderSummary(mContract.cameraNames));
    }
}

void Pi05Policy::validatePixelValues(rt::Tensor const& pixelValues) const
{
    rt::Coords const shape = pixelValues.getShape();
    if (shape.getNumDims() != 4 || shape[1] != 3 || shape[2] != mContract.imageHeight
        || shape[3] != mContract.imageWidth)
    {
        throw std::invalid_argument("pi0.5 pixel values are " + shape.formatString()
            + "; the contract declares [views, 3, " + std::to_string(mContract.imageHeight) + ", "
            + std::to_string(mContract.imageWidth) + "]");
    }
    validateViewCount(static_cast<int32_t>(shape[0]));
    if (pixelValues.getDataType() != nvinfer1::DataType::kHALF)
    {
        throw std::invalid_argument(
            "pi0.5 pixel values must be FLOAT16; the visual engine reads them as fp16 "
            "whatever the tensor declares");
    }
    if (pixelValues.getDeviceType() != rt::DeviceType::kGPU)
    {
        throw std::invalid_argument(
            "pi0.5 pixel values must be a device tensor; the visual engine is handed "
            "their address and never copies them");
    }
}

std::string Pi05Policy::buildPrompt(std::string const& task, std::vector<float> const& state) const
{
    if (static_cast<int32_t>(state.size()) != mContract.stateDim)
    {
        throw std::invalid_argument("pi0.5 state has " + std::to_string(state.size()) + " dims, contract declares "
            + std::to_string(mContract.stateDim));
    }
    if (!mContract.discreteStateInput)
    {
        // openpi's TokenizePrompt with discrete_state_input False: the task text alone,
        // tokenize() appending the newline. The state is validated above but never used.
        return cleanTask(task);
    }
    std::ostringstream bins;
    for (int32_t d = 0; d < mContract.stateDim; ++d)
    {
        float const lo = mStateStats.q01[static_cast<size_t>(d)];
        float const hi = mStateStats.q99[static_cast<size_t>(d)];
        float const denom = hi > lo ? hi - lo : mContract.stateEps;
        float const normalized = 2.0F * (state[static_cast<size_t>(d)] - lo) / denom - 1.0F;
        int32_t const bin = digitize(normalized, mContract.numBins);
        if (bin < 0)
        {
            LOG_WARNING(
                "pi0.5 state dim %d normalizes to %.4f, below the discretization range; the prompt "
                "carries the reference's out-of-range bin -1",
                d, normalized);
        }
        bins << (d ? " " : "") << bin;
    }
    return "Task: " + cleanTask(task) + ", State: " + bins.str() + ";\nAction: ";
}

std::vector<float> Pi05Policy::unnormalizeActions(
    std::vector<float> const& normalized, int32_t horizon, int32_t actionDim) const
{
    if (mContract.robotActionDim > actionDim)
    {
        throw std::invalid_argument("pi0.5 robot action dim exceeds the model action dim");
    }
    size_t const expected = static_cast<size_t>(horizon) * static_cast<size_t>(actionDim);
    if (normalized.size() != expected)
    {
        throw std::invalid_argument("pi0.5 normalized chunk size does not match horizon * actionDim");
    }
    int32_t const robotDim = mContract.robotActionDim;
    std::vector<float> out(static_cast<size_t>(horizon) * static_cast<size_t>(robotDim));
    for (int32_t t = 0; t < horizon; ++t)
    {
        for (int32_t d = 0; d < robotDim; ++d)
        {
            float const normed = normalized[static_cast<size_t>(t) * actionDim + d];
            float const lo = mActionStats.q01[static_cast<size_t>(d)];
            float const hi = mActionStats.q99[static_cast<size_t>(d)];
            out[static_cast<size_t>(t) * robotDim + d] = (normed + 1.0F) * (hi - lo) * 0.5F + lo;
        }
    }
    return out;
}

std::string cameraOrderSummary(std::vector<std::string> const& names)
{
    std::ostringstream summary;
    for (size_t i = 0; i < names.size(); ++i)
    {
        summary << (i ? ", " : "") << i << '=' << (names[i].empty() ? "<unnamed>" : names[i]);
    }
    return summary.str();
}

namespace
{
//! Pillow's triangle resampler, one axis. The support widens with the downscale factor,
//! and the 8-bit passes accumulate in fixed point; a float accumulation lands a level off.
constexpr int32_t kPrecisionBits = 32 - 8 - 2;

struct ResampleAxis
{
    std::vector<int32_t> weights; //!< \p taps entries per output pixel, scaled by 2^kPrecisionBits
    std::vector<int32_t> starts;
    int32_t taps{0};
};

ResampleAxis makeResampleAxis(int32_t srcSize, int32_t dstSize)
{
    double const scale = static_cast<double>(srcSize) / dstSize;
    double const support = std::max(scale, 1.0);
    ResampleAxis axis;
    axis.taps = static_cast<int32_t>(std::ceil(support)) * 2 + 1;
    axis.weights.assign(static_cast<size_t>(dstSize) * axis.taps, 0);
    axis.starts.assign(static_cast<size_t>(dstSize), 0);
    for (int32_t i = 0; i < dstSize; ++i)
    {
        double const center = (static_cast<double>(i) + 0.5) * scale;
        int32_t const begin = std::max(0, static_cast<int32_t>(center - support + 0.5));
        int32_t const end = std::min(srcSize, static_cast<int32_t>(center + support + 0.5));
        axis.starts[static_cast<size_t>(i)] = begin;
        auto const tapAt = [&](int32_t j) {
            return std::max(0.0, 1.0 - std::abs((static_cast<double>(j) + 0.5 - center) / support));
        };
        double total = 0.0;
        for (int32_t j = begin; j < end; ++j)
        {
            total += tapAt(j);
        }
        for (int32_t j = begin; total > 0.0 && j < end; ++j)
        {
            axis.weights[static_cast<size_t>(i) * axis.taps + (j - begin)]
                = static_cast<int32_t>(std::lround(tapAt(j) / total * (1 << kPrecisionBits)));
        }
    }
    return axis;
}

//! Pillow rounds each pass back to 8 bits before the next one reads it.
unsigned char round8(int32_t accumulated)
{
    return static_cast<unsigned char>(std::clamp(accumulated >> kPrecisionBits, 0, 255));
}
} // namespace

void resizeWithPad(unsigned char const* pixels, int32_t srcH, int32_t srcW, int32_t height, int32_t width, float* out)
{
    // Aspect-preserving target box, centred in the padded frame.
    double const ratio = std::max(static_cast<double>(srcW) / width, static_cast<double>(srcH) / height);
    auto const boxH = static_cast<int32_t>(static_cast<double>(srcH) / ratio);
    auto const boxW = static_cast<int32_t>(static_cast<double>(srcW) / ratio);
    int32_t const padTop = (height - boxH) / 2;
    int32_t const padLeft = (width - boxW) / 2;

    size_t const plane = static_cast<size_t>(height) * width;
    std::fill_n(out, 3 * plane, -1.0F); // black in [-1, 1]: the padding value after the [0, 1] shift
    if (boxH <= 0 || boxW <= 0)
    {
        return;
    }

    ResampleAxis const horizontal = makeResampleAxis(srcW, boxW);
    std::vector<unsigned char> rows(static_cast<size_t>(srcH) * boxW * 3);
    for (int32_t y = 0; y < srcH; ++y)
    {
        for (int32_t x = 0; x < boxW; ++x)
        {
            for (int32_t c = 0; c < 3; ++c)
            {
                int32_t acc = 1 << (kPrecisionBits - 1);
                for (int32_t k = 0; k < horizontal.taps; ++k)
                {
                    int32_t const j = horizontal.starts[static_cast<size_t>(x)] + k;
                    if (j >= srcW)
                    {
                        break;
                    }
                    acc += horizontal.weights[static_cast<size_t>(x) * horizontal.taps + k]
                        * pixels[(static_cast<size_t>(y) * srcW + j) * 3 + c];
                }
                rows[(static_cast<size_t>(y) * boxW + x) * 3 + c] = round8(acc);
            }
        }
    }

    ResampleAxis const vertical = makeResampleAxis(srcH, boxH);
    for (int32_t y = 0; y < boxH; ++y)
    {
        for (int32_t x = 0; x < boxW; ++x)
        {
            for (int32_t c = 0; c < 3; ++c)
            {
                int32_t acc = 1 << (kPrecisionBits - 1);
                for (int32_t k = 0; k < vertical.taps; ++k)
                {
                    int32_t const j = vertical.starts[static_cast<size_t>(y)] + k;
                    if (j >= srcH)
                    {
                        break;
                    }
                    acc += vertical.weights[static_cast<size_t>(y) * vertical.taps + k]
                        * rows[(static_cast<size_t>(j) * boxW + x) * 3 + c];
                }
                out[static_cast<size_t>(c) * plane + static_cast<size_t>(padTop + y) * width + (padLeft + x)]
                    = static_cast<float>(round8(acc)) / 255.0F * 2.0F - 1.0F;
            }
        }
    }
}

} // namespace pi05
} // namespace trt_edgellm
