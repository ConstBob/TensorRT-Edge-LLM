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

// End-to-end Cosmos3 policy inference from a raw image + text instruction(s).
//
// The pipeline runs entirely in-process:
//   (a) load + preprocess the image (core imageUtils decode + resize, normalize to
//       [-1,1] pixel_values, broadcast the conditioning frame to the VAE clip [B,3,F,H,W]),
//   (b) tokenize the prompt(s) with the co-located tokenizer (text_tokenizer/tokenizer.json),
//   (c) apply embed_tokens with the core embeddingLookup kernel -> inputs_embeds [B,S,hidden],
//   (d) VAE encode -> UND prefill -> GEN diffusion loop -> action chunk,
//   (e) write the action chunk.
//
// Batching: --prompt may be given multiple times; the request batch B is the prompt count
// (the conditioning image is shared) and requires engines built with --maxBatchSize >= B.
// Batched prompts must tokenize to the SAME length: the UND prefill graph has no attention
// mask input, so padded positions would leak into the gen cross-attention.
//
// All shape parameters (clip F/H/W, hidden size, action dims) come from the exported
// component contracts under --engineDir; nothing is passed or hardcoded here.
//
// I/O convention (project-wide):
//   * The DEFAULT --output is a JSON action file (the inference-response INTERFACE).
//     For B == 1 the "action" field is [chunk][dim] (unchanged single-request layout);
//     for B > 1 it is [B][chunk][dim]. "shape" is always [B, chunk, dim].
//   * An --output <name>.safetensors writes the RAW action tensor for bulk tensor
//     recording under key "action" (float32, [B, chunk, dim]).
//
// The tokenizer + embed_tokens artifacts are looked up under the engine dir
// (co-located by the build step, like the core engine layout).

#include "common/logger.h"
#include "runtime/cosmos3Runtime.h"

#include "common/checkMacros.h"
#include "common/safetensorsUtils.h"
#include "common/tensor.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "runtime/imageUtils.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace trt_edgellm;
namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace
{
enum OptionId : int
{
    HELP = 901,
    ENGINE_DIR = 902,
    IMAGE = 903,
    PROMPT = 904,
    DOMAIN = 905,
    OUTPUT = 906,
    STEPS = 907,
    SEED = 908,
    VIEW_POINT = 909,
    RAW_PROMPT = 911,
    PROMPT_FILE = 912,
    ITERS = 915,
    WARMUP = 916,
    CUDAGRAPH = 917
};

struct Args
{
    std::string engineDir;
    std::string imagePath;               //!< PNG/JPG conditioning observation (shared across the batch)
    std::vector<std::string> prompts;    //!< text instructions; batch = prompt count
    std::string domain{"droid_lerobot"}; //!< action domain
    std::string viewPoint{"ego_view"};   //!< camera viewpoint for the policy prompt framing
    bool rawPrompt{false};               //!< use --prompt / --promptFile verbatim (skip JSON build)
    std::string promptFile;              //!< read the instruction/JSON prompt from a file
    std::string output;                  //!< action chunk out (.json default, or .safetensors for raw tensor)
    int32_t steps{4};                    //!< diffusion steps
    int32_t seed{0};                     //!< initial-noise seed
    // Benchmark controls.
    int32_t iters{1};
    int32_t warmup{2};
    bool cudagraph{false};
    bool help{false};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] --engineDir <dir> --image <path> --prompt \"<instruction>\" [--prompt ...] "
                 "[--output <action.json|.safetensors>] [--domain <name>] [--viewPoint <name>] [--steps <int>] [--seed "
                 "<int>] "
                 "[--iters <int>] [--warmup <int>] [--cudagraph]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help       Display this help message" << std::endl;
    std::cerr << "  --engineDir  Directory with und_prefill/, vae_encoder/, gen/ engines" << std::endl;
    std::cerr << "               (+ text_tokenizer/ and embed_tokens.safetensors). Required." << std::endl;
    std::cerr << "  --image      Conditioning observation (PNG/JPG), shared across the batch. Required." << std::endl;
    std::cerr << "  --prompt     Text instruction. Repeat for a batched request (batch = prompt count;" << std::endl;
    std::cerr << "               requires engines built with --maxBatchSize >= count). Required." << std::endl;
    std::cerr << "  --output     Action chunk output. DEFAULT is JSON (the inference-response" << std::endl;
    std::cerr << "               interface); a .safetensors suffix writes the raw action tensor." << std::endl;
    std::cerr << "  --domain     Action domain. Default = droid_lerobot" << std::endl;
    std::cerr << "  --rawPrompt  Use the prompt / --promptFile text verbatim as the instruction" << std::endl;
    std::cerr << "               (skip structured-JSON prompt construction)." << std::endl;
    std::cerr << "  --promptFile Read the instruction (or full JSON prompt) from a file." << std::endl;
    std::cerr << "  --viewPoint  Camera viewpoint for the policy prompt: ego_view (default)," << std::endl;
    std::cerr << "               third_person_view, wrist_view, or concat_view." << std::endl;
    std::cerr << "  --steps      Diffusion denoise steps. Default = 4" << std::endl;
    std::cerr << "  --seed       Initial-noise seed. Default = 0" << std::endl;
    std::cerr << "  --iters      Benchmark iterations. Default = 1" << std::endl;
    std::cerr << "  --warmup     Benchmark warmup rounds. Default = 2" << std::endl;
    std::cerr << "  --cudagraph  Capture/replay the per-step GEN forward as a CUDA graph." << std::endl;
}

bool parseArgs(Args& args, int argc, char** argv)
{
    static struct option options[] = {{"help", no_argument, nullptr, HELP},
        {"engineDir", required_argument, nullptr, ENGINE_DIR}, {"image", required_argument, nullptr, IMAGE},
        {"prompt", required_argument, nullptr, PROMPT}, {"domain", required_argument, nullptr, DOMAIN},
        {"viewPoint", required_argument, nullptr, VIEW_POINT}, {"rawPrompt", no_argument, nullptr, RAW_PROMPT},
        {"promptFile", required_argument, nullptr, PROMPT_FILE}, {"output", required_argument, nullptr, OUTPUT},
        {"steps", required_argument, nullptr, STEPS}, {"seed", required_argument, nullptr, SEED},
        {"iters", required_argument, nullptr, ITERS}, {"warmup", required_argument, nullptr, WARMUP},
        {"cudagraph", no_argument, nullptr, CUDAGRAPH}, {nullptr, 0, nullptr, 0}};
    int opt;
    while ((opt = getopt_long(argc, argv, "", options, nullptr)) != -1)
    {
        switch (opt)
        {
        case HELP: args.help = true; return true;
        case ENGINE_DIR: args.engineDir = optarg ? optarg : ""; break;
        case IMAGE: args.imagePath = optarg ? optarg : ""; break;
        case PROMPT:
            if (optarg != nullptr)
            {
                args.prompts.emplace_back(optarg);
            }
            break;
        case DOMAIN: args.domain = optarg ? optarg : args.domain; break;
        case VIEW_POINT: args.viewPoint = optarg ? optarg : args.viewPoint; break;
        case RAW_PROMPT: args.rawPrompt = true; break;
        case PROMPT_FILE: args.promptFile = optarg ? optarg : ""; break;
        case OUTPUT: args.output = optarg ? optarg : ""; break;
        case STEPS: args.steps = optarg ? std::stoi(optarg) : args.steps; break;
        case SEED: args.seed = optarg ? std::stoi(optarg) : args.seed; break;
        case ITERS: args.iters = optarg ? std::stoi(optarg) : args.iters; break;
        case WARMUP: args.warmup = optarg ? std::stoi(optarg) : args.warmup; break;
        case CUDAGRAPH: args.cudagraph = true; break;
        default: return false;
        }
    }
    return !args.engineDir.empty() && !args.imagePath.empty() && (!args.prompts.empty() || !args.promptFile.empty());
}

//! Read the [B,3,F,H,W] pixel-clip shape from the vae_encoder component contract.
std::vector<int64_t> readPixelClipShape(fs::path const& engineDir)
{
    std::ifstream f(engineDir / "vae_encoder" / "config.json");
    ELLM_CHECK(f.is_open(), "Failed to open vae_encoder config under " + engineDir.string());
    auto const j = Json::parse(f);
    ELLM_CHECK(j.contains("optimization_profile") && j.at("optimization_profile").contains("pixel_values"),
        "vae_encoder config missing optimization_profile.pixel_values");
    auto shape = j.at("optimization_profile").at("pixel_values").at("opt").get<std::vector<int64_t>>();
    ELLM_CHECK(shape.size() == 5, "vae_encoder pixel_values profile must be [B,3,F,H,W]");
    return shape;
}

//! Read the text hidden size from the und_prefill component contract.
int32_t readHiddenSize(fs::path const& engineDir)
{
    std::ifstream f(engineDir / "und_prefill" / "config.json");
    ELLM_CHECK(f.is_open(), "Failed to open und_prefill config under " + engineDir.string());
    auto const j = Json::parse(f);
    ELLM_CHECK(j.contains("hidden_size"), "und_prefill config missing hidden_size");
    return j.at("hidden_size").get<int32_t>();
}

//! Viewpoint framing sentences used in the structured policy prompt (mirrors the
//! reference action-dataset viewpoint templates).
std::string viewpointFraming(std::string const& viewPoint)
{
    if (viewPoint == "ego_view")
    {
        return "This video is captured from a first-person perspective looking at the scene.";
    }
    if (viewPoint == "third_person_view")
    {
        return "This video is captured from a third-person perspective looking towards the agent from the front.";
    }
    if (viewPoint == "wrist_view")
    {
        return "This video is captured from a wrist-mounted camera.";
    }
    if (viewPoint == "concat_view")
    {
        return "This video contains concatenated views from multiple camera perspectives.";
    }
    ELLM_CHECK(false,
        "Unsupported --viewPoint '" + viewPoint
            + "'; expected ego_view, third_person_view, wrist_view, or concat_view.");
    return {};
}

//! Format integer seconds as M:SS for the policy prompt time range.
std::string formatTimeMSS(int32_t seconds)
{
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%d:%02d", seconds / 60, seconds % 60);
    return buf;
}

//! Canonical width,height aspect-ratio string for the known 480-class buckets;
//! reduced-fraction fallback otherwise.
std::string aspectRatioString(int32_t width, int32_t height)
{
    if (width == 736 && height == 544)
    {
        return "4,3";
    }
    if (width == 832 && height == 480)
    {
        return "16,9";
    }
    if (width == 480 && height == 832)
    {
        return "9,16";
    }
    if (width == 544 && height == 736)
    {
        return "3,4";
    }
    if (width == 640 && height == 640)
    {
        return "1,1";
    }
    int32_t const d = std::gcd(width, height);
    return std::to_string(width / d) + "," + std::to_string(height / d);
}

//! Build the structured JSON policy prompt the reference pipeline conditions the
//! UND tower on. Field order and ": " / ", " spacing must match the reference
//! Python json serialization exactly so the token stream is identical.
std::string buildPolicyPrompt(std::string const& instruction, std::string const& viewPoint, int32_t actionChunkSize,
    float fps, int32_t height, int32_t width)
{
    std::string sentence = instruction;
    while (!sentence.empty() && std::isspace(static_cast<unsigned char>(sentence.back())))
    {
        sentence.pop_back();
    }
    if (!sentence.empty() && sentence.back() != '.' && sentence.back() != '!' && sentence.back() != '?')
    {
        sentence += '.';
    }
    double const endSeconds = static_cast<double>(actionChunkSize) / fps;
    int32_t const timeEnd = static_cast<int32_t>(std::lround(endSeconds));
    int32_t const duration = static_cast<int32_t>(endSeconds); // truncated
    char fpsText[32];
    std::snprintf(fpsText, sizeof(fpsText), "%.1f", static_cast<double>(fps));

    std::string prompt = "{\"cinematography\": {\"framing\": ";
    prompt += Json(viewpointFraming(viewPoint)).dump();
    prompt += "}, \"actions\": [{\"time\": ";
    prompt += Json("0:00-" + formatTimeMSS(timeEnd)).dump();
    prompt += ", \"description\": ";
    prompt += Json(sentence).dump();
    prompt += "}], \"duration\": \"" + std::to_string(duration) + "s\", \"fps\": ";
    prompt += fpsText;
    prompt += ", \"resolution\": {\"H\": " + std::to_string(height) + ", \"W\": " + std::to_string(width) + "}";
    prompt += ", \"aspect_ratio\": " + Json(aspectRatioString(width, height)).dump() + "}";
    return prompt;
}

//! Read action_chunk_size and fps from the GEN component contract.
std::pair<int32_t, float> readActionChunkAndFps(fs::path const& engineDir)
{
    std::ifstream f(engineDir / "gen" / "config.json");
    ELLM_CHECK(f.is_open(), "Failed to open gen config under " + engineDir.string());
    auto const j = Json::parse(f);
    ELLM_CHECK(j.contains("action_chunk_size") && j.contains("fps"), "gen config missing action_chunk_size / fps");
    return {j.at("action_chunk_size").get<int32_t>(), j.at("fps").get<float>()};
}

//! Write the RAW action tensor as a safetensors file (bulk tensor recording).
//! Single tensor "action" (float32, shape [B, chunk, dim]).
void writeActionSafetensors(std::string const& path, std::vector<float> const& action, int32_t batch, int32_t chunk,
    int32_t dim, cudaStream_t stream)
{
    std::vector<int64_t> const shape{batch, chunk, dim};
    rt::Tensor t(shape, rt::DeviceType::kCPU, nvinfer1::DataType::kFLOAT, "action");
    std::memcpy(t.rawPointer(), action.data(), action.size() * sizeof(float));
    std::vector<rt::Tensor> tensors;
    tensors.push_back(std::move(t));
    ELLM_CHECK(rt::safetensors::saveSafetensors(path, tensors, stream), "Failed to write safetensors: " + path);
    LOG_INFO("Wrote raw action tensor (%d,%d,%d) to %s (safetensors)", batch, chunk, dim, path.c_str());
}

//! Write the action chunk as the JSON inference-response interface (the DEFAULT path).
//! For batch == 1 "action" is [chunk][dim] (the single-request layout consumed by the
//! RoboLab wrapper); for batch > 1 it is [B][chunk][dim]. "shape" is always [B, chunk, dim].
void writeActionJson(std::string const& path, std::vector<float> const& action, int32_t batch, int32_t chunk,
    int32_t dim, std::string const& domain, int32_t steps, std::vector<std::string> const& prompts, bool finite,
    int32_t seqLen)
{
    auto chunkRows = [&](int32_t b) {
        Json rows = Json::array();
        for (int32_t c = 0; c < chunk; ++c)
        {
            Json row = Json::array();
            for (int32_t d = 0; d < dim; ++d)
            {
                row.push_back(action[(static_cast<size_t>(b) * chunk + c) * dim + d]);
            }
            rows.push_back(std::move(row));
        }
        return rows;
    };

    Json j;
    if (batch == 1)
    {
        j["action"] = chunkRows(0);
        j["prompt"] = prompts.front();
    }
    else
    {
        Json batched = Json::array();
        for (int32_t b = 0; b < batch; ++b)
        {
            batched.push_back(chunkRows(b));
        }
        j["action"] = std::move(batched);
        j["prompt"] = prompts;
    }
    j["shape"] = {batch, chunk, dim};
    j["dtype"] = "float32";
    j["domain"] = domain;
    j["num_inference_steps"] = steps;
    j["finite"] = finite;
    j["meta"] = {{"seq_len", seqLen}};

    std::ofstream of(path);
    of << j.dump(2) << "\n";
    LOG_INFO("Wrote action JSON (%d,%d,%d) to %s", batch, chunk, dim, path.c_str());
}
} // namespace

int main(int argc, char** argv)
{
    Args args;
    if (!parseArgs(args, argc, argv))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }
    gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);

    if (args.domain != "droid_lerobot")
    {
        LOG_WARNING("Domain '%s' is not the validated 'droid_lerobot' domain; running anyway.", args.domain.c_str());
    }

    cudaStream_t stream{nullptr};
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        LOG_ERROR("Failed to create CUDA stream");
        return EXIT_FAILURE;
    }

    // All shape parameters come from the exported component contracts.
    if (!args.promptFile.empty())
    {
        std::ifstream pf(args.promptFile);
        ELLM_CHECK(pf.is_open(), "Failed to open --promptFile: " + args.promptFile);
        std::stringstream ss;
        ss << pf.rdbuf();
        args.prompts.assign(1, ss.str());
    }
    int32_t const batch = static_cast<int32_t>(args.prompts.size());
    std::vector<int64_t> const clipShape = readPixelClipShape(args.engineDir); // [B,3,F,H,W]
    int32_t const pixelFrames = static_cast<int32_t>(clipShape[2]);
    int32_t const clipH = static_cast<int32_t>(clipShape[3]);
    int32_t const clipW = static_cast<int32_t>(clipShape[4]);
    int32_t const hiddenSize = readHiddenSize(args.engineDir);
    auto const [actionChunkSize, fps] = readActionChunkAndFps(args.engineDir);
    size_t const hw = static_cast<size_t>(clipH) * clipW;

    try
    {
        // ------------------------------------------------------------------ //
        // (a) Conditioning frame -> pixel clip [B,3,F,H,W] in [-1,1].         //
        // ------------------------------------------------------------------ //
        rt::imageUtils::ImageData const img = rt::imageUtils::loadImageFromFile(args.imagePath);
        LOG_INFO("Loaded image %s (%ldx%ld), resizing to %dx%d and normalizing to pixel_values.",
            args.imagePath.c_str(), img.width, img.height, clipW, clipH);
        rt::imageUtils::ImageData resized(
            rt::Tensor({1, clipH, clipW, 3}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8, "cosmos3::resized"));
        rt::imageUtils::ImageData const& frame
            = rt::imageUtils::resizeImage(img, resized, clipW, clipH, rt::imageUtils::InterpolationMode::kLINEAR);

        // HWC uint8 -> planar CHW float in [-1,1] (preprocessor convention: x / 127.5 - 1), with the
        // single conditioning frame broadcast across all F clip frames and all B batch elements.
        size_t const clipElems = static_cast<size_t>(3) * pixelFrames * hw;
        std::vector<float> clip(static_cast<size_t>(batch) * clipElems);
        unsigned char const* srcPixels = frame.data();
        for (int32_t c = 0; c < 3; ++c)
        {
            float* frame0 = clip.data() + static_cast<size_t>(c) * pixelFrames * hw;
            for (size_t i = 0; i < hw; ++i)
            {
                frame0[i] = static_cast<float>(srcPixels[i * 3 + c]) / 127.5F - 1.0F;
            }
            for (int32_t t = 1; t < pixelFrames; ++t)
            {
                std::copy_n(frame0, hw, frame0 + static_cast<size_t>(t) * hw);
            }
        }
        for (int32_t b = 1; b < batch; ++b)
        {
            std::copy_n(clip.data(), clipElems, clip.data() + static_cast<size_t>(b) * clipElems);
        }
        rt::Tensor pixelTensor({batch, 3, pixelFrames, clipH, clipW}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT,
            "cosmos3::pixelValues");
        CUDA_CHECK(cudaMemcpyAsync(
            pixelTensor.rawPointer(), clip.data(), clip.size() * sizeof(float), cudaMemcpyHostToDevice, stream));

        // ------------------------------------------------------------------ //
        // (b) Prompt(s) -> token ids (co-located tokenizer).                  //
        // ------------------------------------------------------------------ //
        fs::path tokDir = fs::path(args.engineDir) / "text_tokenizer";
        if (!fs::exists(tokDir / "tokenizer.json"))
        {
            tokDir = args.engineDir; // fall back to engine dir root
        }
        tokenizer::Tokenizer tok;
        ELLM_CHECK(tok.loadFromHF(tokDir), "Failed to load tokenizer from " + tokDir.string());
        ELLM_CHECK(tok.loadChatTemplate(tokDir / "processed_chat_template.json"),
            "Failed to load chat template from " + tokDir.string());
        std::vector<std::vector<tokenizer::Rank>> ids;
        ids.reserve(args.prompts.size());
        for (std::string const& prompt : args.prompts)
        {
            // The reference policy pipeline wraps the instruction in the VLM chat template with a
            // generation prompt (tokenize_caption: user message + add_generation_prompt); the UND
            // tower must see the same token stream for the exported K/V to match.
            rt::LLMGenerationRequest::Request request;
            rt::Message message;
            message.role = "user";
            std::string const instruction = args.rawPrompt
                ? prompt
                : buildPolicyPrompt(prompt, args.viewPoint, actionChunkSize, fps, clipH, clipW);
            message.contents.push_back({"text", instruction});
            request.messages.push_back(std::move(message));
            rt::LLMGenerationRequest::FormattedRequest formatted;
            ELLM_CHECK(tok.applyChatTemplate(request, formatted, /*applyChatTemplate=*/true,
                           /*addGenerationPrompt=*/true, /*enableThinking=*/true),
                "Failed to apply the chat template to prompt: " + prompt);
            ids.push_back(tok.encode(formatted.formattedCompleteRequest, /*addBos=*/false, /*addEos=*/false));
            ELLM_CHECK(!ids.back().empty(), "Prompt tokenized to zero tokens: " + prompt);
            // The reference terminates the UND span with EOS + the start-of-generation token
            // (<|vision_start|>); the GEN cross-attention attends to these positions.
            ids.back().push_back(tok.getEosId());
            std::vector<tokenizer::Rank> const sog = tok.encode("<|vision_start|>", /*addBos=*/false, /*addEos=*/false);
            ELLM_CHECK(sog.size() == 1, "Tokenizer must map <|vision_start|> to a single id");
            ids.back().push_back(sog.front());
            // The UND prefill graph has no attention mask input, so a batched request requires
            // equal-length prompts (padding would leak into the gen cross-attention).
            ELLM_CHECK(ids.back().size() == ids.front().size(),
                "Batched prompts must tokenize to the same length (got " + std::to_string(ids.front().size()) + " vs "
                    + std::to_string(ids.back().size()) + ")");
        }
        int32_t const seqLen = static_cast<int32_t>(ids.front().size());
        LOG_INFO("Tokenized %d prompt(s) to %d tokens each.", batch, seqLen);

        // ------------------------------------------------------------------ //
        // (c) embed_tokens via the core embeddingLookup kernel (all Tensors). //
        // ------------------------------------------------------------------ //
        std::vector<rt::Tensor> embedTensors;
        fs::path const embedPath = fs::path(args.engineDir) / "embed_tokens.safetensors";
        ELLM_CHECK(rt::safetensors::loadSafetensors(embedPath, embedTensors, stream),
            "Failed to load embed_tokens: " + embedPath.string());
        ELLM_CHECK(!embedTensors.empty(), "embed_tokens artifact has no tensors: " + embedPath.string());
        rt::Tensor const& table = embedTensors.front();
        ELLM_CHECK(table.getShape().getNumDims() == 2 && table.getShape()[1] == hiddenSize,
            "embed_tokens must be [vocab, hidden] matching the und_prefill contract");

        std::vector<int32_t> idsHost;
        idsHost.reserve(static_cast<size_t>(batch) * seqLen);
        for (auto const& promptIds : ids)
        {
            idsHost.insert(idsHost.end(), promptIds.begin(), promptIds.end());
        }
        rt::Tensor idsTensor({batch, seqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "cosmos3::inputIds");
        CUDA_CHECK(cudaMemcpyAsync(
            idsTensor.rawPointer(), idsHost.data(), idsHost.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        rt::Tensor inputsEmbeds(
            {batch, seqLen, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "cosmos3::inputsEmbeds");
        kernel::embeddingLookup(idsTensor, table, std::nullopt, inputsEmbeds, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // ------------------------------------------------------------------ //
        // (d) VAE encode -> UND prefill -> GEN diffusion loop -> action chunk //
        // ------------------------------------------------------------------ //
        cosmos3::Cosmos3Runtime runtime(args.engineDir, stream);
        runtime.setNoiseSeed(args.seed);
        runtime.setNumInferenceSteps(args.steps);
        runtime.setUseCudaGraph(args.cudagraph);

        std::vector<float> action;
        for (int32_t w = 0; w < args.warmup; ++w)
        {
            action = runtime.generatePolicy(pixelTensor, inputsEmbeds, stream);
        }
        std::vector<double> walls;
        walls.reserve(args.iters);
        for (int32_t it = 0; it < args.iters; ++it)
        {
            auto const t0 = std::chrono::high_resolution_clock::now();
            action = runtime.generatePolicy(pixelTensor, inputsEmbeds, stream);
            cudaStreamSynchronize(stream);
            auto const t1 = std::chrono::high_resolution_clock::now();
            walls.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        if (!walls.empty())
        {
            std::vector<double> sorted = walls;
            std::sort(sorted.begin(), sorted.end());
            LOG_INFO(
                "Steady-state full policy round (batch %d): median %.2f ms over %d iters (min %.2f, max %.2f), "
                "warmup %d",
                batch, sorted[sorted.size() / 2], args.iters, sorted.front(), sorted.back(), args.warmup);
        }

        ELLM_CHECK(!action.empty(), "Policy generation returned no action.");

        // ------------------------------------------------------------------ //
        // (e) Emit the action chunk (B, chunk, dim).                          //
        // ------------------------------------------------------------------ //
        int32_t const rawActionDim = runtime.policyConfig().rawActionDim;
        int32_t const chunk = runtime.policyConfig().actionChunkSize;
        bool finite = true;
        float amin = 0.0f, amax = 0.0f;
        for (size_t i = 0; i < action.size(); ++i)
        {
            float const v = action[i];
            if (!std::isfinite(v))
            {
                finite = false;
            }
            else
            {
                if (i == 0 || v < amin)
                {
                    amin = v;
                }
                if (i == 0 || v > amax)
                {
                    amax = v;
                }
            }
        }
        LOG_INFO("Action chunk (%zu values, %dx%dx%d): finite=%s min=%.5g max=%.5g", action.size(), batch, chunk,
            rawActionDim, finite ? "true" : "false", amin, amax);

        if (!args.output.empty())
        {
            bool const asSafetensors
                = args.output.size() >= 12 && args.output.substr(args.output.size() - 12) == ".safetensors";
            if (asSafetensors)
            {
                writeActionSafetensors(args.output, action, batch, chunk, rawActionDim, stream);
            }
            else
            {
                writeActionJson(args.output, action, batch, chunk, rawActionDim, args.domain, args.steps, args.prompts,
                    finite, seqLen);
            }
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Cosmos3 policy inference failed: %s", e.what());
        cudaStreamDestroy(stream);
        return EXIT_FAILURE;
    }

    cudaStreamDestroy(stream);
    return EXIT_SUCCESS;
}
