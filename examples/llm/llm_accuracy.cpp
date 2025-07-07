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

#include "common/common.h"
#include "common/trtUtils.h"
#include "decoder/decoder.h"
#include "engine/llm_engine.h"
#include "llm_param.h"
#include "tokenizer/tokenizer.h"
#include <NvInferRuntime.h>
#include <algorithm>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
namespace fs = std::filesystem;

struct LLMAccuracyArgs
{
    std::string datasetPath;
    BaseParams baseParams;
    EagleParams eagleParams;
    LoraWeights loraWeights;
};

struct TestData
{
    std::string question;
    std::vector<std::string> options;
    std::string ans;

    std::string format(bool includeAnswer = true) const
    {
        std::string prompt = question;
        for (int i = 0; i < 4; i++)
        {
            prompt += fmtstr("\n%c. %s", 'A' + i, options[i].c_str());
        }
        prompt += "\nAnswer:";
        if (includeAnswer)
        {
            prompt += fmtstr(" %s\n\n", ans.c_str());
        }
        return prompt;
    }
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [-h] [-i or --inputString=<input>] [-e or --enginePath=<path to TensorRT engine>] [-s or "
                 "--maxLength=<int>] [-t or --tokenizerPath=<path to HF tokenizer>] "
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --datasetPath    Provide the dataset path for evaluation." << std::endl;
    CommonUsage::printBaseOptions();
    CommonUsage::printEagleOptions();
    CommonUsage::printLoraOptions();
};

bool parseLLMAccuracyArgs(LLMAccuracyArgs& args, int argc, char* argv[])
{
    static struct option accuracyOptions[]
        = {{"datasetPath", required_argument, 0, 'D'}, {"loraWeights", required_argument, 0, 'l'}, {0, 0, 0, 0}};

    struct option long_options[64];
    int idx = 0;
    for (int i = 0; CommonOptions::baseOptions[i].name != 0; ++i)
        long_options[idx++] = CommonOptions::baseOptions[i];
    for (int i = 0; CommonOptions::eagleOptions[i].name != 0; ++i)
        long_options[idx++] = CommonOptions::eagleOptions[i];
    for (int i = 0; accuracyOptions[i].name != 0; ++i)
        long_options[idx++] = accuracyOptions[i];
    long_options[idx] = {0, 0, 0, 0};

    int opt;
    while ((opt = getopt_long(argc, argv, "D:l:e:t:hdga:m:k:p:E:", long_options, nullptr)) != -1)
    {
        if (CommonOptions::parseBaseOptions(args.baseParams, opt, optarg, true))
        {
            continue;
        }

        if (CommonOptions::parseEagleOptions(args.eagleParams, opt, optarg))
        {
            continue;
        }

        switch (opt)
        {
        case 'D':
            if (optarg)
            {
                args.datasetPath = optarg;
            }
            break;
        case 'l':
            if (optarg)
            {
                if (LoraWeights::validateFormat(optarg))
                {
                    auto loraPair = LoraWeights::parse(optarg);
                    args.loraWeights.add(loraPair.first, loraPair.second);
                }
                else
                {
                    return false;
                }
            }
            break;
        default: return false;
        }
    }
    return true;
}

std::vector<TestData> parseCSVFile(fs::path const& csvPath, int maxRecordNum = -1, bool debug = false)
{
    int printFirstThreeLine = debug ? 3 : 0;
    std::vector<TestData> res;
    std::ifstream file(csvPath);
    if (debug)
    {
        LOG_DEBUG(csvPath.c_str());
    }
    while (!file.eof() && (maxRecordNum == -1 || static_cast<int>(res.size()) < maxRecordNum))
    {
        TestData data;
        data.options.resize(4);
        auto getItem = [&file, &csvPath](std::string& item, char const delim = '\n') {
            // Handling this according to http://super-csv.github.io/super-csv/csv_specification.html
            auto c = file.get();
            if (file.eof())
            {
                return false;
            }
            std::string tmp;
            if (c == '\"')
            {
                while (1)
                {
                    if (!std::getline(file, tmp, '\"'))
                    {
                        throw std::runtime_error(
                            "File " + csvPath.string() + " EOF unexpectedly. Current buffer: " + item);
                    }
                    item += tmp;
                    c = file.get();
                    if (file.eof())
                    {
                        throw std::runtime_error(
                            "File " + csvPath.string() + " EOF unexpectedly. Current buffer: " + item);
                    }
                    if (c == '\"')
                    {
                        item += c;
                    }
                    else if (c == delim)
                    {
                        break;
                    }
                    else
                    {
                        throw std::runtime_error("Error parsing line: " + item);
                    }
                }
            }
            else
            {
                item = c;
                std::getline(file, tmp, delim);
                item += tmp;
            }
            return true;
        };
        // Assuming each record only has 6 items: question, 4 options, answer.
        if (!getItem(data.question, ','))
        {
            break;
        }
        getItem(data.options[0], ',');
        getItem(data.options[1], ',');
        getItem(data.options[2], ',');
        getItem(data.options[3], ',');
        getItem(data.ans);
        auto startIter = data.ans.begin(), endIter = data.ans.end();
        do
        {
            --endIter;
        } while (endIter != startIter && std::isspace(*endIter));
        data.ans = std::string(startIter, endIter + 1);
        if (data.ans.size() != 1 || printFirstThreeLine)
        {
            LOG_DEBUG("%s\n%s\n%s\n%s\n%s\n%s\n====================", data.question.c_str(), data.options[0].c_str(),
                data.options[1].c_str(), data.options[2].c_str(), data.options[3].c_str(), data.ans.c_str());

            if (printFirstThreeLine && data.ans.size() == 1)
            {
                printFirstThreeLine--;
            }
            else
            {
                throw std::runtime_error("Error: " + std::to_string(data.ans.size()) + ", content: " + data.ans);
            }
        }
        res.emplace_back(data);
    }
    file.close();
    return res;
}

void mmluAccuracy(LLMAccuracyArgs const& args, Tokenizer* tokenizer, GenerationConfig generationConfig)
{
    std::unordered_map<std::string, std::vector<TestData>> testSubject2Data, devSubject2Data;
    std::vector<std::string> subjects;
    try
    {
        assert(fs::exists(fs::path(args.datasetPath) / "test"));
        assert(fs::exists(fs::path(args.datasetPath) / "dev"));
        for (auto& filename : fs::directory_iterator(fs::path(args.datasetPath) / "test"))
        {
            auto filenameStr = filename.path().filename().string();
            auto idx = filenameStr.find("_test.csv");
            if (filenameStr.find("_test.csv") != filenameStr.size())
            {
                subjects.emplace_back(filenameStr.substr(0, idx));
            }
        }

        std::sort(subjects.begin(), subjects.end());
        for (auto& subject : subjects)
        {
            auto testFile = fs::path(args.datasetPath) / "test" / (subject + "_test.csv");
            auto devFile = fs::path(args.datasetPath) / "dev" / (subject + "_dev.csv");
            assert(fs::exists(testFile));
            assert(fs::exists(devFile));
            [[maybe_unused]] int printFirstThreeLine = args.baseParams.debug ? 3 : 0;

            auto testData = parseCSVFile(testFile, -1, args.baseParams.debug);
            auto devData = parseCSVFile(devFile, 5, args.baseParams.debug);

            [[maybe_unused]] auto formatExample = []() { std::string prompt; };
            [[maybe_unused]] auto genPrompt = [&subject]() {
                std::string prompt
                    = "The following are multiple choice questions (with answers) about " + subject + ".\n\n";
            };

            testSubject2Data[subject] = testData;
            devSubject2Data[subject] = devData;
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to load dataset: %s", e.what());
        return;
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    EngineConfig engineConfig;
    if (args.eagleParams.eagleEnginePath.empty())
    {
        LOG_INFO("Running in standard LLM mode.");
        engineConfig = EngineConfig(args.baseParams.enginePath);
    }
    else
    {
        LOG_INFO("Running in Eagle mode.");
        engineConfig = EngineConfig(args.baseParams.enginePath, args.eagleParams.eagleEnginePath,
            args.eagleParams.maxPathLen, args.eagleParams.topK, args.eagleParams.isEagle3,
            args.eagleParams.maxDecodingTokens, !args.baseParams.noCudaGraph);
    }
    auto llmEngine = std::make_unique<LLMEngineHalf>(engineConfig, stream);
    bool const eagleMode = llmEngine->isEagleModel();
    // Load and switch to LoRA weights if provided
    if (args.loraWeights.hasWeights() && !eagleMode)
    {
        auto& decoderPtr = llmEngine->getDecoder();
        auto loraPair = args.loraWeights.getFirst();
        if (!decoderPtr->addLora(loraPair.first, loraPair.second))
        {
            LOG_ERROR("Failed to load LoRA weights: %s from %s", loraPair.first.c_str(), loraPair.second.c_str());
            return;
        }
        if (!decoderPtr->switchLora(loraPair.first))
        {
            LOG_ERROR("Failed to switch to LoRA: %s", loraPair.first.c_str());
            return;
        }
    }

    std::vector<int64_t> choices
        = {tokenizer->encode("A")[0], tokenizer->encode("B")[0], tokenizer->encode("C")[0], tokenizer->encode("D")[0]},
        lastTokenIds(1);
    std::vector<int32_t> contextLengths(1);
    int64_t total = 0, correct = 0;

    auto genDevPrompt = [](uint16_t n, std::vector<TestData> const& datas, std::string const& subjectFmt) {
        std::string devPrompt
            = "The following are multiple choice questions (with answers) about " + subjectFmt + ".\n\n";
        for (size_t i = 0; i < n && i < datas.size(); i++)
        {
            devPrompt += datas[i].format();
        }
        return devPrompt;
    };

    for (auto const& subject : subjects)
    {
        int64_t subjectTotal, subjectCorrect = 0;
        std::string subjectFmt = subject;
        int printFirstThree = args.baseParams.debug ? 3 : 0;
        std::replace(subjectFmt.begin(), subjectFmt.end(), '_', ' ');
        auto const& testData = testSubject2Data[subject];
        auto const& devData = devSubject2Data[subject];
        subjectTotal = testData.size();

        for (auto const& data : testData)
        {
            std::string prompt;
            std::vector<int64_t> inputIds;

            uint16_t devPromptNum = 5;
            do
            {
                prompt = genDevPrompt(devPromptNum, devData, subjectFmt) + data.format(false);
                inputIds = tokenizer->encode(prompt, true);
                devPromptNum--;
            } while (inputIds.size() > 2048 && devPromptNum >= 0);

            if (args.baseParams.debug && printFirstThree)
            {
                LOG_DEBUG("Prompt: %s", prompt.c_str());
            }

            std::vector<std::vector<int64_t>> outputIds(1);
            generationConfig.maxLength = inputIds.size() + 1;

            if (inputIds.size() > 2048)
            {
                LOG_WARNING("Dropping test case whose length is greater than 2048");
                continue;
            }
            contextLengths[0] = inputIds.size();
            llmEngine->generate(
                inputIds, contextLengths, outputIds, generationConfig, nullptr, nullptr, nullptr, tokenizer);

            std::vector<half> hostLogits;
            llmEngine->getLastHostLogits(hostLogits);

            int bestIdx = 0;
            half val = hostLogits[choices[0]];
            for (int i = 1; i < 4; i++)
            {
                if (val < hostLogits[choices[i]])
                {
                    bestIdx = i;
                    val = hostLogits[choices[i]];
                }
            }
            if (args.baseParams.debug && printFirstThree)
            {
                LOG_DEBUG("Model's answer: %c, expected answer: %c", bestIdx + 'A', data.ans[0]);
                printFirstThree--;
            }
            if (bestIdx + 'A' == data.ans[0])
            {
                subjectCorrect++;
            }
        }

        LOG_INFO("Subject %s evaluation done. Average accuracy: %.3f", subjectFmt.c_str(),
            float(subjectCorrect) / subjectTotal);
        correct += subjectCorrect;
        total += subjectTotal;
    }

    LOG_INFO("MMLU: %.3f", float(correct) / total);
}

int main(int argc, char* argv[])
{
    LLMAccuracyArgs args;
    if ((argc < 2) || (!parseLLMAccuracyArgs(args, argc, argv)))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.baseParams.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    if (args.baseParams.debug)
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    }
    else
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);
    }

    auto pluginHandles = loadPlugins();

    GenerationConfig generationConfig{0, 0, 1, 0};

    auto tokenizer = std::make_unique<Tokenizer>();
    tokenizer->loadFromHF(args.baseParams.tokenizerPath);

    mmluAccuracy(args, tokenizer.get(), generationConfig);
    return EXIT_SUCCESS;
};