
#include "common/common.h"
#include "decoder.h"
#include "tokenizer.h"
#include <NvInferRuntime.h>
#include <algorithm>
#include <bits/getopt_core.h>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace nvinfer1;
namespace fs = std::filesystem;

enum class MODE
{
    kInference,
    kBenchmark,
    kEvaluate
};

struct RuntimeArgs
{
    bool help{false};
    std::vector<std::string> inputString;
    std::string enginePath;
    std::string tokenizerPath;
    std::string datasetPath;
    int maxLength{40};
    int inputLength;
    MODE mode{MODE::kInference};
    int64_t numRuns{10};
    int64_t warmUp{2};
    bool debug{false};
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
                 "--maxLength=<int>] [-t or --tokenizerPath=<path to HF tokenizer>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -h               Display this help message" << std::endl;
    std::cerr << "  --inputString    Provide the input string to the runtime. Required. " << std::endl;
    std::cerr << "  --inputLength    Provide the input string to the runtime. Required. " << std::endl;
    std::cerr << "  --enginePath     Provide the input TensorRT engine file path. Required. " << std::endl;
    std::cerr << "  --tokenizerPath  Provide the path to HF tokenizer. Required. " << std::endl;
    std::cerr << "  --maxLength      Provide the maximum output length for the generation session (including the "
                 "input). Default = 40"
              << std::endl;
    std::cerr << "  --mode           Provide the mode. " << std::endl;
    std::cerr << "  --warmUp         [Benchmark] Provide warm up iterations before benchmark starts." << std::endl;
    std::cerr << "  --numRuns        [Benchmark] Minimal number of iterations to run during benchmarking." << std::endl;
    std::cerr << "  --datasetPath    [Evaluation] Provide the dataset path for evaluation." << std::endl;
    std::cerr << "  --debug          Use debug mode, which outputs tensors." << std::endl;
};

bool parseRuntimeArgs(RuntimeArgs& args, int argc, char* argv[])
{
    static struct option long_options[] = {{"help", no_argument, 0, 'h'}, {"inputString", required_argument, 0, 'i'},
        {"enginePath", required_argument, 0, 'e'}, {"tokenizerPath", required_argument, 0, 't'},
        {"maxLength", required_argument, 0, 's'}, {"inputLength", required_argument, 0, 'c'},
        {"mode", required_argument, 0, 'm'}, {"warmUp", required_argument, 0, 'w'},
        {"numRuns", required_argument, 0, 'r'}, {"datasetPath", required_argument, 0, 'D'},
        {"debug", no_argument, 0, 'd'}, {0, 0, 0, 0}};

    int opt;

    // Loop to process each option
    while ((opt = getopt_long(argc, argv, "h:iest", long_options, nullptr)) != -1)
    {
        switch (opt)
        {
        case 'h': args.help = true; return true;
        case 'i':
            if (optarg)
            {
                args.inputString.emplace_back(optarg);
            }
            else
            {
                std::cerr << "ERROR: --inputString requires option argument" << std::endl;
                return false;
            }
            break;
        case 'e':
            if (optarg)
            {
                args.enginePath = optarg;
            }
            else
            {
                std::cerr << "ERROR: --enginePath requires option argument" << std::endl;
                return false;
            }
            break;
        case 't':
            if (optarg)
            {
                args.tokenizerPath = optarg;
            }
            else
            {
                std::cerr << "ERROR: --tokenizerPath requires option argument" << std::endl;
                return false;
            }
            break;
        case 's':
            if (optarg)
            {
                args.maxLength = std::stoi(optarg);
            }
            break;
        case 'c':
            if (optarg)
            {
                args.inputLength = std::stoi(optarg);
            }
            break;
        case 'm':
            if (optarg)
            {
                if (strcmp(optarg, "benchmark") == 0)
                {
                    args.mode = MODE::kBenchmark;
                }
                else if (strcmp(optarg, "evaluate") == 0)
                {
                    args.mode = MODE::kEvaluate;
                }
                else if (strcmp(optarg, "inference") == 0)
                {
                    args.mode = MODE::kInference;
                }
                else
                {
                    std::cerr << "ERROR: invalid argument for --mode. Get " << optarg
                              << ", valid options: benchmark, evaluate, inference." << std::endl;
                    return false;
                }
            }
            break;
        case 'D':
            if (optarg)
            {
                args.datasetPath = optarg;
            }
            break;
        case 'w':
            if (optarg)
            {
                args.warmUp = std::stoi(optarg);
            }
            break;
        case 'r':
            if (optarg)
            {
                args.numRuns = std::stoi(optarg);
            }
            break;
        case 'd': args.debug = true; break;
        default: return false;
        }
    }
    return true;
}

void prepareBatchInputIds(std::vector<std::string> const& inputString, Tokenizer* tokenizer,
    std::vector<int64_t>& inputIds, std::vector<int32_t>& contextLengths, int maxContextLength)
{
    std::vector<std::vector<int64_t>> batchInputIds;
    int batchSize = inputString.size();
    for (int i = 0; i < batchSize; ++i)
    {
        auto ids = tokenizer->encode(inputString[i], true);
        batchInputIds.emplace_back(ids);
        contextLengths[i] = ids.size();
    }

    // right padding
    int64_t padId = tokenizer->getPadId();
    inputIds.resize(batchSize * maxContextLength, padId);
    for (int i = 0; i < batchSize; ++i)
    {
        std::copy(batchInputIds[i].begin(), batchInputIds[i].end(), inputIds.begin() + i * maxContextLength);
    }
}

std::vector<std::string> decode(fs::path const& enginePath, std::vector<std::string> const& inputString,
    Tokenizer* tokenizer, GenerationConfig const& generationConfig, bool debug = false)
{
    int batchSize = inputString.size();
    auto decoder = new Decoder<half>();
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    decoder->setup(enginePath, stream);

    std::vector<int64_t> inputIds; // flattened
    std::vector<int32_t> contextLengths(batchSize);
    prepareBatchInputIds(inputString, tokenizer, inputIds, contextLengths, decoder->getMaxContextLength());

    if (debug)
    {
        std::ostringstream oss;
        for (int i = 0; i < batchSize; ++i)
        {
            oss << "contextLengths[" << i << "] is " << contextLengths[i] << ". inputIds[" << i << "] is: [";
            for (int j = 0; j < decoder->getMaxContextLength(); ++j)
            {
                oss << inputIds[i * decoder->getMaxContextLength() + j] << ",";
            }
            oss << "]\n";
        }
        LOG_DEBUG(oss.str().c_str());
    }

    std::vector<std::vector<int64_t>> outputIds(batchSize);
    for (int i = 0; i < batchSize; ++i)
    {
        outputIds[i].reserve(generationConfig.maxLength);
    }
    decoder->generate(inputIds, contextLengths, outputIds, generationConfig, tokenizer->getEosId());

    std::vector<std::string> output(batchSize);
    for (int i = 0; i < batchSize; ++i)
    {
        if (debug)
        {
            LOG_DEBUG("Output%d length is %d", i, outputIds[i].size());
        }
        output[i] = tokenizer->decode(outputIds[i]);
        LOG_INFO("Input%d is: %s", i, inputString[i].c_str());
        LOG_INFO("Output%d is: %s", i, output[i].c_str());
    }
    return output;
}

void benchmark(fs::path const& enginePath, int const inputLength, int64_t warmUp, int64_t numRuns,
    GenerationConfig const& generationConfig)
{
    auto profiler = std::make_shared<BenchmarkProfiler>();
    profiler->startTiming();
    profiler->recordDeviceMemStart();
    profiler->recordHostMemStart();
    auto decoder = new Decoder<half>();
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    profiler->recordHostStart("decoder setup");
    decoder->setup(enginePath, stream);
    profiler->recordHostEnd("decoder setup");
    profiler->stopTiming();

    int64_t batchSize = decoder->getModelBatchSize();
    int32_t maxContextLength{128};
    std::vector<int64_t> inputIds(batchSize * decoder->getMaxContextLength(), -1);
    std::vector<int32_t> contextLengths(batchSize, inputLength);
    std::vector<int64_t> lastTokenIds(batchSize, inputLength - 1);
    std::vector<std::vector<int64_t>> outputIds(batchSize);

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, 1000);
    for (int i = 0; i < batchSize; ++i)
    {
        auto beginIter = inputIds.begin() + i * decoder->getMaxContextLength();
        std::generate(beginIter, beginIter + inputLength, [&rng, &dist]() { return dist(rng); });
        outputIds[i].reserve(generationConfig.maxLength);
    }

    for (int64_t i = 0; i < warmUp; i++)
    {
        // Warmup for profiler
        profiler->recordHostStart("seq latency");
        decoder->generate(inputIds, contextLengths, outputIds, generationConfig, -1, profiler);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        profiler->recordHostEnd("seq latency");
        for (int i = 0; i < batchSize; ++i)
        {
            outputIds[i].resize(0);
        }
    }
    cudaDeviceSynchronize();

    profiler->startTiming();
    cudaProfilerStart();
    for (int64_t i = 0; i < numRuns; i++)
    {
        profiler->recordHostStart("seq latency");
        profiler->recordHostStart("first token latency");
        decoder->generate(inputIds, contextLengths, outputIds, generationConfig, -1, profiler);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        profiler->recordHostEnd("seq latency");
        for (int i = 0; i < batchSize; ++i)
        {
            outputIds[i].resize(0);
        }
    }
    cudaProfilerStop();
    profiler->stopTiming();
    auto peakDeviceMem = profiler->recordDeviceMemEnd();
    auto peakHostMem = profiler->recordHostMemEnd();

    auto maxNewTokens = batchSize * (generationConfig.maxLength - inputLength);
    auto [averageSeqLatency, duration, seqLatencies] = profiler->getHostElapsedTimeMs("seq latency");
    auto [averageFirstTokenLatency, totalFirstTokenLatency, firstTokenLatencies]
        = profiler->getHostElapsedTimeMs("first token latency");
    auto [averageGenerationTime, totalGenerationTime, generationTimes] = profiler->getDeviceElapsedTimeMs("generation");
    auto decoderSetupLatency = std::get<1>(profiler->getHostElapsedTimeMs("decoder setup"));
    auto tokensPerSec = maxNewTokens / (averageSeqLatency / 1000);
    auto generationTokensPerSec = maxNewTokens / (averageGenerationTime / 1000);

    LOG_INFO("========================================================");
    LOG_INFO("Benchmarking done. Decoder setup: %.2fs, Iteration: %d, GPU Time: %.2fs.", decoderSetupLatency / 1000,
        seqLatencies.size(), duration / 1000);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "Sequence Latencies(ms): [";
    constexpr int maxPrintedLatencies{20};
    for (int i = 0; i < seqLatencies.size(); ++i)
    {
        oss << seqLatencies[i];
        if (i == seqLatencies.size() - 1)
        {
            oss << "]";
        }
        else if (seqLatencies.size() > maxPrintedLatencies && i == (maxPrintedLatencies / 2 - 1))
        {
            oss << " ... ";
            i = seqLatencies.size() - maxPrintedLatencies / 2;
        }
        else
        {
            oss << ", ";
        }
    }

    LOG_INFO(oss.str().c_str());

    oss.str("");
    oss.clear();
    oss << "First Token Latencies(ms): [";
    for (int i = 0; i < firstTokenLatencies.size(); ++i)
    {
        oss << firstTokenLatencies[i];
        if (i == firstTokenLatencies.size() - 1)
        {
            oss << "]";
        }
        else if (firstTokenLatencies.size() > maxPrintedLatencies && i == (maxPrintedLatencies / 2 - 1))
        {
            oss << " ... ";
            i = firstTokenLatencies.size() - maxPrintedLatencies / 2;
        }
        else
        {
            oss << ", ";
        }
    }
    LOG_INFO(oss.str().c_str());
    LOG_INFO("batch_size: %d", batchSize);
    LOG_INFO("input_length per batch: %d", inputLength);
    LOG_INFO("output_length per batch: %d", maxNewTokens);
    LOG_INFO("seq_latency(ms): %.2f", averageSeqLatency);
    LOG_INFO("first_token_latency(ms): %.2f", averageFirstTokenLatency);
    LOG_INFO("tokens_per_sec: %.2f", tokensPerSec);
    LOG_INFO("generation_time(ms): %.2f", averageGenerationTime);
    LOG_INFO("generation_tokens_per_sec: %.2f", generationTokensPerSec);
    LOG_INFO("cpu_peak_mem(GiB): %.2f", peakHostMem / 1048576.0);
    LOG_INFO("gpu_peak_mem(GiB): %.2f", peakDeviceMem / 1073741824.0);
    LOG_INFO("execution_context_mem(GiB): %.2f", decoder->getDeviceMemorySize() / 1073741824.0);
    LOG_INFO("========================================================");
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
    while (!file.eof() && (maxRecordNum == -1 || res.size() < maxRecordNum))
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

void evaluate(fs::path const& enginePath, fs::path const& datasetPath, Tokenizer* tokenizer,
    GenerationConfig generationConfig, bool debug)
{
    std::unordered_map<std::string, std::vector<TestData>> testSubject2Data, devSubject2Data;
    std::vector<std::string> subjects;
    try
    {
        assert(fs::exists(datasetPath / "test"));
        assert(fs::exists(datasetPath / "dev"));

        for (auto& filename : fs::directory_iterator(datasetPath / "test"))
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
            auto testFile = datasetPath / "test" / (subject + "_test.csv");
            auto devFile = datasetPath / "dev" / (subject + "_dev.csv");
            assert(fs::exists(testFile));
            assert(fs::exists(devFile));
            int printFirstThreeLine = debug ? 3 : 0;

            auto testData = parseCSVFile(testFile, -1, debug);
            auto devData = parseCSVFile(devFile, 5, debug);

            auto formatExample = []() { std::string prompt; };
            auto genPrompt = [&subject]() {
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

    auto decoder = new Decoder<half>();
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    decoder->setup(enginePath, stream);
    std::vector<int64_t> choices
        = {tokenizer->encode("A")[0], tokenizer->encode("B")[0], tokenizer->encode("C")[0], tokenizer->encode("D")[0]},
        lastTokenIds(1);
    std::vector<int32_t> contextLengths(1);
    int64_t total = 0, correct = 0;

    auto genDevPrompt = [](int n, std::vector<TestData> const& datas, std::string const& subjectFmt) {
        std::string devPrompt
            = "The following are multiple choice questions (with answers) about " + subjectFmt + ".\n\n";
        for (int i = 0; i < n && i < datas.size(); i++)
        {
            devPrompt += datas[i].format();
        }
        return devPrompt;
    };

    for (auto const& subject : subjects)
    {
        int64_t subjectTotal, subjectCorrect = 0;
        std::string subjectFmt = subject;
        int printFirstThree = debug ? 3 : 0;
        std::replace(subjectFmt.begin(), subjectFmt.end(), '_', ' ');
        auto const& testData = testSubject2Data[subject];
        auto const& devData = devSubject2Data[subject];
        subjectTotal = testData.size();

        for (auto const& data : testData)
        {
            std::string prompt;
            std::vector<int64_t> inputIds;

            int devPromptNum = 5;
            do
            {
                prompt = genDevPrompt(devPromptNum, devData, subjectFmt) + data.format(false);
                inputIds = tokenizer->encode(prompt, true);
                devPromptNum--;
            } while (inputIds.size() > 2048 && devPromptNum >= 0);

            if (debug && printFirstThree)
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
            decoder->generate(inputIds, contextLengths, outputIds, generationConfig);
            auto const& hostLogits = decoder->getLastHostLogits();
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
            if (debug && printFirstThree)
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
    RuntimeArgs args;
    if ((argc < 2) || (!parseRuntimeArgs(args, argc, argv)))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    if (args.debug)
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    }
    else
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);
    }

    void* handle = dlopen("build/plugins/libLLamaPlugin.so", RTLD_LAZY);
    if (!handle)
    {
        LOG_ERROR("Cannot open library: %s", dlerror());
        return EXIT_FAILURE;
    }

    GenerationConfig generationConfig{args.maxLength, 0, 1, 0};

    switch (args.mode)
    {
    case MODE::kBenchmark:
    {
        if (args.inputLength < 1)
        {
            LOG_ERROR("Please specify the input length");
            return EXIT_FAILURE;
        }
        benchmark(args.enginePath, args.inputLength, args.warmUp, args.numRuns, generationConfig);
        break;
    }
    case MODE::kEvaluate:
    {
        auto tokenizer = new Tokenizer();
        tokenizer->loadFromHF(args.tokenizerPath);
        evaluate(args.enginePath, args.datasetPath, tokenizer, generationConfig, args.debug);
        break;
    }
    case MODE::kInference:
    {
        auto tokenizer = new Tokenizer();
        tokenizer->loadFromHF(args.tokenizerPath);
        auto output = decode(args.enginePath, args.inputString, tokenizer, generationConfig, args.debug);
        break;
    }
    }

    dlclose(handle);

    return EXIT_SUCCESS;
};
