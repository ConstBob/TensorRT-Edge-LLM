
#include "common/common.h"
#include "common/trtUtils.h"
#include "decoder/decoder.h"
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
    bool help{false};
    std::string enginePath;
    std::string tokenizerPath;
    std::string datasetPath;
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
    std::cerr << "  --enginePath     Provide the input TensorRT engine file path. Required. " << std::endl;
    std::cerr << "  --tokenizerPath  Provide the path to HF tokenizer. Required. " << std::endl;
    std::cerr << "  --datasetPath    Provide the dataset path for evaluation." << std::endl;
    std::cerr << "  --debug          Use debug mode, which outputs tensors." << std::endl;
};

bool parseLLMAccuracyArgs(LLMAccuracyArgs& args, int argc, char* argv[])
{
    static struct option long_options[] = {{"help", no_argument, 0, 'h'}, {"enginePath", required_argument, 0, 'e'},
        {"tokenizerPath", required_argument, 0, 't'}, {"datasetPath", required_argument, 0, 'D'},
        {"debug", no_argument, 0, 'd'}, {0, 0, 0, 0}};

    int opt;

    // Loop to process each option
    while ((opt = getopt_long(argc, argv, "he:t:D:d", long_options, nullptr)) != -1)
    {
        switch (opt)
        {
        case 'h': args.help = true; return true;
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
        case 'D':
            if (optarg)
            {
                args.datasetPath = optarg;
            }
            break;
        case 'd': args.debug = true; break;
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

void mmluAccuracy(fs::path const& enginePath, fs::path const& datasetPath, Tokenizer* tokenizer,
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
    LLMAccuracyArgs args;
    if ((argc < 2) || (!parseLLMAccuracyArgs(args, argc, argv)))
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

    auto handle = loadPlugin();

    // The generationConfig will change
    GenerationConfig generationConfig{0, 0, 1, 0};

    auto tokenizer = new Tokenizer();
    tokenizer->loadFromHF(args.tokenizerPath);
    mmluAccuracy(args.enginePath, args.datasetPath, tokenizer, generationConfig, args.debug);
    return EXIT_SUCCESS;
};
