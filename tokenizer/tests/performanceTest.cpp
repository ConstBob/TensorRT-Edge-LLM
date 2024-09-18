/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <chrono>
#include <fstream>
#include <iostream>

#include <tokenizer.h>
#include <tokenizerUtils.h>

std::string strip(std::string& str)
{
    size_t first = str.find_first_not_of(' ');
    size_t last = str.find_last_not_of(' ');
    if (first != std::string::npos && last >= first)
    {
        return str.substr(first, (last - first + 1));
    }
    return "";
}

void testPerformance(std::string const& modelPath, std::string const& dataPath)
{
    std::ifstream data(dataPath);
    std::string line;
    std::vector<std::string> documents;
    float nBytes = 0.0f;

    while (std::getline(data, line))
    {
        line = strip(line);
        if (!line.empty())
        {
            documents.emplace_back(line);
            nBytes += line.size();
        }
    }

    gLogger.info("Loaded data from " + dataPath);
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << "Num_bytes: " << (nBytes / 1024 / 1024) << " MB";
    gLogger.info(ss.str());

    Tokenizer* enc = new LlamaV3Tokenizer;
    if (modelPath.compare(modelPath.size() - 15, 15, "tokenizer.model") == 0)
    {
        enc->loadFromTiktoken(modelPath);
    }
    else
    {
        enc->loadFromHF(modelPath);
    }

    enc->decode(enc->encode("warmup"));

    gLogger.info("Start performance test...");

    std::chrono::duration<float> encodeTimer;
    std::chrono::duration<float> decodeTimer;

    for (auto const& text : documents)
    {
        auto start = std::chrono::steady_clock::now();
        auto token = enc->encode(text);
        auto end = std::chrono::steady_clock::now();
        encodeTimer += end - start;

        start = std::chrono::steady_clock::now();
        auto output = enc->decode(token);
        end = std::chrono::steady_clock::now();
        decodeTimer += end - start;
    }

    auto encodeDuration = std::chrono::duration_cast<std::chrono::microseconds>(encodeTimer).count();
    auto decodeDuration = std::chrono::duration_cast<std::chrono::microseconds>(decodeTimer).count();

    ss.str("");
    ss << "Encode latency: " << (encodeDuration / nBytes) << " μs / Byte"
       << ", throughput: " << (nBytes / 1024 / 1024 / encodeDuration) * 1e6 << " MB / s";
    gLogger.info(ss.str());

    ss.str("");
    ss << "Decode latency: " << (decodeDuration / nBytes) << " μs / Byte"
       << ", throughput: " << (nBytes / 1024 / 1024 / decodeDuration) * 1e6 << " MB / s";
    gLogger.info(ss.str());
}

int main(int argc, char* argv[])
{
    if (argc == 3)
    {
        testPerformance(argv[1], argv[2]);
    }
    else
    {
        gLogger.error(
            "Usage:\n"
            "argv[1]: tokenizer path\n"
            "argv[2]: wiki.tokens path");
    }

    return 0;
}
