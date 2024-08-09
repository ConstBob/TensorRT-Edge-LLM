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

#include <iostream>
#include <fstream>

#include <tokenizer.h>
#include <tokenizerUtils.h>

void testEncodeDecode(Tokenizer* tokenizer, const std::string& input, const std::vector<Rank> expected = {}, 
    bool addSpecialTokens = false)
{
    auto token = tokenizer->encode(input, addSpecialTokens);
    std::string output = tokenizer->decode(token);

    gLogger.info("Input: " + input);

    std::stringstream ss;
    for (const auto& t : token)
    {
        ss << t << ", ";
    }
    gLogger.info("Token: " + ss.str());

    if (!expected.empty())
    {
        ss.str("");
        for (const auto& t : expected)
        {
            ss << t << ", ";
        }
        gLogger.info("Expected: " + ss.str());

        assert(token == expected);
    }

    if (addSpecialTokens)
    {
        assert(output == "<|begin_of_text|>" + input + "<|end_of_text|>");
    }
    else
    {
        assert(output == input);
    }

    gLogger.info("Passed");
}

std::vector<Rank> getGolden(const std::string& modelPath, const std::string& input, bool addSpecialTokens)
{
    if (modelPath.compare(modelPath.size() - 15, 15, "tokenizer.model") != 0)
    {
        return {};
    }

    try
    {
        std::string cmd = "python3 ../tests/llama3Test.py"
            " --model_path=\"" + modelPath + "\""
            " --input=\"" + input + "\""
            " --add_special=" + std::to_string(addSpecialTokens);

        std::array<char, 128> buffer;
        std::string result;
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);

        assert(pipe);

        while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
            result += buffer.data();
        }
        // remove []
        result = result.substr(1, result.size() - 2);

        std::vector<Rank> golden;
        int start = 0;

        while (true)
        {
            int idx = result.find(", ", start);
            if (idx == std::string::npos) {
                break;
            }

            int length = idx - start;
            golden.emplace_back(std::stoi(result.substr(start, idx - start)));
            start += (length + 2);
        };

        golden.emplace_back(std::stoi(result.substr(start)));

        return golden;       
    }
    catch(const std::exception& e)
    {
        return {};
    }
}

int main(int argc, char* argv[])
{
    if (argc >= 2)
    {
        // load tokenizer
        Tokenizer* enc = new LlamaV3Tokenizer;

        auto modelPath = std::string(argv[1]);
        if (modelPath.compare(modelPath.size() - 15, 15, "tokenizer.model") == 0)
        {
            enc->loadFromTiktoken(modelPath);
        }
        else
        {
            enc->loadFromHF(modelPath);
        }

        // get input
        std::string input = (argc >= 3) ? argv[2] : "<|begin_of_text|>hello¢ nvidia，你好！👍<|end_of_text|>";
        bool addSpecial = (argc >= 4) ? std::stoi(argv[3]) : false;

        // get golden from python script
        auto expected = getGolden(modelPath, input, addSpecial);

        testEncodeDecode(enc, input, expected, addSpecial);
    }
    else
    {
        gLogger.error(
            "Usage:\n"
            "argv[1]: tokenizer path\n"
            "argv[2]: (optional) input_text\n"
            "argv[3]: (optional) add_special [0,1]"
        );
    }
    
    return 0;
}
