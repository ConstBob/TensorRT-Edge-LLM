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

#include <limits>
#include <fstream>
#include <cassert>

#include "tokenizer.h"
#include "tokenizerUtils.h"

// BPE
BPE::BPE(BPETokenToRanks& encoder, BPETokenToRanks& specialTokensEncoder, const std::string& patStr)
    : mEncoder{encoder}, mSpecialTokensEncoder{specialTokensEncoder}, mNeedRegexCollapse{false}
{
    mNeedRegexCollapse = unicodeCollapseRegex(patStr, mRegex);

    mDecoder = reverseEncoder(mEncoder);
    mSpecialTokensDecoder = reverseEncoder(mSpecialTokensEncoder);
}

bool BPE::specialTokenPartition(const std::string& text, std::forward_list<textPartition>& partitions) const noexcept
{
    try
    {
        for (const auto& [specialToken, specialId] : mSpecialTokensEncoder)
        {
            for (auto it = partitions.begin(); it != partitions.end(); ++it)
            {
                auto& part = (*it);

                // if part not yet processed
                if (part.type == TEXT_PART_RAW_TEXT)
                {
                    auto& rawText = part.rawText;
                    auto baseOffset = part.offset;
                    auto baseLength = part.length;

                    // find occurences of specialToken in rawText
                    while (true)
                    {
                        auto match = rawText.find(specialToken, baseOffset);
                        if ((match == std::string::npos) || (match + specialToken.length() > baseOffset + baseLength))
                        {
                            break;
                        }

                        auto basePos = std::distance(partitions.begin(), it);

                        // insert left part
                        if (match > baseOffset)
                        {
                            partitions.emplace_after(it, rawText, baseOffset, match - baseOffset);
                            ++it;
                        }

                        // insert special token
                        partitions.emplace_after(it, specialId);
                        ++it;

                        // remove original part
                        if (basePos == 0)
                        {
                            partitions.erase_after(partitions.before_begin());
                        }
                        else
                        {
                            partitions.erase_after(std::next(partitions.begin(), (basePos - 1)));
                        }

                        // insert right part and continue loop
                        if (match + specialToken.length() < baseOffset + baseLength)
                        {
                            int rightOffset = match + specialToken.length();
                            int rightLength = baseLength + baseOffset - (match + specialToken.length());
                            partitions.emplace_after(it, rawText, rightOffset, rightLength);
                            ++it;

                            baseOffset = rightOffset;
                            baseLength = rightLength;
                        }
                        else
                        {
                            break;
                        }
                    }
                }
            }
        }

        return true;
    }
    catch(const std::exception& e)
    {
        LOG_ERROR(fmtstr("BPE::specialTokenPartition failed on text: %s", text.c_str()));
        return false;
    }
}

bool BPE::tokenize(const std::string& piece, std::vector<Rank>& output) const noexcept
{
    try
    {
        auto words = regexSplitText(piece);

        for (const auto& word : words)
        {
            auto it = mEncoder.find(word);
            if (it != mEncoder.end())
            {
                output.emplace_back(it->second);
            }
            else
            {
                bytePairEncode(word, output);
            }
        }
        return true;
    }
    catch(const std::exception& e)
    {
        LOG_ERROR(fmtstr("BPE::tokenize failed on piece: %s", piece.c_str()));
        return false;
    }

}

std::vector<std::string> BPE::regexSplitText(const std::string& text) const
{
    const auto cpts = unicodeCptsFromUtf8(text);

    // collapse for unicode regex match
    std::string textCollapsed;
    if (mNeedRegexCollapse)
    {
        textCollapsed = unicodeCollapseText(cpts);
    }
    else
    {
        textCollapsed = text;
    }

    auto bpeOffsets = unicodeRegexSplit(textCollapsed, mRegex);

    std::vector<std::string> bpeWords;
    bpeWords.reserve(bpeOffsets.size());

    int wordStart = 0;
    for (const auto& offset : bpeOffsets) {
        bpeWords.emplace_back();
        for (int i = wordStart; i < wordStart + offset; ++i) {
            bpeWords.back() += unicodeCptToUtf8(cpts[i]);
        }
        wordStart += offset;
    }

    return bpeWords;
}

void BPE::bytePairEncode(const std::string& piece, std::vector<Rank>& output) const
{
    // init parts, which is a vector of (start, rank).
    std::vector<std::pair<int, Rank>> parts;
    parts.reserve(piece.size() + 1);

    auto MAX_INT = std::numeric_limits<int>::max();
    auto MAX_RANK = std::numeric_limits<Rank>::max();
    std::pair<int, Rank> minRank{MAX_INT, MAX_RANK};

    for (int i = 0; i < piece.size() - 1; ++i)
    {
        Rank rank = MAX_RANK;
        const auto it = mEncoder.find({piece.begin() + i, piece.begin() + i + 2});
        if (it != mEncoder.end())
        {
            rank = it->second;
        }

        if (rank < minRank.second)
        {
            minRank = std::make_pair(i, rank);
        }

        parts.emplace_back(std::make_pair(i, rank));
    }

    parts.emplace_back(std::make_pair(piece.size() - 1, MAX_RANK));
    parts.emplace_back(std::make_pair(piece.size(), MAX_RANK));

    // helper function
    auto getMergedRank = [&](const int i) -> Rank
    {
        Rank rank = MAX_RANK;
        if (i + 3 < parts.size())
        {
            const auto it = mEncoder.find(std::string
                (
                    piece.begin() + parts[i].first,
                    piece.begin() + parts[i + 3].first
                )
            );
            if (it != mEncoder.end())
            {
                rank = it->second;
            }
        }
        return rank;
    };

    while (minRank.second != MAX_RANK)
    {
        int i = minRank.first;

        // update parts[i - 1], parts[i], parts[i + 1]
        if (i > 0)
        {
            parts[i - 1].second = getMergedRank(i - 1);
        }
        parts[i].second = getMergedRank(i);
        parts.erase(parts.begin() + i + 1);

        // update minRank
        minRank = std::make_pair(MAX_INT, MAX_RANK);
        for (int i = 0; i < parts.size() - 1; ++i)
        {
            auto rank = parts[i].second;
            if (rank < minRank.second) {
                minRank = std::make_pair(i, rank);
            }
        }
    }

    // collect tokens from parts
    for (int i = 0; i < parts.size() - 1; ++i)
    {
        const auto it = mEncoder.find({
            piece.begin() + parts[i].first, piece.begin() + parts[i + 1].first
        });
        assert(it != mEncoder.end());
        output.emplace_back(it->second);
    }
}

bool BPE::detokenize(const std::vector<Rank>& tokens, std::string& output) const noexcept
{
    try
    {
        for (const Rank& tok : tokens)
        {
            std::string bytes;
            auto it = mDecoder.find(tok);
            if (it != mDecoder.end())
            {
                bytes = it->second;
            }
            else
            {
                it = mSpecialTokensDecoder.find(tok);
                assert(it != mSpecialTokensDecoder.end());
                bytes = it->second;
            }
            output += bytes;
        }
        return true;
    }
    catch(const std::exception& e)
    {
        LOG_ERROR("BPE::detokenize failed.");
        return false;
    }

}

// Tokenizer
Tokenizer::Tokenizer()
    : mNumVocab{0}, mBosId{-1}, mEosId{-1}, mPadId{-1}
{}

Tokenizer::Tokenizer(const std::string& patStr, BPETokenToRanks& mergeableRanks, BPETokenToRanks& specialTokens,
    const Rank& bosId, const Rank& eosId, const Rank& padId, const std::unordered_set<Rank>& stopTokens)
    : mBosId{bosId}, mEosId{eosId}, mPadId{padId}, mStopTokens{stopTokens}
{
    auto comp = [](const std::pair<std::string, Rank>& p1, const std::pair<std::string, Rank>& p2)
    {
        return p1.second < p2.second;
    };
    auto maxId = std::max_element(mergeableRanks.begin(), mergeableRanks.end(), comp)->second;
    auto maxSpecialId = std::max_element(specialTokens.begin(), specialTokens.end(), comp)->second;
    mNumVocab = std::max(maxId, maxSpecialId) + 1;

    mBpe = std::make_unique<BPE>(mergeableRanks, specialTokens, patStr);
}

std::vector<Rank> Tokenizer::encode(const std::string& text, bool addSpecialTokens) const
{
    std::vector<Rank> output;
    output.reserve(text.size() + 2 * addSpecialTokens);
    std::forward_list<textPartition> partitions;

    if (!text.empty())
    {
        partitions.emplace_front(text, 0, text.length());
        mBpe->specialTokenPartition(text, partitions);
    }

    if (addSpecialTokens)
    {
        appendBos(output);
    }

    for (const auto& part : partitions)
    {
        if (part.type == TEXT_PART_RAW_TEXT)
        {
            auto piece = part.rawText.substr(part.offset, part.length);
            bool success = mBpe->tokenize(piece, output);
            assert(success);
        }
        else
        {
            output.emplace_back(part.token);
        }
    }

    if (addSpecialTokens)
    {
        appendEos(output);
    }

    return output;
}

std::string Tokenizer::decode(const std::vector<Rank>& tokens) const
{
    std::string output;
    output.reserve(tokens.size() * 2);

    bool success = mBpe->detokenize(tokens, output);
    assert(success);

    return output;
}

void Tokenizer::appendBos(std::vector<Rank>& output) const noexcept
{
    if (mBosId != -1)
    {
        output.push_back(mBosId);
    }
    else
    {
        LOG_WARNING("BOS ID is not set. Not appending BOS token.");
    }
}

void Tokenizer::appendEos(std::vector<Rank>& output) const noexcept
{
    if (mEosId != -1)
    {
        output.push_back(mEosId);
    }
    else
    {
        LOG_WARNING("EOS ID is not set. Not appending EOS token.");
    }
}

int Tokenizer::getNumVocab() const noexcept
{
    return mNumVocab;
}

Rank Tokenizer::getBosId() const noexcept
{
    return mBosId;
}

Rank Tokenizer::getEosId() const noexcept
{
    return mEosId;
}

Rank Tokenizer::getPadId() const noexcept
{
    return mPadId;
}

const std::unordered_set<Rank>& Tokenizer::getStopTokens() const noexcept
{
    return mStopTokens;
}

bool Tokenizer::loadTikTokenVocab(std::filesystem::path const& tiktokenFile, BPETokenToRanks& vocab) const noexcept
{
    try
    {
        assert(std::filesystem::exists(tiktokenFile));

        std::ifstream file(tiktokenFile);

        std::string line;
        while (std::getline(file, line))
        {
            auto it = line.find(" ");
            if (it != std::string::npos)
            {
                std::string token = base64Decode(line.substr(0, it));
                Rank rank = std::stoi(line.substr(it + 1));
                vocab[token] = rank;
            }
        }

        file.close();
        return true;
    }
    catch(const std::exception& e)
    {
        LOG_ERROR(fmtstr("Failed to load Tokenizer from Tiktoken: %s", tiktokenFile.c_str()));
        return false;
    }
}

bool Tokenizer::loadHFVocab(std::filesystem::path const& modelDir, BPETokenToRanks& vocab,
    BPETokenToRanks& specialTokens, Rank& bosId, Rank& eosId) const noexcept
{
    try
    {
        std::filesystem::path tokenizerFile = modelDir / "tokenizer.json";
        assert(std::filesystem::exists(tokenizerFile));
        std::ifstream data(tokenizerFile);

        std::string line;
        int indent = 0;
        bool parseVocab = false;
        bool parseSpecial = false;

        std::string specialContent;
        Rank specialId;

        while (std::getline(data, line))
        {
            // parse model.vocab
            if (!parseVocab && line.find("\"vocab\": {") != std::string::npos)
            {
                parseVocab = true;
                indent = line.find("\"");
            }
            else if (parseVocab && line.substr(indent) == "},")
            {
                parseVocab = false;
            }
            else if (parseVocab)
            {
                auto start = indent + 3;  // indent + 2 + "
                auto mid = line.find("\": ", start);
                auto end = line.find(",", mid);

                auto hfToken = line.substr(start, mid - start);
                Rank rank = std::stoi(line.substr(mid + 3, end - mid - 3));

                // remove "escape" character in json pattern: "\"", "\\"
                hfToken = std::regex_replace(hfToken, std::regex(R"(\\([\\\"]))"), "$1");
                auto token = decodeHFTokenToNormal(hfToken);

                vocab[token] = rank;
            }

            // parse added_tokens
            else if (!parseSpecial && line.find("\"added_tokens\": [") != std::string::npos)
            {
                parseSpecial = true;
                indent = line.find("\"");
            }
            else if (parseSpecial && line.substr(indent) == "],")
            {
                parseSpecial = false;
            }
            else if (parseSpecial)
            {
                // Only parse id and content for now
                if (line.find("\"id\": ") != std::string::npos)
                {
                    auto start = line.find(": ");
                    auto end = line.size() - 1;
                    specialId = std::stoi(line.substr(start + 2, end - start - 2));
                }
                else if (line.find("\"content\"") != std::string::npos)
                {
                    auto start = line.find(": ");
                    auto end = line.size() - 2;
                    specialContent = line.substr(start + 3, end - start - 3);

                    specialTokens[specialContent] = specialId;
                }
            }
        }

        data.close();

        // parse bos and eos token from config
        std::filesystem::path tokenizerConfig = modelDir / "tokenizer_config.json";
        if (std::filesystem::exists(tokenizerConfig))
        {
            std::ifstream config(tokenizerConfig);
            std::string line;

            while (std::getline(config, line))
            {
                if (line.find("\"bos_token\"") != std::string::npos)
                {
                    auto start = line.find(": ");
                    auto end = line.size() - 2;
                    std::string token = line.substr(start + 3, end - start - 3);
                    bosId = specialTokens[token];
                }
                else if (line.find("\"eos_token\"") != std::string::npos)
                {
                    auto start = line.find(": ");
                    auto end = line.size() - 2;
                    std::string token = line.substr(start + 3, end - start - 3);
                    eosId = specialTokens[token];
                }
            }

            config.close();
        }

        return true;
    }
    catch(const std::exception& e)
    {
        LOG_ERROR(fmtstr("Failed to load Tokenizer from HF: %s ", modelDir.c_str()));
        return false;
    }
}

// LlamaV3Tokenizer
void LlamaV3Tokenizer::loadFromTiktoken(std::filesystem::path const& modelPath)
{
    BPETokenToRanks mergeableRanks;

    assert(loadTikTokenVocab(modelPath, mergeableRanks));

    // add special tokens
    int numBaseTokens = mergeableRanks.size();

    std::vector<std::string> specialTokensList{
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|reserved_special_token_4|>",
        "<|eot_id|>",  // end of turn
    };

    int numReservedSpecialTokens = 256;
    for (int i = 5; i < numReservedSpecialTokens - 5; ++i)
    {
        specialTokensList.emplace_back("<|reserved_special_token_" + std::to_string(i) + "|>");
    }

    BPETokenToRanks specialTokens;
    for (int i = 0; i < specialTokensList.size(); ++i)
    {
        specialTokens[specialTokensList[i]] = numBaseTokens + i;
    }

    auto comp = [](const std::pair<std::string, Rank>& p1, const std::pair<std::string, Rank>& p2)
    {
        return p1.second < p2.second;
    };
    auto maxId = std::max_element(mergeableRanks.begin(), mergeableRanks.end(), comp)->second;
    auto maxSpecialId = std::max_element(specialTokens.begin(), specialTokens.end(), comp)->second;

    this->mNumVocab = std::max(maxId, maxSpecialId) + 1;
    this->mBpe = std::make_unique<BPE>(mergeableRanks, specialTokens, mRegexExpr);
    this->mBosId = specialTokens["<|begin_of_text|>"];
    this->mEosId = specialTokens["<|end_of_text|>"];
    this->mPadId = -1;
    this->mStopTokens = {
        specialTokens["<|end_of_text|>"],
        specialTokens["<|eot_id|>"]
    };
    LOG_INFO(fmtstr("Loaded LlamaV3Tokenizer from %s", modelPath.c_str()));
}

void LlamaV3Tokenizer::loadFromHF(std::filesystem::path const& modelDir)
{
    BPETokenToRanks mergeableRanks;
    BPETokenToRanks specialTokens;
    Rank bosId = -1;
    Rank eosId = -1;

    assert(loadHFVocab(modelDir, mergeableRanks, specialTokens, bosId, eosId));

    auto comp = [](const std::pair<std::string, Rank>& p1, const std::pair<std::string, Rank>& p2)
    {
        return p1.second < p2.second;
    };
    auto maxId = std::max_element(mergeableRanks.begin(), mergeableRanks.end(), comp)->second;
    auto maxSpecialId = std::max_element(specialTokens.begin(), specialTokens.end(), comp)->second;

    this->mNumVocab = std::max(maxId, maxSpecialId) + 1;
    this->mBpe = std::make_unique<BPE>(mergeableRanks, specialTokens, mRegexExpr);
    this->mBosId = bosId;
    this->mEosId = eosId;
    this->mPadId = -1;
    this->mStopTokens = {
        specialTokens["<|end_of_text|>"],
        specialTokens["<|eot_id|>"]
    };

   LOG_INFO(fmtstr("Loaded LlamaV3Tokenizer from %s", modelDir.c_str()));
}
