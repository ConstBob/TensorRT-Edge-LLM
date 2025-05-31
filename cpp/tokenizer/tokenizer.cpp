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

#include <cassert>
#include <fstream>
#include <limits>

#include "tokenizer.h"
#include "tokenizerUtils.h"

// BPE
BPE::BPE(BPETokenToRanks& encoder, BPETokenToRanks& specialTokensEncoder, std::string const& patStr)
    : mEncoder{encoder}
    , mSpecialTokensEncoder{specialTokensEncoder}
    , mNeedRegexCollapse{false}
{
    mNeedRegexCollapse = unicodeCollapseRegex(patStr, mRegex);

    mDecoder = reverseEncoder(mEncoder);
    mSpecialTokensDecoder = reverseEncoder(mSpecialTokensEncoder);
}

bool BPE::specialTokenPartition(std::string const& text, std::forward_list<textPartition>& partitions) const noexcept
{
    try
    {
        for (auto const& [specialToken, specialId] : mSpecialTokensEncoder)
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

                    // find occurrences of specialToken in rawText
                    while (true)
                    {
                        auto match = rawText.find(specialToken, baseOffset);
                        if ((match == std::string::npos)
                            || (static_cast<int>(match + specialToken.length()) > (baseOffset + baseLength)))
                        {
                            break;
                        }

                        auto basePos = std::distance(partitions.begin(), it);

                        // insert left part
                        if (match > static_cast<size_t>(baseOffset))
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
                        if (match + specialToken.length() < static_cast<size_t>(baseOffset + baseLength))
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
    catch (std::exception const& e)
    {
        LOG_ERROR("BPE::specialTokenPartition failed on text: %s", text.c_str());
        return false;
    }
}

bool BPE::tokenize(std::string const& piece, std::vector<Rank>& output) const noexcept
{
    try
    {
        auto words = regexSplitText(piece);

        for (auto const& word : words)
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
    catch (std::exception const& e)
    {
        LOG_ERROR("BPE::tokenize failed on piece: %s", piece.c_str());
        return false;
    }
}

std::vector<std::string> BPE::regexSplitText(std::string const& text) const
{
    auto const cpts = unicodeCptsFromUtf8(text);

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

    size_t wordStart = 0;
    for (auto const& offset : bpeOffsets)
    {
        bpeWords.emplace_back();
        for (size_t i = wordStart; i < wordStart + offset; ++i)
        {
            bpeWords.back() += unicodeCptToUtf8(cpts[i]);
        }
        wordStart += offset;
    }

    return bpeWords;
}

void BPE::bytePairEncode(std::string const& piece, std::vector<Rank>& output) const
{
    // init parts, which is a vector of (start, rank).
    std::vector<std::pair<int, Rank>> parts;
    parts.reserve(piece.size() + 1);

    auto MAX_INT = std::numeric_limits<int>::max();
    auto MAX_RANK = std::numeric_limits<Rank>::max();
    std::pair<int, Rank> minRank{MAX_INT, MAX_RANK};

    for (size_t i = 0; i < piece.size() - 1; ++i)
    {
        Rank rank = MAX_RANK;
        auto const it = mEncoder.find({piece.begin() + i, piece.begin() + i + 2});
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
    auto getMergedRank = [&](size_t const i) -> Rank {
        Rank rank = MAX_RANK;
        if (i + 3 < parts.size())
        {
            const auto it
                = mEncoder.find(std::string(piece.begin() + parts[i].first, piece.begin() + parts[i + 3].first));
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
        for (size_t i = 0; i < parts.size() - 1; ++i)
        {
            auto rank = parts[i].second;
            if (rank < minRank.second)
            {
                minRank = std::make_pair(i, rank);
            }
        }
    }

    // collect tokens from parts
    for (size_t i = 0; i < parts.size() - 1; ++i)
    {
        auto const it = mEncoder.find({piece.begin() + parts[i].first, piece.begin() + parts[i + 1].first});
        assert(it != mEncoder.end());
        output.emplace_back(it->second);
    }
}

bool BPE::detokenize(std::vector<Rank> const& tokens, std::string& output, bool skipSpecialTokens) const noexcept
{
    try
    {
        for (Rank const& tok : tokens)
        {
            std::string bytes;
            auto it = mDecoder.find(tok);
            if (it != mDecoder.end())
            {
                bytes = it->second;
            }
            else if (!skipSpecialTokens)
            {
                it = mSpecialTokensDecoder.find(tok);
                assert(it != mSpecialTokensDecoder.end());
                bytes = it->second;
            }
            output += bytes;
        }
        return true;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("BPE::detokenize failed.");
        return false;
    }
}

// Tokenizer
Tokenizer::Tokenizer()
    : mNumVocab{0}
    , mBosId{-1}
    , mEosId{-1}
    , mPadId{-1}
    , mUnkId{-1}
{
}

Tokenizer::Tokenizer(std::string const& patStr, BPETokenToRanks& mergeableRanks, BPETokenToRanks& specialTokens,
    Rank const& bosId, Rank const& eosId, Rank const& padId, Rank const& unkId)
    : mBosId{bosId}
    , mEosId{eosId}
    , mPadId{padId}
    , mUnkId{unkId}
{
    auto comp = [](std::pair<std::string, Rank> const& p1, std::pair<std::string, Rank> const& p2) {
        return p1.second < p2.second;
    };
    auto maxId = std::max_element(mergeableRanks.begin(), mergeableRanks.end(), comp)->second;
    auto maxSpecialId = std::max_element(specialTokens.begin(), specialTokens.end(), comp)->second;
    mNumVocab = std::max(maxId, maxSpecialId) + 1;

    mBpe = std::make_unique<BPE>(mergeableRanks, specialTokens, patStr);
}

std::vector<Rank> Tokenizer::encode(std::string const& text, bool addBos, bool addEos) const
{
    std::vector<Rank> output;
    output.reserve(text.size() + addBos + addEos);
    std::forward_list<textPartition> partitions;

    if (!text.empty())
    {
        partitions.emplace_front(text, 0, text.length());
        mBpe->specialTokenPartition(text, partitions);
    }

    if (addBos)
    {
        appendBos(output);
    }

    for (auto const& part : partitions)
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

    if (addEos)
    {
        appendEos(output);
    }

    return output;
}

std::string Tokenizer::decode(std::vector<Rank> const& tokens, bool skipSpecialTokens) const
{
    std::string output;
    output.reserve(tokens.size() * 2);

    bool success = mBpe->detokenize(tokens, output, skipSpecialTokens);
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
        LOG_DEBUG("BOS ID is not set. Not appending BOS token.");
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
        LOG_DEBUG("EOS ID is not set. Not appending EOS token.");
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
    return mPadId == -1 ? mEosId : mPadId;
}

Rank Tokenizer::getUnkId() const noexcept
{
    return mUnkId;
}

void Tokenizer::loadHFSpecialTokens(std::filesystem::path const& modelDir, BPETokenToRanks& specialTokens)
{
    std::string line;
    int indent = 0;
    bool parseSpecial = false;
    std::string specialContent;
    Rank specialId;

    // Load 'added_tokens_decoder' from tokenizer_config.json
    std::filesystem::path tokenizerConfig = modelDir / "tokenizer_config.json";
    if (std::filesystem::exists(tokenizerConfig))
    {
        std::ifstream config(tokenizerConfig);

        while (std::getline(config, line))
        {
            if (!parseSpecial && line.find("\"added_tokens_decoder\": {") != std::string::npos)
            {
                parseSpecial = true;
                indent = line.find("\"");
            }
            else if (parseSpecial && line.substr(indent) == "},")
            {
                break;
            }
            else if (parseSpecial)
            {
                // Only parse id and content for now
                if (line.find("\": {") != std::string::npos)
                {
                    auto start = line.find("\"") + 1;
                    auto end = line.find("\": {");
                    specialId = std::stoi(line.substr(start, end - start));
                }
                else if (line.find("\"content\"") != std::string::npos)
                {
                    auto start = line.find(": ") + 3;
                    auto end = line.size() - 2;
                    specialContent = line.substr(start, end - start);
                    specialTokens[specialContent] = specialId;
                }
            }
        }

        config.close();
    }

    if (!parseSpecial)
    {
        // Load 'added_tokens' from tokenizer.json
        std::filesystem::path tokenizerFile = modelDir / "tokenizer.json";
        assert(std::filesystem::exists(tokenizerFile));
        std::ifstream data(tokenizerFile);

        while (std::getline(data, line))
        {
            if (!parseSpecial && line.find("\"added_tokens\": [") != std::string::npos)
            {
                parseSpecial = true;
                indent = line.find("\"");
            }
            else if (parseSpecial && line.substr(indent) == "],")
            {
                break;
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
    }
}

void Tokenizer::loadHFVocab(std::filesystem::path const& modelDir, BPETokenToRanks& vocab)
{
    std::filesystem::path tokenizerFile = modelDir / "tokenizer.json";
    assert(std::filesystem::exists(tokenizerFile));
    std::ifstream data(tokenizerFile);

    std::string line;
    int indent = 0;
    bool parseVocab = false;

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
            auto start = indent + 3; // indent + 2 + "
            auto mid = line.find("\": ", start);
            auto end = line.find(",", mid);

            auto hfToken = line.substr(start, mid - start);
            Rank rank = std::stoi(line.substr(mid + 3, end - mid - 3));

            // remove "escape" character in json pattern: "\"", "\\"
            hfToken = std::regex_replace(hfToken, std::regex(R"(\\([\\\"]))"), "$1");
            auto token = decodeHFTokenToNormal(hfToken);
            vocab[token] = rank;
        }

        // parse regex
        else if (line.find("\"Regex\": \"") != std::string::npos)
        {
            auto start = line.find(": ");
            auto end = line.size() - 1;
            std::string rawRegex = line.substr(start + 3, end - start - 3);

            // remove "escape" character in json pattern: "\"", "\\"
            rawRegex = std::regex_replace(rawRegex, std::regex(R"(\\([\\\"]))"), "$1");
            this->mRegexExpr = normalizeRegex(rawRegex);
        }
    }

    data.close();
}

void Tokenizer::loadHFConfig(std::filesystem::path const& modelDir, BPETokenToRanks& specialTokens)
{
    std::filesystem::path tokenizerConfig = modelDir / "tokenizer_config.json";

    auto parseSpecialToken = [&specialTokens](std::string line) -> Rank {
        auto start = line.find(": ");
        auto end = line.find(",");
        std::string token = line.substr(start + 2, end - start - 2);
        if (token == "null")
        {
            return -1;
        }

        assert(token[0] == '\"' && token[token.size() - 1] == '\"');
        token = token.substr(1, token.size() - 2);
        return specialTokens[token];
    };

    if (std::filesystem::exists(tokenizerConfig))
    {
        std::ifstream config(tokenizerConfig);
        std::string line;

        while (std::getline(config, line))
        {
            if (line.find("\"bos_token\"") != std::string::npos)
            {
                this->mBosId = parseSpecialToken(line);
            }
            else if (line.find("\"eos_token\"") != std::string::npos)
            {
                this->mEosId = parseSpecialToken(line);
            }
            else if (line.find("\"pad_token\"") != std::string::npos)
            {
                this->mPadId = parseSpecialToken(line);
            }
            else if (line.find("\"unk_token\"") != std::string::npos)
            {
                this->mUnkId = parseSpecialToken(line);
            }
        }

        config.close();
    }
    else
    {
        LOG_WARNING("Cannot find tokenizer_config.json. Use default config.");
    }
}

void Tokenizer::loadFromHF(std::filesystem::path const& modelDir)
{
    BPETokenToRanks mergeableRanks;
    BPETokenToRanks specialTokens;

    loadHFSpecialTokens(modelDir, specialTokens);
    loadHFVocab(modelDir, mergeableRanks);
    loadHFConfig(modelDir, specialTokens);

    auto comp = [](std::pair<std::string, Rank> const& p1, std::pair<std::string, Rank> const& p2) {
        return p1.second < p2.second;
    };
    auto maxId = std::max_element(mergeableRanks.begin(), mergeableRanks.end(), comp)->second;
    auto maxSpecialId = std::max_element(specialTokens.begin(), specialTokens.end(), comp)->second;

    this->mNumVocab = std::max(maxId, maxSpecialId) + 1;
    this->mBpe = std::make_unique<BPE>(mergeableRanks, specialTokens, mRegexExpr);

    LOG_INFO("Loaded tokenizer from %s", modelDir.c_str());
}