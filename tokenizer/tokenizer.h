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

#pragma once
#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <forward_list>
#include <unordered_map>
#include <unordered_set>
#include <regex>
#include <filesystem>
#include <cassert>

using Rank = std::int32_t;
using BPETokenToRanks = std::unordered_map<std::string, Rank>;
using BPERanksToToken = std::unordered_map<Rank, std::string>;

typedef enum TEXT_PART_TYPE
{
    TEXT_PART_SPECIAL_TOKEN,
    TEXT_PART_RAW_TEXT
} TEXT_PART_TYPE;

struct textPartition
{
    textPartition(Rank _token):
        type(TEXT_PART_SPECIAL_TOKEN),
        token(_token),
        rawText(_dummy),
        offset(0),
        length(0)
    {}

    textPartition(const std::string& _rawText, int _offset, int _length):
        type(TEXT_PART_RAW_TEXT),
        token(-1),
        rawText(_rawText),
        offset(_offset),
        length(_length)
    {
        assert(offset >= 0);
        assert(length >= 1);
        assert(offset + length <= rawText.length());
    }

    const TEXT_PART_TYPE type;
    const Rank token;
    const std::string _dummy;
    const std::string& rawText;
    const int offset;
    const int length;
};

// BPE
class BPE
{
public:
    BPE(BPETokenToRanks& encoder, BPETokenToRanks& specialTokensEncoder, const std::string& patStr);

    ~BPE() = default;

    bool tokenize(const std::string& text, std::vector<Rank>& output) const noexcept;

    bool detokenize(const std::vector<Rank>& tokens, std::string& bytes) const noexcept;

    bool specialTokenPartition(const std::string& text, std::forward_list<textPartition>& partitions) const noexcept;

private:
    void initRegex(const std::string* patStr);

    void bytePairEncode(const std::string& piece, std::vector<Rank>& output) const;

    std::vector<std::string> regexSplitText(const std::string& text) const;

    BPETokenToRanks mEncoder;
    BPERanksToToken mDecoder;

    BPETokenToRanks mSpecialTokensEncoder;
    BPERanksToToken mSpecialTokensDecoder;

    std::regex mRegex;
    bool mNeedRegexCollapse;
};

// Tokenizer base class
class Tokenizer
{
public:
    Tokenizer();

    Tokenizer(const std::string& patStr, BPETokenToRanks& mergeableRanks, BPETokenToRanks& specialTokens,
        const Rank& bosId = -1, const Rank& eosId = -1, const Rank& padId = -1, 
        const std::unordered_set<Rank>& stopTokens = {});

    virtual ~Tokenizer() = default;

    virtual std::vector<Rank> encode(const std::string& text, bool addSpecialTokens = false) const;

    virtual std::string decode(const std::vector<Rank>& tokens) const;

    virtual void loadFromTiktoken(std::filesystem::path const& modelPath) = 0;

    virtual void loadFromHF(std::filesystem::path const& modelDir) = 0;

    int getNumVocab() const noexcept;

    Rank getBosId() const noexcept;

    Rank getEosId() const noexcept;

    Rank getPadId() const noexcept;

    const std::unordered_set<Rank>& getStopTokens() const noexcept;

protected:
    bool loadTikTokenVocab(std::filesystem::path const& tiktokenFile, BPETokenToRanks& vocab) const noexcept;

    // manually parse vocab and special tokens tokenizer.json file without using 3rdparty libraries
    bool loadHFVocab(std::filesystem::path const& modelDir, BPETokenToRanks& vocab,
        BPETokenToRanks& specialTokens, Rank& bosId, Rank& eosId) const noexcept;

    void appendEos(std::vector<Rank>& output) const noexcept;

    void appendBos(std::vector<Rank>& output) const noexcept;

    int mNumVocab;
    std::unique_ptr<BPE> mBpe;

    Rank mBosId;
    Rank mEosId;
    Rank mPadId;
    std::unordered_set<Rank> mStopTokens;
};

// Llamav3 tokenizer
class LlamaV3Tokenizer : public Tokenizer
{
public:
    void loadFromTiktoken(std::filesystem::path const& modelPath) override;

    void loadFromHF(std::filesystem::path const& modelDir) override;

private:
    // original regex from tokenizer.json
    // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    // adapted: https://github.com/ggerganov/llama.cpp/pull/6920#issuecomment-2080233989
    static constexpr char mRegexExpr[] = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
};

#endif // TOKENIZER_H
