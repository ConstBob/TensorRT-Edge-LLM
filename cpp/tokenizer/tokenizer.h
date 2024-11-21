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

#include <cassert>
#include <filesystem>
#include <forward_list>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using Rank = std::int64_t;
using BPETokenToRanks = std::unordered_map<std::string, Rank>;
using BPERanksToToken = std::unordered_map<Rank, std::string>;

typedef enum TEXT_PART_TYPE
{
    TEXT_PART_SPECIAL_TOKEN,
    TEXT_PART_RAW_TEXT
} TEXT_PART_TYPE;

struct textPartition
{
    textPartition(Rank _token)
        : type(TEXT_PART_SPECIAL_TOKEN)
        , token(_token)
        , rawText(_dummy)
        , offset(0)
        , length(0)
    {
    }

    textPartition(std::string const& _rawText, int _offset, int _length)
        : type(TEXT_PART_RAW_TEXT)
        , token(-1)
        , rawText(_rawText)
        , offset(_offset)
        , length(_length)
    {
        assert(offset >= 0);
        assert(length >= 1);
        assert(offset + length <= rawText.length());
    }

    const TEXT_PART_TYPE type;
    const Rank token;
    const std::string _dummy;
    std::string const& rawText;
    int const offset;
    int const length;
};

// BPE
class BPE
{
public:
    BPE(BPETokenToRanks& encoder, BPETokenToRanks& specialTokensEncoder, std::string const& patStr);

    ~BPE() = default;

    bool tokenize(std::string const& text, std::vector<Rank>& output) const noexcept;

    bool detokenize(std::vector<Rank> const& tokens, std::string& bytes) const noexcept;

    bool specialTokenPartition(std::string const& text, std::forward_list<textPartition>& partitions) const noexcept;

private:
    void initRegex(std::string const* patStr);

    void bytePairEncode(std::string const& piece, std::vector<Rank>& output) const;

    std::vector<std::string> regexSplitText(std::string const& text) const;

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

    Tokenizer(std::string const& patStr, BPETokenToRanks& mergeableRanks, BPETokenToRanks& specialTokens,
        Rank const& bosId = -1, Rank const& eosId = -1, Rank const& padId = -1);

    virtual ~Tokenizer() = default;

    virtual std::vector<Rank> encode(std::string const& text, bool addBos = false, bool addEos = false) const;

    virtual std::string decode(std::vector<Rank> const& tokens) const;

    virtual void loadFromHF(std::filesystem::path const& modelDir);

    int getNumVocab() const noexcept;

    Rank getBosId() const noexcept;

    Rank getEosId() const noexcept;

    Rank getPadId() const noexcept;

protected:
    // manually parse tokenizer.json and tokenizer_config.json without using 3rdparty libraries
    void loadHFVocab(std::filesystem::path const& modelDir, BPETokenToRanks& vocab, BPETokenToRanks& specialTokens) noexcept;

    void loadHFConfig(std::filesystem::path const& modelDir, BPETokenToRanks& specialTokens) noexcept;

    void appendEos(std::vector<Rank>& output) const noexcept;

    void appendBos(std::vector<Rank>& output) const noexcept;

    int mNumVocab;
    std::unique_ptr<BPE> mBpe;
    Rank mBosId;
    Rank mEosId;
    Rank mPadId;
    std::string mRegexExpr;

};

#endif // TOKENIZER_H
