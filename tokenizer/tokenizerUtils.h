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
#ifndef TOKENIZER_UTILS_H
#define TOKENIZER_UTILS_H

#include "common.h"
#include "tokenizer.h"
#include <cassert>
#include <iostream>

/**
 * Helper functions
 */
// Replace special chars to support std::regex
static const std::regex specialChars{R"([[\^$.|?*+(){}])"};

// base64 decode to bytes
std::string base64Decode(std::string const& encoded);

// reverse map<token, id> to map<id, token>
BPERanksToToken reverseEncoder(BPETokenToRanks const& encoder);

// decode hf format token str to normal utf-8
std::string decodeHFTokenToNormal(std::string const& hfToken);

/**
 * Unicode Utils
 * Reference: https://github.com/ggerganov/llama.cpp/src/unicode.cpp
 */
struct codepointFlags
{
    enum
    {
        UNDEFINED = 0x0001,
        NUMBER = 0x0002,      // regex: \p{N}
        LETTER = 0x0004,      // regex: \p{L}
        SEPARATOR = 0x0008,   // regex: \p{Z}
        ACCENT_MARK = 0x0010, // regex: \p{M}
        PUNCTUATION = 0x0020, // regex: \p{P}
        SYMBOL = 0x0040,      // regex: \p{S}
        CONTROL = 0x0080,     // regex: \p{C}
        MASK_CATEGORIES = 0x00FF,
    };

    // codepoint type
    uint16_t isUndefined : 1;
    uint16_t isNumber : 1;      // regex: \p{N}
    uint16_t isLetter : 1;      // regex: \p{L}
    uint16_t isSeparator : 1;   // regex: \p{Z}
    uint16_t isAccentMark : 1;  // regex: \p{M}
    uint16_t isPunctuation : 1; // regex: \p{P}
    uint16_t isSymbol : 1;      // regex: \p{S}
    uint16_t isControl : 1;     // regex: \p{C}
    // helper flags
    uint16_t isWhitespace : 1; // regex: \s
    uint16_t isLowercase : 1;
    uint16_t isUppercase : 1;
    uint16_t isNfd : 1;

    // decode from uint16
    inline codepointFlags(const uint16_t flags = 0)
    {
        *reinterpret_cast<uint16_t*>(this) = flags;
    }

    inline uint16_t asUint() const
    {
        return *reinterpret_cast<uint16_t const*>(this);
    }

    inline uint16_t categoryFlag() const
    {
        return this->asUint() & MASK_CATEGORIES;
    }
};

// unicode categories
static const std::map<std::string, int> kUatEnum = {
    {"\\p{N}", codepointFlags::NUMBER},
    {"\\p{L}", codepointFlags::LETTER},
    {"\\p{P}", codepointFlags::PUNCTUATION},
};

static const std::map<int, int> kUcatCpt = {
    {codepointFlags::NUMBER, 0xD1},
    {codepointFlags::LETTER, 0xD2},
    {codepointFlags::PUNCTUATION, 0xD3},
};

static const std::map<int, std::string> kUcatMap = {
    {codepointFlags::NUMBER, "\x30-\x39"},                                             // 0-9
    {codepointFlags::LETTER, "\x41-\x5A\x61-\x7A"},                                    // A-Za-z
    {codepointFlags::PUNCTUATION,
        "\x21-\x23\x25-\x2A\x2C-\x2F\x3A-\x3B\x3F-\x40\\\x5B-\\\x5D\x5F\\\x7B\\\x7D"}, // !-#%-*,-/:-;?-@\[-\]_\{\}
};

// generate a "collapsed" representation of regex to handle unicode categories
bool unicodeCollapseRegex(std::string const& expr, std::regex& regex);

// generate a "collapsed" representation of the text, where all codepoints are replaced by a single byte
// ref: https://github.com/ggerganov/llama.cpp/pull/6920#issuecomment-2081479935
std::string unicodeCollapseText(std::vector<uint32_t> const& cpts);

std::vector<size_t> unicodeRegexSplit(std::string const& text, std::regex const& regex);

std::vector<uint32_t> unicodeCptsFromUtf8(std::string const& utf8);

uint32_t unicodeCptFromUtf8(std::string const& utf8, size_t& offset);

std::string unicodeCptToUtf8(uint32_t cp);

codepointFlags unicodeCptFlags(const uint32_t cp);

#endif // TOKENIZER_UTILS_H
