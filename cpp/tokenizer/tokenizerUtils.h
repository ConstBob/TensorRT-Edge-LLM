/*
 * This file is based on MIT-licensed code from
 * https://github.com/ggerganov/llama.cpp/blob/master/src/unicode.h
 *
 * Modifications and enhancements by DriveOS LLM-SDK team, 2025.
 *
 * Original license:
 * MIT License
 *
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include "common/logger.h"
#include "tokenizer.h"
#include <cassert>
#include <iostream>

/**
 * Helper functions
 */
// Replace special chars to support std::regex
static std::regex const specialChars{R"([[\^$.|?*+(){}])"};

// reverse map<token, id> to map<id, token>
BPERanksToToken reverseEncoder(BPETokenToRanks const& encoder);

// decode hf format token str to normal utf-8
std::string decodeHFTokenToNormal(std::string const& hfToken);

// deal with regex expressions that c++ regex don't support
std::string normalizeRegex(std::string const& expr);

/**
 * Unicode Utils
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
    inline codepointFlags(uint16_t const flags = 0)
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
static std::map<std::string, int> const kUatEnum = {
    {"\\p{N}", codepointFlags::NUMBER},
    {"\\p{L}", codepointFlags::LETTER},
    {"\\p{P}", codepointFlags::PUNCTUATION},
};

static std::map<int, int> const kUcatCpt = {
    {codepointFlags::NUMBER, 0xD1},
    {codepointFlags::LETTER, 0xD2},
    {codepointFlags::PUNCTUATION, 0xD3},
};

static std::map<int, std::string> const kUcatMap = {
    {codepointFlags::NUMBER, "\x30-\x39"},          // 0-9
    {codepointFlags::LETTER, "\x41-\x5A\x61-\x7A"}, // A-Za-z
    {codepointFlags::PUNCTUATION,
        "\x21-\x23\x25-\x2A\x2C-\x2F\x3A-\x3B\x3F-\x40\\\x5B-\\\x5D\x5F\\\x7B\\\x7D"}, // !-#%-*,-/:-;?-@\[-\]_\{\}
};

bool unicodeCollapseRegex(std::string const& expr, std::regex& regex);

std::string unicodeCollapseText(std::vector<uint32_t> const& cpts);

std::vector<size_t> unicodeRegexSplit(std::string const& text, std::regex const& regex);

std::vector<uint32_t> unicodeCptsFromUtf8(std::string const& utf8);

uint32_t unicodeCptFromUtf8(std::string const& utf8, size_t& offset);

std::string unicodeCptToUtf8(uint32_t cp);

codepointFlags unicodeCptFlags(uint32_t const cp);