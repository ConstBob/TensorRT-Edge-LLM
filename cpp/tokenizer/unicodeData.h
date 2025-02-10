#pragma once

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct rangeNfd
{
    uint32_t first;
    uint32_t last;
    uint32_t nfd;
};

static const uint32_t MAX_CODEPOINTS = 0x110000;

extern const std::vector<std::pair<uint32_t, uint16_t>> unicodeRangesFlags;
extern const std::unordered_set<uint32_t> unicodeSetWhitespace;
extern const std::unordered_map<uint32_t, uint32_t> unicodeMapLowercase;
extern const std::unordered_map<uint32_t, uint32_t> unicodeMapUppercase;
extern const std::vector<rangeNfd> unicodeRangesNfd;