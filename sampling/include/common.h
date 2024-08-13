
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#if defined(_MSC_VER)
std::string fmtstr(char const *format, ...);
#else
std::string fmtstr(char const *format, ...)
    __attribute__((format(printf, 1, 2)));
#endif

template <typename T>
inline bool allOfBatchSlots(std::int32_t const *batchSlotsHost, T const *data,
                            std::int32_t batchSize, T value) {
  return std::all_of(batchSlotsHost, batchSlotsHost + batchSize,
                     [&](std::int32_t b) { return data[b] == value; });
}
