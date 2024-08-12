
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

#if defined(_WIN32)
#define TLLM_LIKELY(x) (__assume((x) == 1), (x))
#define TLLM_UNLIKELY(x) (__assume((x) == 0), (x))
#else
#define TLLM_LIKELY(x) __builtin_expect((x), 1)
#define TLLM_UNLIKELY(x) __builtin_expect((x), 0)
#endif

#define NEW_TLLM_EXCEPTION(...)                                                \
  TllmException(__FILE__, __LINE__, fmtstr(__VA_ARGS__))

class TllmException : public std::runtime_error {
public:
  static auto constexpr MAX_FRAMES = 128;

  explicit TllmException(char const *file, std::size_t line,
                         std::string const &msg);

  ~TllmException() noexcept override;

  [[nodiscard]] std::string getTrace() const;

  static std::string demangle(char const *name);

private:
  std::array<void *, MAX_FRAMES> mCallstack{};
  int mNbFrames;
};

[[noreturn]] inline void throwRuntimeError(char const *const file,
                                           int const line,
                                           std::string const &info = "") {
  throw TllmException(
      file, line,
      fmtstr("[TensorRT-LLM][ERROR] Assertion failed: %s", info.c_str()));
}

#define TLLM_CHECK(val)                                                        \
  do {                                                                         \
    TLLM_LIKELY(static_cast<bool>(val))                                        \
    ? ((void)0) : throwRuntimeError(__FILE__, __LINE__, #val);                 \
  } while (0)

template <typename T>
inline bool allOfBatchSlots(std::int32_t const *batchSlotsHost, T const *data,
                            std::int32_t batchSize, T value) {
  return std::all_of(batchSlotsHost, batchSlotsHost + batchSize,
                     [&](std::int32_t b) { return data[b] == value; });
}
