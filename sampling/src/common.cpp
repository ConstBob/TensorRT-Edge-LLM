#include "common.h"
#include <cstdarg>
#include <cstdlib>
#if !defined(_MSC_VER)
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#endif
#include <sstream>

namespace {
int constexpr VOID_PTR_SZ = 2 + sizeof(void *) * 2;

std::string vformat(char const *fmt, va_list args) {
  va_list args0;
  va_copy(args0, args);
  auto const size = vsnprintf(nullptr, 0, fmt, args0);
  if (size <= 0)
    return "";

  std::string stringBuf(size, char{});
  auto const size2 = std::vsnprintf(&stringBuf[0], size + 1, fmt, args);

  return stringBuf;
}

} // namespace

std::string fmtstr(char const *format, ...) {
  va_list args;
  va_start(args, format);
  std::string result = vformat(format, args);
  va_end(args);
  return result;
};
