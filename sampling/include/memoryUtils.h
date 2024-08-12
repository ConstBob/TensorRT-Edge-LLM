#pragma once

#include <NvInferRuntime.h>
#include <cassert>
#include <cstdint>
#include <cuda_fp16.h>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

inline size_t calcAlignedSize(std::vector<size_t> const &sizes,
                              const size_t ALIGN_BYTES = 256) {
  const size_t ALIGN_MASK = ~(ALIGN_BYTES - 1);
  // Check ALIGN_BYTES is a power of 2
  assert((ALIGN_BYTES & (ALIGN_BYTES - 1)) == 0);

  size_t total = 0;
  for (auto sz : sizes) {
    total += (sz + ALIGN_BYTES - 1) & ALIGN_MASK;
  }

  // We add extra "ALIGN_BYTES - 1" bytes in case the start address passed to
  // the function calcAlignedPointers() is not aligned.
  return total + ALIGN_BYTES - 1;
}

inline void calcAlignedPointers(std::vector<void *> &outPtrs, void const *p,
                                std::vector<size_t> const &sizes,
                                size_t ALIGN_BYTES = 256) {
  const size_t ALIGN_MASK = ~(ALIGN_BYTES - 1);
  // Check ALIGN_BYTES is a power of 2
  assert((ALIGN_BYTES & (ALIGN_BYTES - 1)) == 0);

  // In case the start address is not aligned
  char *ptr = reinterpret_cast<char *>(
      (reinterpret_cast<size_t>(p) + ALIGN_BYTES - 1) & ALIGN_MASK);

  outPtrs.reserve(sizes.size());
  for (auto sz : sizes) {
    outPtrs.push_back(ptr);
    ptr += (sz + ALIGN_BYTES - 1) & ALIGN_MASK;
  }
}


//! \brief For converting a C++ data type to a TensorRT data type.
template <typename T, bool = false> struct TRTDataType {};

template <> struct TRTDataType<float> {
  static constexpr auto value = nvinfer1::DataType::kFLOAT;
};

template <> struct TRTDataType<half> {
  static constexpr auto value = nvinfer1::DataType::kHALF;
};

template <> struct TRTDataType<std::int8_t> {
  static constexpr auto value = nvinfer1::DataType::kINT8;
};

template <> struct TRTDataType<std::int32_t> {
  static constexpr auto value = nvinfer1::DataType::kINT32;
};

template <> struct TRTDataType<std::int64_t> {
  static constexpr auto value = nvinfer1::DataType::kINT64;
};

template <> struct TRTDataType<bool> {
  static constexpr auto value = nvinfer1::DataType::kBOOL;
};

template <> struct TRTDataType<std::uint8_t> {
  static constexpr auto value = nvinfer1::DataType::kUINT8;
};

#ifdef ENABLE_BF16
template <> struct TRTDataType<__nv_bfloat16> {
  static constexpr auto value = nvinfer1::DataType::kBF16;
};
#endif

#ifdef ENABLE_FP8
template <> struct TRTDataType<__nv_fp8_e4m3> {
  static constexpr auto value = nvinfer1::DataType::kFP8;
};
#endif

template <typename T> struct TRTDataType<T *> {
  static constexpr auto value = nvinfer1::DataType::kINT64;
};

constexpr static size_t getDTypeSize(nvinfer1::DataType type) {
  switch (type) {
  case nvinfer1::DataType::kINT64:
    return 8;
  case nvinfer1::DataType::kINT32:
    [[fallthrough]];
  case nvinfer1::DataType::kFLOAT:
    return 4;
  case nvinfer1::DataType::kBF16:
    [[fallthrough]];
  case nvinfer1::DataType::kHALF:
    return 2;
  case nvinfer1::DataType::kBOOL:
    [[fallthrough]];
  case nvinfer1::DataType::kUINT8:
    [[fallthrough]];
  case nvinfer1::DataType::kINT8:
    [[fallthrough]];
  case nvinfer1::DataType::kFP8:
    return 1;
  // case nvinfer1::DataType::kINT4: TLLM_THROW("Cannot determine size of INT4
  // data type");
  default:
    return 0;
  }
  return 0;
}

class TensorWrapper {
public:
  using DataType = nvinfer1::DataType;

  TensorWrapper() {
    _data = nullptr;
    _dim.clear();
  }

  TensorWrapper(void *data, std::vector<std::int64_t> const &dim,
                DataType dtype)
      : _data(data), _dim(dim), _dtype(dtype) {}

  ~TensorWrapper() = default;

  TensorWrapper(TensorWrapper const &) = delete;
  TensorWrapper &operator=(TensorWrapper const &) = delete;

  TensorWrapper(TensorWrapper &&other) {
    _data = other._data;
    _dim = std::move(other._dim);
    _dtype = other._dtype;
    other._data = nullptr;
    other._dim.clear();
  }

  TensorWrapper &operator=(TensorWrapper &&other) {
    if (this != &other) {
      _data = other._data;
      _dim = std::move(other._dim);
      _dtype = other._dtype;
      other._data = nullptr;
      other._dim.clear();
    }
    return *this;
  }

  auto getDataType() const { return _dtype; }

  auto data() { return _data; }

  void const *data() const { return _data; }

  template <int64_t rank> auto getDimension() { return _dim[rank]; }

  auto getSizeInBytes() const {
    auto size =
        std::accumulate(_dim.begin(), _dim.end(), 1, std::multiplies<size_t>());
    return size * getDTypeSize(_dtype);
  }

private:
  void *_data;
  std::vector<std::int64_t> _dim;
  DataType _dtype;
};

using TensorPtr = std::shared_ptr<TensorWrapper>;
using TensorConstPtr = std::shared_ptr<const TensorWrapper>;

template <typename T> T const *tensorCast(TensorWrapper const &tensor) {
  if (TRTDataType<typename std::remove_cv<T>::type>::value !=
      tensor.getDataType()) {
    throw std::bad_cast();
  }
  return static_cast<T const *>(tensor.data());
}

template <typename T> T *tensorCast(TensorWrapper &tensor) {
  if (TRTDataType<typename std::remove_cv<T>::type>::value !=
      tensor.getDataType()) {
    throw std::bad_cast();
  }
  return static_cast<T *>(tensor.data());
}

template <typename T> T const *tensorCastOrNull(TensorWrapper const *tensor) {
  if (tensor) {
    return tensorCast<T>(*tensor);
  }
  return static_cast<T const *>(nullptr);
}

template <typename T> T *tensorCastOrNull(TensorWrapper *tensor) {
  if (tensor) {
    return tensorCast<T>(*tensor);
  }
  return static_cast<T *>(nullptr);
}

template <typename T> T const *tensorCastOrNull(TensorWrapper const &tensor) {
  if (tensor.data()) {
    return tensorCast<T>(tensor);
  }
  return static_cast<T const *>(nullptr);
}

template <typename T> T *tensorCastOrNull(TensorWrapper &tensor) {
  if (tensor.data()) {
    return tensorCast<T>(tensor);
  }
  return static_cast<T *>(nullptr);
}

template <typename T>
T *tensorCastOrNull(std::optional<TensorPtr> const &tensor) {
  if (tensor) {
    return tensorCast<T>(*tensor.value());
  }
  return static_cast<T *>(nullptr);
}

template <typename T>
T const *tensorCastOrNull(std::optional<TensorConstPtr> const &tensor) {
  if (tensor) {
    return tensorCast<T>(*tensor.value());
  }
  return static_cast<T const *>(nullptr);
}

template <typename T> T *tensorCastOrNull(TensorPtr const &tensor) {
  if (tensor) {
    return tensorCast<T>(*tensor);
  }
  return static_cast<T *>(nullptr);
}

template <typename T> T const *tensorCastOrNull(TensorConstPtr const &tensor) {
  if (tensor) {
    return tensorCast<T>(*tensor);
  }
  return static_cast<T const *>(nullptr);
}

inline auto copyTensorWrapper(TensorWrapper const &src, TensorWrapper &dst,
                              cudaMemcpyKind kind = cudaMemcpyDeviceToDevice) {
  cudaMemcpy(dst.data(), src.data(), src.getSizeInBytes(), kind);
}
