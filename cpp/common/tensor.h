#pragma once

#include <NvInferRuntime.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace drivellm
{
namespace rt
{

enum class DeviceType
{
    kCPU = 0,
    kGPU = 1,
};

constexpr int32_t kMAX_DIMS = 8;

// Extend the is_arithmetic trait to support fp16 and bfloat16 for convenience.
template <typename T>
struct is_arithmetic_ext : std::is_arithmetic<T>
{
};
template <>
struct is_arithmetic_ext<half> : std::true_type
{
};
template <>
struct is_arithmetic_ext<__nv_bfloat16> : std::true_type
{
};

//! Array of dimensions that used to store the shape of a tensor.
//! All dimensions shall be non-negative.
class Coords
{
public:
    Coords() = default;
    Coords(Coords const& coords) noexcept
        : mDims(coords.mDims)
        , mNumDims(coords.mNumDims)
    {
    }
    Coords(Coords&& coords) noexcept = default;
    Coords& operator=(Coords const& coords) noexcept = default;
    Coords& operator=(Coords&& coords) noexcept = default;

    Coords(nvinfer1::Dims const& dims)
        : mNumDims(dims.nbDims)
    {
        std::copy(dims.d, dims.d + mNumDims, mDims.begin());
    }

    template <typename IT>
    Coords(IT begin, IT end)
        : mNumDims(std::distance(begin, end))
    {
        if (mNumDims > kMAX_DIMS)
        {
            throw std::runtime_error("Coords: number of dimensions out of range");
        }
        std::copy(begin, end, mDims.begin());
    }

    Coords(std::initializer_list<int64_t> init)
        : Coords(init.begin(), init.end())
    {
    }

    Coords(std::vector<int64_t> const& vec)
        : Coords(vec.begin(), vec.end())
    {
    }

    int32_t getNumDims() const noexcept
    {
        return mNumDims;
    }

    int64_t& operator[](int32_t idx)
    {
        if (idx < 0 || idx >= mNumDims)
        {
            throw std::out_of_range("Coords: index out of range");
        }
        return mDims[idx];
    }

    int64_t operator[](int32_t idx) const
    {
        if (idx < 0 || idx >= mNumDims)
        {
            throw std::out_of_range("Coords: index out of range");
        }
        return mDims[idx];
    }

    int64_t volume() const
    {
        if (mNumDims == 0)
        {
            return 0;
        }
        int64_t vol = 1;
        for (int32_t i = 0; i < mNumDims; ++i)
        {
            vol *= mDims[i];
        }
        return vol;
    }

    nvinfer1::Dims getTRTDims() const
    {
        nvinfer1::Dims dims;
        dims.nbDims = mNumDims;
        for (int32_t i = 0; i < mNumDims; ++i)
        {
            dims.d[i] = mDims[i];
        }
        return dims;
    }

private:
    std::array<int64_t, kMAX_DIMS> mDims{};
    int32_t mNumDims{0};
};

//! Tensor class that wrap linear layout tensor.
//! The underlying memory can either be owned by the tensor object or be reused from another allocation.
//! The Tensor Object support reshapes when memory is owned by the object and has sufficient capacity.
class Tensor
{
public:
    Tensor() = default;

    //! Disable copy constructor and assignment operator explicitly to enforce explicit
    //! memory ownership transfer.
    Tensor(Tensor const& other) = delete;
    Tensor& operator=(Tensor const& other) = delete;

    //! Allow transfer memory ownership through move constructor and assignment operator.
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    ~Tensor();

    //! Constructor that allocates memory on the specified device.
    //! The memory is owned by the tensor object and will be freed when the tensor object is destroyed.
    //! @param extent: The shape of the tensor (must has non-zero volume).
    //! @param deviceType: The device type to allocate memory on.
    //! @param dataType: The data type of the tensor. (sub-type like kInt4 or kE2M1 are not supported)
    Tensor(Coords const& extent, DeviceType deviceType, nvinfer1::DataType dataType);

    //! Constructor that reuses the memory of another tensor.
    //! Memory is indeed not owned by the tensor object, the caller should ensure the lifecycle of the memory.
    Tensor(void* data, Coords const& extent, DeviceType deviceType, nvinfer1::DataType dataType) noexcept;

    //! Getter methods for tensor attributes.
    Coords getShape() const noexcept;
    DeviceType getDeviceType() const noexcept;
    nvinfer1::DataType getDataType() const noexcept;
    nvinfer1::Dims getTRTDims() const noexcept;
    bool getOwnMemory() const noexcept;
    int64_t getMemoryCapacity() const noexcept;

    //! Get stride of the tensor.
    [[nodiscard]] int64_t getStride(int32_t idx) const;

    //! Get the data pointer of the tensor.
    void const* rawPointer() const noexcept;
    void* rawPointer() noexcept;

    //! Get typed data pointer of the tensor. Mismatching data type will
    //! lead to undefined behavior in the program.
    template <typename T>
    T* dataPointer() noexcept
    {
        if constexpr (!is_arithmetic_ext<T>::value)
        {
            static_assert(is_arithmetic_ext<T>::value, "Only arithmetic types are supported");
        }
        return reinterpret_cast<T*>(data);
    }

    template <typename T>
    T const* dataPointer() const noexcept
    {
        if constexpr (!is_arithmetic_ext<T>::value)
        {
            static_assert(is_arithmetic_ext<T>::value, "Only arithmetic types are supported");
        }
        return reinterpret_cast<T const*>(data);
    }

    //! Reshape the tensor for convenience of development.
    //! Explicitly disallow reshape when the memory is not owned by the tensor object to avoid misuse.
    //! Reshape will not happen when memory capacity is insufficient.
    [[nodiscard]] bool reshape(Coords extent) noexcept;

private:
    Coords mShape;
    std::array<int64_t, kMAX_DIMS> mStrides;
    DeviceType mDeviceType;
    nvinfer1::DataType mDataType;
    void* data{nullptr};
    bool ownMemory{false};

    // Determined once the tensor is constructed.
    int64_t memoryCapacity{0};
};

} // namespace rt
} // namespace drivellm