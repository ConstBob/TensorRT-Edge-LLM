#include "tensor.h"

#include "common.h"

using namespace nvinfer1;

namespace drivellm
{
namespace rt
{

namespace
{

size_t getTypeSize(DataType dataType)
{
    size_t size{0};
    switch (dataType)
    {
    case DataType::kINT64:
    {
        size = 8;
        break;
    }
    case DataType::kFLOAT:
    case DataType::kINT32:
    {
        size = 4;
        break;
    }
    case DataType::kHALF:
    case DataType::kBF16:
    {
        size = 2;
        break;
    }
    case DataType::kFP8:
    case DataType::kINT8:
    case DataType::kUINT8:
    {
        size = 1;
        break;
    }
    default:
    {
        throw std::runtime_error("Other types are not supported");
    }
    }
    return size;
}

std::array<int64_t, kMAX_DIMS> computeStrides(Coords const& shape)
{
    std::array<int64_t, kMAX_DIMS> strides;
    int32_t const numDims = shape.getNumDims();
    strides[numDims - 1] = 1;
    int64_t stride = 1;
    for (int32_t i = numDims - 2; i >= 0; --i)
    {
        stride *= shape[i + 1];
        strides[i] = stride;
    }
    return strides;
}
} // namespace

Tensor::Tensor(Coords const& shape, DeviceType deviceType, nvinfer1::DataType dataType)
{
    if (shape.volume() == 0)
    {
        throw std::runtime_error("Construction of Tensor object with zero volume is prohibited");
    }

    if (dataType == DataType::kINT4 || dataType == DataType::kFP4)
    {
        throw std::runtime_error("Sub-type like kInt4 or kFP4 are not supported");
    }

    mShape = shape;
    mDeviceType = deviceType;
    mDataType = dataType;
    ownMemory = true;
    mStrides = computeStrides(shape);

    memoryCapacity = shape.volume() * getTypeSize(dataType);
    if (deviceType == DeviceType::kCPU)
    {
        data = malloc(memoryCapacity);
        if (data == nullptr)
        {
            throw std::runtime_error("Failed to allocate memory on CPU");
        }
    }
    else
    {
        CUDA_CHECK(cudaMalloc(&data, memoryCapacity));
    }
}

Tensor::Tensor(void* data, Coords const& shape, DeviceType deviceType, nvinfer1::DataType dataType) noexcept
{
    // Populate the tensor information and only serve as a data container with shape.
    mShape = shape;
    mDeviceType = deviceType;
    mDataType = dataType;
    ownMemory = false;
    mStrides = computeStrides(shape);
    this->data = data;
    memoryCapacity = shape.volume() * getTypeSize(dataType);
}

Tensor::~Tensor()
{
    if (ownMemory)
    {
        if (mDeviceType == DeviceType::kCPU)
        {
            free(data);
        }
        else
        {
            CUDA_CHECK(cudaFree(data));
        }
        data = nullptr;
    }
}

Tensor::Tensor(Tensor&& other) noexcept
{
    this->data = other.data;
    this->mShape = other.mShape;
    this->mStrides = other.mStrides;
    this->mDeviceType = other.mDeviceType;
    this->mDataType = other.mDataType;
    this->ownMemory = other.ownMemory;
    this->memoryCapacity = other.memoryCapacity;

    // Reset the other tensor.
    other.data = nullptr;
    other.mShape = Coords{};
    other.mStrides = std::array<int64_t, kMAX_DIMS>{};
    other.mDeviceType = DeviceType::kCPU;
    other.mDataType = DataType::kFLOAT;
    other.ownMemory = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    if (this != &other)
    {
        this->~Tensor();
        this->data = other.data;
        this->mShape = other.mShape;
        this->mStrides = other.mStrides;
        this->mDeviceType = other.mDeviceType;
        this->mDataType = other.mDataType;
        this->ownMemory = other.ownMemory;
        this->memoryCapacity = other.memoryCapacity;

        // Reset the other tensor.
        other.data = nullptr;
        other.mShape = {};
        other.mStrides = {};
        other.mDeviceType = DeviceType::kCPU;
        other.mDataType = DataType::kFLOAT;
        other.ownMemory = false;
    }
    return *this;
}

Coords Tensor::getShape() const noexcept
{
    return mShape;
}

DeviceType Tensor::getDeviceType() const noexcept
{
    return mDeviceType;
}

DataType Tensor::getDataType() const noexcept
{
    return mDataType;
}

bool Tensor::getOwnMemory() const noexcept
{
    return ownMemory;
}

int64_t Tensor::getMemoryCapacity() const noexcept
{
    return memoryCapacity;
}

int64_t Tensor::getStride(int32_t idx) const
{
    if (idx < 0 || idx >= mShape.getNumDims())
    {
        throw std::out_of_range("Tensor: indexing of strides out of range");
    }
    return mStrides[idx];
}

Dims Tensor::getTRTDims() const noexcept
{
    Dims dims;
    dims.nbDims = mShape.getNumDims();
    for (int32_t i = 0; i < mShape.getNumDims(); ++i)
    {
        dims.d[i] = mShape[i];
    }
    return dims;
}

void* Tensor::rawPointer() noexcept
{
    return data;
}

void const* Tensor::rawPointer() const noexcept
{
    return data;
}

bool Tensor::reshape(Coords shape) noexcept
{
    if (!ownMemory)
    {
        return false;
    }

    if (shape.volume() * getTypeSize(mDataType) > memoryCapacity)
    {
        return false;
    }

    mShape = shape;
    mStrides = computeStrides(shape);
    return true;
}

} // namespace rt
} // namespace drivellm