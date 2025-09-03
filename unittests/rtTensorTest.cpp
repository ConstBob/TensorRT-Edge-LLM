/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "common/checkMacros.h"
#include "common/tensor.h"

using namespace drivellm;

TEST(TensorTest, Coords)
{
    // Verify initializ(ed list constructor.
    rt::Coords const extent({2, 3, 4});
    ASSERT_EQ(extent.volume(), 24);
    ASSERT_EQ(extent.getNumDims(), 3);
    ASSERT_EQ(extent[0], 2);
    ASSERT_EQ(extent[1], 3);
    ASSERT_EQ(extent[2], 4);

    // Verify vector constructor.
    rt::Coords extent2{std::vector<int64_t>{5, 6, 7, 8}};
    ASSERT_EQ(extent2.volume(), 1680);
    ASSERT_EQ(extent2.getNumDims(), 4);

    // Verify [] operator.
    extent2[2] = 5;
    extent2[3] = 6;
    ASSERT_EQ(extent2.volume(), 900);

    // Verify nvinfer1::Dims constructor.
    nvinfer1::Dims dims;
    dims.nbDims = 3;
    dims.d[0] = 7;
    dims.d[1] = 8;
    dims.d[2] = 9;
    rt::Coords extent3{dims};
    ASSERT_EQ(extent3.volume(), 504);
    ASSERT_EQ(extent3.getNumDims(), 3);
    ASSERT_EQ(extent3[0], 7);
    ASSERT_EQ(extent3[1], 8);
    ASSERT_EQ(extent3[2], 9);

    // Verify getTRTDims.
    auto const trtDims = extent3.getTRTDims();
    ASSERT_EQ(trtDims.nbDims, 3);
    ASSERT_EQ(trtDims.d[0], 7);
    ASSERT_EQ(trtDims.d[1], 8);
    ASSERT_EQ(trtDims.d[2], 9);
}

TEST(TensorTest, CoordsNegative)
{
    // Verify out of bounds access.
    rt::Coords extent({2, 3, 4});
    ASSERT_THROW(extent[3], std::out_of_range);
    ASSERT_THROW(extent[-1], std::out_of_range);
    ASSERT_THROW(extent[100], std::out_of_range);

    // Verify out of bounds constructor.
    std::vector<int64_t> vec(10, 5);
    ASSERT_THROW(rt::Coords extent2(vec), std::runtime_error);
}

TEST(TensorTest, HostTensorOwnMemory)
{
    rt::Tensor tensor({2, 3, 4}, rt::DeviceType::kCPU, nvinfer1::DataType::kFLOAT);
    ASSERT_TRUE(tensor.getOwnMemory());
    ASSERT_EQ(tensor.getDeviceType(), rt::DeviceType::kCPU);
    ASSERT_EQ(tensor.getDataType(), nvinfer1::DataType::kFLOAT);

    // Memory capacity is initial volume * sizeof(float)
    ASSERT_EQ(tensor.getMemoryCapacity(), 96);

    nvinfer1::Dims dims = tensor.getTRTDims();
    ASSERT_EQ(dims.nbDims, 3);
    ASSERT_EQ(dims.d[0], 2);
    ASSERT_EQ(dims.d[1], 3);
    ASSERT_EQ(dims.d[2], 4);

    ASSERT_EQ(tensor.getStride(0), 12);
    ASSERT_EQ(tensor.getStride(1), 4);
    ASSERT_EQ(tensor.getStride(2), 1);

    ASSERT_NE(tensor.rawPointer(), nullptr);
    ASSERT_NE(tensor.dataPointer<float>(), nullptr);

    // Verify Identical reshape
    ASSERT_TRUE(tensor.reshape({4, 3, 2}));
    ASSERT_EQ(tensor.getMemoryCapacity(), 96);
    ASSERT_EQ(tensor.getStride(0), 6);
    ASSERT_EQ(tensor.getStride(1), 2);
    ASSERT_EQ(tensor.getStride(2), 1);
    rt::Coords tensorExtent = tensor.getShape();
    ASSERT_EQ(tensorExtent[0], 4);
    ASSERT_EQ(tensorExtent[1], 3);
    ASSERT_EQ(tensorExtent[2], 2);

    // Verify non-identical reshape
    ASSERT_TRUE(tensor.reshape({4, 3}));
    // Capacity keeps unchanged
    ASSERT_EQ(tensor.getMemoryCapacity(), 96);
    ASSERT_EQ(tensor.getStride(0), 3);
    ASSERT_EQ(tensor.getStride(1), 1);
    tensorExtent = tensor.getShape();
    ASSERT_EQ(tensorExtent[0], 4);
    ASSERT_EQ(tensorExtent[1], 3);

    // Verify out of range stride access
    ASSERT_THROW(tensor.getStride(3), std::out_of_range);
    ASSERT_THROW(tensor.getStride(-1), std::out_of_range);

    // Expect reshape to fail when reshape to shapes out of capacity
    ASSERT_FALSE(tensor.reshape({4, 4, 4}));
}

TEST(TensorTest, DeviceTensorOwnMemory)
{
    rt::Tensor tensor({8, 16, 32}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    ASSERT_TRUE(tensor.getOwnMemory());
    ASSERT_EQ(tensor.getDeviceType(), rt::DeviceType::kGPU);
    ASSERT_EQ(tensor.getDataType(), nvinfer1::DataType::kHALF);

    // Memory capacity is initial volume * sizeof(half)
    ASSERT_EQ(tensor.getMemoryCapacity(), 8192);

    nvinfer1::Dims dims = tensor.getTRTDims();
    ASSERT_EQ(dims.nbDims, 3);
    ASSERT_EQ(dims.d[0], 8);
    ASSERT_EQ(dims.d[1], 16);
    ASSERT_EQ(dims.d[2], 32);

    ASSERT_EQ(tensor.getStride(0), 512);
    ASSERT_EQ(tensor.getStride(1), 32);
    ASSERT_EQ(tensor.getStride(2), 1);

    half* data = tensor.dataPointer<half>();

    // Verify CUDA calls can succeed on the memory pointer
    CUDA_CHECK(cudaMemset(data, 0, tensor.getMemoryCapacity()));

    // Move assignment of this tensor object.
    rt::Tensor tensor2 = std::move(tensor);
    ASSERT_EQ(tensor2.getDeviceType(), rt::DeviceType::kGPU);
    ASSERT_EQ(tensor2.getDataType(), nvinfer1::DataType::kHALF);
    ASSERT_EQ(tensor2.getMemoryCapacity(), 8192);
    ASSERT_EQ(tensor2.getOwnMemory(), true);
    ASSERT_EQ(tensor2.getShape().volume(), 4096);

    // New tensor own the memory while original tensor is empty.
    ASSERT_NE(tensor2.dataPointer<half>(), nullptr);
    ASSERT_EQ(tensor.dataPointer<half>(), nullptr);
    ASSERT_EQ(tensor.getOwnMemory(), false);
    ASSERT_EQ(tensor.getShape().volume(), 0);
}

TEST(TensorTest, DeviceTensorNonOwnMemory)
{
    void* devicePtr{nullptr};
    CUDA_CHECK(cudaMallocAsync(&devicePtr, 512, 0));

    rt::Tensor tensor(devicePtr, {4, 8, 16}, rt::DeviceType::kGPU, nvinfer1::DataType::kFP8);
    ASSERT_FALSE(tensor.getOwnMemory());
    ASSERT_EQ(tensor.getDeviceType(), rt::DeviceType::kGPU);
    ASSERT_EQ(tensor.getDataType(), nvinfer1::DataType::kFP8);

    // Memory capacity is initial volume * sizeof(fp8)
    ASSERT_EQ(tensor.getMemoryCapacity(), 512);

    nvinfer1::Dims dims = tensor.getTRTDims();
    ASSERT_EQ(dims.nbDims, 3);
    ASSERT_EQ(dims.d[0], 4);
    ASSERT_EQ(dims.d[1], 8);
    ASSERT_EQ(dims.d[2], 16);

    ASSERT_EQ(tensor.getStride(0), 128);
    ASSERT_EQ(tensor.getStride(1), 16);
    ASSERT_EQ(tensor.getStride(2), 1);

    // Reshape on a non-own memory tensor should fail even if the new shape is within capacity.
    ASSERT_FALSE(tensor.reshape({4, 4, 16}));

    // Move construction of a new tensor object.
    rt::Tensor tensor2(std::move(tensor));

    // New tensor does not own the memory. Old tensor is empty.
    ASSERT_EQ(tensor.getOwnMemory(), false);
    ASSERT_EQ(tensor2.getOwnMemory(), false);

    ASSERT_NE(tensor2.rawPointer(), nullptr);
    ASSERT_EQ(tensor.rawPointer(), nullptr);

    // Clean up the manually allocated memory
    CUDA_CHECK(cudaFreeAsync(devicePtr, 0));
}

TEST(TensorTest, TensorNameFunctionality)
{
    // Test tensor with name
    rt::Tensor namedTensor({2, 3}, rt::DeviceType::kCPU, nvinfer1::DataType::kFLOAT, "test_tensor");
    ASSERT_EQ(namedTensor.getName(), "test_tensor");

    // Test tensor with empty name (default)
    rt::Tensor unnamedTensor({2, 3}, rt::DeviceType::kCPU, nvinfer1::DataType::kFLOAT);
    ASSERT_EQ(unnamedTensor.getName(), "");

    // Test tensor with explicit empty name
    rt::Tensor explicitEmptyTensor({2, 3}, rt::DeviceType::kCPU, nvinfer1::DataType::kFLOAT, "");
    ASSERT_EQ(explicitEmptyTensor.getName(), "");

    // Test tensor with long name
    std::string longName = "very_long_tensor_name_with_many_characters_for_testing_purposes";
    rt::Tensor longNamedTensor({1, 1}, rt::DeviceType::kCPU, nvinfer1::DataType::kFLOAT, longName);
    ASSERT_EQ(longNamedTensor.getName(), longName);

    // Test non-own memory tensor with name
    void* devicePtr{nullptr};
    CUDA_CHECK(cudaMallocAsync(&devicePtr, 24, 0));
    rt::Tensor nonOwnNamedTensor(devicePtr, {2, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "non_own_tensor");
    ASSERT_EQ(nonOwnNamedTensor.getName(), "non_own_tensor");

    // Test move constructor preserves name
    rt::Tensor movedTensor = std::move(namedTensor);
    ASSERT_EQ(movedTensor.getName(), "test_tensor");
    ASSERT_EQ(namedTensor.getName(), ""); // Original tensor should have empty name after move

    // Test move assignment preserves name
    rt::Tensor assignedTensor({1, 1}, rt::DeviceType::kCPU, nvinfer1::DataType::kFLOAT, "original_name");
    assignedTensor = std::move(movedTensor);
    ASSERT_EQ(assignedTensor.getName(), "test_tensor");
    ASSERT_EQ(movedTensor.getName(), ""); // Moved tensor should have empty name after assignment

    // Clean up the manually allocated memory
    CUDA_CHECK(cudaFreeAsync(devicePtr, 0));
}
