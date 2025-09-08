/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <NvInferRuntime.h>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>

namespace drivellm
{
namespace plugins
{
constexpr int32_t kDEVICE_ALIGNMENT{128}; // Make sure all device pointers are aligned by 128.

inline int8_t* alignDevicePtr(void* ptr)
{
    // Convert the pointer to an integer
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned_addr = (addr + kDEVICE_ALIGNMENT - 1) & ~static_cast<uintptr_t>(kDEVICE_ALIGNMENT - 1);

    // Convert the aligned address back to a pointer
    return reinterpret_cast<int8_t*>(aligned_addr);
}

template <typename T>
nvinfer1::PluginFieldType toFieldType();
#define SPECIALIZE_TO_FIELD_TYPE(T, type)                                                                              \
    template <>                                                                                                        \
    inline nvinfer1::PluginFieldType toFieldType<T>()                                                                  \
    {                                                                                                                  \
        return nvinfer1::PluginFieldType::type;                                                                        \
    }
SPECIALIZE_TO_FIELD_TYPE(float, kFLOAT32)
SPECIALIZE_TO_FIELD_TYPE(int32_t, kINT32)
#undef SPECIALIZE_TO_FIELD_TYPE

template <typename T>
inline std::optional<T> parsePluginScalarField(std::string const& fieldName, nvinfer1::PluginFieldCollection const* fc)
{
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        nvinfer1::PluginField const& pluginField = fc->fields[i];
        if (fieldName.compare(pluginField.name) == 0)
        {
            assert(toFieldType<T>() == pluginField.type && "Mismatch datatype of plugin field");
            assert(pluginField.length == 1 && pluginField.data != nullptr && "Invalid plugin field");
            return std::optional{*static_cast<T const*>(pluginField.data)};
        }
    }

    return std::nullopt;
}

template <typename T, class Enable = void>
struct Serializer
{
};

template <typename T>
struct Serializer<T, typename std::enable_if_t<std::is_arithmetic_v<T> || std::is_enum_v<T>>>
{
    static void serialize(void** buffer, T const& value)
    {
        ::memcpy(*buffer, &value, sizeof(T));
        reinterpret_cast<char*&>(*buffer) += sizeof(T);
    }
    static void deserialize(void const** buffer, size_t* buffer_size, T* value)
    {
        assert(*buffer_size >= sizeof(T));
        ::memcpy(value, *buffer, sizeof(T));
        reinterpret_cast<char const*&>(*buffer) += sizeof(T);
        *buffer_size -= sizeof(T);
    }
};

template <typename T>
inline void serializeValue(void** buffer, T const& value)
{
    return Serializer<T>::serialize(buffer, value);
}

template <typename T>
inline void deserializeValue(void const** buffer, size_t* buffer_size, T* value)
{
    return Serializer<T>::deserialize(buffer, buffer_size, value);
}

} // namespace plugins
} // namespace drivellm