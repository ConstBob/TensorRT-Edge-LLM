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

#pragma once

#include <memory>
#include <string>

namespace drivellm
{

class JsonNodeImpl;
class JsonRootImpl;

class JsonNode
{
public:
    JsonNode() {}
    JsonNode(JsonNodeImpl* jni)
        : mImpl(jni)
    {
    }
    JsonNode(std::unique_ptr<JsonNodeImpl>& jni)
        : mImpl(jni.get())
    {
    }
    // A member of an object has a name. The name of an array member is "".
    [[nodiscard]] std::string getName() const;
    // Whether the node is an object.
    [[nodiscard]] bool isObject() const;
    // Whether the node is an array.
    [[nodiscard]] bool isArray() const;
    // Number of children if the node is an object or an array.
    [[nodiscard]] size_t size() const;
    // Whether the object has a member
    [[nodiscard]] bool hasMember(std::string const& name) const;
    // Access a child via its name. For object only.
    JsonNode operator[](std::string const& name);
    // Access a child via its index.
    JsonNode operator[](size_t index);

    // Whether the node is a boolean.
    [[nodiscard]] bool isBool() const;
    // Get the boolean value.
    [[nodiscard]] bool getBool() const;
    // Whether the node is an integer.
    [[nodiscard]] bool isInteger() const;
    // Get the integer value.
    [[nodiscard]] int64_t getInteger() const;
    // Whether the node is a float.
    [[nodiscard]] bool isFloat() const;
    // Get the float value.
    [[nodiscard]] float getFloat() const;
    // Whether the node is a string.
    [[nodiscard]] bool isString() const;
    // Get the string.
    [[nodiscard]] std::string getString() const;

protected:
    JsonNodeImpl* mImpl{nullptr};
};

// Developer interface for JsonRoot in JSON format
class JsonRoot
{
public:
    JsonRoot() {}

    ~JsonRoot();

    // Manually destroy the contained JSON root object.
    // This call will destruct the whole JSON tree.
    void destroy();

    // Parse JSON in a string.
    bool parse(std::string const& text);
    // Get the root node of the JSON structure.
    [[nodiscard]] JsonNode getRoot() const;

private:
    // Use raw pointer intentionally to avoid introducing the whole JSMN header to
    // the other compilation unit. The impl will own the memory of the JSON tree.
    JsonRootImpl* mImpl{nullptr};
};

} // namespace drivellm
