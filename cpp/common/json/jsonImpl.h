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

#include "jsmn.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace drivellm
{
class JsonNodeImpl
{
public:
    JsonNodeImpl() = default;
    JsonNodeImpl(std::string const& n, Jsmntok* t, char const* json);
    virtual ~JsonNodeImpl() = default;

    // Copy and move constructors and assignment operators are deleted.
    JsonNodeImpl(JsonNodeImpl const&) = delete;
    JsonNodeImpl(JsonNodeImpl&&) = delete;
    JsonNodeImpl& operator=(JsonNodeImpl const&) & = delete;
    JsonNodeImpl& operator=(JsonNodeImpl&&) & = delete;

    [[nodiscard]] size_t size() const noexcept;
    [[nodiscard]] std::string getName() const noexcept;
    [[nodiscard]] bool isObject() const noexcept;
    [[nodiscard]] bool isArray() const noexcept;
    JsonNodeImpl* operator[](size_t index);

    [[nodiscard]] bool isBool() const noexcept;
    [[nodiscard]] bool getBool() const;
    [[nodiscard]] bool isInteger() const noexcept;
    [[nodiscard]] int64_t getInteger() const;
    [[nodiscard]] bool isFloat() const noexcept;
    [[nodiscard]] float getFloat() const;
    [[nodiscard]] bool isString() const noexcept;
    [[nodiscard]] std::string getString() const;

protected:
    std::string mName;                                    // The name of the node
    Jsmntok* mToken{nullptr};                             // The handler in rapidjson
    char const* mJson{nullptr};                           // The original JSON string
    std::vector<std::unique_ptr<JsonNodeImpl>> mChildren; // The list of children nodes
};

// JSON object implementation, which is a special JSON node.
class JsonObjectImpl : public JsonNodeImpl
{
public:
    // Inherit all constructors from JsonNodeImpl
    using JsonNodeImpl::JsonNodeImpl;
    ~JsonObjectImpl() override = default;

    JsonObjectImpl(JsonObjectImpl const&) = delete;
    JsonObjectImpl(JsonObjectImpl&&) = delete;
    JsonObjectImpl& operator=(JsonObjectImpl const&) & = delete;
    JsonObjectImpl& operator=(JsonObjectImpl&&) & = delete;

    virtual JsonNodeImpl* insertNode(std::string const& name, Jsmntok* node);
    JsonNodeImpl* operator[](std::string const& name);
    bool hasMember(std::string const& name) const;

private:
    // The map from the name to its index in children list.
    std::unordered_map<std::string, size_t> mKeyIndex;
};

class JsonArrayImpl : public JsonNodeImpl
{
public:
    ~JsonArrayImpl() override = default;
    JsonArrayImpl(std::string const& n, Jsmntok* t, char const* j);
    virtual JsonNodeImpl* insertNode(std::string const& name, Jsmntok* node);
};

class JsonRootImpl : public JsonObjectImpl
{
public:
    JsonRootImpl() = default;
    ~JsonRootImpl() override = default;

    // Copy and move constructors and assignment operators are deleted.
    JsonRootImpl(JsonRootImpl const&) = delete;
    JsonRootImpl& operator=(JsonRootImpl const&) & = delete;
    JsonRootImpl(JsonRootImpl&&) = delete;
    JsonRootImpl& operator=(JsonRootImpl&&) & = delete;

    bool parse(std::string const& text);
    bool parseFromPath(std::string const& filePath);

private:
    // Create our JSON nodes via walking through the third-party JSON tree. Thus
    // we will not allocate JSON nodes during the later usage.
    // Return the total nodes (tokens) of the tree.
    // Return zero if an error is encountered.
    int32_t createTree(JsonNodeImpl* parent, char const* name, Jsmntok* node);
    JsmnParser mParser;
    std::vector<Jsmntok> mTokens;
    std::string mSource;
};

} // namespace drivellm
