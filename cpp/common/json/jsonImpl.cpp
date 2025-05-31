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

#include "jsonImpl.h"
#include "common/json.h"
#include "common/logger.h"

#include <algorithm>
#include <cctype>

namespace drivellm
{

JsonNodeImpl::JsonNodeImpl(std::string const& n, Jsmntok* t, char const* json)
    : mName(n)
    , mToken(t)
    , mJson(json)
{
}

size_t JsonNodeImpl::size() const noexcept
{
    return mChildren.size();
}

std::string JsonNodeImpl::getName() const noexcept
{
    return mName;
}
bool JsonNodeImpl::isObject() const noexcept
{
    return mToken->type == kJSMN_OBJECT;
}

bool JsonNodeImpl::isArray() const noexcept
{
    return mToken->type == kJSMN_ARRAY;
}

JsonNodeImpl* JsonNodeImpl::operator[](size_t index)
{
    if (mChildren.size() <= index)
    {
        return nullptr;
    }
    return mChildren[index].get();
}

bool JsonNodeImpl::isBool() const noexcept
{
    if (mToken && mToken->type == kJSMN_PRIMITIVE)
    {
        int len = mToken->end - mToken->start;
        return (len == 4 && strncmp(mJson + mToken->start, "true", 4) == 0)
            || (len == 5 && strncmp(mJson + mToken->start, "false", 5) == 0);
    }
    return false;
}

bool JsonNodeImpl::getBool() const
{
    int len = mToken->end - mToken->start;
    if (len == 4 && strncmp(mJson + mToken->start, "true", 4) == 0)
    {
        return true;
    }
    if (len == 5 && strncmp(mJson + mToken->start, "false", 5) == 0)
    {
        return false;
    }
    // By default, false is returned.
    throw std::runtime_error("Invalid bool value");
}

bool JsonNodeImpl::isInteger() const noexcept
{
    if (!mToken || mToken->type != kJSMN_PRIMITIVE)
    {
        return false;
    }
    char const* start = mJson + mToken->start;
    if (*start == '-')
    {
        ++start;
    }
    return std::all_of(start, mJson + mToken->end, [](char c) { return std::isdigit(c); });
}

int64_t JsonNodeImpl::getInteger() const
{
    // By default, 0 is returned.
    int len = mToken->end - mToken->start;
    std::string intStr(mJson + mToken->start, len);
    return std::stoll(intStr);
}

bool JsonNodeImpl::isFloat() const noexcept
{
    if (!mToken || mToken->type != kJSMN_PRIMITIVE)
    {
        return false;
    }

    int32_t dot = 0;
    int32_t exp = 0;

    char const* start = mJson + mToken->start;
    if (*start == '-')
    {
        ++start;
    }
    char const* p = start;
    while (p != mJson + mToken->end)
    {
        if (*p == '.')
        {
            ++dot;
            // At most one dot is allowed and the dot must be before 'e' or 'E'.
            if (dot > 1 || exp > 0)
            {
                return false;
            }
        }
        else if (*p == 'e' || *p == 'E')
        {
            ++exp;
            if (exp > 1)
            {
                return false;
            }
            // Immediately after 'e' or 'E', the character can be digit or '+' or '-'.
            char const* next = p + 1;
            if (next == mJson + mToken->end)
            {
                return false;
            }
            if (*next == '+' || *next == '-')
            {
                ++p;
            }
        }
        else if (!std::isdigit(*p))
        {
            return false;
        }
        ++p;
    }
    return true;
}

float JsonNodeImpl::getFloat() const
{
    // By default, 0.0 is returned.
    int len = mToken->end - mToken->start;
    std::string floatStr(mJson + mToken->start, len);
    return std::stof(floatStr);
}

bool JsonNodeImpl::isString() const noexcept
{
    return mToken && mToken->type == kJSMN_STRING;
}

std::string JsonNodeImpl::getString() const
{
    if (!mToken)
    {
        // For root
        return "";
    }
    return std::string(mJson + mToken->start, mToken->end - mToken->start);
}

std::unique_ptr<JsonNodeImpl> newJsonImpl(std::string const& name, Jsmntok* node, char const* json)
{
    switch (node->type)
    {
    case kJSMN_OBJECT: return std::make_unique<JsonObjectImpl>(name, node, json);
    case kJSMN_ARRAY: return std::make_unique<JsonArrayImpl>(name, node, json);
    default: return std::make_unique<JsonNodeImpl>(name, node, json);
    }
}

JsonNodeImpl* JsonObjectImpl::insertNode(std::string const& name, Jsmntok* node)
{
    mChildren.push_back(newJsonImpl(name, node, mJson));
    mKeyIndex[name] = mChildren.size() - 1;
    return mChildren.back().get();
};

JsonNodeImpl* JsonObjectImpl::operator[](std::string const& name)
{
    JsonNodeImpl* result{nullptr};
    if (hasMember(name))
    {
        result = mChildren[mKeyIndex[name]].get();
    }
    return result;
}

bool JsonObjectImpl::hasMember(std::string const& name) const
{
    return mKeyIndex.count(name);
}

// JSON array implementation, which is a special JSON node.
JsonArrayImpl::JsonArrayImpl(std::string const& n, Jsmntok* t, char const* j)
    : JsonNodeImpl(n, t, j)
{
}
JsonNodeImpl* JsonArrayImpl::insertNode(std::string const& name, Jsmntok* node)
{
    mChildren.push_back(newJsonImpl(name, node, mJson));
    return mChildren.back().get();
};

bool JsonRootImpl::parse(std::string const& text)
{
    mSource = text;
    mJson = mSource.c_str();
    jsmn_init(&mParser);

    int const INITIAL_TOKEN_NUM = 65536;
    mTokens.resize(INITIAL_TOKEN_NUM);
    while (true)
    {
        int ret = jsmn_parse(&mParser, text.c_str(), text.size(), mTokens.data(), mTokens.size());
        if (ret >= 0)
        {
            mTokens.resize(ret);
            break;
        }
        if (ret == kJSMN_ERROR_NOMEM)
        {
            mTokens.resize(mTokens.size() * 2);
        }
        else if (ret == kJSMN_ERROR_INVAL)
        {
            return false;
        }
        else
        {
            return false;
        }
    }
    int count = 1;
    Jsmntok* root = &mTokens[0];
    mToken = root;

    for (int i = 0; i < root->size; ++i)
    {
        Jsmntok* key = root + count;
        if (key->type != kJSMN_STRING)
        {
            LOG_ERROR("JSON key is not string");
            return false;
        }

        std::string childName(mJson + key->start, key->end - key->start);
        ++count;
        int nbNodes = createTree(this, childName.c_str(), root + count);
        if (nbNodes == 0)
        {
            return false;
        }
        count += nbNodes;
    }
    return true;
}

// Create our JSON nodes via walking through the third-party JSON tree. Thus
// we will not allocate JSON nodes during the later usage.
// Return the total nodes (tokens) of the tree.
// Return zero if an error is encountered.
int32_t JsonRootImpl::createTree(JsonNodeImpl* parent, char const* name, Jsmntok* node)
{
    int count = 1;
    JsonNodeImpl* current = parent->isObject() ? static_cast<JsonObjectImpl*>(parent)->insertNode(name, node)
                                               : static_cast<JsonArrayImpl*>(parent)->insertNode("", node);
    if (node->type == kJSMN_OBJECT)
    {
        for (int i = 0; i < node->size; ++i)
        {
            Jsmntok* key = node + count;
            if (key->type != kJSMN_STRING)
            {
                LOG_ERROR("JSON key is not string");
                return 0;
            }

            std::string childName(mJson + key->start, key->end - key->start);
            ++count;
            int nbNodes = createTree(current, childName.c_str(), node + count);
            if (nbNodes == 0)
            {
                return nbNodes;
            }
            count += nbNodes;
        }
    }
    else if (node->type == kJSMN_ARRAY)
    {
        for (int i = 0; i < node->size; ++i)
        {
            int nbNodes = createTree(current, "", node + count);
            if (nbNodes == 0)
            {
                return nbNodes;
            }
            count += nbNodes;
        }
    }
    return count;
}

// Public interface for JsonNode and JsonRoot
// JsonNode methods
std::string JsonNode::getName() const
{
    return mImpl->getName();
}

size_t JsonNode::size() const
{
    return mImpl->size();
}

bool JsonNode::isObject() const
{
    return mImpl->isObject();
}

bool JsonNode::isArray() const
{
    return mImpl->isArray();
}

bool JsonNode::hasMember(std::string const& name) const
{
    if (!mImpl->isObject())
    {
        return false;
    }
    JsonObjectImpl* obj = static_cast<JsonObjectImpl*>(mImpl);
    return (*obj).hasMember(name);
}

JsonNode JsonNode::operator[](std::string const& key)
{
    if (!mImpl->isObject())
    {
        return JsonNode();
    }
    JsonObjectImpl* obj = static_cast<JsonObjectImpl*>(mImpl);
    return JsonNode(obj->operator[](key));
}

JsonNode JsonNode::operator[](size_t index)
{
    return JsonNode((*mImpl)[index]);
}

bool JsonNode::isBool() const
{
    return mImpl->isBool();
}

bool JsonNode::getBool() const
{
    return mImpl->getBool();
}

bool JsonNode::isInteger() const
{
    return mImpl->isInteger();
}

int64_t JsonNode::getInteger() const
{
    return mImpl->getInteger();
}

bool JsonNode::isFloat() const
{
    return mImpl->isFloat();
}

float JsonNode::getFloat() const
{
    return mImpl->getFloat();
}

bool JsonNode::isString() const
{
    return mImpl->isString();
}

std::string JsonNode::getString() const
{
    return mImpl->getString();
}

JsonRoot::~JsonRoot()
{
    destroy();
}

void JsonRoot::destroy()
{
    delete mImpl;
}

bool JsonRoot::parse(std::string const& text)
{
    if (mImpl != nullptr)
    {
        LOG_WARNING("JsonRoot is already parsed and contains a JSON tree.");
        return false;
    }
    mImpl = new JsonRootImpl();
    return mImpl->parse(text);
}

bool JsonRoot::parseFromPath(std::string const& filePath)
{
    // Open the file in binary mode to prevent CRLF translation issues on Windows
    // and ensure tellg() reports the correct byte size.
    if (!std::filesystem::exists(filePath))
    {
        throw std::runtime_error("Could not find file: " + filePath);
    }
    std::ifstream inputFileStream(filePath, std::ios::binary | std::ios::ate);

    if (!inputFileStream.is_open())
    {
        throw std::runtime_error("Could not open file: " + filePath);
    }

    std::streamsize fileSize = inputFileStream.tellg();
    inputFileStream.seekg(0, std::ios::beg);

    if (fileSize == -1)
    {
        inputFileStream.close();
        throw std::runtime_error("Could not determine file size: " + filePath);
    }

    std::string content;
    if (fileSize > 0)
    {
        content.resize(static_cast<std::string::size_type>(fileSize));

        if (!inputFileStream.read(&content[0], fileSize))
        {
            inputFileStream.close();
            throw std::runtime_error("Could not read file into string: " + filePath);
        }
    }

    inputFileStream.close();
    return parse(content);
}

JsonNode JsonRoot::getRoot() const
{
    auto rootNode = JsonNode(mImpl);
    if (!rootNode.isObject())
    {
        throw std::runtime_error("Root node is not an object");
    }
    return rootNode;
}

} // namespace drivellm