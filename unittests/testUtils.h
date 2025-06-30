#pragma once

#include <cmath>
#include <ostream>
#include <random>
#include <vector>

template <typename T>
bool isclose(T a, T b, float rtol, float atol)
{
    float af = static_cast<float>(a);
    float bf = static_cast<float>(b);
    return fabs(af - bf) <= (atol + rtol * fabs(bf));
}

template <typename T>
void uniformFloatinitialization(std::vector<T>& vec, float a = -5.f, float b = 5.f)
{
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_real_distribution d{a, b};
    for (size_t i = 0; i < vec.size(); ++i)
    {
        vec[i] = T(d(gen));
    }
}

template <typename T>
void uniformIntInitialization(std::vector<T>& vec, int low, int high)
{
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_int_distribution d{low, high};
    for (size_t i = 0; i < vec.size(); ++i)
    {
        vec[i] = T(d(gen));
    }
}

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, std::ostream&>::type operator<<(
    std::ostream& os, std::vector<T> const& vec)
{
    os << "[";

    for (size_t i = 0; i < vec.size(); ++i)
    {
        // Print bools as 'true'/'false' instead of 1/0
        if constexpr (std::is_same_v<T, bool>)
        {
            os << std::boolalpha << vec[i];
        }
        else
        {
            os << vec[i];
        }
        if (i < vec.size() - 1)
        {
            os << ", ";
        }
    }

    os << "]";
    return os;
}

class KvCacheIndexer
{
public:
    KvCacheIndexer(
        int32_t const batchSize, int32_t const kvHeadNum, int32_t const kvCacheCapacity, int32_t const headSize)
    {
        mBatchSize = batchSize;
        mKvHeadNum = kvHeadNum;
        mKvCacheCapacity = kvCacheCapacity;
        mHeadSize = headSize;
    }

    int32_t indexK(int32_t const b, int32_t const hk, int32_t const cacheIdx, int32_t const d)
    {
        // Linear KVCache has layout of [B, 2, Hkv, S_capacity, D].
        return b * 2 * mKvHeadNum * mKvCacheCapacity * mHeadSize + hk * mKvCacheCapacity * mHeadSize
            + cacheIdx * mHeadSize + d;
    }

    int32_t indexV(int32_t const b, int32_t const hv, int32_t const cacheIdx, int32_t const d)
    {
        // Linear KVCache has layout of [B, 2, Hkv, S_capacity, D].
        // V cache need to offset the whole kCache buffer for the sequence.
        return b * 2 * mKvHeadNum * mKvCacheCapacity * mHeadSize + (mKvHeadNum + hv) * mKvCacheCapacity * mHeadSize
            + cacheIdx * mHeadSize + d;
    }

private:
    int32_t mBatchSize;
    int32_t mKvHeadNum;
    int32_t mKvCacheCapacity;
    int32_t mHeadSize;
};