#pragma once

#include <cmath>
#include <vector>
#include <random>
#include <ostream>

template <typename T>
bool isclose(T a, T b, float rtol, float atol) {
    float af = static_cast<float>(a);
    float bf = static_cast<float>(b);
    return fabs(af - bf) <= (atol + rtol * fabs(bf));
}

template <typename T>
void uniformFloatinitialization(std::vector<T>& vec, float a = -5.f, float b = 5.f) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_real_distribution d{a, b};
    for (size_t i = 0; i < vec.size(); ++i) 
    {
        vec[i] = T(d(gen));
    }
}

template <typename T>
void uniformIntInitialization(std::vector<T>& vec, int low, int high) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_int_distribution d{low, high};
    for (size_t i = 0; i < vec.size(); ++i) 
    {
        vec[i] = T(d(gen));
    }
}

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, std::ostream&>::type
operator<<(std::ostream& os, std::vector<T> const& vec) {
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
