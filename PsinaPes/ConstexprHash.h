#pragma once

#include <string_view>
#include <cstdint>
#include <type_traits>
#include <bit>

template<typename Type, typename ...Args>
constexpr Type BitCast(Args... args)
{
    std::common_type_t<Args...> array[] = { args... };
    return std::bit_cast<Type>(array);
}

constexpr std::uint64_t Hash(std::string_view str)
{
    constexpr std::uint64_t multiply = 0xc6a4a7935bd1e995ULL;
    constexpr std::uint64_t shift = 47ULL;
    constexpr std::uint64_t seed = 700924169573080812ULL;

    const std::size_t len = str.size();
    const std::size_t calclen = len ? len + 1 : 0;
    std::uint64_t hash = seed ^ (calclen * multiply);

    if (len > 0)
    {
        const auto* data = str.data();
        const auto first_loop_iterations = calclen / 8;

        for (size_t i = 0; i < first_loop_iterations; ++i)
        {
            std::uint64_t k = BitCast<std::uint64_t>(
                *data, *(data + 1), *(data + 2), *(data + 3),
                *(data + 4), *(data + 5), *(data + 6), *(data + 7));

            k *= multiply;
            k ^= k >> shift;
            k *= multiply;

            hash ^= k;
            hash *= multiply;
            data += 8;
        }

        const auto* data2 = str.data() + 8 * first_loop_iterations;

        switch (calclen & 7)
        {
        case 7: hash ^= static_cast<uint64_t>(data2[6]) << 48ULL; [[fallthrough]];
        case 6: hash ^= static_cast<uint64_t>(data2[5]) << 40ULL; [[fallthrough]];
        case 5: hash ^= static_cast<uint64_t>(data2[4]) << 32ULL; [[fallthrough]];
        case 4: hash ^= static_cast<uint64_t>(data2[3]) << 24ULL; [[fallthrough]];
        case 3: hash ^= static_cast<uint64_t>(data2[2]) << 16ULL; [[fallthrough]];
        case 2: hash ^= static_cast<uint64_t>(data2[1]) << 8ULL; [[fallthrough]];
        case 1: hash ^= static_cast<uint64_t>(data2[0]); [[fallthrough]];
            hash *= multiply;
        };
    }

    hash ^= hash >> shift;
    hash *= multiply;
    hash ^= hash >> shift;

    return hash;
}
