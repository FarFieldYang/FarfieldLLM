#pragma once

#include <vector>
#include <string>
#include <utility>
#include <cstdint>
#include <functional>

using TokenId = int;
using TokenIds = std::vector<TokenId>;
using TokenPair = std::pair<TokenId, TokenId>;
constexpr TokenPair NO_PAIR = {-1, -1};
using Byte = unsigned char;
using Bytes = std::vector<Byte>;

struct PairHash {
    std::size_t operator()(const TokenPair& pair) const noexcept {
        std::uint64_t key =
            (static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(pair.first)
            ) << 32)
            |
            static_cast<std::uint32_t>(pair.second);
        return std::hash<std::uint64_t>{}(key);
    }
};

class Tokenizer {
public:
    virtual TokenIds encode(const std::string& text) = 0;
    virtual std::string decode(const TokenIds& tokens) const = 0;

    virtual ~Tokenizer() = default;
};

