#pragma once

#include <vector>
#include <string>

using TokenId = int;
using TokenIds = std::vector<TokenId>;
using TokenPair = std::pair<TokenId, TokenId>;
using Byte = unsigned char;
using Bytes = std::vector<Byte>;

class Tokenizer {
public:
    virtual std::vector<int> encode(const std::string& text) const = 0;
    virtual std::string decode(const std::vector<int>& tokens) const = 0;

    virtual ~Tokenizer() = default;
};

