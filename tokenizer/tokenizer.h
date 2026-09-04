#pragma once

#include <vector>
#include <string>

class Tokenizer {
public:
    virtual std::vector<int> encode(const std::string& text) const = 0;
    virtual std::string decode(const std::vector<int>& tokens) const = 0;

    virtual ~Tokenizer() = default;
};