#pragma once

#include <string>
#include <vector>

#include "tokenizer.h"

class ByteTokenizer : public Tokenizer {
public:
    TokenIds encode(const std::string& text) override;
    std::string decode(const std::vector<int>& tokens) const override;
};