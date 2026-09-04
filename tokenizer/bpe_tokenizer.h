#pragma once

#include <string>
#include <vector>
#include "tokenizer.h"

class BPETokenizer : public Tokenizer {
public:
    std::vector<int> encode(const std::string& text) const override;
    std::string decode(const std::vector<int>& tokens) const override;
};