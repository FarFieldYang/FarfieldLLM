#pragma once

#include <string>
#include <vector>

class ByteTokenizer {
public:
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& tokens);
};