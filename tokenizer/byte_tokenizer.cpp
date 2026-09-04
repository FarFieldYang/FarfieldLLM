#include "byte_tokenizer.h"

std::vector<int> ByteTokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    for(char c : text){
        tokens.push_back(static_cast<unsigned char>(c));
    }
    return tokens;
}

std::string ByteTokenizer::decode(const std::vector<int>& tokens) const {
    std::string text;
    for(int token : tokens){
        text.push_back(static_cast<char>(token));
    }
    return text;
}