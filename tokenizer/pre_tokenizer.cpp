#include <regex>

#include "pre_tokenizer.h"

Pieces pretokenize(const std::string& text){
    std::regex pattern(PATTERN);
    Pieces pieces;
    for(std::sregex_iterator it(text.begin(), text.end(), pattern), end; it != end; ++it) 
        pieces.push_back(it->str()); 
    return pieces;
}

CountedPieces pretokenize_and_count(const std::string& text){
    std::regex pattern(PATTERN);
    CountedPieces pieces;
    for(std::sregex_iterator it(text.begin(), text.end(), pattern), end; it != end; ++it) 
        ++pieces[it->str()]; 
    return pieces;
}