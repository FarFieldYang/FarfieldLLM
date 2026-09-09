#include <regex>

#include "pre_tokenizer.h"

CountedPieces pretokenize_and_count(const std::string& text){
    std::regex pattern(PATTERN);
    CountedPieces pieces;
    for(std::sregex_iterator it(text.begin(), text.end(), pattern), end; it != end; ++it) 
        ++pieces[it->str()]; 
    return pieces;
}