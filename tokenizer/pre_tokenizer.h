#pragma once

#include <string>
#include <vector>
#include <cstddef>
#include <unordered_map>

using Pieces = std::vector<std::string>;
using CountedPieces = std::unordered_map<std::string, std::size_t>;

//only support English-like text with whitespace-separated text.
inline const std::string PATTERN = R"('s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+)";

Pieces pretokenize(const std::string& text);

CountedPieces pretokenize_and_count(const std::string& text);