#pragma once

#include <vector>
#include <string>
#include <utility>
#include <cstddef>
#include <unordered_set>
#include <queue>
#include <unordered_map>
#include <functional>
#include "bpe_model.h"
#include "pre_tokenizer.h"


struct TokenPiece{
    TokenIds ids;
    std::size_t frequency;
};
using TokenPieces = std::vector<TokenPiece>;
constexpr TokenPair NO_PAIR = {-1, -1};

using PieceId = std::size_t;
struct PairInfo{
    std::size_t count = 0;
    std::unordered_set<PieceId> pieces;
};
struct PairHash {
    std::size_t operator()(const TokenPair& pair) const noexcept {
        std::size_t h1 = std::hash<TokenId>{}(pair.first);
        std::size_t h2 = std::hash<TokenId>{}(pair.second);
        return h1 ^ (h2 << 1);
    }
};
using PairInfoMap = std::unordered_map<TokenPair, PairInfo, PairHash>;

using HeapEntry = std::pair<std::size_t, TokenPair>;
using MaxHeap = std::priority_queue<HeapEntry>;
struct Trainstate{
    PairInfoMap pair_info_map;
    MaxHeap heap;
};

class BPEtrainer {
public:
    BPEModel train(const std::string& text, std::size_t vocab_size);
    BPEModel train_file(const std::string& path, std::size_t vocab_size);
private:
    Vocab vocab_;
    MergeRules merges_;
    Trainstate state_;

    static Vocab init_vocab();
    static CountedPieces pre_tokenize(const std::string& text);
    static TokenPieces text_to_token(const CountedPieces& textpieces);
    void train_merges(TokenPieces& tokenpieces, std::size_t vocab_size);
    void id_to_bytes(const TokenPair& pair);

//train_merge() helper function
    void init_state(const TokenPieces& tokenpieces);
    TokenPair get_best_pair();
    void merge_once(TokenPieces& tokenpieces,const TokenPair& pair);

    static std::string file_to_text(const std::string& path);
};