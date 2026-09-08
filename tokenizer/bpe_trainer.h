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

using PairCounts = std::unordered_map<TokenPair, std::size_t, PairHash>;
using TokenPairs = std::unordered_set<TokenPair, PairHash>;

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
    

//train_merge() helper function
    void init_state(const TokenPieces& tokenpieces);
    TokenPair get_best_pair();
    void merge_once(TokenPieces& tokenpieces,const TokenPair& best_pair);  

//merge_once() helper function
    void add_vocab_bytes(const TokenPair& pair);
    void merge_piece(TokenPiece& piece, TokenId new_id, const TokenPair& best_pair);
    PairCounts count_pairs(const TokenPiece& piece);
    void update_state(PieceId piece_id, TokenPiece& piece, PairCounts& old_counts, PairCounts& new_counts, TokenPairs& touched_pairs);
    void refresh_heap(TokenPairs& touched_pairs);

//trian_file() helper function
    static std::string file_to_text(const std::string& path);
};