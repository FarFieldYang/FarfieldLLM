#pragma once

#include <vector>
#include <string>
#include <utility>
#include <cstddef>
#include <unordered_map>
#include <optional>

#include "tokenizer.h"
#include "pre_tokenizer.h"


using MergeRule = std::pair<TokenPair, TokenId>;
using Vocab = std::vector<Bytes>;
using MergeRules = std::vector<MergeRule>;

struct MergeInfo {
    std::size_t rank;
    TokenId new_id;
};
using MergeMap = std::unordered_map<TokenPair, MergeInfo, PairHash>;
using VocabMap = std::unordered_map<std::string, TokenId>;



class BPEModel{
public:
    BPEModel(Vocab vocab, MergeRules merges);

    const Vocab& vocab() const;
    const MergeRules& merges() const;
    std::size_t vocab_size() const;

    std::optional<TokenId> find_token(const Piece& piece) const;
    std::optional<MergeInfo> find_merge(const TokenPair& pair) const;

    void save(const std::string& path) const;
    static BPEModel load(const std::string& path);
private:
    Vocab vocab_;
    MergeRules merges_;
    VocabMap vocab_map_;
    MergeMap merge_map_;

    void build_vocab_map();
    void build_merge_map();
};