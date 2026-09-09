#pragma once

#include <vector>
#include <string>
#include <utility>
#include <cstddef>

#include "tokenizer.h"


using MergeRule = std::pair<TokenPair, TokenId>;
using Vocab = std::vector<Bytes>;
using MergeRules = std::vector<MergeRule>;

class BPEModel{
public:
    BPEModel(Vocab vocab, MergeRules merges);

    const Vocab& vocab() const;
    const MergeRules& merges() const;
    std::size_t vocab_size() const;

    void save(const std::string& path) const;
    static BPEModel load(const std::string& path);
private:
    Vocab vocab_;
    MergeRules merges_;
};