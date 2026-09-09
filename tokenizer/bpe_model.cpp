#include <utility>

#include "bpe_model.h"

BPEModel::BPEModel(Vocab vocab, MergeRules merges) : vocab_(std::move(vocab)), merges_(std::move(merges)){}

const Vocab& BPEModel::vocab() const {
    return vocab_;
}

const MergeRules& BPEModel::merges() const{
    return merges_;
}

std::size_t BPEModel::vocab_size() const{
    return vocab_.size();
}