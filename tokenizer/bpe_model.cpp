#include <utility>
#include<string>

#include "bpe_model.h"

BPEModel::BPEModel(Vocab vocab, MergeRules merges) : vocab_(std::move(vocab)), merges_(std::move(merges)){
    build_merge_map();
    build_vocab_map();
}

const Vocab& BPEModel::vocab() const {
    return vocab_;
}

const MergeRules& BPEModel::merges() const{
    return merges_;
}

std::size_t BPEModel::vocab_size() const{
    return vocab_.size();
}

std::optional<TokenId> BPEModel::find_token(const Piece& piece) const{
    auto it = vocab_map_.find(piece);
    if(it == vocab_map_.end()){
        return std::nullopt;
    } else{
        return it->second;
    }
}
std::optional<MergeInfo> BPEModel::find_merge(const TokenPair& pair) const{
    auto it = merge_map_.find(pair);
    if(it == merge_map_.end()){
        return std::nullopt;
    }else{
        return it->second;
    }
}

void BPEModel::build_vocab_map(){
    for(std::size_t i = 0; i < vocab_.size(); ++i){
        const Bytes& bytes = vocab_[i];
        std::string key(bytes.begin(), bytes.end());
        vocab_map_.emplace(std::move(key), static_cast<TokenId>(i));
    }
}

void BPEModel::build_merge_map(){
    for(std::size_t rank = 0; rank < merges_.size(); ++rank){
        const auto& [pair, new_id] = merges_[rank];
        merge_map_.emplace(pair, MergeInfo{rank, new_id});
    }
}