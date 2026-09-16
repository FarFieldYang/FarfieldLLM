#include <climits>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <utility>

#include "bpe_tokenizer.h"

BPETokenizer::BPETokenizer(BPEModel model): model_(std::move(model)) {}

std::vector<int> BPETokenizer::encode(const std::string& text) {
    Pieces pieces = pretokenize(text);
    return encode_pieces(pieces);
}

TokenIds BPETokenizer::encode_file(const std::string& path) {
    std::string text = file_to_text(path);
    return encode(text);
}

std::string BPETokenizer::decode(const TokenIds& tokens) const{
    std::string text;
    const Vocab& vocab = model_.vocab();
    for(TokenId id : tokens){
        const Bytes& bytes = vocab.at(id);
        for(Byte byte : bytes){
            text.push_back(static_cast<char>(byte));
        }
    }
    return text;
}

TokenIds BPETokenizer::encode_pieces(const Pieces& pieces){
    TokenIds result;
    for(const Piece& piece : pieces){
        TokenIds ids = encode_piece(piece);
        result.insert(result.end(), ids.begin(), ids.end());
    }
    return result;
}

TokenIds BPETokenizer::encode_piece(const Piece& piece){
    if(auto id = model_.find_token(piece)){
        ++states_.vocab_hits;
        return TokenIds{*id};
    }
    if(auto cached = find_cache(piece)){
        ++states_.cache_hits;
        return *cached;
    }
    ++states_.cache_misses;
    ++states_.merge_calls;
    return merge_piece(piece);
}

TokenIds BPETokenizer::merge_piece(const Piece& piece){
    TokenIds ids = Bt_.encode(piece);
    TokenIds result = do_merge(std::move(ids));
    update_cache(piece, result);
    return result;
}

TokenIds BPETokenizer::do_merge(TokenIds ids){
    while(true){
        std::size_t best_rank = static_cast<std::size_t>(-1);
        TokenPair best_pair;
        TokenId new_id;
        for(std::size_t i = 0; i + 1 < ids.size(); ++i){
            TokenPair pair = {ids[i], ids[i + 1]};
            if(auto info = model_.find_merge(pair)){
                if(info->rank < best_rank){
                    best_rank = info->rank;
                    best_pair = pair;
                    new_id = info->new_id;
                }
            }
        }
        if(best_rank == static_cast<std::size_t>(-1)){
            return ids;
        }else{
            std::size_t read = 0, write = 0;
            while(read < ids.size()){
                if(read + 1 < ids.size() &&
                    ids[read] == best_pair.first && 
                    ids[read + 1] == best_pair.second){                   
                    ids[write++] = new_id;
                    read += 2;
                }else{
                    ids[write++] = ids[read++];
                }   
            }
            ids.resize(write);
        }
    }
}


void BPETokenizer::update_cache(const Piece& piece, const TokenIds& ids){
    if(cache_.size() >= MAX_CACHE_SIZE){
        cache_.erase(recent_.back());
        recent_.pop_back();
    }
    recent_.push_front(piece);
    cache_.emplace(piece, CacheEntry{ids, recent_.begin()});
}

const TokenIds* BPETokenizer::find_cache(const Piece& piece){
    auto it = cache_.find(piece);
    if(it == cache_.end()) return nullptr;
    recent_.splice(recent_.begin(), recent_,it->second.position);
    return &it->second.ids;
}

std::string BPETokenizer::file_to_text(const std::string& path){
    std::ifstream file(path, std::ios::binary);
    if(!file){throw std::runtime_error("Failed to open file: " + path);}
    std::string text(std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{});
    return text;
}

TokenizerStats BPETokenizer::stats() const{
    return states_;
}
void BPETokenizer::reset_stats(){
    states_.cache_hits = 0;
    states_.cache_misses = 0;
    states_.merge_calls = 0;
    states_.vocab_hits = 0;
}