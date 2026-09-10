#include <climits>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "bpe_tokenizer.h"
#include "byte_tokenizer.h"
#include "bpe_model.h"

BPETokenizer::BPETokenizer(BPEModel model): model_(std::move(model)) {}

std::vector<int> BPETokenizer::encode(const std::string& text) const {

}

std::vector<int> BPETokenizer::encode_file(const std::string& path) const {
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

TokenIds BPETokenizer::encode_piece(const Piece& piece){
    if(auto id = model_.find_token(piece)){
        return TokenIds{*id};
    }
    if(auto cached = find_cache(piece)){
        return *cached;
    }
    return merge_piece(piece);
}

TokenIds BPETokenizer::merge_piece(const Piece& piece){

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