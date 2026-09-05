#include "bpe_tokenizer.h"
#include "byte_tokenizer.h"

#include <climits>
#include <fstream>
#include <iterator>
#include <stdexcept>

BPETokenizer::BPETokenizer(const BPEModel& model): model_(model) {}

std::vector<int> BPETokenizer::encode(const std::string& text) const {
    ByteTokenizer bt;
    std::vector<int> ids = bt.encode(text);

    TokenId best_id = INT_MAX;
    TokenPair best_pair = NO_PAIR;
    find_best(ids, best_id, best_pair);
    while(best_pair != NO_PAIR){
        merge_and_find(ids, best_id, best_pair);
    }
    return ids;
}

std::vector<int> BPETokenizer::encode_file(const std::string& path) const {
    std::string text = file_to_text(path);
    return encode(text);
}

std::string BPETokenizer::decode(const std::vector<int>& tokens) const{
    std::string text;
    for(TokenId id : tokens){
        const std::vector<unsigned char>& bytes = model_.vocab[id];
        for(unsigned char byte : bytes){
            text.push_back(static_cast<char>(byte));
        }
    }
    return text;
}

void BPETokenizer::find_best(const std::vector<int>& ids, TokenId& best_id, TokenPair& best_pair) const {
    for(std::size_t i = 0; i + 1 < ids.size(); ++i){
        TokenPair pair = {ids[i], ids[i + 1]};
        auto it = model_.merges.find(pair);
        if(it != model_.merges.end()){
            TokenId new_id = it->second;
            if(new_id < best_id){
                best_id = new_id;
                best_pair = pair;
            }
        }
    }
}

void BPETokenizer::merge_and_find(std::vector<int>& ids, TokenId& best_id, TokenPair& best_pair) const {
    TokenPair new_best_pair = NO_PAIR;
    TokenId new_best_id = INT_MAX;
    std::size_t read = 0, write = 0;
    while(read < ids.size()){
        if(read + 1 < ids.size() && TokenPair{ids[read], ids[read + 1]} == best_pair){
            ids[write++] = best_id;
            read += 2;
        }else{
            ids[write++] = ids[read++];
        }
        if (write >= 2){
            TokenPair pair = {ids[write - 2], ids[write - 1]};
            auto it = model_.merges.find(pair);
            if(it != model_.merges.end()){
                TokenId new_id = it->second;
                if(new_id < new_best_id){
                    new_best_id = new_id;
                    new_best_pair = pair;
                }
            }
        }
    }
    best_id = new_best_id;
    best_pair = new_best_pair;
    ids.resize(write);
}

std::string BPETokenizer::file_to_text(const std::string& path){
    std::ifstream file(path, std::ios::binary);
    if(!file){throw std::runtime_error("Failed to open file: " + path);}
    std::string text(std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{});
    return text;
}