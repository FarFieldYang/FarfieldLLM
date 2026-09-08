#include <fstream>
#include <iterator>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <cstdio>

#include "bpe_trainer.h"
#include "byte_tokenizer.h"
BPEtrainer::BPEtrainer() {vocab_ = init_vocab();}

Vocab BPEtrainer::init_vocab(){
    Vocab v(256);
    for(int i = 0; i < 256; ++i) v[i].push_back(static_cast<Byte>(i));
    return v;
}

CountedPieces BPEtrainer::pre_tokenize(const std::string& text){
    return pretokenize_and_count(text);
}

TokenPieces BPEtrainer::text_to_token(const CountedPieces& textpieces){
    TokenPieces pieces;
    pieces.reserve(textpieces.size());
    ByteTokenizer Bt;
    for(const auto& [text, frequency] : textpieces){
        TokenPiece piece;
        piece.ids = Bt.encode(text);
        piece.frequency = frequency;
        pieces.push_back(std::move(piece));
    }
    return pieces;
}



void BPEtrainer::init_state(const TokenPieces& tokenpieces){
    state_ = Trainstate{};

    for(PieceId i = 0; i < tokenpieces.size(); ++i){
        const TokenPiece& piece = tokenpieces[i];
        for(std::size_t j = 0; j + 1 < piece.ids.size(); ++j){
            TokenPair pair = {piece.ids[j], piece.ids[j + 1]};
            PairInfo& info = state_.pair_info_map[pair];
            info.count += piece.frequency;
            info.pieces.insert(i);
        }
    }

    for(const auto& [pair, info] : state_.pair_info_map) state_.heap.push({info.count, pair});
}

TokenPair BPEtrainer::get_best_pair(){
    while(!state_.heap.empty()){
        auto [count, pair] = state_.heap.top();
        auto it = state_.pair_info_map.find(pair);

        if(it != state_.pair_info_map.end() && it->second.count == count){
            state_.heap.pop();
            if(count < 2) return NO_PAIR;
            return pair;
        }
        state_.heap.pop();
    }
    return NO_PAIR;
}

void merge_once(TokenPieces& tokenpieces,const TokenPair& pair){

}


BPEModel BPEtrainer::train(const std::string& text, int vocab_size, const BPEModel& initial_model){
    if (vocab_size <= static_cast<int>(initial_model.vocab.size())) {
        return initial_model;
    }
    int merge_times = vocab_size - static_cast<int>(initial_model.vocab.size());
    BPEModel model = initial_model;

    std::vector<int> ids = texts_to_ids(text);
    
    if (ids.size() < 2) {
        throw std::runtime_error("Training text must contain at least two bytes.");
    }

    TokenPair bestpair = count(ids);
    TokenId new_id = initial_model.vocab.size();

    while(merge_times > 0 && ids.size() >= 2 && bestpair != NO_PAIR){
        model.merges[bestpair] = new_id;
        id_to_bytes(model, bestpair);
        bestpair = count_and_merge(ids, new_id, bestpair);
        ++new_id;
        --merge_times;
    }
    return model;
}

BPEModel BPEtrainer::train_file(const std::string& path, int vocab_size, const BPEModel& initial_model){
    std::string text = file_to_text(path);
    return train(text, vocab_size, initial_model);

}



void BPEtrainer::id_to_bytes(BPEModel& model, const TokenPair& pair){
    std::vector<unsigned char> bytes = model.vocab[pair.first];
    bytes.insert(bytes.end(), model.vocab[pair.second].begin(), model.vocab[pair.second].end());
    model.vocab.push_back(std::move(bytes));
}

std::vector<int> BPEtrainer::texts_to_ids(const std::string& text){
    ByteTokenizer Bt;
    std::vector<int> ids = Bt.encode(text);
    return ids;
}

std::string BPEtrainer::file_to_text(const std::string& path){
    std::ifstream file(path, std::ios::binary);
    if(!file){throw std::runtime_error("Failed to open file: " + path);}
    std::string text(std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{});
    return text;
}