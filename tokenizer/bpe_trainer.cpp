#include <fstream>
#include <iterator>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <cstdio>
#include <unordered_set>

#include "bpe_trainer.h"
#include "byte_tokenizer.h"

Vocab BPETrainer::init_vocab(){
    Vocab v(256);
    for(int i = 0; i < 256; ++i) v[i].push_back(static_cast<Byte>(i));
    return v;
}

BPEModel BPETrainer::train(const std::string& text, std::size_t vocab_size){
    vocab_ = init_vocab();
    merges_.clear();
    state_ = Trainstate{};

    CountedPieces textpieces = pre_tokenize(text);
    TokenPieces tokenpieces = text_to_token(textpieces);
    train_merges(tokenpieces, vocab_size);

    return BPEModel(std::move(vocab_), std::move(merges_));
}

BPEModel BPETrainer::train_file(const std::string& path, std::size_t vocab_size){
    std::string text = file_to_text(path);
    return train(text, vocab_size);
}

CountedPieces BPETrainer::pre_tokenize(const std::string& text){
    return pretokenize_and_count(text);
}

TokenPieces BPETrainer::text_to_token(const CountedPieces& textpieces){
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

void BPETrainer::train_merges(TokenPieces& tokenpieces, std::size_t vocab_size){
    init_state(tokenpieces);
    while(vocab_.size() < vocab_size){
        TokenPair best_pair = get_best_pair();
        if(best_pair == NO_PAIR) break;
        merge_once(tokenpieces, best_pair);
    }
}

void BPETrainer::init_state(const TokenPieces& tokenpieces){
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

TokenPair BPETrainer::get_best_pair(){
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

void BPETrainer::merge_once(TokenPieces& tokenpieces,const TokenPair& best_pair){
    PairInfo& info = state_.pair_info_map[best_pair];
    TokenId new_id = static_cast<TokenId>(vocab_.size()); 
    merges_.push_back({best_pair, new_id});
    add_vocab_bytes(best_pair);

    std::vector<PieceId> affected(info.pieces.begin(), info.pieces.end());
    std::unordered_set<TokenPair, PairHash> touched_pairs;
    for(auto& piece_id : affected){
        TokenPiece& piece = tokenpieces[piece_id];
        PairCounts old_counts = count_pairs(piece);
        merge_piece(piece, new_id, best_pair);
        PairCounts new_counts = count_pairs(piece);
        update_state(piece_id, piece, old_counts, new_counts, touched_pairs);
    }
    state_.pair_info_map.erase(best_pair);
    refresh_heap(touched_pairs);
}

void BPETrainer::add_vocab_bytes(const TokenPair& pair){
    Bytes bytes = vocab_[pair.first];
    bytes.insert(bytes.end(), vocab_[pair.second].begin(), vocab_[pair.second].end());
    vocab_.push_back(std::move(bytes));
}

void BPETrainer::merge_piece(TokenPiece& piece, TokenId new_id, const TokenPair& best_pair){
    std::size_t read = 0, write = 0;
    while(read < piece.ids.size()){
        if(read + 1 < piece.ids.size() &&
            piece.ids[read] == best_pair.first && 
            piece.ids[read + 1] == best_pair.second){                   
            piece.ids[write++] = new_id;
            read += 2;
        }else{
            piece.ids[write++] = piece.ids[read++];
        }   
    }
    piece.ids.resize(write);
}

PairCounts BPETrainer::count_pairs(const TokenPiece& piece){
    PairCounts counts;
    for(std::size_t index = 0; index + 1 < piece.ids.size(); ++index){
        TokenPair pair = {piece.ids[index], piece.ids[index + 1]};
        ++counts[pair];
    }
    return counts;
}

void BPETrainer::update_state(PieceId piece_id, TokenPiece& piece, PairCounts& old_counts, PairCounts& new_counts, TokenPairs& touched_pairs){
    for(auto& [pair, count] : old_counts){
        PairInfo& info = state_.pair_info_map.at(pair);
        info.count -= piece.frequency * count;
        info.pieces.erase(piece_id);
        touched_pairs.insert(pair);
    }
    for(auto& [pair, count] : new_counts){
        PairInfo& info = state_.pair_info_map[pair];
        info.count += piece.frequency * count;
        info.pieces.insert(piece_id);
        touched_pairs.insert(pair);
    }
}

void BPETrainer::refresh_heap(TokenPairs& touched_pairs){
    for(const TokenPair& pair : touched_pairs){
        auto it = state_.pair_info_map.find(pair);
        if (it != state_.pair_info_map.end() && it->second.count > 0)
            state_.heap.push({it->second.count, pair});
    }
}

std::string BPETrainer::file_to_text(const std::string& path){
    std::ifstream file(path, std::ios::binary);
    if(!file){throw std::runtime_error("Failed to open file: " + path);}
    std::string text(std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{});
    return text;
}