#include "bpe_trainer.h"
#include "byte_tokenizer.h"
#include <fstream>
#include <iterator>
#include <stdexcept>


#include <cstdio>


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

BPEModel BPEtrainer::base_model(){
    BPEModel basemodel;
    int i = 0;
    while(i < 256){
        basemodel.vocab.push_back({static_cast<unsigned char>(i)});
        ++i;
    }
    return basemodel;
}

TokenPair BPEtrainer::count(const std::vector<int>& ids){
    std::map<TokenPair, int> pair_count;
    int max_count = 0;
    TokenPair new_bestpair = NO_PAIR;
    for(std::size_t i = 0; i + 1 < ids.size(); ++i){
        TokenPair pair = {ids[i], ids[i + 1]};
        int count = ++pair_count[pair];
        if(count > max_count || (count == max_count && pair < new_bestpair)){
            max_count = count; new_bestpair = pair;
        }
    }
    return new_bestpair;
}

TokenPair BPEtrainer::count_and_merge(std::vector<int>& ids, TokenId new_id, const TokenPair& bestpair){
    std::map<TokenPair, int> pair_count;
    int max_count = 0;
    TokenPair new_bestpair = NO_PAIR;
    std::size_t read = 0, write = 0;
    while(read < ids.size()){
        if(read + 1 < ids.size() && TokenPair{ids[read], ids[read + 1]} == bestpair){
            ids[write++] = new_id;
            read += 2;
        }else{
            ids[write++] = ids[read++];
        }
        if (write >= 2){
            TokenPair pair = {ids[write - 2], ids[write - 1]};
            int count = ++pair_count[pair];
            if(count > max_count || (count == max_count && pair < new_bestpair)){
                max_count = count; new_bestpair = pair;
            }
        }
    }
    ids.resize(write);
    return new_bestpair;
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