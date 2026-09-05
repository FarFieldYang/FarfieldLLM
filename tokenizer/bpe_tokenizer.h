#pragma once

#include <string>
#include <vector>
#include "tokenizer.h"
#include "bpe_trainer.h"

class BPETokenizer : public Tokenizer {
public:
    explicit BPETokenizer(const BPEModel& model);
    std::vector<int> encode(const std::string& text) const override;
    std::string decode(const std::vector<int>& tokens) const override;
    std::vector<int> encode_file(const std::string& path) const;
private:
    BPEModel model_;
    void find_best(const std::vector<int>& ids, TokenId& best_id, TokenPair& best_pair) const;
    void merge_and_find(std::vector<int>& ids, TokenId& best_id, TokenPair& best_pair) const;
    static std::string file_to_text(const std::string& path);
};