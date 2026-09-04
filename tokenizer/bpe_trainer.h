#pragma once

#include <vector>
#include <string>
#include <map>
#include <utility>
using TokenId = int;
using TokenPair = std::pair<TokenId, TokenId>;
constexpr TokenPair NO_PAIR = {-1, -1};

struct BPEModel{
    std::vector<std::vector<unsigned char>> vocab;
    std::map<TokenPair, TokenId> merges;
};

class BPEtrainer {
public:
    BPEModel train(const std::string& text, int vocab_size, const BPEModel& initial_model = base_model());
    BPEModel train_file(const std::string& path, int vocab_size, const BPEModel& initial_model = base_model());
private:
    static BPEModel base_model();
    static TokenPair count(const std::vector<int>& ids);
    static TokenPair count_and_merge(std::vector<int>& ids, TokenId new_id, const TokenPair& bestpair);
    static void id_to_bytes(BPEModel& model, const TokenPair& pair);
    static std::vector<int> texts_to_ids(const std::string& text);

    static std::string file_to_text(const std::string& path);
};