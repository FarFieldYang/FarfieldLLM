#pragma once

#include <string>
#include <vector>
#include <list>
#include <unordered_map>

#include "tokenizer.h"
#include "pre_tokenizer.h"
#include "bpe_model.h"

struct CacheEntry {
    TokenIds ids;
    std::list<Piece>::iterator position;
};

class BPETokenizer : public Tokenizer {
public:
    explicit BPETokenizer(BPEModel model);
    TokenIds encode(const std::string& text) override;
    std::string decode(const TokenIds& tokens) const override;
    std::vector<int> encode_file(const std::string& path) const;
private:
    BPEModel model_;
    std::list<Piece> recent_;
    std::unordered_map<Piece, CacheEntry> cache_;
    static constexpr std::size_t MAX_CACHE_SIZE = 10000;

    TokenIds encode_pieces(const Pieces& pieces);
    TokenIds encode_piece(const Piece& piece);
    const TokenIds* find_cache(const Piece& piece);

    TokenIds merge_piece(const Piece& piece);
    TokenIds short_merge(TokenIds Ids);
    TokenIds long_merge(TokenIds ids);
    void update_cache(const Piece& piece, const TokenIds& ids);
   
    static std::string file_to_text(const std::string& path);
};