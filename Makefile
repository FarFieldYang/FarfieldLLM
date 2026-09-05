CXX = g++
CXXFLAGS = -O2 -std=c++17 -I tokenizer

BPE_SOURCES = tokenizer/bpe_trainer.cpp tokenizer/byte_tokenizer.cpp
BPE_HEADERS = tokenizer/bpe_trainer.h tokenizer/byte_tokenizer.h tokenizer/tokenizer.h

.PHONY: all test clean test_bpe_file test_bpe_trainer test_byte_tokenizer

all: test

test: build/test_bpe_file build/test_bpe_trainer build/test_byte_tokenizer

test_bpe_file: build/test_bpe_file

test_bpe_trainer: build/test_bpe_trainer

test_byte_tokenizer: build/test_byte_tokenizer

test_bpe_tokenizer: build/test_bpe_tokenizer


build/test_bpe_file: tests/test_bpe_file.cpp $(BPE_SOURCES) $(BPE_HEADERS)
	mkdir -p build
	$(CXX) $(CXXFLAGS) tests/test_bpe_file.cpp $(BPE_SOURCES) -o build/test_bpe_file


build/test_bpe_trainer: tests/test_bpe_trainer.cpp $(BPE_SOURCES) $(BPE_HEADERS)
	mkdir -p build
	$(CXX) $(CXXFLAGS) tests/test_bpe_trainer.cpp $(BPE_SOURCES) -o build/test_bpe_trainer


build/test_byte_tokenizer: tests/test_byte_tokenizer.cpp tokenizer/byte_tokenizer.cpp tokenizer/byte_tokenizer.h tokenizer/tokenizer.h
	mkdir -p build
	$(CXX) $(CXXFLAGS) tests/test_byte_tokenizer.cpp tokenizer/byte_tokenizer.cpp -o build/test_byte_tokenizer

build/test_bpe_tokenizer: tests/test_bpe_tokenizer.cpp tokenizer/bpe_tokenizer.cpp tokenizer/bpe_trainer.cpp tokenizer/byte_tokenizer.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) tests/test_bpe_tokenizer.cpp tokenizer/bpe_tokenizer.cpp tokenizer/bpe_trainer.cpp tokenizer/byte_tokenizer.cpp -o build/test_bpe_tokenizer

clean:
	rm -rf build