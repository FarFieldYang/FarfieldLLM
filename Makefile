CXX := g++
CXXFLAGS := -std=c++20 -O2 -g -fno-omit-frame-pointer -Wall -Wextra -pedantic -Itokenizer


TARGET := build/test_bpe_trainer_v2
STRESS_TARGET := build/test_bpe_stress_v2
CS336_TARGET := build/test_bpe_cs336_benchmark
ENWIK8_TARGET := build/test_bpe_enwik8_benchmark
TOKENIZER_TARGET := build/test_bpe_tokenizer_v2
ENCODE_TARGET := build/bpe_encode_benchmark


COMMON_SRC := \
	tokenizer/bpe_trainer.cpp \
	tokenizer/bpe_model.cpp \
	tokenizer/byte_tokenizer.cpp \
	tokenizer/pre_tokenizer.cpp


SOURCES := \
	$(COMMON_SRC) \
	tests/test_bpe_trainer_v2.cpp


STRESS_SRC := \
	$(COMMON_SRC) \
	benchmarks/tokenizer/stress.cpp


CS336_SRC := \
	$(COMMON_SRC) \
	benchmarks/tokenizer/cs336.cpp


ENWIK8_SRC := \
	$(COMMON_SRC) \
	benchmarks/tokenizer/enwik8.cpp


TOKENIZER_SRC := \
	$(COMMON_SRC) \
	tokenizer/bpe_tokenizer.cpp \
	tests/test_bpe_tokenizer_v2.cpp


ENCODE_SRC := \
	$(COMMON_SRC) \
	tokenizer/bpe_tokenizer.cpp \
	benchmarks/tokenizer/encode.cpp


all: $(TARGET)


$(TARGET): $(SOURCES)
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)


$(STRESS_TARGET): $(STRESS_SRC)
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(STRESS_SRC) -o $(STRESS_TARGET)

stress: $(STRESS_TARGET)
	./$(STRESS_TARGET)


$(CS336_TARGET): $(CS336_SRC)
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(CS336_SRC) -o $(CS336_TARGET)

cs336: $(CS336_TARGET)
	./$(CS336_TARGET)


$(ENWIK8_TARGET): $(ENWIK8_SRC)
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(ENWIK8_SRC) -o $(ENWIK8_TARGET)

enwik8: $(ENWIK8_TARGET)
	./$(ENWIK8_TARGET)


$(TOKENIZER_TARGET): $(TOKENIZER_SRC)
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(TOKENIZER_SRC) -o $(TOKENIZER_TARGET)

tokenizer: $(TOKENIZER_TARGET)
	./$(TOKENIZER_TARGET)


$(ENCODE_TARGET): $(ENCODE_SRC)
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(ENCODE_SRC) -o $(ENCODE_TARGET)

encode: $(ENCODE_TARGET)
	./$(ENCODE_TARGET)


clean:
	rm -rf build


.PHONY: all run stress cs336 enwik8 tokenizer encode clean