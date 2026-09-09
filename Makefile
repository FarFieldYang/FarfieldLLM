CXX := g++
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -pedantic -Itokenizer

TARGET := build/test_bpe_trainer_v2
STRESS_TARGET := build/test_bpe_stress_v2
CS336_TARGET := build/test_bpe_cs336_benchmark
ENWIK8_TARGET := build/test_bpe_enwik8_benchmark

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


clean:
	rm -rf build


.PHONY: all run stress cs336 enwik8 clean