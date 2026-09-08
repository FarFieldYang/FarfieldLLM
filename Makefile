CXX := g++
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -pedantic -Itokenizer

TARGET := build/test_bpe_trainer_v2
STRESS_TARGET := build/test_bpe_stress_v2

SOURCES := \
	tokenizer/bpe_trainer.cpp \
	tokenizer/bpe_model.cpp \
	tokenizer/byte_tokenizer.cpp \
	tokenizer/pre_tokenizer.cpp \
	tests/test_bpe_trainer_v2.cpp

STRESS_SRC := \
	tokenizer/bpe_trainer.cpp \
	tokenizer/bpe_model.cpp \
	tokenizer/byte_tokenizer.cpp \
	tokenizer/pre_tokenizer.cpp \
	tests/test_bpe_stress_v2.cpp

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

clean:
	rm -rf build

.PHONY: all run stress clean