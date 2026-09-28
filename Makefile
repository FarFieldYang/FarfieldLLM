CXX := g++
NVCC := nvcc

BUILD_DIR := build


# ============================================================
# Compiler flags
# ============================================================

CXXFLAGS := \
	-std=c++20 \
	-O2 \
	-g \
	-fno-omit-frame-pointer \
	-Wall \
	-Wextra \
	-pedantic \
	-Itokenizer \
	-Itensor \
	-Itensor/storage \
	-Itensor/backend \
	-Itensor/backend/cuda


NVCCFLAGS := \
	-std=c++20 \
	-O2 \
	-g \
	-Itokenizer \
	-Itensor \
	-Itensor/storage \
	-Itensor/backend \
	-Itensor/backend/cuda


# ============================================================
# Targets
# ============================================================

BPE_TRAINER_TARGET := $(BUILD_DIR)/test_bpe_trainer_v2
BPE_STRESS_TARGET := $(BUILD_DIR)/test_bpe_stress_v2
BPE_CS336_TARGET := $(BUILD_DIR)/test_bpe_cs336_benchmark
BPE_ENWIK8_TARGET := $(BUILD_DIR)/test_bpe_enwik8_benchmark
BPE_TOKENIZER_TARGET := $(BUILD_DIR)/test_bpe_tokenizer_v2
BPE_ENCODE_TARGET := $(BUILD_DIR)/bpe_encode_benchmark

TENSOR_DATA_TARGET := $(BUILD_DIR)/test_tensor_data

CUDA_STORAGE_OBJ := $(BUILD_DIR)/cuda_storage.o
CUDA_BACKEND_TARGET := $(BUILD_DIR)/test_cuda_backend


# ============================================================
# Tokenizer sources
# ============================================================

TOKENIZER_COMMON_SRC := \
	tokenizer/bpe_trainer.cpp \
	tokenizer/bpe_model.cpp \
	tokenizer/byte_tokenizer.cpp \
	tokenizer/pre_tokenizer.cpp


BPE_TRAINER_SRC := \
	$(TOKENIZER_COMMON_SRC) \
	tests/test_bpe_trainer_v2.cpp


BPE_STRESS_SRC := \
	$(TOKENIZER_COMMON_SRC) \
	benchmarks/tokenizer/stress.cpp


BPE_CS336_SRC := \
	$(TOKENIZER_COMMON_SRC) \
	benchmarks/tokenizer/cs336.cpp


BPE_ENWIK8_SRC := \
	$(TOKENIZER_COMMON_SRC) \
	benchmarks/tokenizer/enwik8.cpp


BPE_TOKENIZER_SRC := \
	$(TOKENIZER_COMMON_SRC) \
	tokenizer/bpe_tokenizer.cpp \
	tests/test_bpe_tokenizer_v2.cpp


BPE_ENCODE_SRC := \
	$(TOKENIZER_COMMON_SRC) \
	tokenizer/bpe_tokenizer.cpp \
	benchmarks/tokenizer/encode.cpp


# ============================================================
# TensorData sources
# ============================================================

TENSOR_DATA_SRC := \
	tensor/storage/cpu_storage.cpp \
	tensor/tensor_data.cpp \
	tests/test_tensor_data.cpp


# ============================================================
# CUDA backend test sources
# ============================================================

CUDA_BACKEND_SRC := \
	tensor/storage/cuda_storage.cu \
	tensor/tensor_data.cpp \
	tensor/backend/cuda/cuda_backend.cu \
	tests/test_cuda_backend.cu


# ============================================================
# Default
# ============================================================

all: $(BPE_TRAINER_TARGET) $(TENSOR_DATA_TARGET)


# ============================================================
# Build directory
# ============================================================

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)


# ============================================================
# BPE Trainer
# ============================================================

$(BPE_TRAINER_TARGET): $(BPE_TRAINER_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(BPE_TRAINER_SRC) -o $(BPE_TRAINER_TARGET)

run: $(BPE_TRAINER_TARGET)
	./$(BPE_TRAINER_TARGET)


# ============================================================
# BPE Stress Benchmark
# ============================================================

$(BPE_STRESS_TARGET): $(BPE_STRESS_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(BPE_STRESS_SRC) -o $(BPE_STRESS_TARGET)

stress: $(BPE_STRESS_TARGET)
	./$(BPE_STRESS_TARGET)


# ============================================================
# CS336 Benchmark
# ============================================================

$(BPE_CS336_TARGET): $(BPE_CS336_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(BPE_CS336_SRC) -o $(BPE_CS336_TARGET)

cs336: $(BPE_CS336_TARGET)
	./$(BPE_CS336_TARGET)


# ============================================================
# enwik8 Benchmark
# ============================================================

$(BPE_ENWIK8_TARGET): $(BPE_ENWIK8_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(BPE_ENWIK8_SRC) -o $(BPE_ENWIK8_TARGET)

enwik8: $(BPE_ENWIK8_TARGET)
	./$(BPE_ENWIK8_TARGET)


# ============================================================
# BPE Tokenizer
# ============================================================

$(BPE_TOKENIZER_TARGET): $(BPE_TOKENIZER_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(BPE_TOKENIZER_SRC) -o $(BPE_TOKENIZER_TARGET)

tokenizer: $(BPE_TOKENIZER_TARGET)
	./$(BPE_TOKENIZER_TARGET)


# ============================================================
# Encode Benchmark
# ============================================================

$(BPE_ENCODE_TARGET): $(BPE_ENCODE_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(BPE_ENCODE_SRC) -o $(BPE_ENCODE_TARGET)

encode: $(BPE_ENCODE_TARGET)
	./$(BPE_ENCODE_TARGET)


# ============================================================
# TensorData
# ============================================================

$(TENSOR_DATA_TARGET): \
	$(TENSOR_DATA_SRC) \
	tensor/tensor_data.h \
	tensor/storage/storage.h \
	tensor/device.h \
	| $(BUILD_DIR)

	$(CXX) $(CXXFLAGS) $(TENSOR_DATA_SRC) -o $(TENSOR_DATA_TARGET)


tensor-data: $(TENSOR_DATA_TARGET)
	./$(TENSOR_DATA_TARGET)


# ============================================================
# CUDA Storage compile check
# ============================================================

$(CUDA_STORAGE_OBJ): \
	tensor/storage/cuda_storage.cu \
	tensor/storage/storage.h \
	tensor/device.h \
	| $(BUILD_DIR)

	$(NVCC) $(NVCCFLAGS) \
		-c tensor/storage/cuda_storage.cu \
		-o $(CUDA_STORAGE_OBJ)


cuda-storage: $(CUDA_STORAGE_OBJ)
	@echo "CUDAStorage compile check passed."


# ============================================================
# CUDA Backend test / benchmark
# ============================================================

$(CUDA_BACKEND_TARGET): \
	$(CUDA_BACKEND_SRC) \
	tensor/backend/cuda/cuda_backend.h \
	tensor/tensor_data.h \
	tensor/storage/storage.h \
	tensor/device.h \
	| $(BUILD_DIR)

	$(NVCC) $(NVCCFLAGS) \
		$(CUDA_BACKEND_SRC) \
		-o $(CUDA_BACKEND_TARGET)


cuda-backend: $(CUDA_BACKEND_TARGET)
	./$(CUDA_BACKEND_TARGET)


# ============================================================
# Tests
# ============================================================

test: tensor-data tokenizer run

cuda-test: cuda-storage cuda-backend


# ============================================================
# Clean
# ============================================================

clean:
	rm -rf $(BUILD_DIR)


.PHONY: \
	all \
	run \
	stress \
	cs336 \
	enwik8 \
	tokenizer \
	encode \
	tensor-data \
	cuda-storage \
	cuda-backend \
	cuda-test \
	test \
	clean