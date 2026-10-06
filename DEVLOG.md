# Development Log

## 2026.09.03 Initialization and Byte Tokenizer
easy

## 2026.09.04 BPE Tokenizer and BPE Trainer
not easy

## 2026.09.08 BPE Trainer v2
really difficult!!!

Reworked the BPE trainer with pre-tokenization, deduplicated pieces, incremental pair statistics, affected-piece indexing, and heap-based merge selection.

### benchmarks

| Benchmark | Corpus | Target vocab | Result |
| --- | ---: | ---: | ---: |
| V1 internal sample | ~40 KB | 500 | ~0.665 s |
| V2 internal sample | ~40 KB | 500 | ~0.021 s |
| CS336 `corpus.en` | 0.127 MiB | 500 | ~0.0748 s |
| enwik8, initial V2 | 100 MB | 10,000 | 55.38 s |
| enwik8, improved PairHash | 100 MB | 10,000 | 35.12 s |

## 2026.09.10 BPE Tokenizer V2
difficult!

Completed the runtime BPE tokenizer with model lookup indexes, encode/decode, whole-piece vocab fast path, and a bounded LRU cache.

Added held-out encode benchmarks, runtime cache statistics, round-trip tests, and perf profiling.

### benchmarks

| Benchmark | Train corpus | Eval corpus | Vocab | Cold encode |
| --- | --- | --- | ---: | ---: |
| CS336 `corpus.en` | `test.txt` | `corpus.en` | 2,000 | 4.82 MiB/s |
| Internal sample | `test.txt` | `sample.txt` | 2,000 | 5.16 MiB/s |
| enwik8 held-out | `test.txt` | `enwik8` | 2,000 | 4.29 MiB/s |
| enwik8 in-sample | `enwik8` | `enwik8` | 10,000 | 3.69 MiB/s |

100 MB `enwik8` round-trip test: PASS.

`perf` shows that `std::regex` pre-tokenization is currently the largest runtime bottleneck, so regex optimization is deferred to Tokenizer V2.1.

## 2026.09.22 TensorData
not easy

Started the tensor stack.

Implemented Storage and TensorData with shape, strides, offset, shared storage, and views including reshape, transpose, permute, unsqueeze, squeeze, broadcast_to, narrow, slice, select, and flatten.

Added CPUStorage and GPUStorage.

TensorData tests pass.

## 2026.09.27 CUDA Backend
really difficult!!!

Implemented the CUDA backend for TensorData.

Added stride-aware binary and unary operations, reductions, broadcasting, contiguous conversion, scalar operations, and matrix multiplication through cuBLASLt.

The backend supports non-contiguous views for elementwise operations and reductions. Matmul supports 2D and batched tensors, including contiguous tensors with nonzero offsets. Non-contiguous matmul inputs are materialized before calling cuBLASLt.

### benchmarks

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB VRAM, driver 616.92.

32M-element contiguous tensors, 128 MB each:

| Op | CPU | GPU | Speedup |
| --- | ---: | ---: | ---: |
| exp | 195.72 ms | 1.106 ms | 176.98x |
| log | 203.68 ms | 1.372 ms | 148.41x |
| sqrt | 92.34 ms | 1.563 ms | 59.10x |
| tanh | 297.58 ms | 1.374 ms | 216.59x |
| add | 28.13 ms | 1.994 ms | 14.11x |
| sub | 29.46 ms | 2.258 ms | 13.05x |
| mul | 28.50 ms | 2.093 ms | 13.62x |
| div | 28.10 ms | 2.355 ms | 11.93x |

Reduction benchmarks:

| Case | Op | Time | Logical bandwidth |
| --- | --- | ---: | ---: |
| contiguous last dim | sum | 0.558 ms | 240.45 GB/s |
| contiguous last dim | max | 0.561 ms | 239.30 GB/s |
| contiguous first dim | sum | 1.426 ms | 94.12 GB/s |
| contiguous first dim | max | 1.434 ms | 93.65 GB/s |
| transpose view | sum | 1.416 ms | 94.83 GB/s |
| transpose view | max | 1.419 ms | 94.60 GB/s |
| 3D last dim | sum | 0.553 ms | 243.00 GB/s |
| 3D last dim | max | 0.554 ms | 242.37 GB/s |
| reduction width 128 | sum | 2.221 ms | 60.90 GB/s |
| reduction width 128 | max | 2.226 ms | 60.76 GB/s |

Host-device copies are not included in these timings.

All CUDA backend correctness tests pass, including stride-aware operations, reductions, 2D and batched matmul, offset views, and argument validation.

## 2026.10.05 Tensor v1
very difficult!!! (I think I need a stronger word for this.)

Finished Tensor v1.

The Tensor API now wraps the backend and supports tensor/scalar arithmetic, broadcasting, unary operations, reductions, matmul, contiguous conversion, and TensorData views.

Integration tests cover rank and multi-axis broadcasting, transpose and offset views, 0D scalars, chained views, contiguous conversion, and 2D through 5D batched matmul.

All Tensor API / backend integration tests pass.

Next: autograd.