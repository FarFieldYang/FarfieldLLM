# FarfieldLLM

A from-scratch LLM project in C++20 and CUDA.

The project currently includes a BPE tokenizer and a CUDA-backed Tensor implementation. The next step is autograd, followed by the neural network and Transformer layers.

No PyTorch or TensorFlow is used for the tensor implementation. Matrix multiplication uses cuBLASLt.

## build

The project uses C++20, CUDA, and GNU Make.

```bash
make
```

Run the CPU tests:

```bash
make test
```

Run the CUDA backend and Tensor tests:

```bash
make cuda-test
```

Individual targets are also available:

```bash
make tokenizer
make tensor-data
make cuda-backend
make tensor
```

## tokenizer

The tokenizer includes byte-level tokenization, BPE training, and BPE encode/decode.

The BPE trainer uses pre-tokenization, incremental pair statistics, affected-piece indexing, and heap-based merge selection.

Some existing training benchmarks:

| Corpus | Target vocab | Time |
| --- | ---: | ---: |
| Internal sample (~40 KB), V1 | 500 | ~0.665 s |
| Internal sample (~40 KB), V2 | 500 | ~0.021 s |
| CS336 `corpus.en` (0.127 MiB) | 500 | ~0.0748 s |
| enwik8 (100 MB) | 10,000 | ~35.12 s |

The runtime tokenizer currently reaches about 3.7–5.2 MiB/s on the existing encode benchmarks.

The 100 MB `enwik8` round-trip test passes.

More tokenizer benchmark results are in [DEVLOG.md](DEVLOG.md).

## tensor

Tensor v1 is CUDA-backed and currently supports:

```text
binary
    +  -  *  /

scalar
    +  -  *  /

unary
    exp  log  sqrt  tanh

reduction
    sum  max

matrix
    matmul

views
    reshape
    transpose
    permute
    unsqueeze
    squeeze
    broadcast_to
    narrow
    slice
    select
    flatten

layout
    contiguous
```

`TensorData` stores the shape, strides, offset, and shared storage used by a Tensor.

View operations can therefore represent non-contiguous tensors without copying the underlying storage.

The CUDA backend handles strided elementwise and reduction operations. Binary operations support broadcasting.

Matrix multiplication uses cuBLASLt. Both 2D and batched matrix multiplication are supported. Non-contiguous inputs are converted to contiguous tensors before the cuBLASLt call.

## tests

Tensor tests cover:

- metadata
- tensor-tensor operators
- tensor-scalar operators
- unary operations
- reductions
- tensor views
- broadcasting
- contiguous conversion
- 2D and batched matrix multiplication
- invalid arguments and shape errors

Run them with:

```bash
make tensor
```

or together with the lower-level CUDA backend tests:

```bash
make cuda-test
```

## todo

- autograd
- neural network layers
- Transformer
- training
- inference