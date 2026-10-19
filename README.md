# FarfieldLLM

A from-scratch LLM project in C++20 and CUDA.

The project currently includes a BPE tokenizer, a CUDA-backed Tensor implementation, and first-order reverse-mode autograd. The next steps are SGD, neural network layers, and the first training loop.

No PyTorch or TensorFlow is used for the tensor implementation. Matrix multiplication uses cuBLASLt.

## build

The project uses C++20, CUDA, and GNU Make.

```bash
make
```

Run the CPU tests and compile the autograd objects:

```bash
make test
```

Run the CUDA backend, Tensor, and autograd tests:

```bash
make cuda-test
```

Run all CPU and GPU tests:

```bash
make test-all
```

Individual targets are also available:

```bash
make tokenizer
make tensor-data
make cuda-backend
make tensor
make autograd-test
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
initialization
    full  zeros  ones

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

scatter
    slice_scatter
    max_scatter
```

`TensorData` stores the shape, strides, offset, and shared storage used by a Tensor.

View operations can therefore represent non-contiguous tensors without copying the underlying storage.

The CUDA backend handles strided elementwise and reduction operations. Binary operations support broadcasting.

Matrix multiplication uses cuBLASLt. Both 2D and batched matrix multiplication are supported. Non-contiguous inputs are converted to contiguous tensors before the cuBLASLt call.

## autograd

Autograd v1 wraps the supported Tensor operations with 26 operation Nodes.

`ADvalue` is the user-facing handle. `ADimpl` stores the forward Tensor, optional gradient, and creator Node. Each Node holds its inputs and implements a local backward rule. The graph keeps upstream dependencies alive without storing downstream references.

`Engine::backward(loss)` starts from a scalar loss, collects its dependencies in topological order, and traverses them in reverse to accumulate gradients. Leaf gradients persist across fresh graphs until explicitly cleared with `Engine::zero_grad(parameters)`.

```cpp
#include "ad_value.h"
#include "engine.h"

ADvalue x(Tensor::full({}, 3.0f), true);
Engine::zero_grad({x});
ADvalue loss = x * x;
Engine::backward(loss);
```

After backward, `x.grad()` contains a scalar Tensor with value 6, while `x.value()` remains 3.

The first version supports first-order gradients and 0D scalar losses. Each non-leaf graph can be used for one backward pass; each training iteration builds a fresh graph while reusing the leaf parameters. Parameter updates will be handled by a separate optimizer.

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

Autograd tests cover all 26 operation Nodes, numerical gradients, finite-difference checks for smooth operations, broadcasting, views, reductions, and matmul. They also cover shared subgraphs, repeated inputs, gradient storage aliases, accumulation, zero_grad, and graph reuse restrictions.

All six test programs pass: TensorData, BPE tokenizer, BPE trainer, CUDA backend, Tensor API, and autograd.

Run Tensor tests:

```bash
make tensor
```

Run autograd tests separately:

```bash
make autograd-test
```

Run all GPU tests together:

```bash
make cuda-test
```

## todo

- SGD optimizer
- neural network layers
- Transformer
- training
- inference
