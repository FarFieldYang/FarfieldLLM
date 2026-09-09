# Development Log

## 2026.09.03 Initialization and Byte_tokenizer
easy

## 2026.09.04 BPE_tokenizer and BPE_trainer
not easy

## 2026.09.08 BPE_trainer v2
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