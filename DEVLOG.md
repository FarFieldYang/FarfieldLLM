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

## 2026-09-10 BPE Tokenizer V2
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