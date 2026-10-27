# llama.go

A CPU-only LLM inference engine in pure Go, inspired by [llama.cpp](https://github.com/ggml-org/llama.cpp).

**Status: M0 complete.** GGUF parsing and quantisation dequantisation are implemented and tested. No forward pass yet — see [docs/plan-v0.1.md](docs/plan-v0.1.md) for the full milestone plan.

## Design constraints

- **No cgo.** Pure Go, cross-compiles to a single static binary.
- **float32 throughout.** Matches the float32 reference path in HuggingFace transformers, which is the correctness oracle.
- **Correctness before speed.** Every numeric claim must be backed by a comparison against an independent reference implementation. Performance work starts only after the verifiable-correctness gate passes.

## What works today

| Package | Capability | Test coverage |
|---|---|---|
| `gguf` | GGUF v3 parsing: header, all 13 metadata value types, string/array metadata, tensor table, alignment, mmap-backed payload access | 52.7% |
| `quant` | Dequantisation: F32, F16 (all 65536 half patterns verified), Q8_0, Q4_0 | 93.9% |

### Verified against a real model

`tools/gguf_info.go` parses `tinyllama-1.1b-chat-v1.0.Q8_0.gguf` (1,170,781,568 bytes, 201 tensors, 23 metadata keys) and reports hyperparameters that match the upstream HuggingFace `config.json` exactly:

```
llama.block_count                          uint32    22
llama.embedding_length                     uint32    2048
llama.feed_forward_length                  uint32    5632
llama.attention.head_count                 uint32    32
llama.attention.head_count_kv              uint32    4
llama.attention.layer_norm_rms_epsilon     float32   9.999999747378752e-06
llama.rope.freq_base                       float32   10000
```

## Usage

```sh
go install github.com/xiaoxin-agent/llama.go/cmd/...@latest   # not yet available

go run ./tools/gguf_info.go path/to/model.gguf
```

## Testing

```sh
go test ./... -race
go test ./... -cover
```

## Repository layout

```
gguf/      GGUF v3 reader (header, metadata, tensor table, mmap)
quant/     quantised block formats and dequantisation
tensor/    dense float32 tensors and kernels        (M2)
model/     Llama forward pass and KV cache          (M4/M5)
tokenizer/ SentencePiece-style BPE                  (M3)
sample/    sampling                                 (M6)
cmd/       CLI entry points
tools/     developer utilities
docs/      design documents
research/  verified facts and source citations
```

## Roadmap

| Milestone | Scope | Gate |
|---|---|---|
| M0 | Environment, reference baselines, repo scaffold | complete |
| M1 | GGUF parser | metadata matches `gguf_dump` exactly |
| M2 | Quantisation and tensor kernels | max abs diff ≤ 1e-6 vs independent reference |
| M3 | Tokenizer | token IDs identical to HF `AutoTokenizer` |
| M4 | Single transformer layer | cosine similarity ≥ 0.9999 vs HF hidden states |
| M5 | Full model + KV cache | logits max abs diff ≤ 1e-3; prefill == incremental |
| M6 | Sampling and CLI | 32-token greedy sequence identical to `llama-cli` |
| M7 | Documentation, benchmarks, release | `make verify` passes from a clean checkout |

## License

MIT — see [LICENSE](LICENSE).
