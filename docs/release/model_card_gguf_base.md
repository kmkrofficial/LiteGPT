---
license: apache-2.0
base_model: kmkrworks/LiteGPT-V2-Base
base_model_relation: quantized
pipeline_tag: text-generation
language:
- en
tags:
- gguf
- llama.cpp
- litegpt
- qwen3
---

# LiteGPT V2 Base, GGUF

**[LiteGPT-V2-Base](https://huggingface.co/kmkrworks/LiteGPT-V2-Base) for [llama.cpp](https://github.com/ggml-org/llama.cpp), LM Studio and the apps built on them:** a 201M-parameter language model built from scratch, with every step on record.

The files were converted with llama.cpp build b12084 and checked against the original model in PyTorch before release. Results, charts and run records: [github.com/kmkrofficial/LiteGPT](https://github.com/kmkrofficial/LiteGPT).

## Files

| File | Size | Peak RAM | Decode, 4 CPU threads | Quality against BF16 |
|---|---:|---:|---:|---|
| `LiteGPT-V2-Base-Q8_0.gguf` | 215 MB | 0.35 GiB | 189 tok/s | perplexity within +0.06%: the default choice |
| `LiteGPT-V2-Base-Q4_K_M.gguf` | 127 MB | 0.35 GiB | 254 tok/s | perplexity up to +1.8%: the smallest |
| `LiteGPT-V2-Base-BF16.gguf` | 404 MB | 0.52 GiB | 109 tok/s | V2's exact weights |

Speed and RAM measured on a laptop CPU (Intel i9-14900HX, 4 threads, 128-token prompt; RAM is the whole process at a 4,096-token context). Q8_0 decodes 3.6 times and Q4_K_M 4.8 times as fast as the original in PyTorch float32 (52.5 tok/s in 1.66 GiB).

## Checked against the original

- **Same tokens:** identical token ids on 1,200 validation documents and strings.
- **Same model:** perplexity of the F32 conversion within 0.001% of PyTorch on 5 text domains, and identical greedy output on all 23 test prompts.
- **Small, measured quantization cost:** Q8_0 within +0.06% perplexity of BF16, Q4_K_M within +1.8%, on code, math and web text.

Full tables: [llama.cpp (GGUF)](https://github.com/kmkrofficial/LiteGPT/blob/main/docs/results.md#llamacpp-gguf).

## Use

With llama.cpp:

```bash
llama-completion -hf kmkrworks/LiteGPT-V2-Base-GGUF:Q8_0 -p "The capital of France is" -n 32
```

In LM Studio, search for `kmkrworks/LiteGPT-V2-Base-GGUF`, or run `lms get kmkrworks/LiteGPT-V2-Base-GGUF`.

llama.cpp does not apply Unicode NFC normalisation for BPE tokenizers, so normalise input that may hold decomposed accents first. Ordinary text is unaffected.

On a Raspberry Pi 5 the Q8_0 file is estimated at about 44 tok/s, and every file fits a 1 GB Raspberry Pi 3. These are estimates from laptop runs and published Pi bandwidth, not measurements.

## Model

| Item | Value |
|---|---|
| Parameters | 201,373,952 |
| Shape | 28 layers, 768 wide, 12 heads, 4 KV heads |
| Context | 4,096 tokens |
| Tokenizer | V2's own 32,768-token byte-level BPE |
| llama.cpp labels | architecture `qwen3` (V2 uses the Qwen3 parameter layout), split rule `qwen2` (V2's own regex is exactly this rule) |

**Intended use:** research, learning, and fine-tuning for your own tasks. It is a base model with no safety tuning, so add your own safeguards before putting it in front of people.

## Licence

Apache-2.0 for the weights, with the [NOTICE](NOTICE) file. Training data and its attributions: [the original model card](https://huggingface.co/kmkrworks/LiteGPT-V2-Base).

**Built with Qwen.**
