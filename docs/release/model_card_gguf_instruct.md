---
license: apache-2.0
base_model: kmkrworks/LiteGPT-Llama-V2-Instruct
base_model_relation: quantized
pipeline_tag: text-generation
language:
- en
tags:
- gguf
- llama.cpp
- litegpt
- qwen3
- llama
---

# LiteGPT-Llama-V2-Instruct, GGUF

**Built with Llama. Built with Qwen.**

**[LiteGPT-Llama-V2-Instruct](https://huggingface.co/kmkrworks/LiteGPT-Llama-V2-Instruct) for [llama.cpp](https://github.com/ggml-org/llama.cpp), LM Studio and the apps built on them:** a 201M-parameter language model built from scratch, with every step on record.

The files were converted with llama.cpp build b12084 and checked against the original model in PyTorch before release. Results, charts and run records: [github.com/kmkrofficial/LiteGPT](https://github.com/kmkrofficial/LiteGPT). Try it in your browser: [LiteGPT V2 Chat](https://huggingface.co/spaces/kmkrworks/LiteGPT-V2-Chat).

## Files

| File | Size | Peak RAM | Decode, 4 CPU threads | Quality against BF16 |
|---|---:|---:|---:|---|
| `LiteGPT-Llama-V2-Instruct-Q8_0.gguf` | 215 MB | 0.35 GiB | 189 tok/s | perplexity within +0.02%: the default choice |
| `LiteGPT-Llama-V2-Instruct-Q4_K_M.gguf` | 127 MB | 0.35 GiB | 254 tok/s | perplexity up to +3.1%: the smallest |
| `LiteGPT-Llama-V2-Instruct-BF16.gguf` | 404 MB | 0.52 GiB | 109 tok/s | V2's exact weights |

Speed and RAM measured on a laptop CPU (Intel i9-14900HX, 4 threads, 128-token prompt; RAM is the whole process at a 4,096-token context). Q8_0 decodes 3.6 times and Q4_K_M 4.8 times as fast as the original in PyTorch float32 (52.5 tok/s in 1.66 GiB). Measured on the Base files, which have the same architecture and size.

## Checked against the original

- **Same tokens:** identical token ids on 1,200 validation documents and strings.
- **Same model:** perplexity of the F32 conversion within 0.005% of PyTorch on 5 text domains, and identical greedy output on all 23 test prompts.
- **Small, measured quantization cost:** Q8_0 within +0.02% perplexity of BF16, Q4_K_M within +3.1%, on code, math and web text.
- **Same chat:** llama.cpp renders the stored chat template with one BOS, stops on `<|im_end|>`, and gave the same replies as PyTorch on 10 of 10 conversations.

Full tables: [llama.cpp (GGUF)](https://github.com/kmkrofficial/LiteGPT/blob/main/docs/results.md#llamacpp-gguf).

## Use

With llama.cpp, as an OpenAI-compatible chat server:

```bash
llama-server -hf kmkrworks/LiteGPT-Llama-V2-Instruct-GGUF:Q8_0 --jinja
```

The chat template is stored in the file; `--jinja` turns it on. Roles are `system`, `user` and `assistant`.

In LM Studio, search for `kmkrworks/LiteGPT-Llama-V2-Instruct-GGUF`, or run `lms get kmkrworks/LiteGPT-Llama-V2-Instruct-GGUF`.

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

**Intended use:** research, learning, and small chat or extraction experiments. It has no safety tuning, so add your own safeguards before putting it in front of people.

## Licence

Apache-2.0 for the weights, with the [NOTICE](NOTICE_instruct.txt) file. Training data and its attributions: [the original model card](https://huggingface.co/kmkrworks/LiteGPT-Llama-V2-Instruct).

**Built with Llama. Built with Qwen.** Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved. Use of this model should follow the [Llama Acceptable Use Policy](https://llama.meta.com/llama3_1/use-policy).
