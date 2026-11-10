# Distribution and export

**Status: v2.0.0 is released.** The licence gate passed ([ADR-004](decisions/ADR-004-licensing.md)).

## Where it is

| What | Where |
|---|---|
| Code | [kmkrofficial/LiteGPT](https://github.com/kmkrofficial/LiteGPT), on top of V1's history |
| Base weights | [kmkrworks/LiteGPT-V2-Base](https://huggingface.co/kmkrworks/LiteGPT-V2-Base) |
| Instruct weights | [kmkrworks/LiteGPT-Llama-V2-Instruct](https://huggingface.co/kmkrworks/LiteGPT-Llama-V2-Instruct) |
| Base GGUF (llama.cpp, LM Studio, Ollama) | [kmkrworks/LiteGPT-V2-Base-GGUF](https://huggingface.co/kmkrworks/LiteGPT-V2-Base-GGUF) |
| Instruct GGUF | [kmkrworks/LiteGPT-Llama-V2-Instruct-GGUF](https://huggingface.co/kmkrworks/LiteGPT-Llama-V2-Instruct-GGUF) |

V1's models stay at `koganrath/LiteGPT-Base` and `koganrath/LiteGPT-Instruct` on Hugging Face.

## What a model folder holds

Weights as safetensors at bfloat16, config, tokenizer, chat template, `litegpt.json` metadata, model card, notice and licence files. In float32, logits and greedy output are identical through Transformers and through LiteGPT (EXP-M5-EXPORT).

```bash
uv run --group hf python -m litegpt.export.to_hf --checkpoint milestone-final.pt --kind base --out LiteGPT-V2-Base
```

The frozen report for this release is [`runs/report-v2.0.0/REPORT.md`](../runs/report-v2.0.0/REPORT.md).

## GGUF files

Each GGUF repository holds BF16 (V2's exact weights), Q8_0 and Q4_K_M, converted with llama.cpp build b12084 from the exported folder:

```bash
uv run python -m litegpt.export.to_gguf --hf-dir LiteGPT-V2-Base --kind base --llama-cpp ~/llama.cpp --out gguf
```

`--llama-cpp` points at a llama.cpp folder holding its source in `src/` (with [`benchmarks/gguf/llama_cpp_litegpt.patch`](../benchmarks/gguf/llama_cpp_litegpt.patch) applied), a Python environment for its converter in `venv/`, and its binaries in `b/`. The command checks every file's metadata before it finishes.

- **Architecture label `qwen3`.** V2 uses the Qwen3 parameter layout, so llama.cpp runs it with its existing Qwen3 code: no custom code is needed in llama.cpp, LM Studio or Ollama.
- **Tokenizer label `qwen2`.** llama.cpp stores the regex that splits text before BPE as a name, and V2's regex is character for character its `qwen2` rule. The vocabulary, merges, special tokens and chat template are V2's own. llama.cpp's converter recognises tokenizers by hashing the tokens of a test string, so the patch adds one lookup entry mapping V2's hash to `qwen2`, the same way llama.cpp handles other models with their own vocabulary.
- **Checked before release:** token ids, perplexity, greedy output and the chat path against V2 in PyTorch. Results: [llama.cpp (GGUF)](results.md#llamacpp-gguf).

## Not distributed

The text-to-data fine-tunes. Training data is not redistributed either: it is rebuilt from pinned public revisions.

## Later

ONNX Runtime, for the browser and for phone and PC NPUs.
