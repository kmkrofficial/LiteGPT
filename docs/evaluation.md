# Evaluation

**Status: suite v0 final at 0.1.1; baselines done for 16 models, and V2 scored.** All results: [`results.md`](results.md). Registry: `EXP-M1B-SUITE`,
`EXP-M1B-BASELINES`, `EXP-M3-CHECKPOINT-EVAL`, `EXP-M4-SFT`, `EXP-M4-T2D`.

## What is measured

Suite v0 (`litegpt/eval/suite_v0.yaml`, `litegpt/eval/SUITE_VERSION`) was frozen at 0.1.0 before the first model was scored and
corrected to 0.1.1 before the affected scores were final: the chat code tasks are
corrected copies of lm-eval's (`litegpt/eval/tasks/code_chat`: lm-eval 0.4.13 cut restated functions at `
def` and deleted a
leading `def` in MBPP answers). The base groups are identical in both versions.

| Suite | Tasks | How |
|---|---|---|
| Base (all models) | HellaSwag, ARC-Easy/Challenge, PIQA, OpenBookQA, CommonsenseQA, Winogrande, MMLU (continuation / cloze form), TruthfulQA MC1/MC2 (0-shot); TriviaQA (fixed seeded 2,000-question subset) and GSM8K (5-shot); HumanEval (0-shot) and MBPP (3-shot), pass@1 greedy | lm-evaluation-harness 0.4.13 |
| Instruct (chat models) | IFEval (prompt-level strict), GSM8K chain-of-thought 0-shot (last number in the answer), HumanEval / MBPP instruct (corrected copies), XSTest | lm-eval with each model's own chat template, at most 512 new tokens (GSM8K, XSTest 256); XSTest native |
| Language modelling | Bits per byte on `val_core` and the domain sets (web, synthetic, code, math, chat), each model with its own tokenizer | native (`litegpt/eval/native.py bpb`) |
| Efficiency | Prefill / decode tok/s, **bytes/s**, time to first token, peak VRAM at prompt 128/512 x output 128 x batch 1/8 | native (`litegpt/eval/native.py efficiency`) |

Fixed protocol: seed 1234 everywhere, greedy decoding, bf16, sample-level outputs saved, model revision pinned to the commit
resolved at start, task dataset revisions resolved and recorded per run, code tasks capped at 512 new tokens, no
`trust_remote_code`. Generated code runs only in WSL2. The allocator is capped at free VRAM minus 1 GiB so an overflow fails
instead of spilling into shared memory; a group that does not fit is retried with the batch halved (recorded). Models whose
`generation_config.json` sets a generation length are evaluated from a local copy with those keys removed (recorded).

## Comparing models with different tokenizers

Per-token numbers (perplexity per token, tokens/s) are not comparable across tokenizers. The report therefore uses
**bits per byte** for language modelling and **bytes/s** next to tokens/s for speed. Multiple-choice tasks use lm-eval's
length-normalised metrics where they are the standard headline (`acc_norm` for HellaSwag, ARC-Challenge, OpenBookQA).

## Harness validation (pre-registered)

- GPT-2 124M must land in the widely reported lm-eval ranges: HellaSwag acc_norm in [0.29, 0.33], PIQA acc in [0.61, 0.65].
- The same model run twice must give identical scores.
- The native LiteGPT adapter's log-likelihoods must equal direct model log-probabilities (`tests/test_eval_adapter.py`).

## Running

```bash
uv sync --locked --group eval --group hf-reference --group flash   # lm-eval's HF backend needs accelerate (hf-reference)
python -m litegpt.eval.run --model hf:openai-community/gpt2 --suite base           # in WSL2 (code tasks need Linux)
python -m litegpt.eval.run --model hf:HuggingFaceTB/SmolLM2-135M-Instruct --suite instruct
python -m litegpt.eval.native efficiency --model hf:openai-community/gpt2
python -m litegpt.eval.native xstest --model hf:HuggingFaceTB/SmolLM2-135M-Instruct
python -m litegpt.eval.native bpb --model hf:openai-community/gpt2 --val-dir <data run>/shards/val_text
```

Each run writes `runs/<id>-eval-<suite>-<model>/` with `run.json` (environment fingerprint), `eval/<group>.json`,
per-task sample files and `eval/summary.json` (headline metrics with standard errors, model and task dataset revisions).

## Native LiteGPT models

`litegpt/eval/lm_adapter.py` implements lm-eval's `LM` interface (`loglikelihood`, `loglikelihood_rolling`, `generate_until`)
for LiteGPT checkpoints, so M2 proxies and V2 are scored by exactly the same tasks as the baselines. Tokenisation follows the
lm-eval HF convention (continuation = tail of encode(context + continuation)); every sequence starts with `<|bos|>`.

## Deferred

Spider execution accuracy and MultiPL-E JS/TS execution were not run (they need databases outside the Hub and Node.js in WSL).
Both stay in the decontamination reference list.

## Text-to-data comparison

EXP-M4-T2D, scored by `litegpt/t2d.py` and run by `scripts/run_t2d.py`. Test items are frozen in `benchmarks/m4/t2d_manifest.json` (`scripts/freeze_t2d.py` rebuilds them byte for byte). Text is compared after lowercasing, removing punctuation and collapsing whitespace. Invalid JSON misses every gold value of that output. Decoding is greedy with no grammar help.
