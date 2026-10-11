# Evaluation

**Status: suite v0 built and harness-validated (M1b); baselines in progress.** Registry: `EXP-M1B-SUITE`, `EXP-M1B-BASELINES`.

## What is measured

Suite v0 (`litegpt/eval/suite_v0.yaml`, `litegpt/eval/SUITE_VERSION` = 0.1.0) was frozen before the first model was scored.

| Suite | Tasks | How |
|---|---|---|
| Base (all models) | HellaSwag, ARC-Easy/Challenge, PIQA, OpenBookQA, CommonsenseQA, Winogrande, MMLU (continuation / cloze form), TruthfulQA MC1/MC2 (0-shot); TriviaQA (fixed seeded 2,000-question subset) and GSM8K (5-shot); HumanEval (0-shot) and MBPP (3-shot), pass@1 greedy | lm-evaluation-harness 0.4.13 |
| Instruct (chat models) | IFEval, GSM8K chain-of-thought 0-shot, HumanEval / MBPP instruct, XSTest | lm-eval with each model's own chat template; XSTest native |
| Language modelling | Bits per byte on `val_core` and the domain sets (web, synthetic, code, math, chat), each model with its own tokenizer | native (`litegpt/eval/native.py bpb`) |
| Efficiency | Prefill / decode tok/s, **bytes/s**, time to first token, peak VRAM at prompt 128/512 x output 128 x batch 1/8 | native (`litegpt/eval/native.py efficiency`) |

Fixed protocol: seed 1234 everywhere, greedy decoding, bf16, sample-level outputs saved, model revision pinned to the commit
resolved at start, task dataset revisions resolved and recorded per run, code tasks capped at 512 new tokens, no
`trust_remote_code`. Generated code runs only in WSL2.

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

Spider execution accuracy and MultiPL-E JS/TS execution move to M4 (they need databases outside the Hub and Node.js in WSL).
Both stay in the decontamination reference list.
