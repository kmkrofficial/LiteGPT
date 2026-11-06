# Changelog

Format: newest first. Spec changes are recorded with their ADR (docs/decisions/).

## Unreleased

### 2026-11-05
- **Models published on Hugging Face:** `LiteGPT-V2-Base` and `LiteGPT-Llama-V2-Instruct`. Both export to the Hugging Face Qwen3 layout, with logits and greedy output identical in float32 (EXP-M5-EXPORT). ADR-011 accepted: Apache-2.0, spec 2.0.0-spec.9.
- **M4 done** (EXP-M4-SFT, EXP-M4-T2D): G2 passed; V2 lost the registered text-to-data comparison (0.800 against SmolLM2-135M 0.817, Gemma 3 270M 0.822, Qwen3-0.6B 0.851). Python API and CLI added. All benchmarks in `docs/results.md`.
- M5 documentation and clean-up: READMEs in every top-level folder, `docs/results.md`, `docs/history.md`, draft model cards and notices for Base and Instruct. ADR-011: `LiteGPT-V2-Base` and `LiteGPT-Llama-V2-Instruct` to be released (owner); licence proposed. All charts generated from run records in `docs/figures/`. Personal paths removed from run records. Backup now covers G2's evaluation run. Development history squashed for release.

### 2026-10-13
- **M1b done** (EXP-M1B-SUITE final at suite 0.1.1, EXP-M1B-BASELINES): 16 models (10 base, 6 instruct) measured on this laptop;
  report section 3d with counted parameters, published context, base / instruct / XSTest / bits-per-byte / efficiency tables.
  V1-Base beats GPT-2 on all 10 zero-shot tasks (mean 0.371 vs 0.346) and is weakest of all 16 on code bits-per-byte.
- Suite corrections before the affected scores were final: IFEval at the registered 512 tokens (lm-eval's ifeval extra added to
  the eval group), XSTest at 256, GSM8K chat scored with flexible-extract, corrected chat code tasks (suite 0.1.1).
- M2 training probes the micro-batch under a VRAM cap and halves it (doubling accumulation) until one step fits; tokens per step
  unchanged.

### 2026-10-11
- **Tokenizer frozen** (owner, M1 Checkpoint B): `litegpt/assets/tokenizer/litegpt-32k.json`, sha256 `2f208b58085074d9...` in spec
  `tokenizer.sha256`. 32,768 entries; web 0.966x GPT-2 bytes/token, Python/JS/TS/Shell/SQL 1.2-1.7x better (EXP-M1-TOKENIZER).
- C9 re-plan with the real tokenizer: code files 0-261, code decay fraction 0.08, instruction decay fraction 0.85.
- **Spec 2.0.0-spec.6** (ADR-008): Wikipedia removed from the mixture and redistributed proportionally; focused code mix (Python 63 /
  JavaScript 14 / TypeScript 14 / SQL 4 / Shell 5); decontamination now filters pretraining data; tokenizer, `val_core` and data-order
  hashes added as `pending`; new binding file `docs/spec/data_sources.yaml` (exact revisions, file lists, partitions, filters).
- M1 pre-flight (EXP-M1-PREFLIGHT, C1-C7): shared document filters `litegpt/datafilters.py`, `scripts/audit_sources.py`
  (`preflight-code`, `terms`, `throughput`, `evalsets`), 18 evaluation sets pinned for decontamination.
- Licence register `docs/data_licenses.md`: generator terms (Llama 3.1, Qwen2.5-72B) carried to the M5 gate.

### 2026-10-07
- GPU operating-point sweep (EXP-M0-POWER) completed: `benchmarks/power_sweep.py` (`screen` / `compare` / `soak` and the
  faster single-process `auto` protocol), per-second telemetry incl. the enforced power limit, pre-registered validity /
  candidate / soak rules, report section with tok/s vs power, tok/s/W vs power, tok/s vs temperature and the Pareto front.
  Harness fixes found along the way: memorised-batch loss signal (now fresh random tokens per step), skipped-point reasons,
  measured-span boundary. Decision: keep the current 155 W setting, no clock cap (1935 MHz validated fallback).
- `docs/m0_benchmark_findings.md`: consolidated M0 test results and analysis (throughput, memory, flash-attn, compile,
  power limit, operating-point sweep, test suite).

### 2026-10-06
- Linux flash-attn wheel (2.8.3, cu130/torch2.14/cp312) pinned in `uv.lock` alongside the Windows wheel; correct on WSL2 under
  torch.compile (68/68 tests, 28L equivalence). Sustained flash+compile: 29,944 tok/s, but the GPU power limit had changed from
  95 W to 155 W, so the speedup is not attributable to flash (EXP-M0-FLASH-LINUX: speed inconclusive, not adopted as default).
  Telemetry now records the enforced GPU power limit; the report shows it for sustained runs.
- **Spec 2.0.0-spec.5**: `torch.compile` adopted for production training on WSL2 after passing the pre-registered rule
  (equivalence on 28L; +23.9% tok/s; peak VRAM 11.17 -> 8.28 GiB; 13 s warmup). Sustained 30-min: 22,592 tok/s with compile at
  micro-batch 4x2048, so 10B tokens ~ 5.1 days. WSL2 (Ubuntu 26.04) reproduces the locked environment; it needs `build-essential`.
  New `benchmarks/compile_check.py`; the A/B benchmark gained `--compile-b`; the report shows A/B summaries and the equivalence table.
- **Repository restructure** (ADR-007, spec 2.0.0-spec.4: path pointers only, no value changes): top level is now `litegpt/`,
  `tests/`, `benchmarks/`, `scripts/`, `scratch/`, `runs/`, `docs/`. V1 baseline moved (git mv, history kept) to
  `scripts/legacy_v1/`; spec and registry moved from `experiments/` to `docs/spec/`; the report builder moved from `reports/` to
  `scripts/build_report.py` and now reads `runs/` directly and writes `runs/report-latest/`. The top-level `reports/` directory,
  `reports/collect.py` and the redundant `reports/data/` copies are removed (all five original runs are intact in `runs/`).
  `.gitignore` updated (`runs/` is now tracked except telemetry/checkpoints/shards; `scratch/` ignored except its README).
- **Spec 2.0.0-spec.3**: flash-attn adopted for `attention_backend=auto` after passing the pre-registered parity and +5% speed
  rule (measured +12.2% tok/s, identical VRAM). Installed from a community prebuilt Windows wheel (2.8.3+cu130torch2.14) via
  `uv sync --locked --group flash`; SDPA stays mandatory and the fallback. See ADR-006 evidence / EXP-M0-FLASH.
  Sustained 30-min throughput with flash: 17,553 tok/s (>= the 16.5k requirement; 10B tokens ~ 6.6 days).
- **Spec 2.0.0-spec.1 approved** (ADR-000..005): 28-layer / 201.4M architecture, 10B-token budget with the pre-registered
  10B -> 15B extension gate (G1, threshold 0.010 BPB), SDPA mandatory / flash-attn optional, public synthetic datasets allowed
  but no external teacher models, MIT code licence with the upstream notice and model licence pending dataset audit.
- **Spec 2.0.0-spec.2** (ADR-006): toolchain update - Python 3.12, uv + `uv.lock`, torch 2.14.1+cu130, transformers 5.18.0,
  tokenizers 0.23.2, safetensors 0.8.0, numpy 2.5.3, datasets 5.1.0, trl 1.14.1, peft 0.21.2, accelerate 1.15.0,
  lm-eval 0.4.13, matplotlib 3.11.2; flash-attn 2.8.3.post1 optional/experimental.
- Spec file fix (no semantic change): floats written as `9.0e+9` instead of `9.0e9` because PyYAML loads the latter as a
  string. Guarded by `tests/test_spec.py::test_no_numeric_value_silently_loaded_as_string`. The gate values are unchanged.
- M0 foundations: `litegpt/` package (config, model, spec, records, checkpoint), `benchmarks/`, tests, `LICENSE` restored
  (MIT, upstream notice), `pyproject.toml` + `uv.lock`.
