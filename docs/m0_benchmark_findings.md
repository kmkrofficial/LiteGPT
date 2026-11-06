# M0 benchmark findings: throughput, memory, attention, compile and GPU operating point

Date: 2026-10-06/07. Hardware: Lenovo 83DE (Legion Pro 7i), RTX 4090 Laptop GPU 16 GB (driver 610.88), i9-14900HX.
Workload for every number below unless stated: LiteGPT V2 28L (201.4M params), bf16 autocast, fused AdamW, grad clipping,
synthetic tokens (throughput does not depend on data content). Every number cites the run record it comes from
(`runs/<run-id>/`); the generated report (`runs/report-latest/REPORT.md`) is built from the same records.
Decision records: [ADR-001](decisions/ADR-001-compute-budget-and-extension-gate.md) (budget, evidence),
[ADR-006](decisions/ADR-006-toolchain-oct-2026.md) (toolchain, flash-attn, compile); experiment registry:
[`docs/spec/registry.yaml`](spec/registry.yaml).

## 1. Summary

| Question | Answer | Evidence |
|---|---|---|
| Production training environment | **WSL2 + torch.compile + SDPA, micro-batch 4x2048** (2x4096 for the context-extension phase) | EXP-M0-COMPILE |
| Sustained throughput at the current GPU setting (155 W limit) | **~30k tok/s** (29,944 tok/s over 30 min, flash+compile); **10B tokens ~ 3.9 days** | 20261006T162546Z |
| Does the 16,500 tok/s requirement (10B in 7 days) hold? | Yes, with a large margin (it failed only on Windows SDPA eager, at a 95 W limit) | ADR-001 evidence |
| flash-attn | Correct on Windows and WSL2; +12.2% on Windows (eager); on WSL2 its speed effect is **not attributable** (power limit changed mid-comparison). Not the WSL2 default | EXP-M0-FLASH, EXP-M0-FLASH-LINUX |
| torch.compile | **Adopted on WSL2**: +23.9% tok/s, lower VRAM, outputs equivalent; unavailable on Windows (no Triton) | EXP-M0-COMPILE |
| Dominant performance variable | **The laptop's GPU power limit** (95 W vs 155 W changed throughput by ~30%), not VRAM | ADR-001 evidence |
| GPU operating point (clock cap / undervolt) | **Keep the current 155 W setting, no clock cap.** The 1935 MHz cap is the validated fallback if more thermal margin is wanted | EXP-M0-POWER, section 5 |

## 2. Environment

- Windows (development): Python 3.12.15, torch 2.14.1+cu130, CUDA 13.0, cuDNN 9.24, uv 0.12.23, everything from `uv.lock`.
  No Triton, so `torch.compile` fails (`TritonMissing`). flash-attn 2.8.3 community wheel (Windows).
- WSL2 (production): Ubuntu 26.04, kernel 6.18, same `uv.lock`; Triton 3.8.0 present; needs `build-essential` for Triton.
  flash-attn 2.8.3 community wheel (Linux x86_64). See [reproduce.md](reproduce.md).
- Not available on this machine: GPU voltage readout, hotspot / VRAM temperature, voltage-curve control (no Afterburner;
  NVML offsets need admin). Lenovo power mode is not readable without admin.

## 3. Throughput and memory

| Configuration | Result | Run |
|---|---|---|
| Windows, SDPA eager, short runs | 28L mb3x2048 22,279 tok/s (11.17 GiB); mb2 20,700; mb1 13,163; 1x4096 17,674 | 20261006T122954Z |
| Windows memory boundary | mb4x2048 and 2x4096 OOM (14.72 GiB free with the display attached) | 20261006T122603Z, 20261006T122954Z |
| Windows, SDPA eager, 30 min | **15,553 tok/s**, 20.6k -> 14.8k (thermally throttled 98%, 89 C, 95 W era) | 20261006T130256Z |
| Windows, flash eager, 30 min | **17,553 tok/s**, stable (95 W era) | 20261006T134753Z |
| WSL2, SDPA + compile, mb4x2048, 30 min | **22,592 tok/s**, stable, max 76 C, mean draw 95 W | 20261006T151250Z |
| WSL2, flash + compile, mb4x2048, 30 min | **29,944 tok/s**, ramped 23.7k -> 30k when the power limit rose to 155 W; 0% thermal throttling, max 84 C | 20261006T162546Z |

Harness findings that changed how we measure:
- **Windows silently spills VRAM overflow into shared system memory** (a too-large batch ran ~5x slower instead of failing).
  The benchmark now caps the allocator at free VRAM minus 1 GiB and flags runs near the device limit (`vram_pressure`).
- **Short runs overstate sustained throughput** on a laptop (the first minute is 20-30% faster than steady state), so every
  decision used 30-60 minute sustained runs.
- **The enforced GPU power limit is dynamic** (OEM mode / Dynamic Boost: 80 W default, observed 95 W and 155 W, max 175 W) and
  was invisible in early records; telemetry now records it every second.

## 4. Attention backend and torch.compile

Both were adopted only through rules written into the registry **before** measuring.

**flash-attn (EXP-M0-FLASH, Windows):** parity with SDPA (plain and packed `cu_seqlens`) passed; 4 alternating A/B rounds at 28L
mb3x2048: SDPA 15,375 vs flash 17,250 tok/s = **+12.2%**, identical VRAM (run 20261006T131103Z). Adopted for `attention_backend=auto`.

**torch.compile (EXP-M0-COMPILE, WSL2):** equivalence with identical weights (28L): loss diff 0.0004, logits rel L2 0.88%,
grad-norm rel diff 0.003% (run 20261006T151400Z). A/B 28L mb3x2048: eager 18,561 vs compiled 22,997 tok/s = **+23.9%**, peak VRAM
11.17 -> 8.28 GiB, warmup 4 s -> 13 s (run 20261006T143350Z). Compiled batches: mb4 23,695 tok/s at 10.14 GiB (best), mb5 23,441,
mb6 OOM, 2x4096 20,792 (runs 20261006T143939Z..144221Z). Adopted for production training.

**flash-attn on WSL2 (EXP-M0-FLASH-LINUX):** correctness passed (68/68 tests with flash + compile; 28L compiled-flash equivalence,
run 20261006T155451Z). Speed **inconclusive**: the comparison baseline ran at a 95 W limit and the flash run at 155 W. Not the
WSL2 default; production configs set the backend explicitly.

## 5. GPU operating-point sweep (EXP-M0-POWER)

### 5.1 Goal and method

Find the highest-throughput **stable** operating point with no thermal throttling and a sensible thermal margin. Tooling:
`python -m benchmarks.power_sweep`. Operating points: `observe` (current setting), `manual:<label>` (operator switches a Lenovo
mode), `lgc:<MHz>` (graphics-clock cap via `nvidia-smi --lock-gpu-clocks`, elevated session only, always reset). A clock cap moves
the GPU down its **factory** voltage/frequency curve; the tool never applies voltage offsets, never raises power limits and never
touches the CPU. Points abort at 90 C or on any hardware slowdown.

Pre-registered rules (unchanged throughout): **invalid** on crash / CUDA error / non-finite loss / loss rising >10% above its
running minimum / tok/s CV > 0.08 / >= 90 C / HW slowdown; **candidate** = valid AND 0% thermal throttling AND peak temperature
<= 82 C (85 C target - 3 C) AND CV <= 0.03 AND last/first window >= 0.97; **selection** = highest tok/s, ties within 1% to the
lower power; **soak** = the selected point must still be a candidate over 60 min; **adoption** = owner decision, never automatic.

### 5.2 Protocol evolution and defects found (all runs kept for provenance)

| Run | What | Outcome |
|---|---|---|
| 20261006T164224Z | Staged `screen`, non-elevated validation | Current setting 31,120 tok/s at 155 W, max 80 C; failed decay (0.961): a ~5 min thermal transient dominates short cold-start points. The `lgc` point was safely skipped (no elevation) |
| 20261006T165630Z | Staged `screen`, elevated (owner) | Interrupted after ~3.5 min of its first point; no results |
| 20261006T170001Z | New `auto` protocol smoke test (short) | Pipeline works end-to-end (single continuous process, live window stream) |
| 20261006T171445Z | `auto`, elevated | **Harness defect:** the workload trained on one repeated batch, the loss collapsed to ~1e-7, and the 10% loss-rise rule fired on rounding noise (e.g. 5.2e-8 -> 8.2e-8), invalidating all cap points. Physical measurements valid |
| 20261006T173836Z | Smoke test after the fix | Fresh random tokens every step keep the loss at ~10.40; no false flags; throughput unchanged |
| **20261006T174824Z** | **`auto`, elevated, corrected** | **Final result (below)** |

Why `auto`: one continuous workload process (compiled once, never cooled down); the controller switches operating points live;
caps are derived below the measured natural clock (caps above it never bind). ~81 min instead of ~2.5-3.5 h for the staged
protocol. The schedule change and the workload fix were recorded in the registry before the corrected run.

### 5.3 Final sweep (run 20261006T174824Z; 4-min segments, 1 min settle; WSL2, SDPA + compile, mb4x2048, 155 W limit)

| Point | tok/s | vs current | GPU W mean / peak | SM MHz | Temp C mean / peak | tok/s per W | Thermal throttling | Candidate |
|---|---|---|---|---|---|---|---|---|
| current 155 W (observe) | 31,388 | - | 154.7 / 161.4 | 2,038 | 81.5 / 83 | 202.9 | 0% | no: peak 83 > 82 |
| cap 1980 MHz | 30,774 | -2.0% | 149.3 / 150.9 | 1,980 | 82.0 / 83 | 206.2 | 0% | no: peak 83 > 82 |
| **cap 1935 MHz** | **30,204** | **-3.8%** | 142.5 / 157.7 | 1,935 | 81.1 / 82 | 211.9 | 0% | **yes (selected)** |
| cap 1830 MHz | 28,923 | -7.9% | 133.1 / 139.3 | 1,830 | 79.6 / 80 | 217.3 | 0% | yes |

Pareto front (throughput vs peak temperature): current setting, 1935 MHz, 1830 MHz (1980 MHz is dominated by the current setting).

### 5.4 The 60-minute soak at 1935 MHz

Result: 29,650 tok/s mean, 141.8 W, clock 1,934 MHz (cap held), temperature mean 81.2 C / peak 82 C, **0% thermal throttling**,
loss 10.397-10.398 (healthy). **Formally failed** the pre-registered decay criterion: last/first window = 0.968 < 0.97.

Analysis:
- The decay is a **one-time step, not a trend**: 5-minute block means 30.22k, 30.23k, then flat at ~29.2-29.8k for the remaining
  50 minutes (linear fit -2.8%/h overstates it; first-5-min vs last-5-min 0.966).
- **The GPU was steady throughout**: temperature 81 C in every 5-minute block, power 141-142 W, clock locked at 1,935 MHz (4 of
  3,461 samples below 1,920 MHz), memory clock 9,001 MHz, utilisation 98.7-99.3%, enforced power limit 155 W.
- So the ~2% step is not GPU thermal or power throttling. Likely causes (not measurable here): VRAM temperature (not exposed on
  this GPU) or host-side activity.
- The decay criterion compares single 30-second windows, which makes it fragile (the same weakness as using a single 1-second
  peak-temperature sample for the thermal margin; the current setting failed that by 1 C).

### 5.5 Decision (owner, 2026-10-07)

**Keep the current 155 W setting with no clock cap** (no change to production). Reasons: it is the fastest point and never
thermally throttled in any run (mean ~81 C is normal under full load; it failed only a 1 C single-sample peak margin); a cap buys
only ~1-2 C and ~13 W for ~4% less throughput (~3-4 h more on 10B tokens); a clock cap needs an elevated session and must be
re-applied after every reboot or driver reset, an operational risk over a 4-5 day run. Training keeps per-second telemetry
(temperature, power limit) with an alert/abort at 90 C. **Fallback:** the 1935 MHz cap passed every thermal rule and held steady
after the first 10 minutes. Possible future improvement: robust statistics for the candidate rule (mean temperature, block-trend
decay) pre-registered before any further sweep.

## 6. Tests

### Automated test suite (`uv run pytest`): 87 passed, 2 skipped on Windows (the 2 skips need Triton: compile tests run on WSL2)

| File | What it verifies |
|---|---|
| `tests/test_model.py` | parameter counts, init statistics, KV-cache decoding == full forward, chunked prefill, RoPE properties, GQA == explicit MHA, packed `cu_seqlens` == separate sequences, loss masking, gradients everywhere, flash vs SDPA parity (plain, packed, `auto`), compiled == eager (SDPA and flash) |
| `tests/test_spec.py` | spec loads and hashes, extension gate pre-registered and consistent, config conforms, numeric values not silently strings |
| `tests/test_records.py` | run records valid, every required environment field present, invalid records rejected, JSONL / atomic writes |
| `tests/test_checkpoint.py` | bit-exact resume, keep-last + milestones, interrupted saves ignored, config/spec guards, RNG restore, graceful Ctrl+C |
| `tests/test_reports.py` | report built from `runs/` only, deterministic output, invalid records rejected, stale-report check, A/B and equivalence sections, power-sweep section and plots |
| `tests/test_power_sweep.py` | point parsing and safety bounds (no voltage offsets / power raises in the tool), metric slicing, validity and candidate rules, selection and Pareto front, no clock cap without elevation, cap derivation, segment slicing |

### Hardware validation runs

compile equivalence (SDPA 20261006T151400Z, flash 20261006T155451Z); flash vs SDPA A/B (20261006T131103Z); eager vs compile A/B
(20261006T143350Z); four 30-minute sustained runs (section 3); the power-sweep runs in section 5.2.

## 7. Reproduce

```bash
uv sync --locked [--group flash]
uv run pytest
uv run python -m benchmarks.train_mem ab --a sdpa --b sdpa --compile-b        # eager vs compile (WSL2)
uv run python -m benchmarks.compile_check --backend sdpa                     # equivalence
uv run python -m benchmarks.power_sweep auto --cap-steps 50,100,200          # ~81 min; clock caps need an elevated session
uv run python -m scripts.build_report                                        # regenerate runs/report-latest/
```
