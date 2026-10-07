# ADR-001: Compute budget and pre-registered 10B → 15B extension gate

Status: Accepted
Date: 2026-10-06
Spec fields: `training.pretraining`, `training.extension_gate`, `data.prepared_tokens`, `data.validation.val_core`

## Context

- The 28-layer V2 model costs about 1.47 GFLOP per training token.
- Training 10B tokens in 7 days requires a **sustained 16.5k tok/s** on the RTX 4090 Laptop. The eager-mode estimate is 15–22k tok/s; M0 measures the real figure.
- Small models keep improving past the Chinchilla ratio, so continuing to 15B may be worthwhile. But that choice must not be made by looking at results and picking a convenient threshold.
- With a warmup–stable–decay (WSD) schedule, stable-phase losses at different points are comparable with each other. They are not comparable with post-decay losses.

## Decision

### Budget
- The initial pretraining budget is **10.0B tokens**, with a maximum of **15.0B**.
- The schedule:
  - **Stable phase until 9.0B tokens.**
  - Then a **1.0B-token decay** on the high-quality decay mixture.
  - The 4096-context extension (RoPE θ = 100,000) happens in the final 0.3–0.5B tokens of the decay.
- **All 15.5B tokens (15B plus validation) are prepared before M3 starts.** The first 9B tokens of the data order are identical in both outcomes, and training never stalls waiting on data preparation.
- **If the M0 benchmark shows less than 16.5k tok/s sustained**, the order of response is:
  1. `torch.compile` on WSL2, plus flash-attn if available.
  2. Accept a longer wall clock.
  3. Switch to the 24-layer fallback, only by explicit owner decision recorded in a new ADR.

### Pre-registered extension gate (frozen before M3)
- **Metric:** validation bits-per-byte on `val_core`. This is a fixed held-out slice of the stable mixture with at least 20M tokens, frozen in M1 with its sha256 recorded.
- **Measurement:** stable-phase evaluations run every 0.25B tokens.
  - `BPB_7` = mean of the evals at 6.50B, 6.75B and 7.00B tokens.
  - `BPB_9` = mean of the evals at 8.50B, 8.75B and 9.00B tokens.
- **Statistic:** `Δ = BPB_7 − BPB_9` (improvement over the last 2B stable tokens).
- **Threshold:** **0.010 BPB**.
- **Decision**, taken at the 9.0B stable checkpoint, before any decay:
  - `Δ ≥ 0.010`: **extend.** Continue the stable phase to 14.0B, then decay for 1.0B, for 15.0B total.
  - `Δ < 0.010`: **stop.** Decay from 9.0B to 10.0B, for 10.0B total.
- **Invalidation:** a divergence, a rollback, or any config or data change inside either window means **stop** (the conservative default).
- **Automation:**
  - A script computes the decision from the metrics records and writes a gate record (inputs, Δ, threshold, decision, spec hash).
  - The generated report shows that record.
- **Overrides:** only through a new ADR recorded as a protocol deviation. Reports display any deviation prominently.
- **Freeze:** the threshold, windows, metric and evaluation set are frozen when M3 starts. The M3 `run.json` stores the spec hash. Any edit after M3 starts invalidates the gate.

### Rationale for 0.010 BPB
- A Chinchilla-form data term (L(D) ∝ D^−0.28) predicts about 0.017 BPB of improvement between 7B and 9B tokens for a model of this size. That assumes roughly 4.2 bytes per token.
- 0.010 BPB is about 60% of that prediction. If the observed improvement falls below it, the model is flattening faster than scaling laws predict, so another 5B tokens (about 3.5 GPU-days) is poor value.
- With at least 20M validation tokens and 3-point means, measurement noise is well below 0.010.

## Alternatives rejected
- **Choosing the threshold after seeing curves:** that's post-hoc and could be steered toward any answer.
- **Deciding after a 10B decay:** it wastes about 1B decay tokens if we then extend, and post-decay loss isn't comparable with stable-phase loss.
- **Using a seed-noise band from proxy runs:** noise measured at about 40M parameters doesn't transfer to a 201M model.
- **Using benchmark accuracy as the gate:** it's too noisy at this scale and between these checkpoints.
- **Preparing extension data only if the gate passes:** training would stall for many hours of CPU work at 9B.

## Consequences
- Training disk use rises by about 10 GB (15.5B tokens as uint16 is about 31 GB). That still fits the 50–70 GB budget.
- If the gate passes, compute rises by about 3.5 GPU-days.
- The gate outcome and its inputs are part of the published V2 report.

## Evidence
- Pending M3: gate record `runs/<m3_run>/gate/extension_gate.json`.

### M0 throughput evidence (2026-10-06, Windows native, eager, SDPA; records in `runs/<run-id>/`, generated report section 3 in `runs/report-latest/REPORT.md`)
- **With SDPA the 16.5k tok/s requirement is not met when sustained** (flash-attn changes this, see the next bullet). 28L, micro-batch 3×2048, 30 minutes, run
  `20261006T130256Z-bench-sustained-28l-mb3-30min-101c3f8`: first minute 20,550 tok/s, mean **15,553 tok/s**, last minute
  14,812 tok/s (ratio 0.72). The GPU was thermally throttled in 98% of busy samples (max 89 C, mean SM clock 1,446 MHz).
  Short, un-throttled runs reach 22,279 tok/s, which overstates what a multi-day run delivers.
- Implied wall clock at the sustained mean: **10B tokens = 7.4 days** (about 6% over the 7-day target; 7.8-8.2 days at the
  slowest windows); 15B = 11.2 days. The ADR-001 response order applies: (1) `torch.compile` / WSL2 (not yet measurable:
  Triton is unavailable on Windows and WSL2 is not installed), (2) accept a longer wall clock, (3) 24L only by owner decision.
  No decision is taken on this bullet alone; see the flash-attn result below.
- **With flash-attn (adopted, ADR-006 / EXP-M0-FLASH) the requirement is met.** 28L, micro-batch 3x2048, 30 minutes, run
  `20261006T134753Z-bench-sustained-28l-mb3-flash-30min-101c3f8`: mean **17,553 tok/s**, first minute 17,001, last minute 17,131 (ratio 1.008, stable),
  power-capped 99.9% / thermally throttled 6.9% of busy samples, max 77 C, mean SM clock 1,397 MHz, peak VRAM 11.17 GiB.
  Implied wall clock: **10B tokens = 6.6 days, 15B = 9.9 days**, inside the ~7-day budget. No fallback step (WSL2 compile, longer
  wall clock, 24L) is needed on this evidence. Caveats: one sustained run per backend; the flash run started with a warm GPU
  (first window 17.0k vs 20.6k for the cold SDPA start), so cold-start comparisons favour SDPA's first minute; real training adds
  data loading and periodic evaluation/checkpointing overhead (a few percent); ambient temperature / cooling can shift results.
- **Memory (supersedes the plan estimates for Windows with the display attached):** free VRAM at start was 14.72 GiB.
  28L fits at micro-batch 3×2048 (11.17 GiB peak) and 2×2048 (8.34 GiB), and at 1×4096 (8.34 GiB, 17.7k tok/s) for the
  context-extension phase. Micro-batch 4×2048 and 2×4096 hit a real OOM. Pretraining therefore uses micro-batch 3 with
  gradient accumulation (about 85 steps per 0.5M-token batch).
- **Production training environment (decided on this evidence): WSL2 + torch.compile + SDPA, micro-batch 4x2048**
  (2x4096 for the context-extension phase). Sustained 30-minute mean **22,592 tok/s** (stable, last/first 0.977, max 76 C), run
  `20261006T151250Z-bench-sustained-28l-mb4-compile-wsl-30min-101c3f8`: **10B tokens = 5.1 days, 15B = 7.7 days**. That is +29% over the best Windows result
  (flash, eager, 17,553 tok/s) and +45% over Windows SDPA eager (15,553). 0.5M-token steps need about 61 accumulation
  micro-steps (64 gives 524,288 tokens/step). Caveats as above: one sustained run per configuration, real training adds
  data-loading / evaluation / checkpoint overhead (a few percent). Details: ADR-006 evidence (EXP-M0-COMPILE).
- **The GPU power limit is the dominant variable.** The laptop's enforced GPU power limit is dynamic (OEM power mode / Dynamic
  Boost; max 175 W, default 80 W). All earlier sustained runs ran at about 95 W. The flash+compile sustained run (28L mb4x2048,
  run `20261006T162546Z-bench-sustained-28l-mb4-flash-compile-wsl-30min-101c3f8`) ran with a 155 W limit: **29,944 tok/s** mean (mean draw 151 W, mean clock 1,989 MHz, 0% thermal throttling,
  max 84 C, peak VRAM 10.14 GiB) -> **10B tokens = 3.9 days, 15B = 5.8 days** at that power level. The improvement over the
  95 W runs is mostly the power limit, not the attention kernel (EXP-M0-FLASH-LINUX). Planning consequence: the multi-day run
  should use the high-power mode, and every run now records the enforced power limit in its telemetry.
- **GPU operating point (EXP-M0-POWER, decided 2026-10-07): keep the current 155 W setting, no clock cap.** A pre-registered
  sweep of factory-curve clock caps selected 1935 MHz (30,204 tok/s, -3.8%, 142.5 W, peak 82 C), but its 60-minute soak failed
  the decay rule (0.968 vs 0.97; a one-time ~2% step, GPU steady). The current setting is fastest (31,388 tok/s in the sweep)
  with 0% thermal throttling; a cap would cost ~4% throughput for ~1-2 C and would need re-applying after every reboot.
  1935 MHz is the validated fallback. Analysis: [M0 benchmark findings](../m0_benchmark_findings.md) section 5.
- Harness note: a Windows (WDDM) GPU silently spills VRAM overflow into shared system memory and runs ~5x slower instead of
  failing (observed: 3.9k tok/s at 4×2048). The benchmark now caps the allocator at free-VRAM minus 1 GiB and flags runs
  that approach the device limit (`vram_pressure`); such results are excluded from charts and from the requirement check.
