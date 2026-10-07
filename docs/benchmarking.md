# Benchmarking and records

Everything reported about V2 comes from run records (see the documentation layer in the plan and
[ADR-006](decisions/ADR-006-toolchain-oct-2026.md)).

```
benchmarks/, litegpt/, scripts/   tools that produce results (never write results into their own directories)
        |  every run creates runs/<run-id>/  (run.json + metrics.jsonl, bench/*.json, eval/*.json, gate/*.json, ...)
        v
runs/<run-id>/                    the only place experiment outputs live (and the single source of truth)
        |  python -m scripts.build_report        (--check fails if the committed report is stale)
        v
runs/report-latest/REPORT.md + figures/*.svg    generated, deterministic, never edited by hand
```

See [repository_structure.md](repository_structure.md) for what is committed and what is ignored under `runs/`.

## run.json

Written by `litegpt.records.create_run`; a record that misses a required field is rejected. It contains the spec version and
sha256, git commit and dirty flag, config, resolved attention backend, compile flag and the full environment fingerprint
(Python, uv version, `uv.lock` sha256, every resolved package version, torch + CUDA runtime + cuDNN, NVIDIA driver, GPU and VRAM,
OS and WSL distro/kernel, flash-attn and Triton versions or null).

## Training benchmark

```bash
uv run python -m benchmarks.train_mem ceiling                    # measured bf16 matmul ceiling
uv run python -m benchmarks.train_mem single --config litegpt-v2-28l --micro-batch 4 --seq-len 2048
uv run python -m benchmarks.train_mem single ... --sustained-seconds 1800   # sustained throughput + throttling
uv run python -m benchmarks.train_mem sweep --preset m0 [--compile] [--flash]
uv run python -m benchmarks.train_mem ab --a sdpa --b flash                 # alternating A/B, same config, separate processes
uv run python -m benchmarks.train_mem ab --a sdpa --b sdpa --compile-b      # eager vs torch.compile
uv run python -m benchmarks.compile_check                                   # compiled-vs-eager equivalence (CUDA + Triton)
uv run python -m benchmarks.power_sweep auto                                # GPU operating-point sweep (~81 min; caps need admin)
```

Each config runs in its own subprocess on synthetic tokens with the real recipe (fp32 weights, bf16 autocast, fused AdamW,
grad clipping). MFU is reported against the **measured** matmul ceiling, using both FLOP conventions (`full` = nanoGPT/PaLM,
`causal` = half the attention term). The allocator is capped at 97% of VRAM so that Windows cannot silently spill into shared
memory (which would make a too-large configuration look merely slow). The requirement is 16,500 tokens/s sustained
([ADR-001](decisions/ADR-001-compute-budget-and-extension-gate.md)).

GPU telemetry (power, temperature, clocks, throttle reasons) is sampled with NVML during measurement; laptop GPUs are
power-capped under load, which is reported separately from thermal throttling.

Results and analysis of all M0 benchmarks: [m0_benchmark_findings.md](m0_benchmark_findings.md).
