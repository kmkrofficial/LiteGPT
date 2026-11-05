# Benchmarks and run configs

Results go to [`runs/`](../runs/README.md). Conventions: [docs/benchmarking.md](../docs/benchmarking.md).

## Run configs

| Folder | What it holds |
|---|---|
| `ladder/` | M2 design ladder, one change per step (`s1` to `s8b`), and the M3 learning-rate sweep |
| `m3/` | M3 pretraining (`m3.yaml`), its phase test and a tiny smoke run |
| `m4/` | Chat fine-tuning (`sft.yaml`), the text-to-data recipe (`t2d.yaml`) and the frozen test-item hashes (`t2d_manifest.json`) |

## Tools

| Tool | What it measures |
|---|---|
| `train_mem.py` | Training throughput, memory and utilisation |
| `tune_throughput.py` | The fastest correct training setup on a rented GPU, within a time budget |
| `compile_check.py` | That `torch.compile` and eager mode give equivalent results |
| `eval_speed.py` | Whether a faster evaluation path gives the same answer |
| `power_sweep.py` | The GPU operating point: speed against power and heat |
| `ablation.py` | The proxy model behind the ladder, with switchable V1 and V2 parts |
| `telemetry.py` | GPU power, heat and throttling during a run |

```bash
uv run python -m benchmarks.train_mem single
```
