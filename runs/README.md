# Run records

Every number reported about V2 comes from a record here. A run's folder is named `<UTC time>-<kind>-<name>-<commit>`, where kind is `train`, `eval`, `bench`, `data` or `test`. [`report-latest/REPORT.md`](report-latest/REPORT.md) is generated from them. What each experiment was testing is in the [registry](../docs/spec/registry.yaml).

| File | What it holds |
|---|---|
| `run.json` | Config, spec hash, commit, environment and package versions |
| `metrics.jsonl`, `eval.jsonl`, `events.jsonl` | Training steps, evaluations and events |
| `train_summary.json` | How a training run ended |
| `eval/` | Evaluation scores |
| `dev/`, `test/` | Text-to-data scores and outputs |

**Comparison results:** [`m4_t2d/test_results.json`](m4_t2d/test_results.json), with outputs in `m4_t2d/predictions/`. **Queue records:** `m4_queue/`.

## Read this before relying on them

- **Commit ids don't resolve here.** The commit in a run's name and `run.json` comes from the development history, which was squashed for release. A private copy of that history is kept.
- **Paths were cleaned.** Home-directory paths now read `/home/user`, and two file names no longer carry machine names. Nothing else was changed.
- **Failed runs are kept on purpose.**
- **Not here:** weights, checkpoints, data shards, telemetry and per-sample evaluation outputs, because they are large.
