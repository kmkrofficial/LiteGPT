"""Train a model from a run config (M2 proxy ladder, M2 learning-rate sweeps, M3 pretraining).

    python scripts/train.py --config benchmarks/ladder/s1.yaml --data-run <data run dir> [--seed 2] [--lr 2e-3] [--tokens 0.25e9]
    python scripts/train.py --resume runs/<train run id>        # continue an interrupted run exactly
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    from litegpt.training import TrainConfig, load_config, train
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--resume")
    ap.add_argument("--data-run")
    ap.add_argument("--val-text-dir")
    ap.add_argument("--name")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--tokens", type=float)
    ap.add_argument("--max-steps", type=int, help="smoke runs only: stop early (the schedule keeps its full length)")
    args = ap.parse_args()
    if args.resume:
        run = json.loads((Path(args.resume) / "run.json").read_text(encoding="utf-8"))
        cfg = TrainConfig(**{**run["config"], "betas": tuple(run["config"]["betas"])})
        summary = train(cfg, resume=args.resume)
    else:
        cfg = load_config(args.config, data_run=args.data_run, val_text_dir=args.val_text_dir, name=args.name, seed=args.seed,
                          lr=args.lr, tokens=args.tokens)
        summary = train(cfg, max_steps=args.max_steps)
    print(json.dumps({k: v for k, v in summary.items() if k != "telemetry"}, indent=1, default=str))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
