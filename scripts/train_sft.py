"""M4 chat SFT (EXP-M4-SFT). Running it again continues the newest run of that name, or returns at once if it finished.

    python scripts/train_sft.py --config benchmarks/m4/sft.yaml --name m4-sft-sweep-a --lr 6e-5 --keep-fraction 0.03 --epochs 1
    python scripts/train_sft.py --choose-lr m4-sft-sweep-a m4-sft-sweep-b m4-sft-sweep-c --out runs/m4_sft_lr.json
    python scripts/train_sft.py --config benchmarks/m4/sft.yaml --name m4-sft --lr-from runs/m4_sft_lr.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from litegpt.training import sft as T
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--name")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--lr-from")
    ap.add_argument("--keep-fraction", type=float)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--micro-batch", type=int)
    ap.add_argument("--attention-backend")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--max-steps", type=int, help="smoke and speed runs only")
    ap.add_argument("--runs-root", default=str(ROOT / "runs"))
    ap.add_argument("--choose-lr", nargs="+", help="run names of the sweep")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.choose_lr:
        res = T.choose_lr(Path(a.runs_root), a.choose_lr)
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps(res, indent=1))
        return 0
    lr = json.loads(Path(a.lr_from).read_text(encoding="utf-8"))["chosen_lr"] if a.lr_from else a.lr
    cfg = T.load_config(a.config, name=a.name, lr=lr, keep_fraction=a.keep_fraction, epochs=a.epochs, micro_batch=a.micro_batch, attention_backend=a.attention_backend,
                        compile=False if a.no_compile else None)
    summary = T.train_sft(cfg, runs_root=a.runs_root, max_steps=a.max_steps)
    print(json.dumps(summary, indent=1, default=str))
    return 0 if summary["status"] in ("completed", "stopped_at_max_steps") else 1


if __name__ == "__main__":
    raise SystemExit(main())
