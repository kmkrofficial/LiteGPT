"""M4 gate G2 and the test-scoring speed rule, run by the queue so M4 can finish without a person (owner, 2026-11-03).

    python scripts/m4_g2.py gate          # IFEval on the SFT checkpoint; registered rule; passes or drops V2's chat arm
    python scripts/m4_g2.py speed-rule    # the registered 1.5x rule: moves test scoring to the laptop if the A100 is too slow

Only IFEval runs on the server: its outputs are checked as text and never executed. The rest of the instruct suite, which includes code
tasks, stays on the laptop in WSL2. If the gate cannot be measured it does not pass, because the registered rule says it must pass.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
QUEUE = ROOT / "runs" / "m4_queue"
V1_PASSES, V1_ITEMS = 45, 541           # run 20261012T153246Z-eval-instruct-LiteGPT-Instruct-3afbe97: 0.0832
LAPTOP_SPEED_SECONDS = 241.42           # docs/private/logs/m4/phase1/real_paths.log: starting V2 weights, all 3,087 dev items
MIN_SPEED_RATIO = 1.5


def decide(score: float, items: int, diverged: bool) -> dict:
    """Above V1-Instruct by more than 2 standard errors of the difference, and no divergence. The standard error treats the two scores as
    independent proportions, fixed here before the result is known."""
    v1 = V1_PASSES / V1_ITEMS
    se = math.sqrt(score * (1 - score) / items + v1 * (1 - v1) / V1_ITEMS)
    above = score - v1 > 2 * se
    return {"score": score, "items": items, "v1_instruct": v1, "difference": score - v1, "se_of_difference": se, "above_by_2_se": above, "diverged": diverged, "passed": bool(above and not diverged)}


def speed_rule(seconds: float, laptop_seconds: float = LAPTOP_SPEED_SECONDS) -> dict:
    ratio = laptop_seconds / seconds
    return {"a100_seconds": seconds, "laptop_seconds": laptop_seconds, "ratio": ratio, "minimum": MIN_SPEED_RATIO, "test_on": "cloud" if ratio >= MIN_SPEED_RATIO else "laptop"}


def run_gate() -> dict:
    from litegpt.training.sft import find_run
    run = find_run(ROOT / "runs", "m4-sft")
    if run is None or not (run / "train_summary.json").exists():
        return {"passed": False, "reason": "no finished SFT run"}
    summary = json.loads((run / "train_summary.json").read_text(encoding="utf-8"))
    diverged = summary["status"] != "completed" or summary.get("best") is None or not math.isfinite(summary["best"]["loss"])
    ckpt = run / "checkpoints" / "milestone-final.pt"
    if not ckpt.exists():
        return {"passed": False, "reason": "no final checkpoint", "diverged": diverged}
    p = subprocess.run([sys.executable, "-m", "litegpt.eval.run", "--model", f"litegpt:{ckpt}", "--suite", "instruct", "--groups", "ifeval"], cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)},
                       capture_output=True, text=True, timeout=3 * 3600)
    print(p.stdout[-1500:], p.stderr[-1500:], flush=True)
    record = re.findall(r"record: (\S+)", p.stdout)
    if p.returncode != 0 or not record:
        return {"passed": False, "reason": f"IFEval did not finish: exit {p.returncode}", "diverged": diverged}
    eval_dir = ROOT / "runs" / record[-1] / "eval"
    headline = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))["headline"]["ifeval"]
    items = sum(1 for _ in open(eval_dir / "ifeval.samples.jsonl", encoding="utf-8"))
    return {**decide(headline["value"], items, diverged), "record": record[-1], "sft_run": run.name, "sft_best": summary.get("best")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["gate", "speed-rule"])
    a = ap.parse_args()
    QUEUE.mkdir(parents=True, exist_ok=True)
    if a.command == "gate":
        try:
            res = run_gate()
        except Exception as e:  # noqa: BLE001 - a gate that cannot be measured has not passed; the from-base route still runs
            res = {"passed": False, "reason": f"{type(e).__name__}: {str(e)[:300]}"}
        (QUEUE / "g2.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps(res, indent=1), flush=True)
        if res["passed"]:
            dest = ROOT / "runs" / "m4_t2d" / "v2_instruct.pt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.unlink(missing_ok=True)
            from litegpt.training.sft import find_run
            dest.symlink_to(find_run(ROOT / "runs", "m4-sft") / "checkpoints" / "milestone-final.pt")
        else:
            (QUEUE / "g2_failed").write_text(json.dumps(res), encoding="utf-8")
        return 0
    files = sorted((ROOT / "runs" / "m4_t2d").glob("speed-*-v2-base.json"))
    if not files:
        res = {"test_on": "laptop", "reason": "the speed job did not finish"}
    else:
        res = speed_rule(json.loads(files[-1].read_text(encoding="utf-8"))["seconds"])
    (QUEUE / "speed_rule.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    if res["test_on"] == "laptop":
        (QUEUE / "test_on_laptop").write_text(json.dumps(res), encoding="utf-8")
    print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
