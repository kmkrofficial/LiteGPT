"""Start M3 on the rented server with the tuned settings, plus the guard, each in its own tmux session.

    python scripts/cloud/start_m3.py --rented-at <unix time> --price-usd-h 0.44 --micro-batch 8 --backend sdpa [--compile-mode M] [--eval-fast]

Writes benchmarks/m3/m3_run.yaml from m3.yaml (the tuned settings and the price only; every other value is the spec's), checks the guard's
needs, then starts training and the guard. Refuses if a run or the guard is already running.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TOKENS_PER_STEP = 524_288


def pid1_env() -> dict:
    """Vast puts the instance's restricted key in the container's main process; SSH shells do not have it. Never printed."""
    out = {}
    for kv in Path("/proc/1/environ").read_bytes().split(b"\0"):
        k, _, v = kv.decode("utf-8", "replace").partition("=")
        if k in ("CONTAINER_ID", "CONTAINER_API_KEY"):
            out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rented-at", type=float, required=True)
    ap.add_argument("--price-usd-h", type=float, required=True)
    ap.add_argument("--micro-batch", type=int, required=True)
    ap.add_argument("--backend", choices=["sdpa", "flash", "auto"], required=True)
    ap.add_argument("--compile-mode", default=None)
    ap.add_argument("--eval-fast", action="store_true")
    ap.add_argument("--eval-micro-batch", type=int, default=None)
    ap.add_argument("--backup-repo", default="kmkrworks/litegpt-v2-m3-runs")
    ap.add_argument("--ntfy-topic", default=None)
    a = ap.parse_args()
    if subprocess.run(["pgrep", "-f", "scripts/(train|cloud/m3_guard)[.]py"], capture_output=True).stdout:
        raise SystemExit("a trainer or the guard is already running")
    if 256 % a.micro_batch:
        raise SystemExit("micro-batch must divide 256")
    env = pid1_env()
    if not env.get("CONTAINER_ID") or not env.get("CONTAINER_API_KEY"):
        raise SystemExit("the instance key is missing: the automatic stop would not work")

    cfg = yaml.safe_load((ROOT / "benchmarks" / "m3" / "m3.yaml").read_text(encoding="utf-8"))
    cfg["micro_batch"], cfg["accum"] = a.micro_batch, TOKENS_PER_STEP // (a.micro_batch * 2048)
    ext_mb = max(1, a.micro_batch // 2)                 # 4,096 tokens per row: the same tokens per micro-step
    cfg["schedule"]["extension"]["micro_batch"], cfg["schedule"]["extension"]["accum"] = ext_mb, TOKENS_PER_STEP // (ext_mb * 4096)
    cfg["model"]["attention_backend"] = a.backend
    cfg["compile_mode"] = a.compile_mode
    cfg["eval_fast"] = a.eval_fast
    if a.eval_micro_batch:
        cfg["eval_micro_batch"] = a.eval_micro_batch
    cfg["schedule"]["budget"]["price_usd_h"] = a.price_usd_h
    run_cfg = ROOT / "benchmarks" / "m3" / "m3_run.yaml"
    run_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    home = Path.home()
    py = str(ROOT / ".venv" / "bin" / "python")
    train = f"cd {ROOT} && . {home}/.litegpt_env && PYTHONUNBUFFERED=1 {py} scripts/train.py --config benchmarks/m3/m3_run.yaml --rented-at {a.rented_at} 2>&1 | tee -a {home}/trainer.log"
    guard_args = ["scripts/cloud/m3_guard.py", "--repo-dir", str(ROOT), "--name", "m3", "--rented-at", str(a.rented_at), "--price-usd-h", str(a.price_usd_h),
                  "--backup-repo", a.backup_repo, "--resume-cmd", f"{py} scripts/train.py --resume {{run}}"]
    if a.ntfy_topic:
        guard_args += ["--ntfy-topic", a.ntfy_topic]
    guard = f"cd {ROOT} && . {home}/.litegpt_env && export PATH=/opt/instance-tools/bin:$PATH && PYTHONUNBUFFERED=1 {py} {shlex.join(guard_args)} 2>&1 | tee -a {home}/guard.log"
    # the key reaches the guard through its own environment file, readable by root only; it is not on any command line here
    keyfile = home / ".litegpt_guard_env"
    keyfile.touch(mode=0o600)
    keyfile.write_text("".join(f"export {k}={shlex.quote(v)}\n" for k, v in env.items()), encoding="utf-8")
    guard = f". {keyfile} && {guard}"
    subprocess.run(["tmux", "new-session", "-d", "-s", "m3", train], check=True)
    subprocess.run(["tmux", "new-session", "-d", "-s", "guard", guard], check=True)
    print("started: tmux sessions m3, guard; settings", {k: cfg[k] for k in ("micro_batch", "accum", "compile_mode", "eval_fast")}, cfg["model"]["attention_backend"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
