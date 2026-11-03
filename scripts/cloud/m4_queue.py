"""The M4 job queue on the rented server: runs a fixed job list in order, works with the laptop off, and survives restarts.

    python scripts/cloud/m4_queue.py --session a         # SFT sweep, SFT, verified backup
    python scripts/cloud/m4_queue.py --session b [--test-on cloud|laptop] [--without-arm v2-chat]

The list is written once to runs/m4_queue/session_<x>_jobs.json, so a registered cut is an edit of that file that leaves a trace. Finished jobs
are recorded and skipped on restart. A failed job stops the queue with an error, and the guard decides whether to restart it. Each job
carries on from its own records: SFT from its checkpoint, an interrupted fine-tune from the start. The done file appears only after the
last backup is verified, and that is what lets the guard stop the server.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUEUE_DIR = Path("runs/m4_queue")
# anchored, so the guard's own --resume-cmd argument does not match
WATCH_PATTERN = r"^[^ ]*python[^ ]* scripts/(cloud/m4_queue|cloud/m4_backup|cloud/m4_fetch_models|train_sft|run_t2d)[.]py"
POLL_SECONDS = 20
SFT_LRS = {"a": 6e-5, "b": 2e-4, "c": 6e-4}
SWEEP_KEEP = 0.03
T2D_CONFIG = "benchmarks/m4/t2d.yaml"
SFT_CONFIG = "benchmarks/m4/sft.yaml"


def session_a_jobs(py: str) -> list[dict]:
    jobs = [{"id": f"sft-sweep-{k}", "cmd": f"{py} scripts/train_sft.py --config {SFT_CONFIG} --name m4-sft-sweep-{k} --lr {lr:g} --keep-fraction {SWEEP_KEEP} --epochs 1"} for k, lr in SFT_LRS.items()]
    jobs.append({"id": "sft-choose-lr", "cmd": f"{py} scripts/train_sft.py --choose-lr m4-sft-sweep-a m4-sft-sweep-b m4-sft-sweep-c --out {QUEUE_DIR}/sft_lr.json"})
    jobs.append({"id": "sft", "cmd": f"{py} scripts/train_sft.py --config {SFT_CONFIG} --name m4-sft --lr-from {QUEUE_DIR}/sft_lr.json"})
    jobs.append({"id": "backup", "cmd": f"{py} scripts/cloud/m4_backup.py --session a", "quiet": True})
    return jobs


def session_b_jobs(py: str, arms: list[str], lr_grid: list[float], seeds: list[int], models: list[str], test_on: str, billsum: bool = True) -> list[dict]:
    """Every grid point of every arm, then the extra seeds, routes, test, BillSum last. The order follows the registered cut order: what
    may be cut comes after what may not."""
    run = f"{py} scripts/run_t2d.py"
    c = f"--config {T2D_CONFIG}"
    jobs = [{"id": "fetch-models", "cmd": f"{py} scripts/cloud/m4_fetch_models.py", "quiet": True}]
    rivals_last = sorted(arms, key=lambda a: a.startswith("qwen3"))   # the reference arm can be cut, so it runs after the rivals
    for arm in rivals_last:
        jobs += [{"id": f"fit-{arm}-lr{lr:g}-s{seeds[0]}", "cmd": f"{run} fit {c} --arm {arm} --lr {lr:g} --seed {seeds[0]}"} for lr in lr_grid]
        jobs.append({"id": f"choose-lr-{arm}", "cmd": f"{run} choose-lr {c} --arm {arm}"})
        jobs += [{"id": f"fit-{arm}-chosen-s{s}", "cmd": f"{run} fit {c} --arm {arm} --lr-from-choice --seed {s}"} for s in seeds[1:]]
    jobs.append({"id": "routes", "cmd": f"{run} routes {c}"})
    jobs.append({"id": "backup-weights", "cmd": f"{py} scripts/cloud/m4_backup.py --session b --weights", "quiet": True})
    if test_on == "cloud":
        jobs.append({"id": "test", "cmd": f"{run} test {c}"})
        jobs.append({"id": "backup-test", "cmd": f"{py} scripts/cloud/m4_backup.py --session b", "quiet": True})
    if billsum:
        jobs += [{"id": f"billsum-{m}", "cmd": f"{run} billsum {c} --model {m}"} for m in models]
    jobs.append({"id": "backup", "cmd": f"{py} scripts/cloud/m4_backup.py --session b --weights", "quiet": True})
    return jobs


def newest_activity(root: Path) -> float:
    """The latest sign of work by any job: a run's metrics or a scoring progress file."""
    best = 0.0
    for pattern in ("runs/*-train-m4-*/metrics.jsonl", "runs/m4_t2d/progress"):
        for p in root.glob(pattern):
            try:
                best = max(best, p.stat().st_mtime)
            except OSError:
                pass
    return best


def write_heartbeat(path: Path, job: str, done: int, total: int) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"job": job, "done": done, "total": total, "t": time.time()}), encoding="utf-8")
    os.replace(tmp, path)


def run_jobs(jobs: list[dict], root: Path, session: str, poll: float = POLL_SECONDS, log=None) -> int:
    qdir = root / QUEUE_DIR
    qdir.mkdir(parents=True, exist_ok=True)
    state_file, heartbeat, events = qdir / f"session_{session}_finished.json", qdir / f"session_{session}.heartbeat", qdir / f"session_{session}_events.jsonl"
    finished = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}
    stop = {"asked": False}

    def ask_stop(*_):
        stop["asked"] = True        # the job got the same signal from the guard; wait for it to save and exit

    signal.signal(signal.SIGINT, ask_stop)
    signal.signal(signal.SIGTERM, ask_stop)

    def note(**kw):
        with open(events, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), **kw}) + "\n")

    note(event="queue_start", finished=len(finished), total=len(jobs))
    for job in jobs:
        if job["id"] in finished:
            continue
        if stop["asked"]:
            note(event="queue_stopped", before=job["id"])
            return 1
        t0 = time.time()
        note(event="job_start", job=job["id"])
        write_heartbeat(heartbeat, job["id"], len(finished), len(jobs))
        with open(log or (qdir / f"session_{session}.log"), "a", encoding="utf-8") as f:
            f.write(f"\n=== {job['id']}: {job['cmd']}\n")
            f.flush()
            p = subprocess.Popen(job["cmd"].split(), cwd=root, stdout=f, stderr=subprocess.STDOUT)
            seen = newest_activity(root)
            while p.poll() is None:
                time.sleep(poll)
                now_seen = newest_activity(root)
                if job.get("quiet") or now_seen > seen:      # no output is a stall unless the job is a long upload
                    seen = now_seen
                    write_heartbeat(heartbeat, job["id"], len(finished), len(jobs))
        code = p.returncode
        note(event="job_end", job=job["id"], code=code, seconds=round(time.time() - t0, 1))
        if code != 0:
            return 1
        finished[job["id"]] = {"seconds": round(time.time() - t0, 1), "t": time.time()}
        state_file.write_text(json.dumps(finished, indent=1), encoding="utf-8")
    (qdir / f"session_{session}.done").write_text(json.dumps({"t": time.time(), "jobs": len(jobs)}), encoding="utf-8")
    note(event="queue_done", total=len(jobs))
    return 0


def main() -> int:
    import yaml
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", choices=["a", "b"], required=True)
    ap.add_argument("--test-on", choices=["cloud", "laptop"], default="cloud")
    ap.add_argument("--without-arm", action="append", default=[])
    a = ap.parse_args()
    py = sys.executable
    jobs_file = ROOT / QUEUE_DIR / f"session_{a.session}_jobs.json"
    if jobs_file.exists():
        jobs = json.loads(jobs_file.read_text(encoding="utf-8"))
    else:
        if a.session == "a":
            jobs = session_a_jobs(py)
        else:
            cfg = yaml.safe_load((ROOT / T2D_CONFIG).read_text(encoding="utf-8"))
            arms = [x for x in cfg["arms"] if x not in a.without_arm]
            models = [m for m, routes in cfg["models"].items() if any(r in arms for r in routes)]
            jobs = session_b_jobs(py, arms, cfg["recipe"]["lr_grid"], cfg["recipe"]["seeds"], models, a.test_on)
        jobs_file.parent.mkdir(parents=True, exist_ok=True)
        jobs_file.write_text(json.dumps(jobs, indent=1), encoding="utf-8")
    return run_jobs(jobs, ROOT, a.session)


if __name__ == "__main__":
    raise SystemExit(main())
