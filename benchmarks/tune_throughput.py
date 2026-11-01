"""EXP-M3-OPT: pick the fastest correct training configuration on the rented GPU, inside a fixed time budget.

    python -m benchmarks.tune_throughput [--budget-minutes 45] [--micro-batches 4,8,16,32]

Every measurement is benchmarks.train_mem `single` in its own subprocess (clean VRAM; an OOM or a compile failure is a result),
with the real recipe (fp32 master weights, bf16 autocast, fused AdamW, clipping) and the real step: 524,288 tokens
(accumulation = 256 / micro-batch). Stages:
  1 screen   micro-batch x {sdpa, flash}, default compile mode, a few steps each
  2 autotune the best screened setting with compile mode max-autotune-no-cudagraphs
  3 confirm  the two fastest settings alternating, >= 2 rounds x >= 3 minutes each (the pre-registered speed rule)
Selection (pre-registered): highest confirmed mean tok/s with peak VRAM <= 90% of the card; ties within 2% go to the simpler
setting (SDPA before flash, default before max-autotune, smaller micro-batch). Correctness of flash is checked separately with
benchmarks.compile_check --backend flash before its numbers count. Writes a run record (kind bench, bench/tune.json).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TOKENS_PER_STEP = 524_288
SEQ = 2048
AUTOTUNE = "max-autotune-no-cudagraphs"


def measure(mb: int, backend: str, mode: str | None, steps: int, warmup: int, sustained: float = 0.0, timeout: float = 1800) -> dict:
    accum = TOKENS_PER_STEP // (mb * SEQ)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out = f.name
    cmd = [sys.executable, "-m", "benchmarks.train_mem", "single", "--config", "litegpt-v2-28l", "--micro-batch", str(mb),
           "--seq-len", str(SEQ), "--backend", backend, "--compile", "1", "--grad-accum", str(accum), "--steps", str(steps),
           "--warmup", str(warmup), "--fresh-batches", "--json-out", out]
    if mode:
        cmd += ["--compile-mode", mode]
    if sustained:
        cmd += ["--sustained-seconds", str(sustained), "--window-seconds", "60"]
    t = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        r = json.loads(Path(out).read_text()) if Path(out).stat().st_size else {"status": "crash", "stderr": p.stderr[-1500:]}
    except subprocess.TimeoutExpired:
        r = {"status": "timeout"}
    r.update(micro_batch=mb, accum=accum, backend=backend, compile_mode=mode or "default", wall_s=round(time.time() - t, 1))
    if r.get("sustained"):
        r["tok_per_s_measured"] = r["sustained"]["mean_tok_per_s"]
    else:
        r["tok_per_s_measured"] = r.get("tok_per_s")
    print(json.dumps({k: r.get(k) for k in ("micro_batch", "backend", "compile_mode", "status", "tok_per_s_measured",
                                            "peak_reserved_gib", "warmup_s", "wall_s")}), flush=True)
    return r


def simplicity(r: dict) -> tuple:
    return (r["backend"] != "sdpa", r["compile_mode"] != "default", r["micro_batch"])


def pick(rows: list[dict], vram_gib: float) -> dict | None:
    ok = [r for r in rows if r.get("status") == "ok" and (r.get("peak_reserved_gib") or 0) <= 0.9 * vram_gib and r.get("tok_per_s_measured")]
    if not ok:
        return None
    best = max(r["tok_per_s_measured"] for r in ok)
    return min((r for r in ok if r["tok_per_s_measured"] >= best / 1.02), key=simplicity)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget-minutes", type=float, default=45)
    ap.add_argument("--micro-batches", default="4,8,16,32")
    ap.add_argument("--backends", default="sdpa,flash")
    ap.add_argument("--confirm-seconds", type=float, default=180)
    a = ap.parse_args(argv)
    import torch
    from benchmarks.train_mem import matmul_ceiling_tflops
    from litegpt.records import create_run
    t0 = time.time()
    left = lambda: a.budget_minutes * 60 - (time.time() - t0)  # noqa: E731
    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / 2**30
    rec = {"gpu": props.name, "vram_gib": round(vram, 1), "budget_minutes": a.budget_minutes, "ceiling": matmul_ceiling_tflops(),
           "screen": [], "autotune": [], "confirm": []}
    print(json.dumps({"gpu": rec["gpu"], "vram_gib": rec["vram_gib"], "ceiling_tflops": rec["ceiling"]["tflops"]}), flush=True)

    # 1 screen: larger micro-batches first stop as soon as one does not fit
    for be in a.backends.split(","):
        for mb in sorted(int(m) for m in a.micro_batches.split(",")):
            if left() < a.budget_minutes * 60 * 25 / 45:      # keep ~25 of 45 minutes for stages 2-3
                break
            r = measure(mb, be, None, steps=3, warmup=2)
            rec["screen"].append(r)
            if r.get("status") == "oom":
                break
    best = pick(rec["screen"], vram)
    if best is None:
        print("no screened setting ran; stopping", flush=True)
    else:
        # 2 autotune on the best screened setting
        if left() > a.budget_minutes * 60 * 18 / 45:
            rec["autotune"].append(measure(best["micro_batch"], best["backend"], AUTOTUNE, steps=3, warmup=2,
                                           timeout=max(120, min(1200, left() - a.budget_minutes * 60 * 15 / 45))))
        # 3 confirm: the two fastest distinct settings, alternating rounds
        cands = sorted([r for r in rec["screen"] + rec["autotune"] if r.get("status") == "ok" and r.get("tok_per_s_measured")],
                       key=lambda r: -r["tok_per_s_measured"])[:2]
        rnd = 0
        while cands and left() > len(cands) * (a.confirm_seconds + 120):
            rnd += 1
            for c in cands:
                r = measure(c["micro_batch"], c["backend"], None if c["compile_mode"] == "default" else c["compile_mode"],
                            steps=1, warmup=2, sustained=a.confirm_seconds)
                r["round"] = rnd
                rec["confirm"].append(r)
            if rnd >= 2 and left() < len(cands) * (a.confirm_seconds + 120) * 2:
                break
        groups = {}
        for r in rec["confirm"]:
            if r.get("status") == "ok" and r.get("tok_per_s_measured"):
                k = (r["micro_batch"], r["backend"], r["compile_mode"])
                groups.setdefault(k, []).append(r)
        summary = []
        for (mb, be, mode), rs in groups.items():
            summary.append({"micro_batch": mb, "backend": be, "compile_mode": mode, "rounds": len(rs),
                            "tok_per_s_measured": round(sum(r["tok_per_s_measured"] for r in rs) / len(rs), 1),
                            "peak_reserved_gib": max(r.get("peak_reserved_gib") or 0 for r in rs), "status": "ok"})
        rec["confirm_summary"] = summary
        eligible = [s for s in summary if s["rounds"] >= 2]
        rec["selected"] = pick(eligible, vram) or pick(rec["screen"] + rec["autotune"], vram)
        rec["selected_from"] = "confirm (>= 2 rounds)" if pick(eligible, vram) else "screen (confirm incomplete)"
    rec["minutes_used"] = round((time.time() - t0) / 60, 1)
    run = create_run("bench", "tune-throughput", config=vars(a))
    run.write_json("bench/tune.json", rec)
    print(json.dumps({"selected": rec.get("selected"), "selected_from": rec.get("selected_from"), "minutes": rec["minutes_used"],
                      "record": str(run.path)}, indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
