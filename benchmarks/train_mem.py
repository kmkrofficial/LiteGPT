"""
Training throughput / VRAM / MFU benchmark for LiteGPT V2 (plan M0.6, ADR-001/002/006).

  python -m benchmarks.train_mem ceiling                      # sustained bf16 matmul ceiling of this GPU
  python -m benchmarks.train_mem single --config litegpt-v2-28l --micro-batch 4 --seq-len 2048 --backend sdpa
  python -m benchmarks.train_mem single ... --sustained-seconds 1800      # throttling / sustained throughput
  python -m benchmarks.train_mem sweep --preset m0 [--compile]            # every config in its own subprocess

Each measured config trains on synthetic random tokens (throughput does not depend on the data) with the real training
recipe: fp32 master weights, bf16 autocast, fused AdamW, grad clipping. Sweeps write a run record (kind 'bench') whose
bench/train_mem.json is the only source for reported numbers. MFU is reported against the *measured* matmul ceiling of the
GPU (not a datasheet number) and uses both FLOPs conventions (PaLM 'full' attention term, and 'causal' = half).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REQUIRED_TOK_PER_S = 16500  # 10B tokens in 7 days (ADR-001)
VRAM_MARGIN_GIB = 1.0       # headroom kept free for the CUDA context / display; allocations beyond free-minus-margin raise OOM
VRAM_PRESSURE_FRACTION = 0.97  # device memory in use above this fraction of total => result flagged (spill risk)


def matmul_ceiling_tflops(n: int = 8192, iters: int = 100) -> dict:
    import torch
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(30):
        a @ b
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    dt = time.time() - t
    return {"n": n, "dtype": "bfloat16", "tflops": round(iters * 2 * n ** 3 / dt / 1e12, 2)}


def run_single(config: str, micro_batch: int, seq_len: int, backend: str, compile_: bool, steps: int, warmup: int,
               sustained_seconds: float = 0.0, grad_accum: int = 1, window_seconds: float = 60.0,
               windows_jsonl: str | None = None, fresh_batches: bool = False, compile_mode: str | None = None) -> dict:
    import torch
    from benchmarks.telemetry import TelemetrySampler
    from litegpt.config import get_config
    from litegpt.model import LiteGPT

    res = {"config": config, "micro_batch": micro_batch, "seq_len": seq_len, "backend": backend, "compile": compile_,
           "compile_mode": compile_mode, "grad_accum": grad_accum, "data": "synthetic-random", "status": "ok"}
    overrides = {"attention_backend": backend, "block_size": seq_len}
    cfg = get_config(config, **overrides)
    torch.manual_seed(0)
    # Windows (WDDM) silently spills VRAM overflow into shared system memory, which makes a too-large config run
    # 10x slower instead of failing. Cap the allocator at (currently free - margin) so that such a config reports a real
    # OOM (ADR-001: must fit 16 GB) instead of a meaningless slow result. The context + display already use some VRAM.
    free_b, total_b = torch.cuda.mem_get_info()
    cap_b = max(free_b - int(VRAM_MARGIN_GIB * 2 ** 30), 2 ** 30)
    torch.cuda.set_per_process_memory_fraction(cap_b / total_b)
    res.update(vram_total_gib=round(total_b / 2 ** 30, 2), vram_free_at_start_gib=round(free_b / 2 ** 30, 2),
               vram_cap_gib=round(cap_b / 2 ** 30, 2))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = LiteGPT(cfg).cuda()
        opt = model.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cuda")
        res.update(params=model.num_params(), flops_per_token_full=model.flops_per_token(seq_len, "full"),
                   flops_per_token_causal=model.flops_per_token(seq_len, "causal"))
        run_model = torch.compile(model, mode=compile_mode) if compile_ else model
        x = torch.randint(0, cfg.vocab_size, (micro_batch, seq_len), device="cuda")
        y = torch.roll(x, -1, 1)
        gen = torch.Generator(device="cuda").manual_seed(0)
        res["data"] = "synthetic-random-fresh-per-step" if fresh_batches else "synthetic-random"

        def step():
            nonlocal x, y
            for _ in range(grad_accum):
                if fresh_batches:  # unlearnable stream: loss stays near ln(vocab), a meaningful numerical-health signal
                    x = torch.randint(0, cfg.vocab_size, (micro_batch, seq_len), device="cuda", generator=gen)
                    y = torch.roll(x, -1, 1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = run_model(x, y)
                (loss / grad_accum).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            return loss

        t0 = time.time()
        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        res["warmup_s"] = round(time.time() - t0, 2)  # includes compile time when compile_=True

        tokens_per_step = micro_batch * seq_len * grad_accum
        times, windows, win_loss, win_end, tel = [], [], [], [], TelemetrySampler(interval=1.0)
        tel.start()
        t_start = time.time()
        res["measure_start_t"] = t_start
        stream = open(windows_jsonl, "a", encoding="utf-8") if windows_jsonl else None
        if stream:  # live record for an external controller; survives termination of this process
            stream.write(json.dumps({"event": "start", "t": t_start}) + "\n")
            stream.flush()
        win_t, win_n = time.time(), 0
        i = 0
        while (i < steps) or (sustained_seconds and time.time() - t_start < sustained_seconds):
            s0 = time.time()
            loss = step()
            torch.cuda.synchronize()
            dt = time.time() - s0
            times.append(dt)
            i += 1
            win_n += 1
            if sustained_seconds and time.time() - win_t >= window_seconds:
                windows.append(round(win_n * tokens_per_step / (time.time() - win_t), 1))
                win_loss.append(float(loss.item()))
                win_end.append(time.time())
                if stream:
                    stream.write(json.dumps({"event": "window", "t_end": win_end[-1], "seconds": time.time() - win_t,
                                             "tok_per_s": windows[-1], "loss": win_loss[-1], "step": i}) + "\n")
                    stream.flush()
                win_t, win_n = time.time(), 0
        if stream:
            stream.close()
        res["telemetry"] = tel.stop()
        used = res["telemetry"].get("mem_used_mib_max")
        res["vram_pressure"] = bool(used is not None and used >= VRAM_PRESSURE_FRACTION * total_b / 2 ** 20)
        res["final_loss"] = float(loss.item())
        if not torch.isfinite(loss):
            res["status"] = "nonfinite_loss"
        mean_dt, med_dt = statistics.fmean(times), statistics.median(times)
        res.update(steps_measured=len(times), tokens_per_step=tokens_per_step, step_ms_mean=round(mean_dt * 1e3, 2),
                   step_ms_median=round(med_dt * 1e3, 2), tok_per_s=round(tokens_per_step / mean_dt, 1),
                   tok_per_s_median=round(tokens_per_step / med_dt, 1))
        res["tflops_full"] = round(res["flops_per_token_full"] * res["tok_per_s"] / 1e12, 2)
        res["tflops_causal"] = round(res["flops_per_token_causal"] * res["tok_per_s"] / 1e12, 2)
        if windows:
            res["sustained"] = {"seconds": round(time.time() - t_start), "window_seconds": window_seconds,
                                "window_tok_per_s": windows, "window_loss": win_loss, "window_end_t": win_end,
                                "first_window": windows[0], "last_window": windows[-1],
                                "last_over_first": round(windows[-1] / windows[0], 3),
                                "mean_tok_per_s": round(statistics.fmean(windows), 1)}
    except torch.OutOfMemoryError:
        res["status"] = "oom"
    except Exception as e:  # compile / backend failures are results, not crashes
        res["status"] = "compile_error" if compile_ else "error"
        res["error"] = f"{type(e).__name__}: {str(e)[:600]}"
    res["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    res["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)
    return res


M0_MATRIX = [  # (config, micro_batch, seq_len); sizes beyond mb4 / 2x4096 were measured to OOM on the 16 GB laptop (first M0 sweep)
    ("litegpt-v2-28l", 1, 2048), ("litegpt-v2-28l", 2, 2048), ("litegpt-v2-28l", 3, 2048), ("litegpt-v2-28l", 4, 2048),
    ("litegpt-v2-28l", 1, 4096), ("litegpt-v2-24l", 2, 2048), ("litegpt-v2-24l", 3, 2048),
]


def sweep(preset: str, with_compile: bool, with_flash: bool, steps: int, warmup: int) -> Path:
    from litegpt.model import flash_attn_available
    from litegpt.records import create_run

    assert preset == "m0", "only preset 'm0' exists"
    skipped: list[dict] = []
    jobs = [(c, mb, T, "sdpa", False) for c, mb, T in M0_MATRIX]
    if with_compile:
        jobs += [("litegpt-v2-28l", 2, 2048, "sdpa", True)]
    if with_flash:
        if flash_attn_available():
            jobs += [("litegpt-v2-28l", 2, 2048, "flash", False)]
        else:
            print("flash-attn not installed: flash job recorded as skipped")
            skipped.append({"config": "litegpt-v2-28l", "micro_batch": 2, "seq_len": 2048, "backend": "flash",
                            "compile": False, "status": "skipped", "error": "flash-attn is not installed"})
    rec = create_run("bench", f"train_mem-{preset}", config={"preset": preset, "steps": steps, "warmup": warmup,
                                                              "jobs": jobs})
    results = list(skipped)
    ceiling = None
    try:
        import torch
        ceiling = matmul_ceiling_tflops()
        print("matmul ceiling:", ceiling, flush=True)
    except Exception as e:
        print("ceiling failed:", e)
    for c, mb, T, backend, comp in jobs:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "r.json"
            cmd = [sys.executable, "-m", "benchmarks.train_mem", "single", "--config", c, "--micro-batch", str(mb),
                   "--seq-len", str(T), "--backend", backend, "--compile", str(int(comp)), "--steps", str(steps),
                   "--warmup", str(warmup), "--json-out", str(out)]
            print("running:", " ".join(cmd[3:]), flush=True)
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
                r = json.loads(out.read_text()) if out.exists() else {
                    "config": c, "micro_batch": mb, "seq_len": T, "backend": backend, "compile": comp,
                    "status": "crash", "error": (p.stderr or p.stdout)[-600:]}
            except subprocess.TimeoutExpired:
                r = {"config": c, "micro_batch": mb, "seq_len": T, "backend": backend, "compile": comp, "status": "timeout"}
        if ceiling and r.get("tflops_full") is not None:
            r["mfu_full_vs_matmul_ceiling"] = round(r["tflops_full"] / ceiling["tflops"], 3)
            r["mfu_causal_vs_matmul_ceiling"] = round(r["tflops_causal"] / ceiling["tflops"], 3)
        valid = r.get("status") == "ok" and not r.get("vram_pressure")
        r["meets_16_5k_tok_per_s"] = (bool(valid and r.get("tok_per_s", 0) >= REQUIRED_TOK_PER_S)
                                      if r["seq_len"] == 2048 else None)
        print(f"  -> {r['status']} tok/s={r.get('tok_per_s')} peak={r.get('peak_alloc_gib')}GiB "
              f"vram_pressure={r.get('vram_pressure')}", flush=True)
        results.append(r)
    rec.write_json("bench/train_mem.json", {"required_tok_per_s": REQUIRED_TOK_PER_S, "matmul_ceiling": ceiling,
                                            "results": results})
    return rec.path


def _job(c, mb, T, backend, comp, steps, warmup) -> dict:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "r.json"
        cmd = [sys.executable, "-m", "benchmarks.train_mem", "single", "--config", c, "--micro-batch", str(mb),
               "--seq-len", str(T), "--backend", backend, "--compile", str(int(comp)), "--steps", str(steps),
               "--warmup", str(warmup), "--json-out", str(out)]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
        return json.loads(out.read_text()) if out.exists() else {
            "config": c, "micro_batch": mb, "seq_len": T, "backend": backend, "compile": comp, "status": "crash",
            "error": (p.stderr or p.stdout)[-600:]}


def ab(config: str, micro_batch: int, seq_len: int, backend_a: str, backend_b: str, rounds: int, steps: int,
       warmup: int, compile_b: bool = False) -> Path:
    """Alternating A/B/A/B... runs in separate processes so thermal state affects both sides equally.
    A = (backend_a, eager); B = (backend_b, compile if compile_b else eager)."""
    from litegpt.records import create_run
    rec = create_run("bench", f"ab-{backend_a}-vs-{backend_b}{'-compile' if compile_b else ''}", attention_backend=backend_b,
                     compile_enabled=compile_b,
                     config={"config": config, "micro_batch": micro_batch, "seq_len": seq_len, "rounds": rounds,
                             "steps": steps, "warmup": warmup, "A": backend_a, "B": backend_b, "compile_b": compile_b})
    ceiling = matmul_ceiling_tflops()
    results = []
    sides = (("A", backend_a, False), ("B", backend_b, compile_b))
    for rnd in range(rounds):
        for side, be, comp in sides:
            r = _job(config, micro_batch, seq_len, be, comp, steps, warmup)
            r["ab_round"], r["ab_side"] = rnd, side
            print(f"round {rnd} {side}={be}{'+compile' if comp else ''}: {r['status']} tok/s={r.get('tok_per_s')} peak={r.get('peak_alloc_gib')} "
                  f"clk={r.get('telemetry', {}).get('sm_clock_mhz_mean_busy')}", flush=True)
            if r.get("tflops_full"):
                r["mfu_full_vs_matmul_ceiling"] = round(r["tflops_full"] / ceiling["tflops"], 3)
            results.append(r)
    summ = {}
    for side, be, comp in sides:
        key = f"{side}:{be}{'+compile' if comp else ''}"
        xs = [r["tok_per_s"] for r in results if r.get("ab_side") == side and r.get("status") == "ok"]
        pk = [r["peak_alloc_gib"] for r in results if r.get("ab_side") == side and r.get("status") == "ok"]
        wu = [r["warmup_s"] for r in results if r.get("ab_side") == side and r.get("status") == "ok"]
        fl = [r["final_loss"] for r in results if r.get("ab_side") == side and r.get("status") == "ok"]
        summ[key] = {"n": len(xs), "final_loss_mean": round(statistics.fmean(fl), 5) if fl else None,
                     "warmup_s_mean": round(statistics.fmean(wu), 1) if wu else None, "tok_per_s_mean": round(statistics.fmean(xs), 1) if xs else None,
                    "tok_per_s_min": min(xs) if xs else None, "tok_per_s_max": max(xs) if xs else None,
                    "peak_alloc_gib_max": max(pk) if pk else None}
    ka, kb = (f"{sd}:{be}{'+compile' if c else ''}" for sd, be, c in sides)
    if summ[ka]["tok_per_s_mean"] and summ[kb]["tok_per_s_mean"]:
        summ["speedup_B_over_A"] = round(summ[kb]["tok_per_s_mean"] / summ[ka]["tok_per_s_mean"], 4)
    if summ[ka].get("final_loss_mean") and summ[kb].get("final_loss_mean"):
        summ["final_loss_rel_diff_B_vs_A"] = round(abs(summ[kb]["final_loss_mean"] - summ[ka]["final_loss_mean"])
                                                  / summ[ka]["final_loss_mean"], 5)
    rec.write_json("bench/train_mem.json", {"required_tok_per_s": REQUIRED_TOK_PER_S, "matmul_ceiling": ceiling,
                                            "results": results, "ab_summary": summ})
    print(json.dumps(summ, indent=2))
    return rec.path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ceiling")
    s = sub.add_parser("single")
    s.add_argument("--config", default="litegpt-v2-28l")
    s.add_argument("--micro-batch", type=int, default=4)
    s.add_argument("--seq-len", type=int, default=2048)
    s.add_argument("--backend", default="sdpa", choices=["sdpa", "flash"])
    s.add_argument("--compile", type=int, default=0)
    s.add_argument("--compile-mode", default=None, help="torch.compile mode, e.g. max-autotune-no-cudagraphs (EXP-M3-OPT)")
    s.add_argument("--steps", type=int, default=20)
    s.add_argument("--warmup", type=int, default=5)
    s.add_argument("--grad-accum", type=int, default=1)
    s.add_argument("--sustained-seconds", type=float, default=0.0)
    s.add_argument("--window-seconds", type=float, default=60.0)
    s.add_argument("--windows-jsonl", help="append each measurement window as a JSON line (live, for controllers)")
    s.add_argument("--fresh-batches", action="store_true",
                   help="new random tokens every micro-step (unlearnable; keeps the loss a meaningful health signal)")
    s.add_argument("--json-out")
    s.add_argument("--record", metavar="NAME", help="also write a run record (kind bench) with this name")
    w = sub.add_parser("sweep")
    w.add_argument("--preset", default="m0")
    w.add_argument("--compile", action="store_true")
    w.add_argument("--flash", action="store_true")
    w.add_argument("--steps", type=int, default=20)
    w.add_argument("--warmup", type=int, default=5)
    x = sub.add_parser("ab")
    x.add_argument("--config", default="litegpt-v2-28l")
    x.add_argument("--micro-batch", type=int, default=3)
    x.add_argument("--seq-len", type=int, default=2048)
    x.add_argument("--a", default="sdpa")
    x.add_argument("--b", default="flash")
    x.add_argument("--compile-b", action="store_true", help="side B runs with torch.compile")
    x.add_argument("--rounds", type=int, default=4)
    x.add_argument("--steps", type=int, default=60)
    x.add_argument("--warmup", type=int, default=10)
    a = ap.parse_args(argv)
    if a.cmd == "ab":
        print("run record:", ab(a.config, a.micro_batch, a.seq_len, a.a, a.b, a.rounds, a.steps, a.warmup,
                                   a.compile_b))
    elif a.cmd == "ceiling":
        print(json.dumps(matmul_ceiling_tflops()))
    elif a.cmd == "single":
        r = run_single(a.config, a.micro_batch, a.seq_len, a.backend, bool(a.compile), a.steps, a.warmup,
                       a.sustained_seconds, a.grad_accum, a.window_seconds, a.windows_jsonl,
                       a.fresh_batches, a.compile_mode)
        if a.json_out:
            Path(a.json_out).write_text(json.dumps(r))
        if a.record:
            from litegpt.records import create_run
            ceiling = matmul_ceiling_tflops()
            if r.get("tflops_full") is not None:
                r["mfu_full_vs_matmul_ceiling"] = round(r["tflops_full"] / ceiling["tflops"], 3)
                r["mfu_causal_vs_matmul_ceiling"] = round(r["tflops_causal"] / ceiling["tflops"], 3)
            valid = r.get("status") == "ok" and not r.get("vram_pressure")
            r["meets_16_5k_tok_per_s"] = (bool(valid and (r.get("sustained", {}).get("mean_tok_per_s") or r.get("tok_per_s", 0))
                                               >= REQUIRED_TOK_PER_S) if r["seq_len"] == 2048 else None)
            rec = create_run("bench", a.record, config=vars(a), attention_backend=a.backend, compile_enabled=bool(a.compile))
            rec.write_json("bench/train_mem.json", {"required_tok_per_s": REQUIRED_TOK_PER_S, "matmul_ceiling": ceiling,
                                                    "results": [r]})
            print("run record:", rec.path)
        print(json.dumps(r, indent=2))
    else:
        print("run record:", sweep(a.preset, a.compile, a.flash, a.steps, a.warmup))


if __name__ == "__main__":
    main()
