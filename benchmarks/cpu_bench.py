"""CPU speed and memory of V2 and its nearest rivals, and Raspberry Pi estimates from those measurements (EXP-M5-CPU).

    python -m benchmarks.cpu_bench all                               # every model and precision, each in a fresh process, then bandwidth and Pi
    python -m benchmarks.cpu_bench run --model v2-litegpt --dtype float32
    python -m benchmarks.cpu_bench bandwidth
    python -m benchmarks.cpu_bench pi                                # estimates from the newest records; measures nothing

Batch 1, greedy, every row runs to the full number of new tokens. Decode time is total generation minus one prefill, as in
litegpt/eval/native.py `cmd_efficiency`. Pi figures are estimates from laptop measurements and published Pi figures, never measurements.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS = {
    "v2-litegpt": {"kind": "litegpt", "path": "~/m5_export/LiteGPT-V2-Base", "label": "LiteGPT V2 (LiteGPT engine)"},
    "v2-hf": {"kind": "hf", "repo": "~/m5_export/LiteGPT-V2-Base", "revision": None, "label": "LiteGPT V2 (Transformers)"},
    "smollm2": {"kind": "hf", "repo": "HuggingFaceTB/SmolLM2-135M", "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2", "label": "SmolLM2-135M"},
    "gemma": {"kind": "hf", "repo": "google/gemma-3-270m", "revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1", "label": "Gemma 3 270M"},
    "v1": {"kind": "hf", "repo": "koganrath/LiteGPT-Base", "revision": "62ccc46d556de38b54f021eca71826debc59a96a", "label": "LiteGPT V1"},
}
DTYPES = ("float32", "bfloat16")
THREADS = (1, 4, 8, 16)
PROMPTS = (128, 512)
NEW_TOKENS = 64
REPEATS = 3
TEXT = ("The history of science is the study of the development of science and scientific knowledge, including both the "
        "natural and social sciences. Science is a body of empirical, theoretical, and practical knowledge about the natural "
        "world, produced by scientists who emphasize the observation, explanation, and prediction of real-world phenomena. ")

# Published figures for the estimates. Sustained read bandwidth in GB/s; per-core peak FP32 relative to one AVX2 P-core, from specs.
PI = {
    "Raspberry Pi 5": {"bandwidth_gbs": 13.8, "fp32_core_ratio": 4.0, "ram_gb": (4, 8), "usable_gb": {4: 3.2, 8: 7.2},
                       "cpu": "4x Cortex-A76 at 2.4 GHz", "source": "Arm learning path llm-decode-roofline: 13.8 GB/s sustained read"},
    "Raspberry Pi 3 Model B": {"bandwidth_gbs": 2.2, "fp32_core_ratio": 16.0, "ram_gb": (1,), "usable_gb": {1: 0.75},
                               "cpu": "4x Cortex-A53 at 1.2 GHz", "source": "OpenBenchmarking RAMspeed 1604209-GA-RPI3RAMSP14: about 2.2 GB/s"},
}
PREFILL_RANGE = (0.5, 1.5)


def rss_gib() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 30


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20         # kB on Linux


def load(model: str, dtype):
    """(prefill(ids), generate(ids, n) -> new token count, encode, weight bytes, parameters) for one model at one dtype, on CPU."""
    import torch
    spec = MODELS[model]
    if spec["kind"] == "litegpt":
        from litegpt import api
        from litegpt.inference import GenerationConfig
        m = api.load(Path(spec["path"]).expanduser(), device="cpu", dtype=dtype)
        net = m.model

        def prefill(ids):
            net(torch.tensor([ids]))

        def generate(ids, n):
            return len(m.engine.generate_ids([ids], GenerationConfig(max_new_tokens=n, stop_token_ids=()))[0].ids)
        encode = m.tok.encode_text
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        src = str(Path(spec["repo"]).expanduser()) if spec["repo"].startswith("~") else spec["repo"]
        kw = {"revision": spec["revision"]} if spec["revision"] else {}
        tok = AutoTokenizer.from_pretrained(src, trust_remote_code=False, **kw)
        net = AutoModelForCausalLM.from_pretrained(src, dtype=dtype, trust_remote_code=False, **kw).eval()
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        def prefill(ids):
            net(input_ids=torch.tensor([ids]), use_cache=False)

        def generate(ids, n):
            x = torch.tensor([ids])
            return int(net.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=n, min_new_tokens=n, do_sample=False, pad_token_id=pad).shape[1] - len(ids))

        def encode(text):
            return tok(text, add_special_tokens=False)["input_ids"]
    params = sum(p.numel() for p in net.parameters())
    weight_bytes = sum(p.numel() * p.element_size() for p in net.parameters())
    return prefill, generate, encode, weight_bytes, params


def measure(prefill, generate, encode, threads=THREADS, prompts=PROMPTS, new_tokens=NEW_TOKENS, repeats=REPEATS) -> list[dict]:
    """One row per thread count and prompt length: medians over repeats, with min and max."""
    import torch
    rows = []
    for t in threads:
        torch.set_num_threads(t)
        for length in prompts:
            ids = encode(TEXT * 60)[:length]
            if len(ids) < length:
                raise SystemExit(f"prompt text gives only {len(ids)} tokens")
            with torch.no_grad():
                generate(ids, 4)                                    # warm-up
                runs = []
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    prefill(ids)
                    p = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    n = generate(ids, new_tokens)
                    total = time.perf_counter() - t0
                    d = max(total - p, 1e-9)
                    runs.append({"prefill_tok_s": length / p, "decode_tok_s": n / d, "ttft_s": p + d / n, "new_tokens": n})
            row = {"threads": t, "prompt_tokens": length}
            for k in ("prefill_tok_s", "decode_tok_s", "ttft_s"):
                vals = [r[k] for r in runs]
                row[k] = statistics.median(vals)
                row[k + "_min"], row[k + "_max"] = min(vals), max(vals)
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def cmd_run(a) -> int:
    import torch
    from litegpt.records import create_run
    dtype = getattr(torch, a.dtype)
    torch.set_num_interop_threads(1)
    t0 = time.perf_counter()
    prefill, generate, encode, weight_bytes, params = load(a.model, dtype)
    load_s = time.perf_counter() - t0
    after_load = rss_gib()
    rows = measure(prefill, generate, encode, a.threads, a.prompts, a.new_tokens, a.repeats)
    out = {"model": a.model, "label": MODELS[a.model]["label"], "dtype": a.dtype, "parameters": params, "weight_bytes": weight_bytes, "load_s": load_s,
           "rss_after_load_gib": after_load, "peak_rss_gib": peak_rss_gib(), "rows": rows, "cpu": cpu_name(), "logical_cpus": os.cpu_count(),
           "note": "batch 1, greedy, full length; decode = total minus one prefill; RSS is the whole process"}
    rec = create_run("bench", f"cpu-{a.model}-{a.dtype}", config={"experiment": "EXP-M5-CPU", "model": MODELS[a.model], "dtype": a.dtype, "threads": list(a.threads),
                                                                    "prompts": list(a.prompts), "new_tokens": a.new_tokens, "repeats": a.repeats})
    rec.write_json("bench/cpu.json", out)
    print("record:", rec.path.name, flush=True)
    return 0


def cpu_name() -> str:
    try:
        return next(x.split(":", 1)[1].strip() for x in open("/proc/cpuinfo") if x.startswith("model name"))
    except (OSError, StopIteration):
        return "unknown"


def cmd_bandwidth(a) -> int:
    import torch
    from litegpt.records import create_run
    x = torch.ones(2 ** 28, dtype=torch.float32)
    y = torch.empty_like(x)
    size = x.numel() * x.element_size()
    rows = []
    for t in a.threads:
        torch.set_num_threads(t)
        x.sum()
        y.copy_(x)
        read, copy = [], []
        for _ in range(5):
            t0 = time.perf_counter()
            x.sum()
            read.append(size / (time.perf_counter() - t0) / 1e9)
            t0 = time.perf_counter()
            y.copy_(x)
            copy.append(2 * size / (time.perf_counter() - t0) / 1e9)
        rows.append({"threads": t, "read_gbs": statistics.median(read), "copy_gbs": statistics.median(copy)})
        print(json.dumps(rows[-1]), flush=True)
    rec = create_run("bench", "cpu-bandwidth", config={"experiment": "EXP-M5-CPU", "buffer_gib": size / 2 ** 30, "threads": list(a.threads)})
    rec.write_json("bench/bandwidth.json", {"rows": rows, "cpu": cpu_name(), "note": "torch sum (read) and copy (read plus write), float32, 1 GiB, median of 5"})
    print("record:", rec.path.name, flush=True)
    return 0


def estimate_pi(model: dict, bandwidth_4t_gbs: float, threads: int = 4) -> dict:
    """Estimates for each Pi from one model's laptop record. Decode: scaled by bandwidth, capped by the roofline. Prefill: scaled by per-core peak FP32."""
    out = {}
    for name, pi in PI.items():
        rows = {r["prompt_tokens"]: r for r in model["rows"] if r["threads"] == threads}
        ceiling = pi["bandwidth_gbs"] * 1e9 / model["weight_bytes"]
        per_prompt = {}
        for length, r in rows.items():
            decode = min(r["decode_tok_s"] * pi["bandwidth_gbs"] / bandwidth_4t_gbs, ceiling)
            prefill = r["prefill_tok_s"] / pi["fp32_core_ratio"]
            per_prompt[length] = {"decode_tok_s": decode, "decode_ceiling_tok_s": ceiling, "prefill_tok_s": prefill,
                                  "prefill_tok_s_range": [prefill * PREFILL_RANGE[0], prefill * PREFILL_RANGE[1]],
                                  "ttft_s": length / prefill + 1 / decode}
        fits = {ram: model["peak_rss_gib"] * 2 ** 30 / 1e9 <= usable for ram, usable in pi["usable_gb"].items()}
        out[name] = {"prompts": per_prompt, "fits_in_ram_gb": fits, "cpu": pi["cpu"], "source": pi["source"], "bandwidth_gbs": pi["bandwidth_gbs"],
                     "fp32_core_ratio": pi["fp32_core_ratio"]}
    return out


def newest(pattern: str) -> Path | None:
    found = sorted((ROOT / "runs").glob(pattern))
    return found[-1] if found else None


def cmd_pi(a) -> int:
    from litegpt.records import create_run
    bw_dir = newest("*-bench-cpu-bandwidth-*")
    if bw_dir is None:
        raise SystemExit("no bandwidth record: run `bandwidth` first")
    bw = json.loads((bw_dir / "bench" / "bandwidth.json").read_text(encoding="utf-8"))
    bw4 = next(r["read_gbs"] for r in bw["rows"] if r["threads"] == 4)
    result = {"laptop_read_gbs_4_threads": bw4, "bandwidth_record": bw_dir.name, "models": {}, "assumptions": {
        "usable_ram_gb": {k: v["usable_gb"] for k, v in PI.items()}, "prefill_range": PREFILL_RANGE,
        "platform": "64-bit OS, PyTorch aarch64 build; float32 recommended because neither Pi has native bf16 maths"}}
    for model in MODELS:
        for dtype in DTYPES:
            d = newest(f"*-bench-cpu-{model}-{dtype}-*")
            if d is None:
                continue
            rec = json.loads((d / "bench" / "cpu.json").read_text(encoding="utf-8"))
            result["models"][f"{model}/{dtype}"] = {"record": d.name, "label": rec["label"], "dtype": dtype, "peak_rss_gib": rec["peak_rss_gib"],
                                                    "estimate": estimate_pi(rec, bw4)}
    rec = create_run("bench", "cpu-pi-estimate", config={"experiment": "EXP-M5-CPU", "pi": {k: {**v, "ram_gb": list(v["ram_gb"])} for k, v in PI.items()}})
    rec.write_json("bench/pi_estimate.json", result)
    print(json.dumps({k: {p: round(v["estimate"][p]["prompts"][128]["decode_tok_s"], 1) for p in PI} for k, v in result["models"].items()}, indent=1))
    print("record:", rec.path.name, flush=True)
    return 0


def cmd_all(a) -> int:
    py = [sys.executable, "-m", "benchmarks.cpu_bench"]
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    for model in a.models:
        for dtype in DTYPES:
            r = subprocess.run([*py, "run", "--model", model, "--dtype", dtype], cwd=ROOT, env=env)
            print(f"{model} {dtype}: exit {r.returncode}", flush=True)
    subprocess.run([*py, "bandwidth"], cwd=ROOT, env=env, check=True)
    return subprocess.run([*py, "pi"], cwd=ROOT, env=env).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True, choices=list(MODELS))
    r.add_argument("--dtype", required=True, choices=list(DTYPES))
    r.add_argument("--threads", type=int, nargs="+", default=list(THREADS))
    r.add_argument("--prompts", type=int, nargs="+", default=list(PROMPTS))
    r.add_argument("--new-tokens", type=int, default=NEW_TOKENS)
    r.add_argument("--repeats", type=int, default=REPEATS)
    b = sub.add_parser("bandwidth")
    b.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8])
    sub.add_parser("pi")
    al = sub.add_parser("all")
    al.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    a = ap.parse_args(argv)
    return {"run": cmd_run, "bandwidth": cmd_bandwidth, "pi": cmd_pi, "all": cmd_all}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
