"""The text-to-data comparison (EXP-M4-T2D). Each command can be run again: finished work is read back, never repeated.

    python scripts/run_t2d.py check     --config benchmarks/m4/t2d.yaml
    python scripts/run_t2d.py fit       --config ... --arm v2-base --lr 3e-5 --seed 1
    python scripts/run_t2d.py choose-lr --config ... --arm v2-base
    python scripts/run_t2d.py routes    --config ...
    python scripts/run_t2d.py test      --config ...        # once, after every route is chosen
    python scripts/run_t2d.py billsum   --config ... --model v2
    python scripts/run_t2d.py speed     --config ... --arm v2-base     # the same dev scoring job on this machine
    python scripts/run_t2d.py cpu       --config ... --model v2 --threads 8
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import torch
    import yaml
    from litegpt import t2d as X
    from litegpt import t2d_run as R
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "fit", "choose-lr", "routes", "test", "billsum", "speed", "cpu"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm")
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--model")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--lr-from-choice", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, help="smoke runs only: items per task")
    ap.add_argument("--runs-root", default=str(ROOT / "runs"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = R.expand_paths(yaml.safe_load(Path(a.config).read_text(encoding="utf-8")))
    runs, device = Path(a.runs_root), torch.device(a.device)
    results = Path(cfg["results_dir"])
    results.mkdir(parents=True, exist_ok=True)
    choices_file, routes_file = results / "lr_choices.json", results / "routes.json"

    def read(p: Path) -> dict:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    if a.command == "check":
        res = R.check_fits(cfg, arms=a.arms)
        (results / "fits.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(res, indent=1, ensure_ascii=False))
        return 0 if all(r.get("fits", True) for r in res.values()) else 1
    if a.command == "fit":
        lr = read(choices_file)[a.arm]["chosen_lr"] if a.lr_from_choice else a.lr
        res = R.run_fit(cfg, a.arm, lr, a.seed, runs, device, limit=a.limit)
        print(json.dumps({k: res.get(k) for k in ("status", "arm", "lr", "seed", "score", "cuad", "massive", "banking77", "run")}))
        return 0 if res.get("status") == "completed" else 1
    if a.command == "choose-lr":
        choices = read(choices_file)
        choices[a.arm] = R.choose_lr(cfg, a.arm, runs)
        choices_file.write_text(json.dumps(choices, indent=1), encoding="utf-8")
        print(json.dumps(choices[a.arm]))
        return 0
    if a.command == "routes":
        routes = R.choose_routes(cfg, runs, read(choices_file))
        routes_file.write_text(json.dumps(routes, indent=1), encoding="utf-8")
        print(json.dumps({m: r["chosen"] for m, r in routes.items()}, indent=1))
        return 0
    routes = read(routes_file)
    if a.command == "test":
        if not routes:
            raise SystemExit("choose the routes first")
        res = R.score_test(cfg, routes, runs, results, device)
        print(json.dumps(res["comparisons"], indent=1))
        return 0
    if a.command == "billsum":
        print(json.dumps({k: v for k, v in R.run_billsum(cfg, a.model, routes[a.model]["chosen"], runs, results, device, limit=a.limit).items() if k not in ("train", "generation")}))
        return 0
    manifest = X.load_manifest(cfg["manifest"])
    if a.command == "speed":                    # the registered test-scoring rule: the same job on both machines, with the starting weights both have
        R.cap_memory(device)
        dev = R.build_examples(cfg["items_dir"], manifest, list(X.PRIMARY), "dev", a.limit)
        obj = R.load_arm({**cfg["arms"][a.arm], "name": a.arm}, device)
        obj.as_eval()
        _, speed = R.predict(obj, dev, cfg["recipe"])
        speed["device"] = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        out = results / f"speed-{os.uname().nodename}-{a.arm}.json"
        out.write_text(json.dumps(speed, indent=1), encoding="utf-8")
        print(json.dumps(speed))
        return 0
    if a.command == "cpu":
        torch.set_num_threads(a.threads)
        chosen = routes[a.model]["chosen"]
        run = sorted(runs.glob(f"*-train-{R.run_name(chosen['arm'], chosen['lr'], cfg['recipe']['seeds'][0])}-*"))[-1]
        obj = R.load_arm({**cfg["arms"][chosen["arm"]], "name": chosen["arm"]}, torch.device("cpu"), weights=run / "weights")
        obj.as_eval()
        rows = {}
        recipe = {**cfg["recipe"], "gen_items": 1}
        for task in X.PRIMARY:
            items = R.build_examples(cfg["items_dir"], manifest, [task], "dev", cfg["cpu_items"][task])
            t0 = time.perf_counter()
            obj.generate([obj.encode(items[0])[0]], 1)
            first = time.perf_counter() - t0
            _, speed = R.predict(obj, items, recipe)
            rows[task] = {"items": len(items), "documents_per_minute": 60 * speed["items_per_s"], "time_to_first_token_s": first}
        out = {"model": a.model, "arm": chosen["arm"], "threads": a.threads, "tasks": rows, "peak_ram_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, "note": "bfloat16, one document at a time, same laptop and thread count"}
        (results / f"cpu-{a.model}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(json.dumps(out))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
