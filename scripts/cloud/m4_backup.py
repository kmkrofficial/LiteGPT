"""M4 backup to the private repo, every large file proved by hash. Exits 0 only if everything was verified.

    python scripts/cloud/m4_backup.py --session a             # records, the SFT final checkpoint (or its newest if unfinished)
    python scripts/cloud/m4_backup.py --session b [--weights] # records, and the fine-tuned weights of each chosen route

Files go under m4/ in new paths; nothing earlier is overwritten or deleted, except records of a run, which only grow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = "kmkrworks/litegpt-v2-m3-runs"
RECORD_IGNORE = ["checkpoints/*", "weights/*"]


def git_blob_sha1(path: Path) -> str:
    h = hashlib.sha1()
    h.update(f"blob {path.stat().st_size}\0".encode())
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def hub_matches(info, path: Path) -> bool:
    """The Hub's own hash of what it stored against the local file: sha256 for large files, the git blob id otherwise."""
    if info.lfs:
        return info.lfs.sha256 == sha256(path)
    return info.blob_id == git_blob_sha1(path)


def upload_verified(api, local: Path, dest: str) -> bool:
    """Upload unless the Hub already holds exactly this file, then compare hashes."""
    info = api.get_paths_info(REPO, [dest])
    if not (info and hub_matches(info[0], local)):
        api.upload_file(path_or_fileobj=str(local), path_in_repo=dest, repo_id=REPO, commit_message=dest)
        info = api.get_paths_info(REPO, [dest])
    ok = bool(info and hub_matches(info[0], local))
    print(("verified " if ok else "NOT VERIFIED ") + dest, flush=True)
    return ok


def m4_runs(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("runs/*-train-m4-*") if p.is_dir())


def chosen_weight_dirs(root: Path, seeds: list[int]) -> list[Path]:
    from litegpt.t2d_run import run_name
    from litegpt.training.sft import find_run
    routes_file = root / "runs" / "m4_t2d" / "routes.json"
    if not routes_file.exists():
        return []
    out = []
    for r in json.loads(routes_file.read_text(encoding="utf-8")).values():
        c = r["chosen"]
        for s in seeds:
            run = find_run(root / "runs", run_name(c["arm"], c["lr"], s))
            if run and (run / "weights").exists():
                out.append(run)
    return out


def main() -> int:
    from huggingface_hub import HfApi
    sys.path.insert(0, str(ROOT))
    import yaml
    from litegpt.training.sft import find_run
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", choices=["a", "b"], required=True)
    ap.add_argument("--weights", action="store_true")
    a = ap.parse_args()
    api = HfApi()
    ok = True
    for run in m4_runs(ROOT):
        api.upload_folder(repo_id=REPO, folder_path=str(run), path_in_repo=f"m4/{run.name}", ignore_patterns=RECORD_IGNORE, commit_message="records")
    for sub in ("runs/m4_queue", "runs/m4_t2d"):
        if (ROOT / sub).exists():
            api.upload_folder(repo_id=REPO, folder_path=str(ROOT / sub), path_in_repo=f"m4/{Path(sub).name}", ignore_patterns=["*.pt"], commit_message="records")
    if a.session == "a":
        run = find_run(ROOT / "runs", "m4-sft")
        if run:
            final = run / "checkpoints" / "milestone-final.pt"
            regular = sorted((run / "checkpoints").glob("ckpt-*.pt"))
            ck = final if final.exists() else (regular[-1] if regular else None)
            if ck:
                ok &= upload_verified(api, ck, f"m4/{run.name}/{ck.name}")
    elif a.weights:
        seeds = yaml.safe_load((ROOT / "benchmarks" / "m4" / "t2d.yaml").read_text(encoding="utf-8"))["recipe"]["seeds"]
        for run in chosen_weight_dirs(ROOT, seeds):
            for f in sorted(p for p in (run / "weights").iterdir() if p.is_file()):
                ok &= upload_verified(api, f, f"m4/{run.name}/weights/{f.name}")
    print("backup", "verified" if ok else "NOT VERIFIED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
