"""Data preparation entry point (M1). Subcommands are added step by step; see PLAN_M1.md.

verify: read-only check of every candidate source at its current revision. Records revision SHA, gating, licence from the
dataset card, the file list and sizes per config, and whether a sample row carries non-empty inline text. Nothing is
downloaded beyond a handful of streamed rows, and no terms are accepted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# (domain, repo, config or None, split). Candidates only: docs/spec/data_sources.yaml (step 2) is the binding list.
CANDIDATES = [
    ("web", "HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", "train"),
    ("web-fallback", "HuggingFaceFW/fineweb-edu", "sample-100BT", "train"),
    ("synthetic", "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "train"),
    ("math-3plus", "HuggingFaceTB/finemath", "finemath-3plus", "train"),
    ("math-4plus", "HuggingFaceTB/finemath", "finemath-4plus", "train"),
    ("wiki", "wikimedia/wikipedia", "20231101.en", "train"),
    ("instruction", "HuggingFaceTB/smoltalk", "all", "train"),
    ("code", "HuggingFaceTB/smollm-corpus", "python-edu", "train"),          # expected to FAIL (blob ids only)
    ("code", "HuggingFaceTB/stack-edu", "Python", "train"),
    ("code", "bigcode/the-stack-dedup", None, "train"),
    ("code", "bigcode/starcoderdata", None, "train"),
    ("code", "codeparrot/github-code-clean", None, "train", "data/train-00000-of-00880.parquet"),   # script dataset: read the parquet directly
]
TEXT_KEYS = ("text", "content", "messages", "code")
EXTRA_KEYS = ("license", "language", "lang", "repo_name", "path", "blob_id", "max_stars_repo_licenses")


def repo_info(api, repo: str) -> dict:
    info = api.dataset_info(repo, files_metadata=True)
    card = info.card_data.to_dict() if info.card_data else {}
    files = [(s.rfilename, s.size or 0) for s in (info.siblings or [])]
    return {"sha": info.sha, "gated": info.gated, "private": info.private, "license": card.get("license"),
            "license_name": card.get("license_name"), "license_link": card.get("license_link"),
            "card_tags": card.get("tags"), "configs": [c.get("config_name") for c in card.get("configs", [])] if card.get("configs") else None,
            "n_files": len(files), "total_bytes": sum(s for _, s in files), "files": files}


def config_size(files: list, config: str | None) -> dict:
    sel = [(n, s) for n, s in files if n.endswith((".parquet", ".jsonl", ".json.gz", ".jsonl.zst", ".arrow", ".gz", ".zst"))]
    if config:
        sel = [(n, s) for n, s in sel if f"/{config}/" in f"/{n}" or n.startswith(config) or f"{config}/" in n]
    return {"n_data_files": len(sel), "bytes": sum(s for _, s in sel), "examples": [n for n, _ in sel[:3]]}


def sample_rows(repo: str, config: str | None, split: str, data_files: str | None = None, n: int = 3) -> dict:
    from datasets import load_dataset
    if data_files:   # script-based dataset: read the parquet file straight from the Hub
        ds = load_dataset("parquet", data_files=f"hf://datasets/{repo}/{data_files}", split=split, streaming=True)
    else:
        ds = load_dataset(repo, config, split=split, streaming=True) if config else load_dataset(repo, split=split, streaming=True)
    rows, cols = [], None
    for i, row in enumerate(ds):
        cols = cols or list(row.keys())
        key = next((k for k in TEXT_KEYS if k in row), None)
        val = row.get(key) if key else None
        text = json.dumps(val)[:300] if not isinstance(val, str) else val[:300]
        rows.append({"len": len(val) if isinstance(val, (str, list)) else None, "head": text,
                     "meta": {k: (v if isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) < 80) else None)
                              for k, v in row.items() if k != key}})
        if i + 1 >= n:
            break
    inline = bool(rows) and all((r["len"] or 0) > 0 for r in rows)
    return {"columns": cols, "text_inline_nonempty": inline, "rows": rows}


def cmd_verify(args) -> int:
    from huggingface_hub import HfApi
    from litegpt.records import create_run
    api = HfApi()
    rec = create_run("data", "verify-sources", config={"candidates": [list(c) for c in CANDIDATES]})
    out, cache = [], {}
    for cand in CANDIDATES:
        domain, repo, config, split = cand[:4]
        data_files = cand[4] if len(cand) > 4 else None
        entry = {"domain": domain, "repo": repo, "config": config, "split": split, "data_files": data_files}
        try:
            if repo not in cache:
                cache[repo] = repo_info(api, repo)
            info = cache[repo]
            entry.update({k: info[k] for k in ("sha", "gated", "license", "license_name", "license_link", "configs")})
            entry["size"] = config_size(info["files"], config)
        except Exception as e:
            entry["info_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        try:
            entry["sample"] = sample_rows(repo, config, split, data_files)
        except Exception as e:
            entry["sample_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"{domain:12} {repo}:{config} -> license={entry.get('license')} gated={entry.get('gated')} "
              f"inline={entry.get('sample', {}).get('text_inline_nonempty')} "
              f"{'ERR ' + (entry.get('info_error') or entry.get('sample_error') or '') if ('info_error' in entry or 'sample_error' in entry) else ''}")
        out.append(entry)
    rec.write_json("verify.json", out)
    print("record:", rec.run_id)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify", help="read-only source verification (step 1)")
    args = ap.parse_args()
    return {"verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
