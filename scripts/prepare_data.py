"""Data preparation entry point (M1). See PLAN_M1.md and ADR-008.

prepare: resumable pipeline over the fixed input-file lists of docs/spec/data_sources.yaml. mix: indexes + frozen hashes.

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


def source_jobs(sources: dict, only: list | None = None, limit: int | None = None) -> list:
    """[(source, file_tag, repo, filename, revision, split, gpt2_proxy)] for the fixed input lists in data_sources.yaml."""
    from huggingface_hub import HfApi
    jobs = []
    for name, cfg in sources["sources"].items():
        if cfg.get("domain") == "m4_sft_only" or (only and name not in only):
            continue
        rev = cfg["revision"]
        n_before = len(jobs)
        if name == "instruction_style":
            names = [s.rfilename for s in HfApi().dataset_info(cfg["repo"], revision=rev).siblings]
            for sub in cfg["subsets"]:
                for fn in sorted(f for f in names if f.startswith(f"data/{sub}/") and f.endswith(".parquet")):
                    split = "test" if "/test-" in fn else "train"
                    jobs.append((name, f"{sub}-{split}-{Path(fn).stem}", cfg["repo"], fn, rev, split, False))
        else:
            pat = cfg["files"]["pattern"]
            idxs = cfg["files"].get("indices") or list(range(cfg["files"]["range"][0], cfg["files"]["range"][1] + 1))
            for rank, i in enumerate(idxs):
                jobs.append((name, f"{i:05d}", cfg["repo"], pat.format(i=i), rev, "train", name == "fineweb_edu" and rank < 2))
        if limit:
            jobs[n_before + limit:] = []
    return jobs


def _prepare_one(job):
    import os
    import shutil
    import tempfile
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from litegpt import dataprep as P
    from litegpt.decontam import NgramIndex
    from litegpt.tokenizer import LiteTokenizer
    name, tag, repo, fn, rev, split, gpt2, out_dir, tok_path, ngram_path, cfg, salt = job
    if P.file_done(out_dir, name, tag):
        return {"source": name, "file": tag, "skipped": True}
    tok = LiteTokenizer.load(tok_path)
    ngrams = NgramIndex.load(ngram_path) if ngram_path else None
    gpt2_tok = Tokenizer.from_pretrained("openai-community/gpt2") if gpt2 else None
    tmp = tempfile.mkdtemp(prefix="_dl_", dir=str(Path(out_dir)))
    try:
        local = hf_hub_download(repo, fn, repo_type="dataset", revision=rev, local_dir=tmp)
        return P.process_file(name, cfg, local, out_dir, tag, tok, salt, ngrams, split=split, gpt2_tok=gpt2_tok)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def cmd_prepare(args) -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from litegpt import dataprep as P
    from litegpt.records import RunRecord, create_run
    sources = P.load_sources()
    jobs_meta = source_jobs(sources, args.sources.split(",") if args.sources else None, args.limit_files)
    if args.resume:
        run_dir = Path(args.resume).resolve()
        rec = RunRecord(run_dir, json.loads((run_dir / "run.json").read_text(encoding="utf-8")))
    else:
        rec = create_run("data", "prepare", config={"sources_sha256": P.sources_sha256(), "tokenizer": str(args.tokenizer),
                         "ngrams": str(args.ngrams), "n_jobs": len(jobs_meta), "workers": args.workers})
    out = rec.path / "shards"
    out.mkdir(exist_ok=True)
    salt = sources["partitioning"]["salt"]
    jobs = [(n, t, r, f, v, s, g, str(out), str(args.tokenizer), str(args.ngrams) if args.ngrams else None, sources["sources"][n], salt)
            for (n, t, r, f, v, s, g) in jobs_meta]
    print(f"{len(jobs)} files; run {rec.run_id}", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_prepare_one, j): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                st = fut.result()
            except Exception as e:
                rec.log("error", source=j[0], file=j[1], error=f"{type(e).__name__}: {str(e)[:300]}")
                print(f"ERROR {j[0]} {j[1]}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            done += 1
            rec.log("progress", source=j[0], file=j[1], skipped=bool(st.get("skipped")), tokens=st.get("tokens"), docs_read=st.get("docs_read"))
            print(f"[{done}/{len(jobs)}] {j[0]} {j[1]} {'skipped' if st.get('skipped') else sum((st.get('tokens') or {}).values())} tokens", flush=True)
    print("record:", rec.run_id)
    return 0


def cmd_mix(args) -> int:
    import hashlib
    import numpy as np
    from litegpt import data as D, dataprep as P
    from litegpt.records import RunRecord
    run_dir = Path(args.run).resolve()
    rec = RunRecord(run_dir, json.loads((run_dir / "run.json").read_text(encoding="utf-8")))
    sources = P.load_sources()
    mix = sources["mixture"]
    langs = {k: v["weight"] for k, v in sources["sources"]["code"]["languages"].items()}
    man = P.build_mix(run_dir / "shards", {"stable": mix["stable"], "decay": mix["decay"]}, seed=mix["index"]["seed"],
                      stable_tokens=mix["stable_tokens"], decay_tokens=mix["decay_tokens"],
                      val_core_blocks=sources["validation"]["val_core"]["blocks"], domain_blocks=sources["validation"]["domains"]["blocks_each"],
                      code_langs=langs)
    # val_core sha256 = hash of the actual token content of its blocks, in index order
    src, blk, names, _ = D.load_index(run_dir / "shards" / "indexes" / "val_core.npz")
    shards = P.collect_shards(run_dir / "shards")
    h = hashlib.sha256()
    total = 0
    for s, b in zip(src, blk):
        arr = shards[f"{names[int(s)]}/val"].read(int(b) * D.BLOCK_TOKENS, D.BLOCK_TOKENS + 1)
        h.update(arr.tobytes())
        total += D.BLOCK_TOKENS
    man["val_core"] = {"sha256": h.hexdigest(), "tokens": total, "blocks": int(len(src))}
    rec.write_json("mix_manifest.json", man)
    print(json.dumps({k: man[k] for k in ("data_order_sha256", "val_core", "shortfalls")}, indent=1))
    print("record:", rec.run_id)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify", help="read-only source verification (step 1)")
    pp = sub.add_parser("prepare", help="download -> filter -> decontaminate -> partition -> tokenize -> shards (resumable)")
    pp.add_argument("--tokenizer", required=True, help="frozen tokenizer json")
    pp.add_argument("--ngrams", default=None, help="13-gram index (scripts/decontaminate.py build); omit only for dry runs")
    pp.add_argument("--workers", type=int, default=12)
    pp.add_argument("--sources", default=None, help="comma-separated subset of sources")
    pp.add_argument("--limit-files", type=int, default=None, help="process only the first N files of each source (dry run)")
    pp.add_argument("--resume", default=None, help="existing data run directory to continue")
    pm = sub.add_parser("mix", help="build stable/decay/validation/proxy indexes and freeze hashes")
    pm.add_argument("--run", required=True, help="data run directory holding shards/")
    args = ap.parse_args()
    return {"verify": cmd_verify, "prepare": cmd_prepare, "mix": cmd_mix}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
