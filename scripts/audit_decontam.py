"""Decontamination audit (EXP-M1-DECONTAM audit rule): which 13-grams make documents of one source match a benchmark.

Reads one input file of a source, finds every matching window, and reports the most frequent matched 13-grams per benchmark
(with the benchmark item they come from), so boilerplate false positives can be told apart from real contamination.
Read-only; writes a `data` record.

    python scripts/audit_decontam.py --source math --file-index 0 --ngrams <run>/ngrams.npz
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from litegpt.dataprep import load_sources, read_docs
    from litegpt.decontam import NGRAM, NgramIndex, window_hashes, words
    from litegpt.records import create_run
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--file-index", type=int, default=0)
    ap.add_argument("--ngrams", required=True)
    ap.add_argument("--max-docs", type=int, default=200000)
    ap.add_argument("--repo-file", default=None, help="explicit file in the source repo (sources without a fixed file list)")
    args = ap.parse_args()
    sources = load_sources()
    cfg = sources["sources"][args.source]
    ix = NgramIndex.load(args.ngrams)
    rec = create_run("data", f"decontam-audit-{args.source}", config=vars(args))
    tmp = tempfile.mkdtemp(prefix="_audit_", dir=str(Path.home()))
    try:
        if args.repo_file:
            fn = args.repo_file
        else:
            idxs = cfg["files"].get("indices") or list(range(cfg["files"]["range"][0], cfg["files"]["range"][1] + 1))
            fn = cfg["files"]["pattern"].format(i=idxs[args.file_index])
        local = hf_hub_download(cfg["repo"], fn, repo_type="dataset", revision=cfg["revision"], local_dir=tmp)
        grams = collections.Counter()
        gram_bench = {}
        docs = hit_docs = 0
        hit_bytes = total_bytes = 0
        per_doc_hits = []
        for doc in read_docs(args.source, cfg, local):
            text = doc["text"] if isinstance(doc["text"], str) else json.dumps(doc["text"])
            docs += 1
            total_bytes += len(text.encode("utf-8", "replace"))
            h = window_hashes(text)
            if not len(h):
                continue
            pos = np.searchsorted(ix.hashes, h)
            pos[pos == len(ix.hashes)] = 0
            hit = ix.hashes[pos] == h
            if hit.any():
                hit_docs += 1
                hit_bytes += len(text.encode("utf-8", "replace"))
                ws = words(text)
                where = np.nonzero(hit)[0]
                per_doc_hits.append(len(where))
                for w in where[:20]:
                    g = " ".join(ws[w:w + NGRAM])
                    grams[g] += 1
                    gram_bench[g] = ix.names[int(ix.bench_ids[pos[w]])]
            if docs >= args.max_docs:
                break
        top = [{"ngram": g, "docs_windows": c, "benchmark": gram_bench[g]} for g, c in grams.most_common(60)]
        by_bench = collections.Counter()
        for g, c in grams.items():
            by_bench[gram_bench[g]] += c
        res = {"source": args.source, "file": fn, "docs": docs, "hit_docs": hit_docs, "hit_bytes_fraction": hit_bytes / max(total_bytes, 1),
               "windows_by_benchmark": dict(by_bench), "median_hits_per_doc": float(np.median(per_doc_hits)) if per_doc_hits else 0,
               "top_ngrams": top}
        rec.write_json("decontam_audit.json", res)
        print(json.dumps({k: v for k, v in res.items() if k != "top_ngrams"}, indent=1))
        for t in top[:40]:
            print(f"{t['docs_windows']:6}  {t['benchmark']:12} {t['ngram']}")
        print("record:", rec.run_id)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
