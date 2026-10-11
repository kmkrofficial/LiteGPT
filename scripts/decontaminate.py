"""13-gram decontamination tool (EXP-M1-DECONTAM).

build: download the pinned evaluation sets listed in docs/spec/data_sources.yaml, normalise to word 13-grams and write one
       index (npz) plus a record with per-benchmark counts. Used by `prepare_data.py prepare --ngrams`.
scan:  report which documents of a jsonl file (field `text`, or `--field`) overlap the index; the same code is applied to
       SFT / reasoning / DPO data in M4.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _strings(v)


def cmd_build(args) -> int:
    from datasets import load_dataset
    from litegpt.dataprep import load_sources, sources_sha256
    from litegpt.decontam import NgramIndex
    from litegpt.records import create_run
    ref = load_sources()["decontamination"]["reference"]
    rec = create_run("data", "decontam-index", config={"sources_sha256": sources_sha256(), "reference": ref})
    texts, counts = {}, {}
    for name, r in ref.items():
        kw = {"revision": r["revision"]}
        ds = (load_dataset(r["repo"], r["config"], split=r["split"], streaming=True, **kw) if r["config"]
              else load_dataset(r["repo"], split=r["split"], streaming=True, **kw))
        rows = [s for row in ds for s in _strings(row)]
        texts[name] = rows
        counts[name] = {"strings": len(rows), "repo": r["repo"], "revision": r["revision"]}
    ix = NgramIndex.from_texts(texts)
    out = Path(args.out) if args.out else rec.path / "ngrams.npz"
    ix.save(out)
    import numpy as np
    per = {n: int((ix.bench_ids == i).sum()) for i, n in enumerate(ix.names)}
    rec.write_json("ngram_counts.json", {"unique_ngrams": len(ix), "first_owner_counts": per, "benchmarks": counts})
    print(f"unique 13-grams: {len(ix)}; written to {out}")
    print("record:", rec.run_id)
    return 0


def cmd_scan(args) -> int:
    from litegpt.decontam import NgramIndex
    ix = NgramIndex.load(args.index)
    hits, total, by = 0, 0, {}
    for line in Path(args.file).read_text(encoding="utf-8").splitlines():
        text = json.loads(line)[args.field]
        total += 1
        m = ix.match(text if isinstance(text, str) else json.dumps(text, ensure_ascii=False))
        if m is not None:
            hits += 1
            by[ix.names[m]] = by.get(ix.names[m], 0) + 1
    print(json.dumps({"documents": total, "contaminated": hits, "by_benchmark": by}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default=None)
    s = sub.add_parser("scan")
    s.add_argument("--index", required=True)
    s.add_argument("--file", required=True)
    s.add_argument("--field", default="text")
    args = ap.parse_args()
    return {"build": cmd_build, "scan": cmd_scan}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
