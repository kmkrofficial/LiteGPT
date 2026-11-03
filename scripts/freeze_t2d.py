"""Build the EXP-M4-T2D items from pinned local files, deterministically. No model is involved.

    python scripts/freeze_t2d.py --downloads ~/m4_audit_downloads --out ~/m4_t2d/final --check benchmarks/m4/t2d_manifest.json

Splits come from a salted hash, CUAD by filer so the same parties are never on both sides. Test items the owner-approved rules remove
are listed in the committed manifest, and the output must match its hashes byte for byte, or this exits with an error. Removal lists came from the
13-gram scan (`docs/private/logs/m4/phase0/`) and the context check (`docs/private/logs/m4/phase1/billsum_fit.json`).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
from pathlib import Path

SALT = "litegpt-m4-t2d-20261103"
HEAD, WIN = 3000, 1500
BILLSUM_MAX_CHARS, BILLSUM_DEV = 6000, 300
CUAD_TEST, CUAD_DEV = 102, 51
KEY = re.compile(r"govern(ed|ing)\s+(in\s+all\s+respects\s+)?(by\s+)?(the\s+)?(law|internal\s+law)|governing\s+law|laws?\s+of\s+the\s+state|construed\s+(and\s+\w+\s+)?(in\s+accordance\s+with|under)|shall\s+govern", re.I)
SLOT = re.compile(r"\[(\w+) : ([^\]]+)\]")


def norm(t: str) -> str:
    t = re.sub(r"_{4,}", "____", t.replace("\xa0", " "))
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\s*\n\s*", "\n", t).strip()


def cuad_input(ctx: str) -> str:
    """The first 3,000 characters, then 1,500 around the last governing-law keyword after them, or the end of the contract if there is none."""
    c = norm(ctx)
    if len(c) <= HEAD + WIN:
        return c
    last = None
    for m in KEY.finditer(c, HEAD):
        last = m
    start = max(HEAD, last.start() - WIN // 4) if last else len(c) - WIN
    return c[:HEAD] + "\n[...]\n" + c[start:start + WIN]


def fname_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\.pdf$", "", s.strip(), flags=re.I).lower())


def group_of(title: str) -> str:
    return re.split(r"_| - ", title)[0].strip().lower()


def blank(v: str) -> bool:
    return not v or "[]" in v


def parties(v: str) -> list[str]:
    out = []
    for p in v.split(";"):
        p = re.sub(r"\s+", " ", re.sub(r"\([^)]*\)?", "", p)).strip(" ,.;:")
        if p and not re.search(r"\bpart(y|ies)\b", p, re.I) and p not in out:
            out.append(p)
    return out


def build_cuad(dl: Path) -> dict:
    csv.field_size_limit(10**9)
    data = json.load(open(dl / "cuad/CUAD_v1/CUAD_v1.json", encoding="utf-8"))["data"]
    rows = {fname_key(r["Filename"]): r for r in csv.DictReader(open(dl / "cuad/CUAD_v1/master_clauses.csv", encoding="utf-8", newline=""))}
    groups: dict = {}
    for d in data:
        groups.setdefault(group_of(d["title"]), []).append(d)
    filled = {"test": 0, "dev": 0}
    out: dict = {"train": [], "dev": [], "test": []}
    for g in sorted(groups, key=lambda g: hashlib.sha256((SALT + g).encode()).hexdigest()):
        split = "test" if filled["test"] < CUAD_TEST else "dev" if filled["dev"] < CUAD_DEV else "train"
        if split != "train":
            filled[split] += len(groups[g])
        for d in groups[g]:
            r = rows.get(fname_key(d["title"]))
            if r is None:
                continue
            date, law = r["Agreement Date-Answer"].strip(), r["Governing Law-Answer"].strip()
            gold = {"document_name": re.sub(r"\s+", " ", r["Document Name-Answer"]).strip() or None, "parties": parties(r["Parties-Answer"]),
                    "agreement_date": None if blank(date) else date.split(";")[0].strip(), "governing_law": None if blank(law) else [x.strip() for x in law.split(";") if x.strip()]}
            out[split].append({"id": d["title"], "input": cuad_input(d["paragraphs"][0]["context"]), "gold": gold})
    return out


def build_massive(dl: Path) -> dict:
    import pyarrow.parquet as pq
    return {out: [{"id": str(r["id"]), "input": r["utt"], "gold": [{"type": t, "value": v.strip()} for t, v in SLOT.findall(r["annot_utt"])]}
                  for r in pq.read_table(dl / f"massive/en-US/{s}/0000.parquet").to_pylist()] for s, out in [("train", "train"), ("validation", "dev"), ("test", "test")]}


def build_banking(dl: Path) -> dict:
    import pyarrow.parquet as pq
    table = pq.read_table(dl / "banking77/data/train-00000-of-00001.parquet")
    names = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]["label"]["names"]
    per: dict = {}
    for i, r in enumerate(table.to_pylist()):
        per.setdefault(r["label"], []).append((hashlib.sha256(f"{SALT}{i}{r['text']}".encode()).hexdigest(), i, r))
    train, dev = [], []
    for label, v in sorted(per.items()):
        v.sort()
        k = max(1, round(len(v) * 0.1))
        dev += [{"id": f"tr{i}", "input": r["text"], "gold": names[label]} for _, i, r in v[:k]]
        train += [{"id": f"tr{i}", "input": r["text"], "gold": names[label]} for _, i, r in v[k:]]
    test = [{"id": f"te{i}", "input": r["text"], "gold": names[r["label"]]} for i, r in enumerate(pq.read_table(dl / "banking77/data/test-00000-of-00001.parquet").to_pylist())]
    return {"train": train, "dev": dev, "test": test}


def build_billsum(dl: Path) -> dict:
    import pyarrow.parquet as pq
    out = {}
    for s in ["train", "test"]:
        keep = []
        for i, r in enumerate(pq.read_table(dl / f"billsum/data/{s}-00000-of-00001.parquet").to_pylist()):
            t = norm(r["text"])
            if len(t) <= BILLSUM_MAX_CHARS and not re.match(r"-\w-\w-\w", r["text"]):
                keep.append({"id": f"{s}{i}", "input": t, "gold": html.unescape(r["summary"]).replace("\xa0", " ").strip()})
        out[s] = keep
    dev_ids = {h[64:] for h in sorted(hashlib.sha256(x["id"].encode()).hexdigest() + x["id"] for x in out["train"])[:BILLSUM_DEV]}
    out["dev"] = [x for x in out["train"] if x["id"] in dev_ids]
    out["train"] = [x for x in out["train"] if x["id"] not in dev_ids]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--downloads", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", required=True, help="the committed manifest")
    a = ap.parse_args()
    dl, dest = Path(os.path.expanduser(a.downloads)), Path(os.path.expanduser(a.out))
    dest.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(a.check).read_text(encoding="utf-8"))
    tasks = {"cuad": build_cuad(dl), "massive": build_massive(dl), "banking77": build_banking(dl), "billsum": build_billsum(dl)}
    bad = 0
    for task, splits in tasks.items():
        for split, items in splits.items():
            want = manifest["tasks"][task][split]
            if split == "test":
                removed = set(want["removed_ids"]) | set(want.get("removed_for_context_ids", []))
                items = [x for x in items if x["id"] not in removed]
            data = "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in items).encode("utf-8")
            (dest / f"{task}_{split}.jsonl").write_bytes(data)
            got = hashlib.sha256(data).hexdigest()
            ok = got == want["sha256"] and len(items) == want["items"]
            bad += not ok
            print(task, split, len(items), got[:12], "matches" if ok else f"DIFFERS, expected {want['sha256'][:12]}", flush=True)
    print("all splits match the manifest" if not bad else f"{bad} splits differ")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
