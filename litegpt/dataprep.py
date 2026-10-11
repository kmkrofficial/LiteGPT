"""Data preparation: source readers, per-file processing, mixing and validation sets (ADR-008).

`process_file` turns one input parquet file into per-(sub-source, partition) uint16 shard files plus a stats record. Everything
is deterministic and per-file independent, so the run is parallel and resumable (a file is done when its stats file exists).
`build_mix` then turns the shards into the stable / decay / validation / proxy indexes. Sources and weights come from
docs/spec/data_sources.yaml; filters from litegpt/datafilters.py; contamination from litegpt/decontam.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

from litegpt import chat_template, data as D
from litegpt.datafilters import filter_code
from litegpt.decontam import NgramIndex
from litegpt.tokenizer import BOS, EOT, LiteTokenizer

ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = ROOT / "docs" / "spec" / "data_sources.yaml"


def load_sources(path=None) -> dict:
    import yaml
    return yaml.safe_load(Path(path or SOURCES_PATH).read_text(encoding="utf-8"))


def sources_sha256(path=None) -> str:
    return hashlib.sha256(Path(path or SOURCES_PATH).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def content_hash(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:24]


# --- readers: one generator of dicts {key, text, eligible, sub, lang, split} per source ---------------------------------
def _rows(path, columns):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=columns):
        d = batch.to_pydict()
        for i in range(len(d[columns[0]])):
            yield {c: d[c][i] for c in columns}


def read_docs(name: str, cfg: dict, path, split: str = "train"):
    """Yield dicts with: key, text (str, or messages list for instruction_style), eligible, sub (sub-source), lang, split."""
    if name == "fineweb_edu":
        for r in _rows(path, ["text", "id", "metadata"]):
            m = r["metadata"] or {}
            yield {"key": str(r["id"]), "text": r["text"], "eligible": (m.get("int_score") or 0) >= 4, "sub": name, "split": split}
    elif name == "cosmopedia_v2":
        for r in _rows(path, ["text"]):
            yield {"key": content_hash(r["text"]), "text": r["text"], "eligible": True, "sub": name, "split": split}
    elif name == "math":
        for r in _rows(path, ["text", "url", "int_score"]):
            yield {"key": str(r["url"]), "text": r["text"], "eligible": (r["int_score"] or 0) >= 4, "sub": name, "split": split}
    elif name == "code":
        langs = cfg["languages"]
        permissive = set(cfg["permissive_licences"])
        for r in _rows(path, ["code", "repo_name", "path", "language", "license", "size"]):
            if r["language"] not in langs or r["license"] not in permissive or not r["code"]:
                continue
            yield {"key": r["repo_name"], "text": r["code"], "eligible": True, "sub": f"code_{r['language'].lower()}",
                   "lang": r["language"], "path": r["path"], "size": r["size"], "split": split}
    elif name == "instruction_style":
        for r in _rows(path, ["messages"]):
            msgs = [{"role": m["role"], "content": m["content"]} for m in r["messages"]]
            yield {"key": content_hash(msgs), "text": msgs, "eligible": True, "sub": name, "split": split}
    else:
        raise KeyError(name)


# --- per-file processing -------------------------------------------------------------------------------------------------
def _raw_text(doc) -> str:
    return doc["text"] if isinstance(doc["text"], str) else chat_template.render_text(doc["text"])


def process_file(source: str, cfg: dict, local_path, out_dir, file_tag: str, tok: LiteTokenizer, salt: str,
                 ngrams: NgramIndex | None = None, split: str = "train", gpt2_tok=None, shard_tokens: int = 256_000_000) -> dict:
    """Read one parquet file and write shards for every (sub-source, partition). Returns the stats record."""
    out_dir = Path(out_dir)
    part_cfg = cfg.get("partition", {})
    langs = cfg.get("languages", {})
    stats = {"source": source, "file": file_tag, "docs_read": 0, "docs_kept": {}, "tokens": {}, "bytes": {}, "dropped": {},
             "decontam": {}, "emails_redacted_docs": 0}
    writers: dict = {}
    raw_val: dict = {}
    pending: dict = {}

    def bump(d, k, n=1):
        d[k] = d.get(k, 0) + n

    def flush(sub, part):
        docs = pending.pop((sub, part), [])
        if not docs:
            return
        ids = tok.encode_batch([d for d, _ in docs]) if isinstance(docs[0][0], str) else None
        for j, (payload, _) in enumerate(docs):
            toks = ids[j] if ids is not None else payload
            arr = [BOS] + list(toks) + [EOT] if ids is not None else list(toks) + [EOT]
            w = writers.get((sub, part))
            if w is None:
                w = writers[(sub, part)] = D.ShardWriter(out_dir / sub / part, prefix=file_tag, shard_tokens=shard_tokens)
            w.add(arr)
            bump(stats["tokens"], f"{sub}/{part}", len(arr))

    def add(sub, part, payload, raw):
        pending.setdefault((sub, part), []).append((payload, raw))
        if len(pending[(sub, part)]) >= 256:
            flush(sub, part)

    for doc in read_docs(source, cfg, local_path, split):
        stats["docs_read"] += 1
        text, sub = doc["text"], doc["sub"]
        if source == "code":
            c: dict = {}
            kept, rule = filter_code(text, language=doc["lang"], path=doc.get("path"), size=doc.get("size"), counters=c, repo=doc["key"])
            stats["emails_redacted_docs"] += c.get("email_redacted_docs", 0)
            if rule:
                bump(stats["dropped"], f"{sub}:{rule}")
                continue
            text = kept
        raw = text if isinstance(text, str) else chat_template.render_text(text)
        if ngrams is not None:
            hit = ngrams.match(raw)
            if hit is not None:
                bump(stats["decontam"], f"{sub}:{ngrams.names[hit]}")
                bump(stats["bytes"], f"{sub}/decontam_dropped", len(raw.encode("utf-8", "replace")))
                continue
        # partition
        if source == "instruction_style":
            if split == "test":
                part = "val"
            else:
                u = D.partition_u(salt, source, doc["key"])
                part = "decay" if u < part_cfg["decay_fraction_of_train"] else "sft_reserved"
        else:
            u = D.partition_u(salt, source, doc["key"])
            part = D.assign_partition(u, part_cfg["val_fraction"], part_cfg["decay_fraction"], doc["eligible"])
            if part == "stable":
                kf = langs.get(doc.get("lang"), {}).get("keep_fraction", part_cfg.get("keep_fraction", 1.0))
                if kf < 1.0 and D.partition_u(salt, source, doc["key"], ":keep") >= kf:
                    bump(stats["dropped"], f"{sub}:keep_fraction")
                    continue
        if part == "sft_reserved":
            bump(stats["dropped"], f"{sub}:reserved_for_sft")
            continue
        bump(stats["docs_kept"], f"{sub}/{part}")
        bump(stats["bytes"], f"{sub}/{part}", len(raw.encode("utf-8", "replace")))
        if part == "val":
            raw_val.setdefault(sub, []).append({"key": doc["key"], "text": raw})
        if source == "instruction_style":
            add(sub, part, chat_template.render_ids(tok, text), raw)
        else:
            add(sub, part, text, raw)
        if gpt2_tok is not None and source == "fineweb_edu" and part in ("stable", "val"):
            ids = gpt2_tok.encode(text).ids + [50256]
            w = writers.get(("proxy_gpt2", part))
            if w is None:
                w = writers[("proxy_gpt2", part)] = D.ShardWriter(out_dir / "proxy_gpt2" / part, prefix=file_tag, shard_tokens=shard_tokens)
            w.add(ids)
            bump(stats["tokens"], f"proxy_gpt2/{part}", len(ids))
    for key in list(pending):
        flush(*key)
    shards = {}
    for (sub, part), w in writers.items():
        shards[f"{sub}/{part}"] = w.close()
    for sub, docs in raw_val.items():
        p = out_dir / sub / "val"
        p.mkdir(parents=True, exist_ok=True)
        with open(p / f"{file_tag}.jsonl", "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    stats["shards"] = shards
    stats_dir = out_dir / "_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    tmp = stats_dir / f"{source}-{file_tag}.json.tmp"
    tmp.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    os.replace(tmp, stats_dir / f"{source}-{file_tag}.json")
    return stats


def file_done(out_dir, source: str, file_tag: str) -> bool:
    return (Path(out_dir) / "_stats" / f"{source}-{file_tag}.json").exists()


# --- mixing ---------------------------------------------------------------------------------------------------------------
def collect_shards(out_dir) -> dict:
    """{ 'sub/part': ShardSet } over every .bin written under out_dir/<sub>/<part>/ (sorted for a stable order)."""
    out_dir = Path(out_dir)
    res = {}
    for sub_dir in sorted(p for p in out_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for part_dir in sorted(p for p in sub_dir.iterdir() if p.is_dir()):
            bins = sorted(part_dir.glob("*.bin"))
            if bins:
                res[f"{sub_dir.name}/{part_dir.name}"] = D.ShardSet(bins)
    return res


def _blocks(shards: dict, part: str, block_len=D.BLOCK_TOKENS) -> dict:
    return {k.split("/")[0]: D.block_count(len(v), block_len) for k, v in shards.items() if k.endswith("/" + part)}


def build_mix(out_dir, mixture: dict, seed: int, stable_tokens: float, decay_tokens: float, val_core_blocks: int = 10240,
              domain_blocks: int = 1024, code_langs: dict | None = None, block_len: int = D.BLOCK_TOKENS,
              proxy_tokens: float = 1.0e9) -> dict:
    """Write indexes under out_dir/indexes and return a manifest with counts and sha256 values."""
    out_dir = Path(out_dir)
    idx_dir = out_dir / "indexes"
    idx_dir.mkdir(exist_ok=True)
    shards = collect_shards(out_dir)
    man: dict = {"seed": seed, "block_len": block_len, "indexes": {}, "available_blocks": {}, "shortfalls": []}
    for part in ("stable", "decay", "val"):
        man["available_blocks"][part] = _blocks(shards, part, block_len)

    def make(name, part, weights, total):
        nb = man["available_blocks"][part]
        src, blk, names = D.build_index(nb, weights, total, seed)
        h = D.save_index(idx_dir / f"{name}.npz", src, blk, names, {"part": part, "weights": weights})
        man["indexes"][name] = {"blocks": int(total), "tokens": int(total) * block_len, "sha256": h, "sources": names}

    make("stable", "stable", mixture["stable"], int(stable_tokens // block_len))
    make("decay", "decay", mixture["decay"], int(decay_tokens // block_len))
    make("val_core", "val", mixture["stable"], val_core_blocks)
    web = {"fineweb_edu": 1.0}
    domains = {"web": web, "synthetic": {"cosmopedia_v2": 1.0}, "math": {"math": 1.0}, "chat": {"instruction_style": 1.0}}
    if code_langs:
        domains["code"] = {f"code_{k.lower()}": v for k, v in code_langs.items()}
    for dom, w in domains.items():
        try:
            make(f"val_{dom}", "val", w, domain_blocks)
        except ValueError as e:
            man["shortfalls"].append(f"val_{dom}: {e}")
    man["data_order_sha256"] = hashlib.sha256((man["indexes"]["stable"]["sha256"] + man["indexes"]["decay"]["sha256"]).encode()).hexdigest()
    # proxy views for M2 (steps 7-8): 1B-token prefix of the stable order, and the same without code and math
    nb = man["available_blocks"]["stable"]
    proxy_blocks = int(proxy_tokens // block_len)
    make("proxy_1b", "stable", mixture["stable"], proxy_blocks)
    nocm = {k: v for k, v in mixture["stable"].items() if not (k.startswith("code_") or k == "math")}
    make("proxy_1b_nocodemath", "stable", nocm, proxy_blocks)
    (idx_dir / "mix_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man


def sample_tokenizer_text(sources: dict, mixture_stable: dict, files: dict, total_bytes: int, salt: str):
    """Generator of training text for the tokenizer: stable-partition documents at the stable mixture weights.
    `files` maps source -> list of local parquet paths. Never reads validation or decay documents."""
    budget = {sub: int(total_bytes * w) for sub, w in mixture_stable.items()}
    used = {sub: 0 for sub in budget}
    for source, paths in files.items():
        cfg = sources["sources"][source]
        subs = [s for s in budget if (s.startswith("code_") if source == "code" else s == source)]
        p = cfg["partition"]
        for path in paths:
            for doc in read_docs(source, cfg, path):
                sub = doc["sub"]
                if sub not in budget or used[sub] >= budget[sub]:
                    continue
                u = D.partition_u(salt, source, doc["key"])
                if D.assign_partition(u, p["val_fraction"], p["decay_fraction"], doc["eligible"]) != "stable":
                    continue
                text = doc["text"]
                if source == "code":
                    text, rule = filter_code(text, language=doc["lang"], path=doc.get("path"), size=doc.get("size"), repo=doc["key"])
                    if rule:
                        continue
                used[sub] += len(text.encode("utf-8", "replace"))
                yield text
            if subs and all(used[s] >= budget[s] for s in subs):
                break
