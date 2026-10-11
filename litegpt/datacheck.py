"""M1 step 6 checks and derived sets (EXP-M1-DATA): cross-model validation text, exact-duplicate analysis on the shards,
disjointness, index proportions and loader resume on the real data.

Everything here reads the shards and per-file validation jsonl written by `litegpt/dataprep.py`; nothing is downloaded.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from litegpt import data as D
from litegpt.tokenizer import EOT

VAL_TEXT_BYTES = 2_000_000
DOMAIN_SUBS = {"web": ["fineweb_edu"], "synthetic": ["cosmopedia_v2"], "math": ["math"], "chat": ["instruction_style"]}
CORE_WEIGHTS = {"web": 0.6458, "synthetic": 0.1042, "code": 0.125, "math": 0.125}
_TURN = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>\n?", re.S)


def chat_to_plain(rendered: str) -> str:
    """'<|bos|><|im_start|>user\\nhi<|im_end|>\\n...' -> 'user: hi\\n\\n...' (no special-token strings)."""
    return "\n\n".join(f"{role}: {content}" for role, content in _TURN.findall(rendered))


def _val_docs(out_dir: Path, sub: str) -> list[dict]:
    docs = []
    for f in sorted((out_dir / sub / "val").glob("*.jsonl")):
        docs += [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines()]
    return docs


def build_val_text(out_dir, salt: str, code_langs: dict, exclude_keys: set | None = None, budget: int = VAL_TEXT_BYTES) -> dict:
    """Write out_dir/val_text/<domain>.jsonl (fields: key, text) and return {domain: {docs, bytes, sha256}}."""
    out_dir = Path(out_dir)
    dest = out_dir / "val_text"
    dest.mkdir(exist_ok=True)
    exclude_keys = exclude_keys or set()
    plan = {d: [(s, 1.0) for s in subs] for d, subs in DOMAIN_SUBS.items()}
    plan["code"] = [(f"code_{k.lower()}", w) for k, w in code_langs.items()]
    res = {}
    for dom, parts in plan.items():
        rows = []
        for sub, w in parts:
            docs = [d for d in _val_docs(out_dir, sub) if (sub, d["key"]) not in exclude_keys]
            docs.sort(key=lambda d: hashlib.sha256(f"{salt}:val_text:{dom}:{d['key']}".encode()).hexdigest())
            used, cap = 0, int(budget * w)
            for d in docs:
                if used >= cap:
                    break
                text = chat_to_plain(d["text"]) if dom == "chat" else d["text"]
                rows.append({"key": d["key"], "sub": sub, "text": text})
                used += len(text.encode("utf-8"))
        payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        (dest / f"{dom}.jsonl").write_text(payload, encoding="utf-8")
        res[dom] = {"docs": len(rows), "bytes": sum(len(r["text"].encode("utf-8")) for r in rows),
                    "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}
    return res


def doc_hashes(shards: D.ShardSet, chunk: int = 64_000_000) -> np.ndarray:
    """64-bit hashes of every document (tokens between consecutive EOT ids, EOT included) in a token stream."""
    out: list[int] = []
    carry = np.empty(0, dtype=np.uint16)
    n = len(shards)
    for start in range(0, n, chunk):
        arr = np.concatenate([carry, shards.read(start, min(chunk, n - start))])
        ends = np.nonzero(arr == EOT)[0]
        prev = 0
        for e in ends:
            out.append(int.from_bytes(hashlib.blake2b(arr[prev:e + 1].tobytes(), digest_size=8).digest(), "little"))
            prev = e + 1
        carry = arr[prev:]
    return np.array(out, dtype=np.uint64)


def duplicate_report(out_dir) -> dict:
    """Within-source, cross-source and validation-vs-training exact duplicates (by document token hashes)."""
    shards = _collect(out_dir)
    hashes = {}
    for key, ss in shards.items():
        sub, part = key.split("/")
        if sub == "proxy_gpt2":
            continue
        hashes[(sub, part)] = doc_hashes(ss)
    subs = sorted({s for s, _ in hashes})
    train = {s: np.concatenate([hashes[(s, p)] for p in ("stable", "decay") if (s, p) in hashes] or [np.empty(0, np.uint64)]) for s in subs}
    rep = {}
    for s in subs:
        h = train[s]
        uniq, counts = np.unique(h, return_counts=True)
        within = int((counts - 1).sum())
        others = np.concatenate([train[o] for o in subs if o != s] or [np.empty(0, np.uint64)])
        cross = int(np.isin(h, others).sum()) if len(others) else 0
        val = hashes.get((s, "val"), np.empty(0, np.uint64))
        all_train = np.concatenate([train[o] for o in subs])
        leak = np.isin(val, all_train)
        val_docs = _val_docs(Path(out_dir), s)
        aligned = len(val_docs) == len(val)            # shard documents and jsonl rows are written in the same order
        leak_keys = sorted({val_docs[i]["key"] for i in np.nonzero(leak)[0]}) if aligned else []
        rep[s] = {"train_docs": int(len(h)), "within_source_duplicate_docs": within, "within_rate": within / max(len(h), 1),
                  "cross_source_duplicate_docs": cross, "cross_rate": cross / max(len(h), 1),
                  "val_docs": int(len(val)), "val_docs_also_in_training": int(leak.sum()), "val_rows_aligned": aligned,
                  "val_leak_keys": leak_keys}
    return rep


def _collect(out_dir) -> dict:
    from litegpt.dataprep import collect_shards
    return collect_shards(out_dir)


def check_val_keys(out_dir, sources: dict) -> dict:
    """Every validation key recomputes to u < val_fraction (cannot be in a training partition)."""
    out_dir = Path(out_dir)
    salt = sources["partitioning"]["salt"]
    res = {}
    for name, cfg in sources["sources"].items():
        p = cfg.get("partition", {})
        if "val_fraction" not in p:
            continue
        subs = [f"code_{k.lower()}" for k in cfg["languages"]] if name == "code" else [name]
        bad = total = 0
        for sub in subs:
            for d in _val_docs(out_dir, sub):
                total += 1
                bad += D.partition_u(salt, name, d["key"]) >= p["val_fraction"]
        res[name] = {"val_docs": total, "violations": int(bad)}
    return res


def prefix_proportions(src: np.ndarray, names: list[str], weights: dict, step_blocks: int) -> dict:
    """Largest deviation of the cumulative source share from its weight at every `step_blocks` prefix."""
    total_w = sum(weights[n] for n in names)
    worst = 0.0
    counts = np.zeros(len(names))
    for i in range(0, len(src), step_blocks):
        seg = src[i:i + step_blocks]
        counts += np.bincount(seg, minlength=len(names))
        n = counts.sum()
        dev = max(abs(counts[k] / n - weights[names[k]] / total_w) for k in range(len(names)))
        worst = max(worst, dev)
    return {"max_abs_deviation": float(worst), "step_blocks": step_blocks}


def resume_check(out_dir, index_path, batch_size: int = 4, k: int = 5, block_len: int = D.BLOCK_TOKENS) -> bool:
    """Batches after resuming at batch k equal those of an uninterrupted loader (real shards)."""
    src, blk, names, meta = D.load_index(index_path)
    part = meta["part"]
    all_shards = _collect(out_dir)
    shards = {n: all_shards[f"{n}/{part}"] for n in names}
    a = D.MixtureLoader(src, blk, names, shards, batch_size=batch_size, block_len=block_len)
    ref = [a.next_batch() for _ in range(k + 3)]
    b = D.MixtureLoader(src, blk, names, shards, batch_size=batch_size, block_len=block_len)
    b.load_state_dict({"position": k * batch_size, "block_len": block_len, "batch_size": batch_size})
    return all(bool((b.next_batch()[0] == ref[j][0]).all()) for j in range(k, k + 3))
