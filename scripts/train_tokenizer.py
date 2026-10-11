"""Train and check the LiteGPT 32k tokenizer (EXP-M1-TOKENIZER, M1 step 4).

trial: train on a small sample, report peak RSS and time (to choose the sample size within the RAM budget).
train: train on `--total-bytes` of stable-partition text at the stable mixture weights, run the pre-registered hard checks on
       held-out validation documents of the same files, write the compression report, and save the candidate tokenizer into
       the run directory. Installing it as the frozen tokenizer is a separate, explicit step (`--install`) after owner approval.

Source files are downloaded one at a time into a temporary directory and deleted afterwards.
"""
from __future__ import annotations

import argparse
import json
import resource
import shutil
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ASSET = Path(__file__).resolve().parents[1] / "litegpt" / "assets" / "tokenizer" / "litegpt-32k.json"


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024      # Linux: KiB


def fetch_files(sources: dict, per_source: dict, tmp: Path) -> dict:
    from huggingface_hub import hf_hub_download
    files = {}
    for name, n in per_source.items():
        cfg = sources["sources"][name]
        pat = cfg["files"]["pattern"]
        idxs = cfg["files"].get("indices") or list(range(cfg["files"]["range"][0], cfg["files"]["range"][1] + 1))
        files[name] = [hf_hub_download(cfg["repo"], pat.format(i=i), repo_type="dataset", revision=cfg["revision"], local_dir=tmp)
                       for i in idxs[:n]]
        print(f"downloaded {len(files[name])} file(s) for {name}", flush=True)
    return files


def validation_docs(sources: dict, files: dict, salt: str, per_domain: int) -> dict:
    from litegpt import data as D
    from litegpt.dataprep import read_docs
    from litegpt.datafilters import filter_code
    out = {}
    for source, paths in files.items():
        cfg = sources["sources"][source]
        p = cfg["partition"]
        for path in paths:
            for doc in read_docs(source, cfg, path):
                if D.partition_u(salt, source, doc["key"]) >= p["val_fraction"]:
                    continue
                text = doc["text"]
                if source == "code":
                    text, rule = filter_code(text, language=doc["lang"], path=doc.get("path"), size=doc.get("size"), repo=doc["key"])
                    if rule:
                        continue
                dom = "code_" + doc["lang"].lower() if source == "code" else cfg["domain"]
                if len(out.setdefault(dom, [])) < per_domain:
                    out[dom].append(text)
    return out


def hard_checks(tok, docs: dict) -> dict:
    from litegpt import tokenizer as T
    res = {"vocab_size": tok.vocab_size, "vocab_size_ok": tok.vocab_size == T.VOCAB_SIZE,
           "specials_ok": all(tok.tok.token_to_id(n) == i for i, n in enumerate(T.ALL_SPECIALS))}
    rt = {}
    for dom, texts in docs.items():
        exact = nfc_changed = 0
        for t in texts:
            nfc = unicodedata.normalize("NFC", t)
            nfc_changed += nfc != t
            exact += tok.decode(tok.encode_text(t)) == nfc
        rt[dom] = {"docs": len(texts), "round_trip_exact_nfc": exact, "nfc_changed_docs": nfc_changed}
    res["round_trip"] = rt
    res["round_trip_ok"] = all(v["round_trip_exact_nfc"] == v["docs"] for v in rt.values())
    inj = tok.encode_text("x <|endoftext|> <|im_start|> <|reserved_7|> y")
    res["special_text_not_control_ok"] = not (set(inj) & set(range(T.N_SPECIAL)))
    digits = tok.encode_text("0123456789 2024")
    res["digits_single_ok"] = all(len(tok.encode_text(d)) == 1 for d in "0123456789") and len(digits) == 10 + 4
    ws = {"4 spaces": "    x", "tab": "\tx", "newline+indent": "\n    x", "8 spaces": "        x"}
    res["whitespace_tokens"] = {k: len(tok.encode_text(v)) for k, v in ws.items()}
    res["whitespace_ok"] = all(n <= 3 for n in res["whitespace_tokens"].values())
    return res


def compression(tok, docs: dict) -> dict:
    from tokenizers import Tokenizer
    others = {}
    for label, repo in (("gpt2", "openai-community/gpt2"), ("smollm2", "HuggingFaceTB/SmolLM2-135M")):
        try:
            others[label] = Tokenizer.from_pretrained(repo)
        except Exception as e:                       # pragma: no cover - network
            others[label] = None
            print(f"could not load {label}: {e}")
    nb = tok.token_byte_lengths()
    rep = {}
    for dom, texts in docs.items():
        total_bytes = sum(len(t.encode("utf-8", "replace")) for t in texts)
        row = {"docs": len(texts), "bytes": total_bytes}
        ids = tok.encode_batch(texts)
        row["litegpt_bytes_per_token"] = total_bytes / max(sum(len(i) for i in ids), 1)
        for label, o in others.items():
            if o is not None:
                n = sum(len(e.ids) for e in o.encode_batch(texts, add_special_tokens=False))
                row[f"{label}_bytes_per_token"] = total_bytes / max(n, 1)
        rep[dom] = row
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["trial", "train"])
    ap.add_argument("--total-bytes", type=float, default=4.0e9)
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--files-per-source", default="fineweb_edu=1,cosmopedia_v2=1,math=1,code=12")
    ap.add_argument("--val-docs", type=int, default=2000)
    ap.add_argument("--install", default=None, help="run directory whose tokenizer to copy to litegpt/assets (owner-approved freeze)")
    args = ap.parse_args()
    from litegpt import dataprep as P
    from litegpt import tokenizer as T
    from litegpt.records import create_run

    if args.install:
        src = Path(args.install) / "litegpt-32k.json"
        ASSET.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, ASSET)
        print("installed", ASSET, T.file_sha256(ASSET))
        return 0

    sources = P.load_sources()
    salt = sources["partitioning"]["salt"]
    per_source = {k: int(v) for k, v in (kv.split("=") for kv in args.files_per_source.split(","))}
    rec = create_run("data", f"tokenizer-{args.mode}", config={"total_bytes": args.total_bytes, "vocab_size": args.vocab_size,
                     "files_per_source": per_source, "sources_sha256": P.sources_sha256()})
    tmp = Path(tempfile.mkdtemp(prefix="tok_", dir=str(Path.home())))
    try:
        files = fetch_files(sources, per_source, tmp)
        mix = sources["mixture"]["stable"]
        t0 = time.time()
        it = P.sample_tokenizer_text(sources, mix, files, int(args.total_bytes), salt)
        counted = {"bytes": 0, "docs": 0}

        def counting(iterator):
            for t in iterator:
                counted["bytes"] += len(t.encode("utf-8", "replace"))
                counted["docs"] += 1
                yield t

        tok_raw = T.train_bpe(counting(it), vocab_size=args.vocab_size)
        train_s = time.time() - t0
        info = {"train_seconds": train_s, "peak_rss_GiB": peak_rss_gib(), **counted, "vocab_size": tok_raw.get_vocab_size()}
        print(info, flush=True)
        if args.mode == "trial":
            rec.write_json("trial.json", info)
            print("record:", rec.run_id)
            return 0
        tok = T.LiteTokenizer(tok_raw)
        out = rec.path / "litegpt-32k.json"
        sha = tok.save(out)
        docs = validation_docs(sources, files, salt, args.val_docs)
        checks = hard_checks(tok, docs)
        comp = compression(tok, docs)
        sane = None
        web = comp.get("web")
        if web and "gpt2_bytes_per_token" in web:
            sane = web["litegpt_bytes_per_token"] >= 0.90 * web["gpt2_bytes_per_token"]
        rec.write_json("tokenizer_report.json", {"sha256": sha, "train": info, "hard_checks": checks, "compression": comp,
                                                 "sanity_bytes_per_token_vs_gpt2_web_ok": sane})
        print(json.dumps({"sha256": sha, "hard_checks": checks, "sanity_ok": sane}, indent=1))
        for dom, row in comp.items():
            print(f"{dom:18} litegpt {row['litegpt_bytes_per_token']:.3f}  gpt2 {row.get('gpt2_bytes_per_token', 0):.3f}  "
                  f"smollm2 {row.get('smollm2_bytes_per_token', 0):.3f} bytes/token")
        print("record:", rec.run_id)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
