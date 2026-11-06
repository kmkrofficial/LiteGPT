"""Prepare the M4 chat SFT data from pinned local SmolTalk files (EXP-M4-SFT): the M1 reserved split plus smol-magpie-ultra.

    python scripts/prepare_sft.py --smoltalk <dir with data/<subset>/*.parquet> --decontam-index <index.npz> --out <dir> [--workers 14]

Writes train and heldout splits, and a manifest with every drop count and hash. No model is involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_NGRAMS = None


def _init(index_path) -> None:
    global _NGRAMS
    from litegpt.decontam import NgramIndex
    _NGRAMS = NgramIndex.load(index_path)


def _work(args):
    from litegpt import sft
    path, subset, salt, frac = args
    return sft.process_file(path, subset, _NGRAMS, salt, frac)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    from litegpt import sft
    from litegpt.dataprep import load_sources
    from litegpt.records import create_run
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoltalk", required=True)
    ap.add_argument("--decontam-index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    sources = load_sources()
    src, salt = sources["sources"], sources["partitioning"]["salt"]
    frac = src["instruction_style"]["partition"]["decay_fraction_of_train"]
    if src["instruction_style"]["revision"] != src["smol_magpie_ultra"]["revision"]:
        raise SystemExit("the two SmolTalk sources differ in revision")
    dest = Path(a.out)
    dest.mkdir(parents=True, exist_ok=True)
    jobs = []
    for sub in sft.ALL_SUBSETS:
        files = sorted((Path(a.smoltalk) / "data" / sub).glob("train-*.parquet"))
        if not files:
            raise SystemExit(f"no train files for {sub} under {a.smoltalk}")
        jobs += [(str(f), sub, salt, frac) for f in files]
    config = {"args": vars(a), "revision": src["instruction_style"]["revision"], "decontam_index_sha256": sha256(a.decontam_index), "max_tokens": sft.MAX_TOKENS,
              "heldout_fraction": sft.HELDOUT_FRACTION, "salt": sft.SALT, "files": len(jobs)}
    rec = create_run("data", "sft-prepare", config=config)
    train, heldout, per_file = [], [], []
    with mp.Pool(a.workers, initializer=_init, initargs=(a.decontam_index,)) as pool:
        for i, r in enumerate(pool.imap(_work, jobs), 1):       # imap keeps the file order, so the output is deterministic
            train += r.pop("train")
            heldout += r.pop("heldout")
            per_file.append(r)
            rec.log("metrics", files=i, of=len(jobs), **{k: r["counts"][k] for k in ("read", "kept", "tokens")})
            print(i, "of", len(jobs), r["subset"], r["file"], r["counts"], flush=True)
    per_subset: dict = {}
    for r in per_file:
        s = per_subset.setdefault(r["subset"], {"counts": {}, "decontam": {}})
        for k, v in r["counts"].items():
            s["counts"][k] = s["counts"].get(k, 0) + v
        for k, v in r["decontam"].items():
            s["decontam"][k] = s["decontam"].get(k, 0) + v
    manifest = {"splits": {"train": sft.write_split(dest, "train", train), "heldout": sft.write_split(dest, "heldout", heldout)}, "subsets": per_subset,
                "row": "uint16: id in the low 15 bits, loss mask in the top bit", **config}
    manifest["sha256"] = {p.name: sha256(p) for p in sorted(dest.iterdir()) if p.suffix in (".bin", ".npy")}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    rec.write_json("sft_manifest.json", manifest)
    print(json.dumps(manifest["splits"]), json.dumps(per_subset, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
