"""Convert an exported V2 folder to GGUF with llama.cpp, and check every file's metadata (EXP-M5-GGUF, gate G0).

    python -m litegpt.export.to_gguf --hf-dir ~/m5_export/LiteGPT-V2-Base --kind base --llama-cpp ~/llama.cpp --out ~/m5_gguf

Uses the pinned llama.cpp (registry: build b12084): its converter from <llama-cpp>/src with benchmarks/gguf/llama_cpp_litegpt.patch applied,
run in its own venv at <llama-cpp>/venv, and its prebuilt binaries in <llama-cpp>/b. The patch adds one lookup entry so the converter reads
V2's own 32k tokenizer as using the qwen2 split regex, which it does character for character. Files: F32 (an exact upcast of the bf16
weights, for checks only), BF16, then Q8_0 and Q4_K_M quantized from F32. Nothing is added to the project venv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from litegpt.export.to_hf import GITHUB, HF_OWNER, KINDS

QUANTS = ("F32", "BF16", "Q8_0", "Q4_K_M")
FILE_TYPE = {"F32": 0, "BF16": 32, "Q8_0": 7, "Q4_K_M": 15}            # general.file_type (llama_ftype)
EXPECTED_ARCH = {"general.architecture": "qwen3", "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen2",
                 "qwen3.block_count": 28, "qwen3.attention.head_count": 12, "qwen3.attention.head_count_kv": 4,
                 "qwen3.embedding_length": 768, "qwen3.context_length": 4096, "qwen3.rope.freq_base": 10000.0,
                 "tokenizer.ggml.bos_token_id": 0, "tokenizer.ggml.add_bos_token": True}


def gguf_name(kind: str, quant: str) -> str:
    return f"{KINDS[kind]['name']}-{quant}.gguf"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def metadata_override(kind: str) -> dict:
    name = KINDS[kind]["name"]
    return {"general.name": name, "general.basename": name, "general.license": "apache-2.0",
            "general.url": f"https://huggingface.co/{HF_OWNER}/{name}-GGUF", "general.repo_url": GITHUB.rsplit("/blob/", 1)[0]}


def expected(kind: str, hf_dir: Path, quant: str) -> dict:
    """What gate G0 requires of one file."""
    eos = KINDS[kind]["eos_ids"][0]
    return {**EXPECTED_ARCH, "tokenizer.ggml.eos_token_id": eos, "general.name": KINDS[kind]["name"], "general.license": "apache-2.0",
            "general.file_type": FILE_TYPE[quant], "tokenizer.chat_template": (hf_dir / "chat_template.jinja").read_text(encoding="utf-8"),
            "token_count": 32768, "tied_embeddings": True}


def check_metadata(meta: dict, want: dict) -> list[str]:
    """Problems found, empty if gate G0 passes. `meta` is the flattened dump from read_gguf."""
    problems = []
    for key, value in want.items():
        got = meta.get(key)
        same = abs(got - value) < 1e-6 if isinstance(value, float) and isinstance(got, (int, float)) else got == value
        if not same:
            shown = (repr(got)[:80], repr(value)[:80])
            problems.append(f"{key}: got {shown[0]}, want {shown[1]}")
    return problems


def read_gguf(path: Path, llama_cpp: Path) -> dict:
    """Metadata and tensor names through gguf-py's own dump script (run in the converter venv), flattened to key -> value."""
    script = llama_cpp / "src" / "gguf-py" / "gguf" / "scripts" / "gguf_dump.py"
    out = subprocess.run([str(llama_cpp / "venv" / "bin" / "python"), str(script), "--json", "--json-array", str(path)],
                         check=True, capture_output=True, text=True).stdout
    dump = json.loads(out)
    meta = {k: v.get("value") for k, v in dump["metadata"].items()}
    tokens = meta.get("tokenizer.ggml.tokens") or []
    meta["token_count"] = len(tokens)
    meta["tied_embeddings"] = "output.weight" not in dump.get("tensors", {})
    meta["tensor_count"] = len(dump.get("tensors", {}))
    meta.pop("tokenizer.ggml.tokens", None)
    meta.pop("tokenizer.ggml.merges", None)
    meta.pop("tokenizer.ggml.token_type", None)
    return meta


def convert(hf_dir: Path, kind: str, llama_cpp: Path, out: Path) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    override = out / f"metadata-{kind}.json"
    override.write_text(json.dumps(metadata_override(kind), indent=1), encoding="utf-8")
    py, src, bins = llama_cpp / "venv" / "bin" / "python", llama_cpp / "src", llama_cpp / "b"
    files = {q: out / gguf_name(kind, q) for q in QUANTS}
    for quant, outtype in (("F32", "f32"), ("BF16", "bf16")):
        subprocess.run([str(py), str(src / "convert_hf_to_gguf.py"), str(hf_dir), "--outtype", outtype, "--outfile", str(files[quant]),
                        "--metadata", str(override)], check=True, cwd=src)
    for quant in ("Q8_0", "Q4_K_M"):
        subprocess.run([str(bins / "llama-quantize"), str(files["F32"]), str(files[quant]), quant], check=True)
    return files


def main(argv=None) -> int:
    from litegpt.records import create_run
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-dir", required=True, help="an exported folder from litegpt.export.to_hf")
    ap.add_argument("--kind", required=True, choices=list(KINDS))
    ap.add_argument("--llama-cpp", required=True, help="folder holding src/ (patched), venv/ and b/ (binaries)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-only", action="store_true", help="skip conversion, check existing files")
    a = ap.parse_args(argv)
    hf_dir, llama_cpp, out = (Path(x).expanduser() for x in (a.hf_dir, a.llama_cpp, a.out))
    files = {q: out / gguf_name(a.kind, q) for q in QUANTS} if a.check_only else convert(hf_dir, a.kind, llama_cpp, out)
    result, passed = {}, True
    for quant, path in files.items():
        meta = read_gguf(path, llama_cpp)
        problems = check_metadata(meta, expected(a.kind, hf_dir, quant))
        passed &= not problems
        result[quant] = {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path), "tensor_count": meta["tensor_count"],
                         "problems": problems}
        print(quant, path.stat().st_size, "OK" if not problems else problems, flush=True)
    rec = create_run("export", f"gguf-{KINDS[a.kind]['name']}", config={"experiment": "EXP-M5-GGUF", "gate": "G0", "kind": a.kind,
                                                                        "hf_dir": a.hf_dir, "llama_cpp": "b12084", "quants": list(QUANTS)})
    rec.write_json("export_gguf.json", {"passed": passed, "files": result, "metadata_override": metadata_override(a.kind)})
    print("G0", "PASS" if passed else "FAIL", "record:", rec.path.name, flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
