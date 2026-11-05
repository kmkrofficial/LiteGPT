"""Download every Hugging Face arm of EXP-M4-T2D at its pinned revision: weights, config and tokenizer files only, never repository code.

    python scripts/cloud/m4_fetch_models.py

Gemma is gated: this uses the access the owner already has, through the token the owner enters on the server. It accepts no terms.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ALLOW = ["*.safetensors", "*.safetensors.index.json", "config.json", "generation_config.json", "tokenizer*", "special_tokens_map.json", "vocab.json", "merges.txt", "*.jinja", "*.model"]


def main() -> int:
    from huggingface_hub import snapshot_download
    arms = yaml.safe_load((ROOT / "benchmarks" / "m4" / "t2d.yaml").read_text(encoding="utf-8"))["arms"]
    for name, arm in arms.items():
        if arm["kind"] == "hf":
            print(name, snapshot_download(arm["repo"], revision=arm["revision"], allow_patterns=ALLOW), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
