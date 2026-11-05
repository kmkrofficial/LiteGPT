"""Load and hash the machine-readable V2 specification (docs/spec/v2.yaml) and validate configs against it."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from .config import ModelConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = REPO_ROOT / "docs" / "spec" / "v2.yaml"


def spec_sha256(path: Path = SPEC_PATH) -> str:
    """sha256 of the spec file with line endings normalised (so Windows/Linux checkouts hash identically)."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=None)
def load_spec(path: Path = SPEC_PATH) -> dict:
    import yaml  # lazy: only needed by tooling, not by inference
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def spec_version(path: Path = SPEC_PATH) -> str:
    return load_spec(path)["spec_version"].split()[0]


# spec `model` field -> ModelConfig attribute
_MODEL_FIELDS = {
    "n_layer": "n_layer", "n_embd": "n_embd", "n_head": "n_head", "n_kv_head": "n_kv_head",
    "ffn_hidden": "ffn_hidden", "norm_eps": "norm_eps", "rope_theta": "rope_theta", "vocab_size": "vocab_size",
    "tie_embeddings": "tie_embeddings", "qk_norm": "qk_norm", "head_dim": "head_dim",
}


def validate_model_config(cfg: ModelConfig, spec: dict | None = None) -> list[str]:
    """Return a list of deviations between `cfg` and the spec's model section (empty list = conforms).
    rope_theta may equal the context extension's value; since ADR-009 that is the same theta (training.pretraining.context_extension)."""
    spec = spec or load_spec()
    m = spec["model"]
    problems = []
    for key, attr in _MODEL_FIELDS.items():
        if key in m and getattr(cfg, attr) != m[key]:
            if key == "rope_theta" and cfg.rope_theta == spec["training"]["pretraining"]["context_extension"]["rope_theta"]:
                continue
            problems.append(f"{key}: config={getattr(cfg, attr)!r} spec={m[key]!r}")
    return problems
