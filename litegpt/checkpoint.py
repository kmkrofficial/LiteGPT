"""
Training checkpoints with reliable resume. These are *training* artifacts (model + optimizer + RNG + data
position); they are never distributed. Distribution uses safetensors (see export/).

- atomic writes: written to a temp file in the same directory, fsynced, then os.replace()d into place, so a crash or power
  loss never leaves a half-written checkpoint under a valid name (unlike V1, which also saved only on val improvement).
- keep the last `keep_last` regular checkpoints; milestone checkpoints (e.g. stable_end, decay_end) are never pruned.
- full state: model, optimizer, training counters, data-loader position, python/numpy/torch(+cuda) RNG states.
- guards: spec sha256 and model-config hash must match on resume (the extension gate is only valid for a frozen spec).
- GracefulInterrupt: SIGINT/SIGTERM set a flag so the loop can save before exiting; a second SIGINT force-quits.

Checkpoints are pickled with torch.save (they contain numpy RNG state): only load files you created yourself.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import signal
from pathlib import Path

import numpy as np
import torch

CHECKPOINT_FORMAT = 1
_REGULAR = re.compile(r"^ckpt-(\d{8})\.pt$")


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])


class CheckpointError(RuntimeError):
    pass


class CheckpointManager:
    def __init__(self, directory, keep_last: int = 3, spec_sha256: str | None = None):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.spec_sha256 = spec_sha256
        for tmp in self.dir.glob("*.tmp"):  # leftovers of an interrupted save are never valid checkpoints
            tmp.unlink(missing_ok=True)

    # ----- paths ---------------------------------------------------------------------------------------------------
    def regular_checkpoints(self) -> list[tuple[int, Path]]:
        found = []
        for p in self.dir.iterdir():
            m = _REGULAR.match(p.name)
            if m:
                found.append((int(m.group(1)), p))
        return sorted(found)

    def milestone_path(self, name: str) -> Path:
        return self.dir / f"milestone-{name}.pt"

    def latest(self) -> Path | None:
        ckpts = self.regular_checkpoints()
        return ckpts[-1][1] if ckpts else None

    # ----- save / load ---------------------------------------------------------------------------------------------
    def save(self, *, model, optimizer, step: int, tokens: int, model_config: dict, data_position: dict | None = None,
             extra: dict | None = None, milestone: str | None = None) -> Path:
        raw = getattr(model, "_orig_mod", model)  # unwrap torch.compile
        state = {
            "format": CHECKPOINT_FORMAT,
            "model": raw.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "step": step,
            "tokens": tokens,
            "model_config": model_config,
            "config_hash": config_hash(model_config),
            "spec_sha256": self.spec_sha256,
            "data_position": data_position,
            "rng": capture_rng(),
            "extra": extra or {},
        }
        dest = self.milestone_path(milestone) if milestone else self.dir / f"ckpt-{step:08d}.pt"
        self._atomic_save(state, dest)
        if not milestone:
            for _, old in self.regular_checkpoints()[:-self.keep_last]:
                old.unlink(missing_ok=True)
        return dest

    @staticmethod
    def _atomic_save(state: dict, dest: Path) -> None:
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        with open(tmp, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)

    def load(self, path: Path | None = None, *, model=None, optimizer=None, expected_config: dict | None = None,
             map_location="cpu", restore_rng_state: bool = True, allow_spec_mismatch: bool = False) -> dict:
        """Load a checkpoint (default: the latest regular one). Restores model/optimizer/RNG when objects are given and
        returns the full state dict (step, tokens, data_position, extra, ...)."""
        path = Path(path) if path else self.latest()
        if path is None or not path.exists():
            raise CheckpointError(f"no checkpoint found in {self.dir}")
        state = torch.load(path, map_location=map_location, weights_only=False)
        if state.get("format") != CHECKPOINT_FORMAT:
            raise CheckpointError(f"unsupported checkpoint format {state.get('format')}")
        if expected_config is not None and state["config_hash"] != config_hash(expected_config):
            raise CheckpointError("model config differs from the checkpoint's config (refusing to resume)")
        if (self.spec_sha256 and state.get("spec_sha256") and state["spec_sha256"] != self.spec_sha256
                and not allow_spec_mismatch):
            raise CheckpointError("spec sha256 differs from the checkpoint's; the pre-registered spec is frozen "
                                  "(use allow_spec_mismatch=True only as a recorded protocol deviation)")
        if model is not None:
            raw = getattr(model, "_orig_mod", model)
            sd = {k.removeprefix("_orig_mod."): v for k, v in state["model"].items()}
            raw.load_state_dict(sd)
        if optimizer is not None and state["optimizer"] is not None:
            optimizer.load_state_dict(state["optimizer"])
        if restore_rng_state and state.get("rng"):
            restore_rng(state["rng"])
        return state


class GracefulInterrupt:
    """Context manager: SIGINT/SIGTERM set `.requested`; the training loop checks it and saves before exiting.
    A second SIGINT raises KeyboardInterrupt immediately."""

    def __init__(self):
        self.requested = False
        self._old = {}

    def _handler(self, signum, frame):
        if self.requested and signum == signal.SIGINT:
            raise KeyboardInterrupt
        self.requested = True

    def __enter__(self):
        for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
            if sig is not None:
                try:
                    self._old[sig] = signal.signal(sig, self._handler)
                except ValueError:  # not in main thread
                    pass
        return self

    def __exit__(self, *exc):
        for sig, old in self._old.items():
            signal.signal(sig, old)
        return False
