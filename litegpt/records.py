"""
Run records: the single source of truth for everything reported about LiteGPT V2 (ADR-006, plan section 13).

Every train / eval / bench / data-prep / export run creates a directory `runs/<run_id>/` containing `run.json`
(environment fingerprint, spec version + sha256, git state, config) plus JSONL metric streams and result files.
Reports are generated from these records; nothing is typed by hand. A record that is missing a required field is invalid.

Stdlib only, except that torch is imported lazily to read CUDA/cuDNN/GPU information when available.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .spec import REPO_ROOT, spec_sha256, spec_version

RECORD_SCHEMA_VERSION = 1
RUNS_ROOT = REPO_ROOT / "runs"

# Fields required in run.json["environment"]. Keys must always be present; values may be null only where noted.
ENV_REQUIRED = [
    "python", "uv_version", "uv_lock_sha256", "resolved_package_versions", "torch_version", "torch_cuda_version",
    "cudnn_version", "nvidia_driver_version", "gpu_name", "gpu_vram_gib", "os", "os_version",
    "wsl_distro_and_kernel", "flash_attn_version_or_none", "triton_version_or_none", "cuda_available",
]
ENV_NULLABLE_ALWAYS = {"wsl_distro_and_kernel", "flash_attn_version_or_none", "triton_version_or_none"}
ENV_NULLABLE_WITHOUT_GPU = {"torch_cuda_version", "cudnn_version", "nvidia_driver_version", "gpu_name", "gpu_vram_gib"}
RUN_REQUIRED = ["schema_version", "run_id", "kind", "name", "created_utc", "spec_version", "spec_sha256", "git_commit",
                "git_dirty", "attention_backend", "compile_enabled", "config", "environment"]
KINDS = {"train", "eval", "bench", "data", "export", "test"}


def _run(cmd: list[str], timeout: float = 20.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=REPO_ROOT)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def file_sha256(path: Path) -> str | None:
    p = Path(path)
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() if p.exists() else None


def git_state() -> tuple[str | None, bool | None]:
    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return commit, (None if status is None else bool(status))


def _dist_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def _wsl_info() -> str | None:
    if sys.platform != "linux":
        return None
    try:
        text = Path("/proc/version").read_text()
    except OSError:
        return None
    if "microsoft" not in text.lower():
        return None
    return f"{os.environ.get('WSL_DISTRO_NAME', 'unknown')} / {platform.release()}"


def collect_environment() -> dict:
    """Environment fingerprint required by ADR-006: exact versions, CUDA/driver, OS/WSL and the uv.lock hash."""
    uv_out = _run(["uv", "--version"])
    env = {
        "python": platform.python_version(),
        "uv_version": uv_out.split()[1] if uv_out else None,
        "uv_lock_sha256": file_sha256(REPO_ROOT / "uv.lock"),
        "resolved_package_versions": dict(sorted(
            {d.metadata["Name"]: d.version for d in md.distributions() if d.metadata["Name"]}.items(),
            key=lambda kv: kv[0].lower())),
        "os": platform.system(),
        "os_version": platform.version() if platform.system() == "Windows" else platform.platform(),
        "wsl_distro_and_kernel": _wsl_info(),
        "flash_attn_version_or_none": _dist_version("flash_attn") or _dist_version("flash-attn"),
        "triton_version_or_none": _dist_version("triton") or _dist_version("triton-windows"),
        "torch_version": None, "torch_cuda_version": None, "cudnn_version": None, "nvidia_driver_version": None,
        "gpu_name": None, "gpu_vram_gib": None, "cuda_available": False,
    }
    try:
        import torch
        env["torch_version"] = torch.__version__
        env["torch_cuda_version"] = torch.version.cuda
        env["cuda_available"] = bool(torch.cuda.is_available())
        if env["cuda_available"]:
            env["cudnn_version"] = str(torch.backends.cudnn.version())
            props = torch.cuda.get_device_properties(0)
            env["gpu_name"] = props.name
            env["gpu_vram_gib"] = round(props.total_memory / 2**30, 2)
            drv = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
            env["nvidia_driver_version"] = drv.splitlines()[0].strip() if drv else None
    except ImportError:
        pass
    return env


def validate_run_record(rec: dict) -> list[str]:
    """Return a list of problems (empty = valid). A missing field is a problem; so are unexpected nulls."""
    problems = [f"missing field: {k}" for k in RUN_REQUIRED if k not in rec]
    if problems:
        return problems
    if rec["schema_version"] != RECORD_SCHEMA_VERSION:
        problems.append(f"schema_version {rec['schema_version']} != {RECORD_SCHEMA_VERSION}")
    if rec["kind"] not in KINDS:
        problems.append(f"unknown kind {rec['kind']!r}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(rec["spec_sha256"])):
        problems.append("spec_sha256 is not a sha256 hex digest")
    if not isinstance(rec["git_dirty"], bool) or not rec["git_commit"]:
        problems.append("git_commit/git_dirty missing (run inside the git repository)")
    if not isinstance(rec["compile_enabled"], bool):
        problems.append("compile_enabled must be a bool")
    if rec["attention_backend"] not in ("sdpa", "flash"):
        problems.append("attention_backend must be the resolved backend: 'sdpa' or 'flash'")
    env = rec["environment"]
    for k in ENV_REQUIRED:
        if k not in env:
            problems.append(f"environment missing field: {k}")
        elif env[k] is None:
            allowed = ENV_NULLABLE_ALWAYS | (ENV_NULLABLE_WITHOUT_GPU if env.get("cuda_available") is False else set())
            if k not in allowed:
                problems.append(f"environment field {k} is null")
    if not isinstance(env.get("uv_lock_sha256", ""), str):
        problems.append("uv_lock_sha256 must be a string (is uv.lock committed?)")
    if not env.get("resolved_package_versions"):
        problems.append("resolved_package_versions is empty")
    return problems


class RunRecord:
    """Handle to a run directory. Metrics are appended as JSON lines; results are written atomically."""

    def __init__(self, path: Path, run: dict):
        self.path = Path(path)
        self.run = run

    @property
    def run_id(self) -> str:
        return self.run["run_id"]

    def log(self, stream: str, **fields) -> None:
        """Append one JSON line to <run>/<stream>.jsonl (e.g. stream='metrics' or 'telemetry')."""
        with open(self.path / f"{stream}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), **fields}, default=_json_default) + "\n")

    def write_json(self, relpath: str, obj) -> Path:
        """Atomically write <run>/<relpath> (e.g. 'bench/train_mem.json', 'gate/extension_gate.json')."""
        dest = self.path / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
        os.replace(tmp, dest)
        return dest

    def update(self, **fields) -> None:
        """Merge fields (e.g. status/result summary) into run.json."""
        self.run.update(fields)
        self.write_json("run.json", self.run)


def _json_default(o):
    try:
        import torch
        if isinstance(o, torch.Tensor):
            return o.item() if o.numel() == 1 else o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def create_run(kind: str, name: str, config: dict | None = None, attention_backend: str = "sdpa",
               compile_enabled: bool = False, root: Path | None = None, extra: dict | None = None) -> RunRecord:
    """Create runs/<run_id>/ and write a validated run.json (raises if the record would be invalid)."""
    assert kind in KINDS, f"kind must be one of {sorted(KINDS)}"
    commit, dirty = git_state()
    now = datetime.now(timezone.utc)
    run = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "run_id": f"{now.strftime('%Y%m%dT%H%M%SZ')}-{kind}-{re.sub(r'[^A-Za-z0-9_.-]+', '-', name)}-{(commit or 'nogit')[:7]}",
        "kind": kind,
        "name": name,
        "created_utc": now.isoformat(),
        "spec_version": spec_version(),
        "spec_sha256": spec_sha256(),
        "git_commit": commit,
        "git_dirty": dirty,
        "attention_backend": attention_backend,
        "compile_enabled": compile_enabled,
        "config": config or {},
        "environment": collect_environment(),
        **(extra or {}),
    }
    problems = validate_run_record(run)
    if problems:
        raise ValueError("invalid run record:\n  " + "\n  ".join(problems))
    path = Path(root or RUNS_ROOT) / run["run_id"]
    path.mkdir(parents=True, exist_ok=False)
    rec = RunRecord(path, run)
    rec.write_json("run.json", run)
    return rec


def load_run(path: Path) -> dict:
    return json.loads((Path(path) / "run.json").read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
