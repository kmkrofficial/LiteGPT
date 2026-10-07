# Reproducing the V2 environment

Pinned in [`pyproject.toml`](../pyproject.toml) and [`uv.lock`](../uv.lock) ([ADR-006](decisions/ADR-006-toolchain-oct-2026.md)):
Python 3.12, torch 2.14.1+cu130 and the rest of the stack. `uv sync --locked` is the only supported install path.

## Windows (development, eager mode)

```powershell
winget install --id astral-sh.uv        # uv 0.12.23 is pinned via [tool.uv] required-version
uv python install 3.12
uv sync --locked
uv run pytest
```

Optional flash-attn fast path (Windows, community prebuilt wheel pinned in `uv.lock`; SDPA is used without it):

```powershell
uv sync --locked --group flash
```

GPU runs need an NVIDIA driver supporting CUDA 13.x (verified with driver 610.88). CPU-only runs (the test suite) need no GPU.
`torch.compile` needs Triton, which has no official Windows build; compile is an experiment (see the benchmark records).

## WSL2 (preferred for production training / compile)

Verified on Ubuntu 26.04 under WSL2 (kernel 6.18, RTX 4090 Laptop visible with the Windows driver, no Linux driver needed): the
same `uv.lock` reproduces the environment (Python 3.12.15, torch 2.14.1+cu130, CUDA 13.0, Triton 3.8.0) and the test suite passes.

```bash
curl -LsSf https://astral.sh/uv/0.12.23/install.sh | sh        # pinned uv version
sudo apt-get update && sudo apt-get install -y build-essential   # a C compiler is REQUIRED by Triton / torch.compile
git clone <repo> ~/LiteGPT && cd ~/LiteGPT                       # keep repo and data on the Linux filesystem (ext4), not /mnt/d
uv python install 3.12
uv sync --locked                                                 # same uv.lock as Windows
uv run python -m benchmarks.train_mem sweep --preset m0 --compile
```

Notes:
- `torch.compile` fails without a C compiler (`Failed to find C compiler`): Triton builds a small CUDA driver stub on first use.
- The Windows flash-attn wheel does not apply on Linux; on WSL2 the model uses SDPA unless a Linux flash-attn wheel is installed.
- WSL2 exposes about half of the machine's RAM by default; raise it in `%UserProfile%\.wslconfig` if data preparation needs more.

## Verify

```bash
uv run pytest -q
uv run python -m benchmarks.train_mem single --config litegpt-v2-28l --micro-batch 2 --steps 10
uv run python -m scripts.build_report --check
```
