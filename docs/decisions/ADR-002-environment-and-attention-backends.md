# ADR-002: Environment, attention backends and `torch.compile`

Status: Accepted — amended by [ADR-006](ADR-006-toolchain-oct-2026.md): the Python/torch/CUDA versions and pip lock files below are superseded (now Python 3.12, uv, torch 2.14.1+cu130). The SDPA-mandatory / flash-optional / WSL2-preferred policy stands.
Date: 2026-10-06
Spec fields: `environment`, `model.attention_backend`

## Context

- V1 was tested on Windows with Python 3.11.9. On this machine, the project `.venv` is currently an empty Python 3.13 environment, and no Python 3.11 is installed.
- The existing flash-attn wheel (2.8.2) targets cp311 with torch 2.8, so it is not compatible with torch 2.9.
- No Triton is installed on Windows, which is why V1 ran with `compile=False`. Without compile, each op runs as a separate kernel, which is probably a 1.3–2× throughput loss.

## Decision
- **Python 3.11, PyTorch 2.9.0 + CUDA 12.8** for V2, on both Windows and WSL2.
- **Environments:**
  - **Production training and compile:** WSL2 (Ubuntu), where Triton ships with Linux PyTorch and flash-attn builds are standard.
  - **Development:** native Windows is supported in eager mode with SDPA.
- **Attention backends:** `attention_backend: auto | flash | sdpa`.
  - **SDPA is mandatory.** It is always implemented, always tested, and is the default whenever flash-attn is unavailable.
  - **flash-attn is optional.** It is used for the varlen path (packed sequences with `cu_seqlens`) and as a speed path.
  - A parity test checks that flash and SDPA outputs match within tolerance.
- **`torch.compile`:** try it on Windows (triton-windows) and on WSL2. If triton-windows is unreliable, prefer WSL2. Eager mode is always a working fallback.
- **Pinned lock files** for each environment (Windows and WSL2). The environment fingerprint (Python, torch, CUDA, driver, flash-attn, Triton versions) is recorded in every `run.json`.

## Alternatives rejected
- **Python 3.13:** it diverges from V1's tested environment, and the ecosystem (flash-attn wheels) lags behind.
- **Making flash-attn mandatory:** it's fragile on Windows, so the project would depend on a build step.
- **Native Windows only:** we'd lose compile and most of the potential throughput.

## Consequences
- M0 includes setting up both environments and a measured comparison: tok/s, VRAM and stability for eager vs compile and for SDPA vs flash, on each environment. The result is recorded in the experiment registry and the report.
- Anything that depends on flash-attn (e.g. cross-conversation masking in packed SFT) needs an SDPA equivalent, for example padding or an explicit mask.

## Evidence
- Pending M0: bench records `runs/<m0_bench>/bench/*.json`.
