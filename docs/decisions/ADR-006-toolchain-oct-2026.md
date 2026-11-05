# ADR-006: Toolchain: Python 3.12, uv, pinned stable stack (October 2026)

Status: Accepted
Date: 2026-10-06
Amends: [ADR-000](ADR-000-v2-scope-decisions.md) D2 and [ADR-002](ADR-002-environment-and-attention-backends.md). Python 3.11 and torch 2.9 + cu128 are superseded; the backend and compile policy is kept and tightened.
Spec fields: `environment` (spec `2.0.0-spec.2`)

## Context

The owner asked for the latest stable, mutually compatible ecosystem as of October 2026, managed with uv. Every requested version was checked against PyPI and download.pytorch.org on 2026-10-06:

| Package | Pinned | Latest stable on PyPI | Note |
|---|---|---|---|
| Python | 3.12 | — | NumPy 2.5.3 requires ≥3.12 |
| uv | 0.12.23 | 0.12.23 | — |
| torch | **2.14.1+cu130** | 2.14.1 | 2.14.1 is built **only for cu126 and cu130**; there's no cu128 build. Win + Linux cp312 wheels exist for both |
| transformers | 5.18.0 | 5.18.0 | requires tokenizers ≥0.23.1,<0.24, safetensors ≥0.8.0, huggingface-hub ≥1.31,<3 |
| tokenizers | 0.23.2 | 0.23.2 | — |
| safetensors | 0.8.0 | 0.8.0 | — |
| numpy | 2.5.3 | 2.5.3 | — |
| datasets | 5.1.0 | 5.1.0 | requires huggingface-hub ≥1.31,<3 and pyarrow ≥24 (compatible with transformers) |
| trl | 1.14.1 | 1.14.1 | requires transformers ≥4.56.2, datasets ≥4.7, accelerate ≥1.4 |
| peft | 0.21.2 | 0.21.2 | — |
| accelerate | 1.15.0 | 1.15.0 | — |
| lm-eval | 0.4.13 | 0.4.13 | requires datasets ≥2.16 |
| matplotlib | 3.11.2 | 3.11.2 | — |
| flash-attn | 2.8.3.post1 (optional) | 2.8.3.post1 | **PyPI has only a source distribution**, so it needs a local CUDA build |

- **Host:** NVIDIA driver 610.88, which supports CUDA 13.3, so the cu130 build of PyTorch is the right choice. The RTX 4090 Laptop (Ada, sm_89) is supported by CUDA 13.
- **Final check:** pairwise constraints are consistent on paper. `uv lock` is the final arbiter: any conflict is reported back to the owner, never silently downgraded.

## Decision

- **Python 3.12**, installed and managed by uv (`uv python install 3.12`). No system Python dependency.
- **uv + `pyproject.toml` + `uv.lock`.**
  - One universal lock covers Windows and WSL2/Linux through platform markers.
  - `[tool.uv] required-version` pins the uv version.
  - torch comes from an explicit `pytorch-cu130` index via `[tool.uv.sources]`.
  - `uv sync --locked` is the only supported install path.
- **Dependency groups** keep the core small:
  - `core`: torch, tokenizers, safetensors, numpy.
  - `train`: datasets, matplotlib.
  - `eval`: lm-eval, transformers, datasets.
  - `hf`: transformers.
  - `hf-reference`: trl, peft, accelerate.
- **TRL, PEFT and Accelerate are optional reference tools.** Example use: cross-checking our DPO loss against TRL on the HF-exported model. The V2 architecture and training loop remain our own implementation and don't depend on them.
- **Attention:**
  - PyTorch **SDPA is mandatory**: the default, always tested, always shipped.
  - **flash-attn 2.8.3.post1 is optional and experimental.** It is built from source in WSL2 first. It is adopted only if (a) the flash-vs-SDPA parity tests pass and (b) the M0 benchmark shows a measured speed or memory benefit.
  - **Python and torch are never downgraded to obtain it.** If it fails to build, the result is recorded and SDPA is used.
- **`torch.compile` is an optimization experiment**, measured in M0 on Windows and WSL2. WSL2 is preferred if the Windows toolchain is problematic; eager mode always remains a working fallback.
- **No pre-release packages.** "Latest" means the latest stable release compatible with the stack. Any upgrade is a new ADR plus a lock update, recorded in `CHANGELOG.md`.
- **Every run record must include:**
  - Python version, uv version, **uv.lock sha256**, and all resolved package versions (from `importlib.metadata`).
  - torch version, CUDA runtime (`torch.version.cuda`), cuDNN version, NVIDIA driver version.
  - GPU name and VRAM.
  - OS and version, plus WSL distro and kernel when applicable.
  - flash-attn and Triton versions (or none), compile flag, attention backend.
  - git commit and dirty flag.

## Alternatives rejected
- **Python 3.11:** NumPy 2.5.3 requires Python ≥3.12.
- **torch 2.14.1+cu128:** that build doesn't exist.
- **torch +cu126:** works, but it's older CUDA with no benefit, since the driver supports CUDA 13.3.
- **pip + requirements lock files:** weaker reproducibility than `uv.lock`, and no single cross-platform lock.
- **Making flash-attn mandatory, or downgrading the stack for prebuilt flash-attn wheels:** explicitly rejected by the owner.
- **Building the training loop on TRL/Accelerate:** it would couple the custom architecture to HF trainer abstractions, and Accelerate isn't needed on a single GPU.

## Consequences
- M0.0 becomes:
  1. Install uv.
  2. `uv python install 3.12`.
  3. Write `pyproject.toml`.
  4. `uv lock`, then `uv sync --locked`, on Windows and WSL2.
  5. Run the environment fingerprint smoke test.
- The empty Python 3.13 `.venv` is replaced by a uv-managed Python 3.12 `.venv`.
- The flash-attn and compile results become recorded experiments, with an evidence update to this ADR.

## Evidence
- Version verification: the table above, from PyPI JSON and download.pytorch.org on 2026-10-06.
- **Windows native (2026-10-06):** `uv lock` resolved 118 packages with no conflicts; every pin resolved exactly as specified
  (torch `2.14.1+cu130` from the cu130 index). `uv sync --locked` succeeded. Smoke test: Python 3.12.15, CUDA 13.0, cuDNN 9.24,
  RTX 4090 Laptop (15.99 GiB, sm_89), driver 610.88; SDPA with GQA and bf16 matmul run on the GPU. `uv.lock` sha256 is
  recorded in every run record.
- **torch.compile: adopted for production training on WSL2 (EXP-M0-COMPILE, spec 2.0.0-spec.5); fails on Windows.**
  - Windows: `TritonMissing` (no Triton) in every benchmark record, so Windows runs eager.
  - WSL2 (Ubuntu 26.04, kernel 6.18): the same `uv.lock` reproduces the environment (Python 3.12.15, torch 2.14.1+cu130,
    CUDA 13.0, Triton 3.8.0); the suite passes (62 passed, 3 flash tests skipped). Compile additionally needs a C compiler
    (`build-essential`; without it: "Failed to find C compiler").
  - Pre-registered rule (written before measuring): output equivalence + >= +5% tok/s in alternating A/B within the VRAM cap
    + warmup <= 10 min. **All met:** equivalence on 28L (run `20261006T151400Z-bench-compile-equivalence-litegpt-v2-28l-101c3f8`): loss diff 0.0004, logits rel L2 0.88%, grad-norm
    rel diff 0.003%. A/B (run `20261006T143350Z-bench-ab-sdpa-vs-sdpa-compile-101c3f8`, 28L mb3x2048 SDPA, 4 rounds): eager 18,561 vs compiled 22,997 tok/s
    (**+23.9%**, every compiled round beat every eager round); peak VRAM 11.17 -> 8.28 GiB; warmup 4.1 -> 13.1 s.
    (The 2.6% final-loss gap in that run is training drift after ~70 steps on one repeated synthetic batch, which is why
    equivalence was checked directly with identical weights.)
  - Compiled micro-batch exploration: 4x2048 23,695 tok/s (10.14 GiB, best), 5x2048 23,441 (12.04), 6x2048 OOM, 2x4096 20,792 (10.14).
  - Sustained 30 min, 28L mb4x2048 compiled (run `20261006T151250Z-bench-sustained-28l-mb4-compile-wsl-30min-101c3f8`): mean **22,592 tok/s**, first minute 22,850 / last 22,327
    (0.977), power-capped 100% / thermally throttled 16%, max 76 C, peak VRAM 10.14 GiB.
  - **flash-attn on WSL2 (EXP-M0-FLASH-LINUX):** the Linux wheel (`flash_attn-2.8.3+cu130torch2.14-cp312-cp312-linux_x86_64.whl`,
    mjun0812 release v0.10.0, the only linux_x86_64 build for this stack there) is pinned in the same `uv.lock` via a second
    platform-marked source (`uv sync --locked --group flash`). Correctness passed: 68/68 tests with flash + compile active,
    28L compiled-flash equivalence passed (run `20261006T155451Z-bench-compile-equivalence-litegpt-v2-28l-flash-101c3f8`). Speed: **inconclusive** - see ADR-001 evidence; the GPU power limit
    changed (95 W -> 155 W) between the SDPA+compile baseline and the flash+compile run, so the +32.5% cannot be attributed to
    flash. Not adopted as the WSL2 default on this evidence (`attention_backend: auto` would still pick flash if installed;
    production configs set the backend explicitly).
- **flash-attn: adopted for `attention_backend=auto` (EXP-M0-FLASH, spec 2.0.0-spec.3).** The spec target 2.8.3.post1 is
  sdist-only and cannot be built here (no `nvcc`, no MSVC, no WSL2). At the owner's direction the community prebuilt wheel
  `flash_attn-2.8.3+cu130torch2.14-cp312-cp312-win_amd64.whl` (github.com/mjun0812/flash-attention-prebuild-wheels, release
  v0.10.2) was installed through uv (`[dependency-groups] flash` + `[tool.uv.sources]`, so its sha256 is pinned in `uv.lock`:
  `090e78a6...e7134`; it also adds `einops`). This is a third-party binary: provenance is the pinned URL + hash; version 2.8.3
  is not the `.post1` named in spec.2. Install: `uv sync --locked --group flash`.
  - Parity (pre-registered, passed): flash vs SDPA logits within bf16 tolerance, plain causal and packed `cu_seqlens`
    (`tests/test_model.py`), plus `auto` equivalence. The kernel also matches SDPA directly (max abs diff 0.0039 in bf16).
  - Speed (pre-registered rule: >= +5% tok/s at identical config, same VRAM): 4 alternating A/B rounds, 28L, micro-batch
    3x2048, run `20261006T131103Z-bench-ab-sdpa-vs-flash-101c3f8`: SDPA 15,375 tok/s mean (15.2-15.7k) vs flash 17,250 tok/s mean
    (16.9-17.4k): **+12.2%**, peak VRAM identical (11.17 GiB). Every flash round beat every SDPA round. Adopted.
  - Behaviour: `auto` = flash only if installed AND tensors are on CUDA in bf16/fp16 AND no KV cache; otherwise SDPA (CPU, fp32,
    cached decoding). `sdpa` forces SDPA; `flash` forces flash. SDPA remains mandatory and fully tested.
  - Sustained (30 min, 28L mb3x2048, flash): 17,553 tok/s mean, stable (see ADR-001 evidence); run
    `20261006T134753Z-bench-sustained-28l-mb3-flash-30min-101c3f8`.
