# LiteGPT

A small language model (SLM) trained from scratch, derived from Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

- **V1**: GPT-2 Small (124M), trained on FineWeb-Edu and fine-tuned on Alpaca. Preserved in [`scripts/legacy_v1/`](scripts/legacy_v1) as the baseline.
- **V2** (in development): an independently implemented modern architecture, trained on a single RTX 4090 Laptop (16 GB).

## LiteGPT V2 (in development)

RoPE, RMSNorm + QK-norm, SwiGLU, grouped-query attention, a 32k tokenizer, 28 layers x 768, 2048 context, ~201M parameters,
with a reproducible pipeline and an evaluation / benchmarking layer.

- Specification: [`docs/spec/v2.yaml`](docs/spec/v2.yaml); decisions: [`docs/decisions/`](docs/decisions/README.md).
- Repository layout: [`docs/repository_structure.md`](docs/repository_structure.md).
- Environment (Python 3.12 + uv, locked): [`docs/reproduce.md`](docs/reproduce.md).

```bash
uv sync --locked                                   # create .venv from uv.lock
uv run pytest                                      # V2 test suite (CPU, SDPA path)
uv run python -m benchmarks.train_mem single       # training throughput / VRAM benchmark (needs a GPU)
uv run python -m scripts.build_report              # regenerate runs/report-latest/REPORT.md from runs/
```

Status: M0 (foundations) in progress. No V2 model has been trained yet.

## Repository layout

```
litegpt/        canonical production source          tests/       canonical tests
benchmarks/     reusable benchmark tooling           scripts/     reusable utilities (+ scripts/legacy_v1/, the V1 baseline)
scratch/        temporary experiments (git-ignored)  runs/        all generated experiment outputs and reports
docs/           documentation, ADRs, spec/
```

## V1 baseline

The V1 scripts live in `scripts/legacy_v1/` and are run from inside that directory (their relative paths are unchanged).

1. **Install V1 dependencies** (separate from the V2 environment):
   ```bash
   cd scripts/legacy_v1
   pip install -r requirements.txt
   ```
2. **Prepare data.** The datasets (FineWeb-Edu, Alpaca) are not included. Download and tokenize them with the `prepare.py`
   scripts in `scripts/legacy_v1/data/<dataset>/`.
3. **Train / fine-tune:**
   ```bash
   python train.py config/train_fineweb.py        # base model
   python train.py config/finetune_alpaca.py      # instruction fine-tuning
   ```
4. **Chat with the models:**
   ```bash
   python check_model.py       # base model (FineWeb-Edu)
   python check_instruct.py    # instruction-tuned model; supports /temp, /tokens
   ```

Checkpoints are written to `out-fineweb/` and `out-instruct/` inside `scripts/legacy_v1/` (git-ignored).

### V1 on Hugging Face

The published V1 models are `koganrath/LiteGPT-Instruct` (fine-tuned on Alpaca) and `koganrath/LiteGPT-Base` (pre-trained on
FineWeb). The V1 export script (`export_to_hub.py`) is not part of this repository (it is git-ignored); V2 will ship a tested
export (`litegpt/export/to_hf.py`) with a logit-parity check.

```bash
cd scripts/legacy_v1/hugging_face_inference
pip install -r requirements.txt
python inference_instruct.py     # instruct model
python inference_base.py         # base model
```

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("koganrath/LiteGPT-Instruct")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
```

V1 was tested on Windows with Python 3.11.9; the full package list is in `scripts/legacy_v1/test_env_requirements.txt`.

## Licence and attribution

Code is MIT licensed (see [`LICENSE`](LICENSE)). LiteGPT is derived from [nanoGPT](https://github.com/karpathy/nanoGPT)
(Copyright (c) 2022 Andrej Karpathy, MIT), whose copyright notice is retained. The licence of the V2 model weights is not
final: it will be chosen only after every training dataset's licence has been audited ([ADR-004](docs/decisions/ADR-004-licensing.md)).

## Acknowledgements

- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) Dataset
- [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) Dataset
