# Scripts

Run from the repository root with `uv run python`. Each script's first lines give its usage.

| Stage | Script | What it does |
|---|---|---|
| Data | `audit_sources.py` | Exact revisions, licences and token counts of the sources, from the data itself |
| Data | `prepare_data.py` | Resumable data pipeline; builds the frozen mixture indexes |
| Data | `decontaminate.py`, `audit_decontam.py` | Build the 13-word benchmark filter; check what it matches |
| Data | `train_tokenizer.py` | Train and check the 32k tokenizer |
| Data | `prepare_sft.py` | Chat fine-tuning data |
| Data | `freeze_t2d.py` | Rebuild the text-to-data test sets byte for byte |
| Training | `train.py` | Pretraining, and the M2 ladder |
| Training | `train_sft.py` | Chat fine-tuning and its learning-rate choice |
| Scoring | `run_t2d.py` | Text-to-data: fit, choose learning rate and route, score test once, CPU numbers |
| Scoring | `m4_g2.py` | The G2 chat gate and the test-machine speed rule |
| Release | `python -m litegpt.export.to_hf` | Export a checkpoint to the Hugging Face layout and run the registered checks |
| Reports | `build_report.py` | Generate `runs/report-latest/REPORT.md` from run records |
| Reports | `build_figures.py` | Generate every chart in `docs/figures/` from run records. `--check` finds stale charts |

Evaluation runs through `python -m litegpt.eval.run` and `python -m litegpt.eval.native` ([docs/evaluation.md](../docs/evaluation.md)).

## Other folders

- [`cloud/`](cloud/README.md): rented-server tools.
- [`legacy_v1/`](legacy_v1/README.md): the preserved V1 baseline.
