# The `litegpt` package

Start with [`api.py`](api.py) and [`cli.py`](cli.py). Docs: [inference](../docs/inference.md), [architecture](../docs/architecture.md).

| Module | What it does |
|---|---|
| `api.py` | Python API: `load`, `generate`, `chat`, `chat_stream`, `extract`, `info` |
| `cli.py` | The `litegpt` command: `chat`, `generate`, `info`, `extract` |
| `model.py` | The V2 architecture, in one file |
| `config.py` | Model configs and presets |
| `tokenizer.py`, `assets/` | The 32k tokenizer and the chat template's Jinja twin |
| `chat_template.py` | `litegpt-chat-v1` rendering and the assistant-only loss mask |
| `inference/` | Generation engine, sampling, stop conditions |
| `training/` | Training loops, schedule, gate, chat fine-tuning |
| `eval/` | Evaluation suite and the adapter for lm-evaluation-harness |
| `t2d.py`, `t2d_run.py` | The text-to-data comparison: prompts and scoring, and running every model the same way |
| `data.py`, `dataprep.py`, `datafilters.py`, `datacheck.py`, `decontam.py`, `sft.py` | Data preparation, licence and secret filters, checks, benchmark overlap removal, chat data |
| `checkpoint.py` | Atomic, resumable training checkpoints |
| `records.py` | Run records, the source of every reported number |
| `spec.py` | Reads and checks the specification |
