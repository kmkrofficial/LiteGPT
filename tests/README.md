# Tests

```bash
uv run pytest
```

The suite runs on CPU with the SDPA path in about a minute, on a tiny model. GPU-only tests skip themselves when no GPU is present.

Run it under WSL or Linux for full coverage: some tests use `torch.compile` or POSIX signals.

| Tests | What they guard |
|---|---|
| `test_model`, `test_inference`, `test_api` | Architecture, cached decoding against full passes, API, CLI, safe loading |
| `test_tokenizer`, `test_chat_template` | Frozen tokenizer, and identical Python and Jinja chat ids |
| `test_data*`, `test_decontaminate`, `test_prepare_smoke` | Data pipeline, licence and secret filters, overlap removal |
| `test_training`, `test_checkpoint`, `test_m3_schedule`, `test_sft*` | Schedule, gate, bit-exact resume, chat fine-tuning |
| `test_eval_*`, `test_t2d*` | Scoring, and running every model the same way |
| `test_m3_guard`, `test_m4_queue` | Server guard, job queue and backups |
| `test_records`, `test_reports`, `test_spec` | Run records, generated reports, specification |
| `test_docs` | Tagged doc snippets run, and relative links resolve |
| `test_ablation`, `test_power_sweep` | Benchmark tools |

Real-weights checks don't run here because weights aren't in the repository. They caught a loading bug the unit tests missed, so run `litegpt info` and `litegpt chat` on real checkpoints after changing `litegpt/api.py`.
