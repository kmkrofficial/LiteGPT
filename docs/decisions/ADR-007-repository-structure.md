# ADR-007: Repository structure

Status: Accepted
Date: 2026-10-06
Spec fields: pointer fields only (`evaluation.suite_version_file`, `licensing.audit_file`); no values changed (spec 2.0.0-spec.4)

## Context

By the end of M0 the repository mixed three things at the top level: the V1 baseline (13 tracked files and directories,
including a second `model.py`), V2 production code, and generated outputs (a separate `reports/` directory holding report code,
committed copies of run summaries and the generated report). The specification lived in `experiments/`, outside any
documented home. Before more experiments are run, ownership of every path needs to be unambiguous.

## Decision

Seven top-level directories, each with one role (details in [docs/repository_structure.md](../repository_structure.md)):

| Directory | Role |
|---|---|
| `litegpt/` | canonical production source (model, config, records, checkpointing, and future inference, training, eval, export) |
| `tests/` | canonical tests |
| `benchmarks/` | reusable benchmark tooling |
| `scripts/` | reusable utilities, including `build_report.py` and the preserved V1 baseline in `scripts/legacy_v1/` |
| `scratch/` | temporary experiments and prototypes (git-ignored) |
| `runs/` | all generated experiment outputs: logs, metrics, telemetry, figures, reports, manifests |
| `docs/` | permanent documentation, ADRs, and the specification / experiment registry (`docs/spec/`) |

Rules: production code stays in `litegpt/` with no experimental or duplicate implementations; benchmark tools stay in
`benchmarks/` and their results go to `runs/<run-id>/`; there is no top-level `reports/` directory (the generated report is
`runs/report-latest/`); useful historical runs are never deleted.

## Alternatives rejected

- **Top-level `legacy/`:** an eighth directory. The V1 files are runnable baseline scripts, so `scripts/legacy_v1/` fits the
  seven-directory structure.
- **Keeping `reports/data/` as committed summaries:** a second, slimmer copy of every run record. The report builder reads
  `runs/` directly and validates each record, so the copies were redundant and could drift.
- **Future components as new top-level directories** (`training/`, `eval/`, `export/`, `data/`): they are production code and
  become `litegpt/` subpackages; data-preparation entry points are `scripts/`.

## Consequences

- `docs/spec/v2.yaml` moved from `experiments/spec/` (content unchanged except pointer fields and the version), as did the
  registry. The spec hash recorded in older run records differs from the current one by design: the hash is recorded per run.
- V1 history is preserved (`git mv`). V1 scripts are run from `scripts/legacy_v1/` (their relative paths are unchanged).
- Run records and result files are committed; `telemetry.jsonl`, checkpoints and data shards under `runs/` are ignored.
