# scratch/

Temporary experiments and prototypes. Everything here except this file is git-ignored.

- Nothing in `litegpt/`, `scripts/` or `benchmarks/` may import from `scratch/`.
- If something here proves useful, promote it: reusable tooling to `scripts/` or `benchmarks/`, tested production code to `litegpt/` (with tests in `tests/`). Then delete it from here.
- Results of any experiment that matters belong in `runs/<run-id>/`, written through `litegpt.records.create_run`, not in this directory.
