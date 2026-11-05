# Architecture Decision Records

Each ADR records one decision for LiteGPT V2: its context, the decision, the alternatives that were rejected, and the evidence behind it. The machine-readable form of every approved decision lives in [`docs/spec/v2.yaml`](../spec/v2.yaml). Any change to that spec needs a new ADR and a bumped `spec_version`.

| ADR | Title | Status | Date |
|---|---|---|---|
| [ADR-000](ADR-000-v2-scope-decisions.md) | V2 scope decisions (D1–D4) | Accepted | 2026-10-06 |
| [ADR-001](ADR-001-compute-budget-and-extension-gate.md) | Compute budget and pre-registered 10B → 15B extension gate | Accepted | 2026-10-06 |
| [ADR-002](ADR-002-environment-and-attention-backends.md) | Environment, attention backends and `torch.compile` | Accepted (amended by ADR-006) | 2026-10-06 |
| [ADR-003](ADR-003-synthetic-data-and-provenance.md) | Synthetic data policy and provenance | Accepted | 2026-10-06 |
| [ADR-004](ADR-004-licensing.md) | Licensing | Accepted | 2026-10-06 |
| [ADR-005](ADR-005-model-architecture.md) | V2 model architecture | Accepted (evidence pending M2) | 2026-10-06 |
| [ADR-006](ADR-006-toolchain-oct-2026.md) | Toolchain: Python 3.12, uv, pinned stable stack (Oct 2026) | Accepted (amends ADR-000 D2, ADR-002) | 2026-10-06 |
| [ADR-007](ADR-007-repository-structure.md) | Repository structure: litegpt / tests / benchmarks / scratch / runs / docs / scripts | Accepted | 2026-10-06 |
| [ADR-008](ADR-008-data-sources-mixture-and-freeze.md) | Data sources, mixture, partitions, tokenizer and validation freeze | Accepted | 2026-10-11 |
| [ADR-009](ADR-009-m3-cloud-budget-context-and-mixture.md) | M3 on a rented GPU, budget cap, 4,096 context without a theta switch, text-to-data mixture | Accepted | 2026-11-01 |
| [ADR-010](ADR-010-m4-scope-text-to-data-comparison-and-cloud.md) | M4 scope, text-to-data comparison and where M4 runs | Accepted | 2026-11-03 |
| [ADR-011](ADR-011-model-licence.md) | Model licence and what v2.0.0 releases | Partly decided: release scope by the owner, licence proposed | 2026-11-05 |

## Template

```
# ADR-NNN: Title
Status: Proposed | Accepted | Superseded by ADR-XXX
Date: YYYY-MM-DD
Spec fields: <paths in docs/spec/v2.yaml>

## Context
## Decision
## Alternatives rejected
## Consequences
## Evidence   (links to experiment registry IDs and generated report sections)
```
