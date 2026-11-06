# Architecture Decision Records

Each ADR records one decision for LiteGPT V2: its context, the decision, the alternatives that were rejected, and the evidence behind it. The machine-readable form of every approved decision lives in [`docs/spec/v2.yaml`](../spec/v2.yaml). Any change to that spec needs a new ADR and a bumped `spec_version`.

| ADR | Title | Status | Date |
|---|---|---|---|
| [ADR-000](ADR-000-v2-scope-decisions.md) | V2 scope decisions (D1–D4) | Accepted | 2026-10-06 |
| [ADR-002](ADR-002-environment-and-attention-backends.md) | Environment, attention backends and `torch.compile` | Accepted (amended by ADR-006) | 2026-10-06 |
| [ADR-003](ADR-003-synthetic-data-and-provenance.md) | Synthetic data policy and provenance | Accepted | 2026-10-06 |
| [ADR-004](ADR-004-licensing.md) | Licensing | Accepted | 2026-10-06 |
| [ADR-005](ADR-005-model-architecture.md) | V2 model architecture | Accepted (evidence pending M2) | 2026-10-06 |
| [ADR-007](ADR-007-repository-structure.md) | Repository structure: litegpt / tests / benchmarks / scratch / runs / docs / scripts | Accepted | 2026-10-06 |

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
