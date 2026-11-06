# ADR-004: Licensing

Status: Accepted
Date: 2026-10-06
Spec fields: `licensing`

## Context

LiteGPT V1's initial design came from nanoGPT, which is MIT-licensed, copyright (c) 2022 Andrej Karpathy, and V1's code is kept in this repository. Commit `8b8f4df` removed the repository's LICENSE file. The MIT licence requires the copyright and permission notice to be kept in copies and substantial portions. The V2 model weights will be trained on many datasets, each with its own licence and terms.

## Decision
- **Code:** restore the **MIT** licence in M0. It includes that notice alongside the LiteGPT copyright, and the README gets an attribution section.
- **Model weights:** **not finalised.** The status is `pending-dataset-audit` until:
  1. every dataset used in pretraining, SFT, reasoning and DPO is listed in the licence register with its licence, terms, redistribution and derivative-work constraints, and terms-of-use origin flags;
  2. compatibility is confirmed and a model licence is chosen in a new ADR.
- **The audit is a release-blocking gate for M5 (v2.0.0).** No weights are published before that gate passes.

## Alternatives rejected
- **Picking Apache-2.0 or MIT for the weights now:** it could conflict with dataset terms that are discovered later.

## Consequences
- M0 restores `LICENSE` and the README attribution.
- M1 starts the licence register.
- M5 must contain the model-licence ADR.

## Evidence
- Pending: `LICENSE` (M0), the licence register (M1 onwards), model-licence ADR (M5).
