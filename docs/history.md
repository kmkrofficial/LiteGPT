# How V2 was built

The milestones, the decisions that shaped the model, what went wrong and what it cost. Detail: [experiment registry](spec/registry.yaml), [ADRs](decisions/README.md), and the [run records](../runs/README.md). Final numbers: [results.md](results.md).

## Working method

Every experiment was registered, with its thresholds, before it ran. The owner approved each phase. Corrections were recorded before acting on them. A recipe change after seeing test scores would void a comparison and be recorded as a deviation. Results are published whichever way they went.

## Milestones

| Milestone | Date | What it did | Outcome |
|---|---|---|---|
| **M0** Foundations | Oct 2026 | Environment, model code, run records, GPU benchmarks | Model code and tooling in place |
| **M1** Data and tokenizer | Oct 2026 | Licence audit, data preparation, own 32k tokenizer, benchmark text filtered from training data | 20.79B tokens prepared |
| **M1b** Baselines | Oct 2026 | Evaluation suite, 16 models scored before V2 existed | Targets known before training |
| **M2** Experiments | Oct to Nov 2026 | Small proxy models, one design change per step; learning-rate sweep on the real V2 | Design and learning rate 2.4e-3 chosen |
| **M3** Pretraining | Nov 1 to 3, 2026 | The real V2 on one rented A100 | 10.0B tokens, 37.2 h. Better than V1, behind SmolLM2-135M and Gemma 3 270M |
| **M4** Chat and comparison | Nov 3 to 5, 2026 | Chat fine-tuning, registered text-to-data comparison, API and CLI | Chat gate passed. **Comparison lost** |
| **M5** Release | Nov 2026 | Licence decision, export, model cards, this clean-up | In progress |

## The design ladder (M2)

Each step changed one thing on a small proxy model, 1B tokens each. Web bits per byte, lower is better:

| Step | Change | Web | Code |
|---|---|---:|---:|
| s1 | V1 baseline (mean of two stable seeds) | 1.1096 | 1.8147 |
| s2 | + RoPE | 1.0750 | 1.8433 |
| s3 | + RMSNorm and QK-norm | 1.0640 | 1.7490 |
| s4 | + SwiGLU | 1.0609 | 1.7652 |
| s5 | + grouped-query attention | 1.0593 | 1.8463 |
| s6 | + deeper and thinner | 1.0599 | 1.7704 |
| s7 | + 32k tokenizer | 1.0632 | 1.8957 |
| s8a | + full data mixture | 1.0958 | 0.7123 |

s7 and s8a were labelled "hurts" on web by the registered rule. The owner kept both: the mixture cut code bits per byte by more than half, and web was not the only goal.

## Decisions that shaped the model

| Decision | Choice | Why | Record |
|---|---|---|---|
| Architecture | RoPE, RMSNorm with QK-norm, SwiGLU, GQA, 28 x 768 | Modern parts tested one at a time | [ADR-005](decisions/ADR-005-model-architecture.md) |
| Licensing of data | No Wikipedia; only permissive-labelled code; Apache-2.0-labelled SmolTalk subsets | Avoid share-alike and ambiguous terms | [register](data_licenses.md) |
| Benchmark overlap | Remove training documents sharing a 13-word run with evaluation sets | Scores should not reflect memorised test text | EXP-M1-DECONTAM |
| Length | 10B tokens, more only through a registered gate | Decide extension from evidence, not hope | [ADR-001](decisions/ADR-001-compute-budget-and-extension-gate.md) |
| Compute | Move pretraining from the laptop to one rented A100 | Laptop thermals set the speed; a multi-day run needs no interruptions | [ADR-009](decisions/ADR-009-m3-cloud-budget-context-and-mixture.md) |
| Purpose | Private, offline text-to-data | The owner's use case. Judge by results, not by how little data was used | [ADR-009](decisions/ADR-009-m3-cloud-budget-context-and-mixture.md), [ADR-010](decisions/ADR-010-m4-scope-text-to-data-comparison-and-cloud.md) |
| M4 scope | Chat fine-tuning and a real comparison; reasoning and preference tuning deferred | A comparison tells us more than more stages | [ADR-010](decisions/ADR-010-m4-scope-text-to-data-comparison-and-cloud.md) |
| Autonomy | M4 ran without owner stops; registered rules decided | Owner's choice, made before any G2 or test result | EXP-M4-SFT |
| After the loss | No plan for more tokens or longer training | Owner's decision. What comes next is open | This page |
| Release | Base and Instruct, Instruct without the "Llama" prefix | Owner's decision, with a known licence risk | [ADR-011](decisions/ADR-011-model-licence.md) |
| Repository | Development history squashed for release | Owner's decision; also removes personal paths from history | [runs/README.md](../runs/README.md) |

## Recorded deviations

- **M2 thermal rule:** runs were throttled 95 to 98% of the time, breaking the registered 5% limit. A power profile fixed it; the affected work was handled by a recorded owner decision.
- **M2 noise floor:** s1 seed 2 diverged during decay. Taken at face value, it would have set a threshold no step could pass. A divergence rule and a third seed were added, recorded before use.
- **M4 speed check:** real packed steps ran at 46.7k tokens per second, under the registered 55k stop. The owner chose to continue on flash attention.
- **M4 evaluation machine:** G2 was measured on the A100, its baselines on the laptop. A later laptop run differed by one question of 541.
- **Instruct name:** see [ADR-011](decisions/ADR-011-model-licence.md).

## What went wrong

| When | What | Result |
|---|---|---|
| M2 | First smoke run filled GPU memory and spilled into shared memory: no step in 4 minutes | Memory cap and probe added |
| M2 | Final evaluations ran out of memory at the last step | Fixed and resumed |
| M2 | WSL updated itself and restarted mid-run; Windows Update rebooted the laptop twice; a charger came loose and the laptop slept | Resumed from checkpoints, worst loss about 10 minutes |
| M3 | First start crashed on an index too short; "too many open files" | Fixed; the failed run is kept |
| M3 | The guard backed up only the newest checkpoints, so milestones were on the server alone | Found in time; all uploaded and verified |
| M4 | Fine-tunes first ran at 224 tokens per second: GPU memory spill | Memory cap added |
| M4 | The A100 was no faster than the laptop on generation | Registered rule moved test scoring to the laptop |
| M4 | The backup list missed G2's evaluation run, and the A100 was then rented out | Recovered from the volume by a cheap second server. Backup fixed |
| M4 | Test scoring died 5 minutes in and went unnoticed for a day | Resumed from saved outputs. Nothing had been scored |
| M5 | Safe checkpoint loading refused real training checkpoints | Caught only by testing real weights. Tests now cover it |

## Cost

| Item | Cost |
|---|---:|
| M3, one A100 | $16.64 |
| M4, one A100, plus storage, bandwidth and a small copy server | $8.09 |
| **Cloud total** | **$24.73** |

The laptop's work in M0 to M2 isn't costed.

## What the project concluded

1. V2 improved on V1 at general tasks, chat and code tasks.
2. V2 did not beat SmolLM2-135M or Gemma 3 270M, near its size, at general tasks or at the use case it was aimed at.
3. Its strength is the record: this page, the registry and the runs.

## Not done

- No one has rebuilt the model independently.
- Reasoning and preference tuning were not attempted.
- Spider and MultiPL-E evaluations were not run.
- Safety work: none. V2-Instruct refuses nothing.
