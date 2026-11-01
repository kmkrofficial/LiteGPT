# ADR-009: M3 on a rented GPU, budget cap, context extension without a theta switch, and the text-to-data mixture

Status: Accepted (owner, 2026-11-01); spec 2.0.0-spec.7, applied before M3 started
Date: 2026-11-01
Amends: ADR-001 (budget in the gate), ADR-002 / ADR-006 (environment), ADR-005 (theta at the extension), ADR-008 (mixture)
Spec fields: `model.rope_theta` (unchanged), `training.pretraining.context_extension`, `training.extension_gate`, `data.mixture`,
`data.validation.data_order_sha256`

## Context

M2 finished on the laptop. The laptop would need about 4 days for 10B tokens, so the owner moved M3 to one rented GPU. Four things
then changed, each recorded in `docs/spec/registry.yaml` before any M3 measurement:
- A price: ~$0.60/h, with a cap on spending.
- A measurement: changing RoPE theta from 10k to 100k on the same weights costs +0.036 validation BPB at the moment of the switch (+0.023 web, +0.038 code, +0.035 math), on the 200M-token LR checkpoint.
- A use case: the owner wants V2 to solve something specific, and a very small model is weak at writing programs.
- A size argument: at theta 10k, the slowest RoPE component turns only 31 degrees by position 4,096, so 4,096 does not need a higher theta.

## Decision

1. **Hardware.** One rented A100 40 GB, on-demand, Docker template, 20 GB container disk + 100 GB local disk, ~$0.60/h, driver >= 580. The 80 GB card (~$0.95/h) would have to train 1.61x faster to cost the same per token.
2. **Budget.** Hard cap **INR 2,500** on all M3 cloud spend. The server guard stops the instance at INR 2,450. The gate projects against **INR 2,400**, so a run it allows is not cut off mid-decay. This tightens EXP-G1-EXTENSION-GATE's first text (INR 2,500) before M3; the BPB rule is unchanged. If delta >= 0.010 and 14.0B does not fit, the stable phase ends at the largest affordable 0.5B step; if none fits, stop at 9.0B.
3. **Peak LR 2.4e-3** (EXP-M2-M3LR, owner-approved).
4. **Theta 10k from start to finish; final context 4,096.** The last 0.4B tokens of the decay read contiguous 4,096-token sequences, so long documents stay whole. Positions up to 2,048 do not change. `context_extension.rope_theta` becomes 10000. A `pre_extension` milestone keeps the last 2,048-context weights.
5. **Use case: private, offline text-to-data.** Extract fields, classify and route, find entities and write short summaries, after fine-tuning on the user's documents, on their own machine. The reusable recipe and the tokenizer research are also goals (owner: a mix of recipe, product and research).
6. **Mixture for that use case**, by judgement between two measured ends, not measured itself (the owner skipped a proxy test as not needed):

   | Share | FineWeb-Edu | Cosmopedia | Code | Math | Instruction |
   |---|---|---|---|---|---|
   | stable, was | 64.58 | 10.42 | 12.50 | 12.50 | - |
   | **stable, now** (`stable_v2`) | **72.00** | **13.00** | **7.50** | **7.50** | - |
   | decay, was | 43.18 | 12.95 | 17.27 | 21.59 | 5.00 |
   | **decay, now** (`decay_v2`) | **50.00** | **20.00** | **12.00** | **13.00** | **5.00** |

   The code languages keep their shares. Measured ends at proxy scale: 25% code + math gave web 1.0958 and code 0.7123 (s8a); 0% gave web 1.0674 and code 1.3629 (s8b). Index sha256: `stable_v2` cfecb807…, `decay_v2` 79f35692…. The shards, the tokenizer and `val_core` do not change; the old indexes stay for provenance. `val_core` stays the gate's yardstick: its weights are the old stable mixture, which does not matter for a comparison of the run with itself.
7. **Other schedule values** the spec left open: warmup 0.1B tokens; final LR 10% of peak; evaluation every 0.25B on `val_core` and all five `val_text` domains; checkpoints every 30 minutes, keep 3; milestones at 2.5B, 5B, 9B, `stable_end`, `pre_extension`, `final` (= the spec's `decay_end`).
8. **Tuning and safety on the server.** EXP-M3-OPT selects micro-batch, backend and compile mode (<= 45 minutes), leaving the training computation unchanged. `scripts/cloud/m3_guard.py` resumes crashes, backs up to a private repo every 3 hours, and stops the instance when the final checkpoint is verified, at the budget stop, or in a crash loop. It only ever stops; the owner destroys the instance.

## Alternatives rejected
- **Switch theta to 100k at the extension (ADR-005 as written):** a measured penalty with only 0.4B low-LR tokens to recover; SmolLM2 did this with 75B tokens.
- **Theta 100k from step 0:** no penalty, but all M2 evidence is at 10k and 4,096 does not need it. Its test was stopped by the owner at step 97 of 381 with no validation result, so nothing here compares the two.
- **8,192 context:** about INR 42 more, but half the practice per position, and text-to-data does not need it.
- **No code or math at all:** the small web gain (2.6%) does not justify the 91% code loss, and code is expected (not measured) to teach strict formats.
- **Direct upload to the instance:** bills the GPU while a home connection uploads. The data went to a private Hugging Face repo instead.

## Consequences
- M2's ladder evidence stays valid for the architecture, but s8a was not the final mixture. The first measurement of `stable_v2` is M3 itself.
- `data_order_sha256` changes, so the M1 freeze value is superseded by this ADR.
- The final context is trained for only about 4% of the run. Quality in the 2,048–4,096 range is unmeasured until M3.
- Headline claims move from "best code model" to the text-to-data comparison in `docs/private/LITEGPT_V2_POSITIONING.md`, registered before any result.
