# ADR-010: M4 scope, the text-to-data comparison and where M4 runs

Status: Accepted (owner, 2026-11-03); spec 2.0.0-spec.8, applied before any M4 run
Date: 2026-11-03
Amends: ADR-000 and `PLAN.md` (M4 order), ADR-009 (cloud budget, which covered M3 only)
Spec fields: `training.sft`, `training.reasoning`, `training.preference` (deferred), `data.sources` (`docs/spec/data_sources.yaml`), new `m4`

## Context

M3 finished at 10.0B tokens. V2's zero-shot mean is 0.383, behind Gemma 3 270M (0.393) and SmolLM2-135M (0.403), and it solves 0.8% of GSM8K. The owner chose private, offline text-to-data as the use case, and people will judge results. The planned self-generated reasoning data keeps only answers the model gets right, so at this rate it would yield very little. Neither reasoning nor DPO targets extraction.

## Decision

1. **Scope (owner, D1).** M4 = chat SFT, then a registered text-to-data comparison, then the offline API and CLI. Reasoning and DPO are deferred to a decision after the comparison, with their own plan.
2. **SFT data (owner, D2).** `sft_reserved` plus `smol-magpie-ultra`. The from-base route stays free of Llama terms. The legal naming call stays at M5.
3. **Tasks (owner, D3, as corrected by the Phase 0 audit).** CUAD fields, MASSIVE en-US slots and Banking77 intents are primary. The score is their plain mean. BillSum is secondary and reported only.
4. **CUAD corrections from the audit (owner approved):**
   - Effective Date fails the registered 80% presence rule (76.5%) and is replaced by Document Name (100%).
   - Gold comes from the `-Answer` columns of `master_clauses.csv`, because the raw spans contain fragments and blanks.
   - Input is the first 3,000 characters plus a 1,500-character window around the last governing-law keyword, after whitespace normalisation. "First page" alone would contain only 7.8% of governing-law evidence even at 8,000 characters.
   - Evidence outside the input cannot be found by any model: on train, 12.5% of agreement dates and 14.9% of governing-law clauses.
5. **Overlap with V2's training data (owner, after the audit).** The rule set before the scan was to remove any test item sharing a 13-word run with V2's training data, and to stop and report above 1%. The scan found CUAD 60 of 102, BillSum 741 of 1,044, MASSIVE 1 of 2,974 and Banking77 0 of 3,080, and the matches are standard wording. The new rule removes only items with a longest run of at least 30 words or at least 10% matched, for every model. That leaves CUAD 98, MASSIVE 2,973, Banking77 3,080 and BillSum 1,001. Scores are also reported on zero-match items (CUAD 42, BillSum 303) next to the main score.
6. **Comparison protocol.** As in `EXP-M4-T2D`: same examples byte for byte, same recipe, 3-point learning-rate grid chosen on dev, 3 seeds, route chosen on dev only, test scored once, paired bootstrap, win or loss beyond 2 standard errors, otherwise tie. Every outcome is published.
7. **Where it runs (owner, D4).** GPU-heavy work on the cloud; simple work, or work taking about the same time, on the laptop:
   - Cloud: SFT and all fine-tune arms with dev scoring, on one A100 PCIe 40 GB, on-demand, Docker, at most $0.50/h.
   - Laptop: audit, code, data preparation, instruct suite and G2, CPU numbers. Model-generated code still runs only in WSL2.
   - Test scoring moves to the laptop if the A100 is under 1.5x the laptop's generation speed on the same dev scoring job. All models use the same machine for test scoring.
8. **Budget.** The guard stops at $9.00, or at the balance minus $1 if lower, across all M4 sessions. If the projection passes the stop, cut in this order: SFT at 2 epochs, BillSum, the Qwen3 reference arm; then stop and ask the owner. V2 and rival arms, their seeds and test scoring are never cut.
9. **Gate G2.** IFEval above V1-Instruct (0.083) by more than 2 standard errors, and no divergence. If it fails, stop and report; the from-base route can still run.
10. **Safety data (owner, D5).** A safety slice only if a cleanly licensed training source exists. One search found none, so none is planned, and the model card will say so.

## Alternatives rejected
- **Keep SFT, reasoning and DPO as planned:** high cost, no link to the use case, and V2's measured math and code scores would give the self-generated data little to keep.
- **Limit CUAD to fields on the first page:** drops the only clause that requires reading past the header, and makes the task too easy.
- **Keep Effective Date:** breaks a rule registered before any model ran.
- **Expiration Date as the replacement:** 81% present but with a median position of 38% of the contract, so a fixed window would miss most of it.
- **Slide windows across whole contracts:** the median contract has 33,143 characters, so generation cost would multiply on a limited budget.
- **Token-based input caps:** each tokenizer would cut different text, which breaks the byte-identical rule.
- **Few-NERD:** share-alike licence. **paraloq/json_data_extraction:** synthetic. **GenSIE, ExtractBench, LLMStructBench:** language, format or unconfirmed licence (`docs/private/PLAN_M4.md` section 3).
- **All training on the laptop:** no spending, but about 25 to 46 GPU hours; the owner chose the cloud for GPU-heavy work.
- **Apply the original overlap rule as written:** CUAD would fall to 42 contracts, with very wide error bars, and contracts would be removed for using common legal wording. Those 42 are still reported as the stricter check.

## Consequences
- The original M4 order in `PLAN.md` changes. V2-Instruct will likely stay weak at math and code chat, and we say so.
- The overlap rule was relaxed, with its cut-offs chosen after seeing overlap figures and before any model output. Results must state this, and that V2's overlap is measured while rivals' cannot be.
- CUAD's score partly reflects easy fields (Document Name, Parties), and its 98 test contracts give wide error bars.
- MASSIVE and Banking77 are short texts, so the 13-gram overlap check has little power there: it can test only 173 and 761 of their test items. Rival models' exposure to these public datasets cannot be measured at all.
- Banking77 has published, unverified reports of up to 14% possible label errors. They affect all models alike but cap the reachable score.
- M3 cost $16.64, so M3 plus M4 could reach $25.64 (about Rs 2,523 at Rs 98.40/$). The Rs 2,500 cap covered M3 only, so any total-cost claim must state this.

## Evidence
`EXP-M4-SFT`, `EXP-M4-T2D` in `docs/spec/registry.yaml`; final item hashes in `benchmarks/m4/t2d_manifest.json`; audit figures in `docs/private/M4_AUDIT.md` and `docs/private/logs/m4/phase0/`.
