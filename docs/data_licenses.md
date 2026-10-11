# Data licence and provenance register

Status: **M1 step 1 audit complete; mixture not yet finalised.** Feeds the D4 release gate: the model licence is decided at M5 only
after this register is complete. Facts come from the Hub dataset cards/API and from reading the data itself at the exact
revisions below. Evidence records (in `runs/`): `20261007T025701Z-data-verify-sources` (availability, gating),
`20261007T124946Z-data-audit-code`, `20261007T125037Z-data-audit-pretrain`, `20261007T125330Z-data-audit-smoltalk`.
Token counts are estimates with the GPT-2 tokenizer as a proxy; final counts come from the frozen LiteGPT tokenizer.
Owner decisions of 2026-10-07 are recorded in `docs/spec/registry.yaml` (EXP-M1-CODE, EXP-M1-SOURCES).

## Pretraining sources (approved)

| Domain | Source (config) | Exact revision | Licence (card) | Original generator | Size (rows / est. tokens) | Flags |
|---|---|---|---|---|---|---|
| Web | `HuggingFaceTB/smollm-corpus` (`fineweb-edu-dedup`) | `3ba9d605774198c5868892d7a8deda78031a781f` | ODC-By 1.0 | none (Common Crawl) | 190.17M / ~204.8B (int_score>=4: ~27.3B) | Common Crawl terms of use |
| Synthetic textbooks | `smollm-corpus` (`cosmopedia-v2`) | same revision | ODC-By 1.0 | Mixtral-8x7B-Instruct (Apache-2.0) | 39.13M / ~28.5B | D3: public synthetic data allowed; generator recorded |
| Math | `HuggingFaceTB/finemath` (`finemath-3plus`) | `e92b25a616738fe95dc186b64dfb19f9c8525594` | ODC-By 1.0 | none (web) | 21.41M / ~37.7B (int_score>=4: ~9.7B) | Common Crawl terms. `finemath-4plus` (6.70M / ~10.4B) is a subset and is not used separately |
| Code | `codeparrot/github-code-clean` | `c48d40f9e70f0196f8236901ee35807f7d6c44c0` | card: apache-2.0; **files carry their own repo licences, only permissive ones kept** | none | 880 parquet files; ~116.4B kept tokens in the retained languages | see below |
| Instruction-style (decay 5%) | `HuggingFaceTB/smoltalk` (4 subsets) | `5feaf2fd3ffca7c237fc38d1861bc30365d48ffa` | Apache-2.0 (per card, those subsets only) | Llama-3.1-405B-Instruct (Smol-Magpie-Ultra) and others | ~727M tokens | see Llama terms below |

**Wikipedia (`wikimedia/wikipedia`, `20231101.en`, revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`): EXCLUDED** by owner decision
2026-10-07 (CC BY-SA 3.0 + GFDL share-alike). Its 4% stable and 7% decay allocation is redistributed across the approved sources.
It is not used for training or validation, so the release model carries no share-alike data.

## Code: audit of `github-code-clean` (12 of 880 files, evenly spaced)

- Homogeneous: per-file variation of kept tokens is 2-7% for the large languages (14-20% for Rust/Shell/SQL), so a fixed contiguous
  file list is representative. Exact duplicates among kept documents across files: 0.000%.
- Licence mix of all documents (bytes): MIT 27.7%, GPL-2.0 24.4%, Apache-2.0 19.9%, GPL-3.0 11.6%, BSD-3 6.1%, AGPL-3.0 2.6%, LGPL 3.9%,
  other. Permissive = {mit, apache-2.0, bsd-2-clause, bsd-3-clause, isc, cc0-1.0, unlicense} = **55.7% of bytes**; everything else is dropped.
- Registered filters (file <= 100 KB, longest line <= 1,000, alphanumeric fraction >= 0.25, auto-generated header) keep 82.8% of
  permissive Python bytes (most drops are files over 100 KB).
- **Python:** 12.7B kept tokens available (6.2% of all bytes, 61.6% permissive); `ast.parse` succeeds on 87.9% of a 3,014-document sample.
  Python licence mix: Apache-2.0 24.7%, MIT 21.4%, GPL-3.0 16.3%, BSD-3 12.4%, GPL-2.0 12.2%.

| Language | Kept tokens, all 880 files (B) | Natural share of retained | Proposed share of code |
|---|---|---|---|
| Java | 26.3 | 22.6% | 10.0% |
| JavaScript | 16.2 | 13.9% | 11.0% |
| C | 14.0 | 12.0% | 8.0% |
| C++ | 13.0 | 11.1% | 8.0% |
| **Python** | **12.7** | **10.9%** | **36.0%** |
| PHP | 10.4 | 8.9% | 3.0% |
| C# | 9.7 | 8.3% | 4.0% |
| Go | 4.35 | 3.7% | 5.0% |
| Ruby | 3.5 | 3.0% | 2.0% |
| TypeScript | 3.0 | 2.6% | 6.0% |
| Scala | 1.3 | 1.1% | 1.5% |
| Rust | 0.76 | 0.7% | 2.0% |
| Shell | 0.68 | 0.6% | 2.0% |
| SQL | 0.48 | 0.4% | 1.5% |
| (excluded: HTML 25.6, Markdown 6.3, CSS 2.8, TeX, others) | 38.1 | | 0% |

The proposed shares are a proposal awaiting owner confirmation (they are not in the spec yet). Caveats carried to M5: the licence field is
a repository-level label, not a per-file audit; attribution requirements of MIT/BSD/Apache files are not reproduced in the shards.
Not used: `HuggingFaceTB/stack-edu` and `smollm-corpus` python-edu (blob ids only, no text), `bigcode/starcoderdata` and
`bigcode/the-stack-dedup` (gated, not needed: the audit shows ample Python and other code tokens).

## SmolTalk (instruction-style decay data and the M4 SFT split)

Owner decision 2026-10-07: only subsets explicitly labelled Apache-2.0 on the dataset card are used. For the other subsets the card
says to "refer to the original dataset for the license", so their licensing is ambiguous and they are **excluded**.

| Subset | Train rows | Est. tokens | Status |
|---|---|---|---|
| smol-magpie-ultra | 409,537 | ~654M | **allowed** (generated with Llama-3.1-405B-Instruct; see below) |
| smol-summarize | 96,356 | ~48M | **allowed** |
| smol-rewrite | 53,342 | ~19M | **allowed** |
| smol-constraints | 34,424 | ~7M | **allowed** |
| numina-cot-100k, openhermes-100k, apigen-80k, self-oss-instruct, metamathqa-50k, systemchats-30k, explore-instruct-rewriting, longalign, everyday-conversations | 1,100-106,147 each | | **excluded** (ambiguous) |

Consequence: the 5% decay slice (~51M tokens) and the M4 SFT split are disjoint hash partitions of these four subsets (~727M tokens in
total); math/code reasoning data for M4 needs its own licence audit.

### Llama terms for Smol-Magpie-Ultra (generator: Llama-3.1-405B-Instruct)

Read from the Llama 3.1 Community License text on 2026-10-07 (a summary of the fetched page, **to be verified against the licence file at M5**):
- Section 1(b): anyone who uses Llama materials *or any outputs or results of Llama materials* to create, train, fine-tune or improve an
  AI model that is distributed or made available must **include "Llama" at the beginning of that model's name**, and display
  "Built with Llama" on related websites and documentation.
- The notice "Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright (c) Meta Platforms, Inc. All Rights Reserved"
  must be retained in distributed copies of Llama materials.
- Use must comply with Meta's Acceptable Use Policy.
- A licence from Meta is needed above 700 million monthly active users.

**Open question for the M5 release gate (the owner's call):** whether outputs of Llama 3.1 used inside the 5% decay slice and the SFT data make
the released model subject to the "Llama" naming requirement (for example `Llama-LiteGPT-V2-Instruct` instead of `LiteGPT-V2-Instruct`).
This is recorded, not decided. Options include accepting the naming and notice, or dropping Smol-Magpie-Ultra and keeping the remaining three subsets
(~74M tokens), which would not cover the decay slice plus an SFT split.

## Generator terms of the decay-slice subsets (pre-flight check C5, 2026-10-11)

| Subset | Generator (source) | Generator licence | Naming / attribution clause |
|---|---|---|---|
| smol-magpie-ultra | Llama-3.1-405B-Instruct-FP8 (dataset card, SmolLM2 paper) | Llama 3.1 Community License | "Llama" prefix + "Built with Llama" (see above) |
| smol-summarize, smol-rewrite | Qwen2.5-72B-Instruct (SmolLM2 paper, arXiv 2502.02737; **not stated on the dataset card**) | Qwen LICENSE AGREEMENT (Hub licence tag "other") | "Built with Qwen" / "Improved using Qwen" in related documentation (section 5(b)), plus the Qwen notice file (section 3(c)) |
| smol-constraints | Qwen2.5-72B-Instruct (same paper) | same | same |
| cosmopedia-v2 | Mixtral-8x7B-Instruct-v0.1 (dataset card) | Apache-2.0 (Hub licence tag) | none |

The Qwen clause is read from a summary of the licence page; verify against the licence file at M5. Section 5(b) is conditional on the
trained model being "distributed or made available". The licence does not define "outputs", and it is unclear whether a party that
only trains on a public dataset of outputs is bound at all. This is a legal question for the owner, not a technical one.

**Consequence for the plan:** the pre-registered pass rule of C5 (no naming or attribution clause on the decay subsets) is **not met**.
Pretraining (decay slice) would give the base model the Qwen display requirement. Options for the owner:
1. Accept the "Built with Qwen" notice for the base model (documentation only), keeping the 5% instruction-style slice.
2. Drop the instruction-style slice from pretraining (its 5% goes to the other decay sources proportionally). The base model then
   carries no Llama or Qwen terms; they apply only to the Instruct model, which uses these subsets in M4 SFT.

**Owner decision (2026-10-11):** ignore the Qwen clause for now and proceed with option 1 (the instruction slice stays in the
decay mix). The clause is not removed from this register: it is carried to the M5 release gate so that the model card, naming and
notice files can be handled there.

## Pre-flight evidence for the code source (2026-10-11, 12 files, GPT-2 tokenizer proxy)

- Final code languages and weights: Python 63, JavaScript 14, TypeScript 14, SQL 4, Shell 5 (percent of code tokens); input files 0-222.
- GPL/AGPL/LGPL header notices found in permissive-labelled files before the rule: 0.09-0.76% of files per language; dropped (rescan finds none).
- Secrets before the rule (private keys, AWS/GCP/GitHub/Slack token formats) in 12 files: Python 16 documents, JavaScript 96,
  TypeScript 20, Shell 2; dropped. E-mail addresses redacted to `<EMAIL>`.
- Evidence records: `20261011T140430Z-data-preflight-code` (final), `20261011T135845Z-data-preflight-code` (first pass).

## Evaluation datasets (decontamination reference)

Used only to build the 13-gram filter and for scoring, never as training data. Revisions and licences are pinned in step 2 with `data_sources.yaml`.

## Items carried to the M5 release gate

1. Llama 3.1 naming/notice terms (Smol-Magpie-Ultra), above.
2. Code: repository-level licence labels, attribution requirements not reproduced.
3. Common Crawl terms of use (web, math sources).
4. Cosmopedia generator (Mixtral, Apache-2.0): no restriction found (C5).
5b. Qwen2.5-72B-Instruct display requirement (smol-summarize / rewrite / constraints), above.
5. Wikipedia: excluded, nothing carried.
