# Dataset audit — 2026-09-17 UTC

**The highest-value immediate change is a stronger training-data exclusion, followed by a small paired acquisition/rehearsal experiment. Broad quality improvement and absence of forgetting remain unproved.** This audit used CPU file inspection and open TRAIN/development artifacts only. No protected `ANY_READ` bank, fresh confirmation outputs, model weights or GPU was opened by this audit.

## New finding: the supposedly clean rehearsal pool still contains panel templates

A full-file recount reproduces the old 29,764-row large-v3 pool after excluding the three panel classes, multi-turn rows and user-only eight-word contact. It also finds a gap: a six-word question cannot trigger an eight-word check. The panel stems `What is 10% of 100?` and `What is 25% of 100?` survive under different operands and the wrong class label.

| TRAIN source | Original rows | After old class/turn/8-gram exclusions | After additional blinded-stem exclusion |
|---|---:|---:|---:|
| v3 integrated reasoning/dialogue | 30,000 | 8,637 | **8,637** |
| v3 natural communication | 30,000 | 9,132 | **9,132** |
| v3 reasoning | 50,000 | 11,995 | **11,937** |
| v3 preservation slice | 14,000 | 10,353 | **10,319** |

The three large sources therefore offer **29,706** rows after both protections, with zero ID overlap against the preservation pool. This is a text-contact filter, not a semantic contamination certificate or a solver review of these answers. The original pool is balanced by class; the filtered pool is not exactly balanced and should not be described as such.

The preservation slice contains 34 stem-contact rows: 20 `arithmetic_short_answer` and **14 `insufficient_information`**. For example, `rineya_grnc_v2_insufficient_information_00596` asks `What is 25% of 72?` and correctly answers `18.`. Its answer is correct; its class does not identify an abstention task. All 58 contact rows in the large reasoning pool are likewise labeled `insufficient_information`. Class-based rehearsal is not a semantic answerability label.

The preservation abstention selector yields **1,471 → 1,457** eligible rows after this correction. A future fixed-token draw must be rebuilt from the filtered pool, and its accepted rows, rejected rows and actual token totals frozen. Do not silently remove contacts from an already registered experiment.

**The gap reached historical training.** Directly reading retained compiled TRAIN files finds 12 contact rows in M4, six in M5, and seven in each of the three M1 mixtures. Historical score artifacts remain unchanged, but interpreting their panel gains as entirely off-task transfer is unsupported. This audit does not estimate the numerical effect of those contacts. The earlier statement that `splits_onpanel` never reached training concerns that particular bank and cannot establish that every later rehearsal corpus was contact-free.

Reproducible evidence, including source SHA256s, all excluded IDs, representative prompts and hashes of the consumed TRAIN files:

- [Full source contact census](DATASET_TEXT_CONTACT_AUDIT.json).
- [Historical consumed-contact census](DATASET_CONSUMED_CONTACT_AUDIT.json).
- [Reusable filter](../../../experiments/reasoning_posttraining_v1/text_contact.py): IDs, user-only eight-word spans and digit-blinded last-user panel stems; preserves source objects/order and records every exclusion. Eight focused CPU tests passed.

## Inventory and actual uses

| Asset | Evidence and usable scope | Constraint |
|---|---|---|
| Forge RL v2, release card dated September 14 | 16,000 canonical TRAIN worlds across eight families; 32,000 preference pairs; independently recounted **36,752 transition rows**, from 2,000 ledger, 2,000 register and 1,155 prerequisite parents | Derivatives are not independent worlds. Preserve parent grouping and exclude both process instruments. Source-reported oracle verification is stronger than unchecked generated prose, but does not prove broad efficacy. |
| SELECT-VIEWS-001, built September 16 | Recounted 384 rows, 384 parents, 128/family; all `ADMITTED_FOR_SELECTION`; SHA256 `7dd53f2a751b54c349c2ef571e8f9c5153737f4a1ed3df5eafe25f95980e78a3` | Selection/DEV only, never TRAIN or confirmation. Historical source-Supa arms do not license reuse of their trained weights as a fresh source. Future training must exclude these parents. |
| Large v3 TRAIN banks | 29,706 rows survive the combined guards above; 37 classes across three sources | Short-form content cannot establish long-context retention. The current source-specific preservation adapter must be extended explicitly before accepting a new source. |
| Preservation slice | 10,319 guarded single-turn rows; 1,457 in the metadata-based abstention selector | 3,402 of the original 10,353 eligible rows lack `capability_group`; filters on that field omit them. Audit actual answerability, not just class names. |
| Grounded replay | Retained admission audit verifies 4,189/4,320 rows, excludes 131: 129 same-result conflict labels and two unresolved latency ties | Use evidence/procedure rendering and its verifier. Bare conversation fields omit the evidence and are not a safe equivalent serialization. |
| CC0 long-context v1 | 44,016 TRAIN + 3,000 validation + 3,000 test; 10,000 DPO; release manifest dated August 25 | **Not admitted.** Existing full audit reports 0/50,016 fitting 2048 and a perfect question-only abstention cue. Arrival in the workspace is not the release date. |
| LCR SFT v1 | Generator and 12-pack/48-row sample exist; advertised full 8,000-row file is absent | Regeneration is possible CPU work, but would need tokenizer/shortcut/partition review; it is not an already qualified bank. |
| `../rineya-dataset-retention-records/lcr-v2-corpus` | At inspection, only `lcr/__init__.py`, `lcr/common.py` and bytecode; `data/` and `reports/` contain no files | A scaffold, not a delivered corpus or validation receipt. |

Source references: [Forge data card](../../../rineya_reasoning_forge_rl_v2/DATA_CARD.md), [CC0 audit](../../../LCR_CC0_AUDIT.md), [corrected LCR specification](../../../docs/LCR_GENERATION_PROMPT_V2_ADDENDUM.md), [replay admission audit](../../../research/reasoning_posttraining_v1/rrp005/REPLAY_ADMISSION_AUDIT.json), [quarantine inventory](../../../research/reasoning_posttraining_v1/QUARANTINE.json).

The Forge process file was independently counted and hashed: `386bc03c7a9a883f60986cfe59ea0cac2cce34ea7dbaec71595e018eb1e0ce43`. No dataset version/import date was inferred merely from a directory timestamp. Large source data are not necessarily represented by recent Git commits.

## LCR completion needs a corrected contract, not just more rows

The scaffold already has useful deterministic entity partitions, variant-invariant question checks and a bridge to `compile_row`. Its current `common.py` nevertheless does not meet the v2 addendum: it uses context-token bands 300–600/600–1100/1100–1600 instead of rendered-prompt bands 600–1000/1000–1400/1400–1920, four task names differ from the prescribed names, `CORPUS_VERSION` remains `lcr_v1`, and `verified_by` is left unset. Task generators, independent checker, allocation driver and receipt files are absent. Do not describe those intended checks as completed.

Complete a separate small pilot generator first, preserving the existing scaffold. Split scenario graphs, templates and lexicons **before** generation; keep answerable, missing-evidence and counterfactual descendants together. Require the real pinned tokenizer over every rendered row, independently derived answers, matched question distributions, surface-feature adversaries and recorded rejection reasons. A prompt contract containing `UNKNOWN` can be appropriate for TRAIN, but it cannot silently replace the separately specified sentinel-absent confirmation contract.

The prior survey's strongest defensible conclusion is that its **seven audited short-form corpora** cannot supply matched long-prose rehearsal. Raw long inputs do exist elsewhere (Forge and CC0). No audited, fitting, clean, matched long-prose rehearsal bank is currently delivered; that is narrower than claiming no long-context data exists anywhere.

## Experiment implications

1. Freeze a new future-use preparation contract with both process-view exclusions and the text-contact guard. Preserve historical builders/artifacts explicitly when reproducing old experiments. The new filtered mixture is a new scientific intervention.
2. Pair the same transition draw from exact frozen Supa with and without filtered abstention-shaped rehearsal. Freeze dose, supervised/processed tokens, source hashes and decision rules before training. This first pair is exploratory and resolves mechanics plus the direction of the trade; three independent draws are required before a replicated training-draw claim.
3. Report per-family acquisition against the instrument's **own** published floor using the exact Clopper–Pearson lower-bound predicate in [the erratum](../../../ERRATUM_FLOOR_RULE.md). Do not transcribe obsolete count triples or use the report-only process probe for selection.
4. Keep strict and permissive arithmetic, source successes retained/lost, literal versus semantic abstention, false abstention, termination and cost separate. Existing long-context outputs remain report-only; nonsignificance is not preservation. Gate v2 also requires arithmetic improvement over Supa, so passing it is not synonymous with avoiding forgetting.
5. Build fresh long-context/answerability DEV and protected confirmation partitions before a broad-retention training claim. Current spent or other-programme banks do not become admissible merely because a new model is being tested. Do not open protected outputs to choose a recipe.

Existing experiments show that atomic transition targets can be learned and that particular rehearsal/dose choices changed particular measured endpoints. They do **not** prove universal 5% supervision-density thresholds, a universally optimal 50/50 mix, that only one skill can ever be preserved, or that more data cannot help. Those statements in inherited documents extrapolate beyond the measured sample sizes and recipes. No current corpus or experiment covers every aspect of intelligence, multilingual ability, coding, open-world knowledge and long-horizon reasoning.
