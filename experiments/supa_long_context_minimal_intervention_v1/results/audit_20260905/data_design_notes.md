# Data and design audit, 2026-09-05

Reproduce from repository root:

```powershell
../Chat-Bubble/.venv/Scripts/python.exe experiments/supa_long_context_minimal_intervention_v1/scripts/audit_data_design.py
```

The adjacent `data_design_before_fixes.json` is the preserved **pre-fix** audit snapshot; `data_design.json` is the latest replay. It records the actual interpreter, source hashes, both canonical checkpoint hashes before and after (`canonical_checkpoint_unchanged: true`), every curriculum item and measured length, and one-based `per_example.jsonl` source lines. No model was loaded and no GPU was used. Re-running after changing runner constants measures the changed curriculum and would replace this snapshot; preserve it before doing so.

## 4.2 Design

**CONFIRMED — format/content confound.** `src/slcmi/context_selection.py::chunk_context` normalizes sentence boundaries; `assemble` adds newlines and blank-line gaps. A uses the untouched raw context. Reassembling **all** chunks, without deleting content, changes the raw text on 256/256 demo rows and 200/200 evaluated LCR rows. Reconstructing each cached condition with the current tokenizer exactly reproduces every cached selected token count (928 demo records, 750 LCR records). Demo mean tokens A/B/C/D: 1260.046875 / 560.33203125 / 543.94921875 / 541.19375; D has its own eligible denominator. LCR: 2296.365 / 524.45 / 520.06 / 523.5066666666667. Thus the report's LCR mean **2381 is REFUTED**. Mean per-row B retention is 45.05845% demo and 22.91986% LCR; a ratio of pooled means is a different statistic and must be labeled. B and C share chunk count, not exact token count, and their assembly family is matched while the number/placement of gaps can differ.

**REFUTED — causal closure and retrieval/capability ceilings.** B−A and D−A test the joint formatting/content intervention. B−C establishes a relative advantage of this query selector over its random control under approximately matched lengths. It does not isolate interference or repair the confounded A comparisons. D uses privileged metadata, but its worse score does not make it the optimum over all evidence subsets, formats, or retrievers. Retaining all annotated LCR facts does not establish the annotations' completeness or the optimum model input. Withdraw assertions that 38% is the best any retriever could do, that perfect retrieval leaves a fixed intrinsic-capability deficit, that a negative D−A closes the interference question, and that shorter context cannot expose capability. The differences between demo and LCR also change task composition, scoring targets, and oracle localization, so length is not separately identified.

**CONFIRMED with denominator correction — answer-informed oracle.** Case-insensitive gold span in at least one supporting fact: demo 168/192 answerable = 87.5%, and 144/160 eligible = 90%; full LCR dev 553/600 answerable = 92.1667%; **evaluated LCR sample 144/150 = 96%**. All 2688/2688 facts over the full 800-row dev split are verbatim; evaluated answerable sample has 507/507 verbatim facts. D preserves all those 507 facts on all 150 rows. Demo facts are 0/424 verbatim over all rows, 0/360 answerable, 0/304 eligible. The selector does not read `answer` directly, but its localization metadata often contains that answer.

**CONFIRMED disclosure; UNCERTAIN history — selector revision.** `preregistrations/phase0_context_refinement.md:65` discloses an earlier whole-chunk anchor and 768 ceiling, changed to first-sentence anchor and current 384/576 budget/ceiling. It does not record the old 512 budget. `git log --format="%H %s" -- experiments/supa_long_context_minimal_intervention_v1` returns no committed history. Manifests record different selector source hashes in early and later stages; that alone cannot reconstruct old bytes or prove how many changes occurred. The claimed CPU-only timing and exactly-one-revision history remain unverified. Selecting parameters after inspecting retained content is a material researcher choice even absent outcome labels; do not describe this as an immutable initial configuration or assert undocumented timing.

**CONFIRMED class mismatch/disclosed selection rationale; UNCERTAIN prior probe rates.** `src/slcmi/preservation.py:14` says classes were selected after model probes; actual `PRESERVATION_CLASSES` is `counting_sets, conditional_rules`, echoed by smoke records. The Phase 1 preregistration instead names `arithmetic_chain, counting_sets` at line 141. The recorded smoke output has 3/8 correct for each actual class, recomputed with `score_preservation`. No prior class-probe receipt is present in lab results, so the claimed 24-item 42%/38–42%/12%/0% probe rates and timing cannot be independently verified. Selecting non-floor classes can make regression detectable, but narrows coverage and conditions selection on observed baseline performance; the gate is a selected two-class tripwire, not evidence of general preservation. Amend the protocol to disclose the mismatch before interpreting a gate.

**CONFIRMED caveat; REFUTED blanket no-significance claim.** `RESULT.md:321` says cells are robustness, not independent confirmation; the amendment repeats that and no correction is implemented. Yet `RESULT.md:235–236` explicitly says "significantly worse" and "significantly better". Remove those phrases. Shared rows, contexts, and researcher-selected variants preclude treating results as independent replications. The LCR sample is 200 rows, while four core/gate cells share 256 demo rows; the statement that both suites are the same 256 rows is imprecise.

**CONFIRMED no lab validity gate; UNCERTAIN global baseline nonexistence.** `src/slcmi/suites.py` gives LCR no expected-count validity gate and replaces the original verbose accepted answers with `short_target`. However, the source corpus does ship `../Chat-Bubble/Rineya_LCR_SFT_v1/Rineya_LCR_SFT_v1/scripts/score_predictions.py`, an accepted-answer-or-all-required/no-forbidden lexical scorer. The lab's short-answer protocol is distinct; “no published baseline exists anywhere” cannot be established merely from absent lab counts.

## 4.3 Short-answer rule

**CONFIRMED mechanical uniqueness.** Every one of 6000 answerable rows in the full 8000-row corpus has exactly one `must_include` span absent from its question; every extracted target occurs in its original answer. This proves deterministic extraction, not unique semantic correctness of the scoring rule.

**CONFIRMED A sensitivity; complete all-condition sensitivity:** scores recomputed from raw outputs in `results/phase0_eval_lcr_dev/per_example.jsonl`, 150 answerable rows per condition:

| Condition | Strict | Gold substring anywhere | Difference |
|---|---:|---:|---:|
| A | 49/150 = 32.6667% | 52/150 = 34.6667% | +2.0000 points |
| B | 45/150 = 30% | 45/150 = 30% | 0 |
| C | 15/150 = 10% | 15/150 = 10% | 0 |
| D | 57/150 = 38% | 59/150 = 39.3333% | +1.3333 points |

Raw case-insensitive substring and normalized substring produce the same counts. Cached strict `correct` flags agree with rescoring. Strict D−A is +5.3333 points; lenient D−A is +4.6667 points. The lenient-only outputs add harmless descriptive prefixes: A lines 33, 102, 151; D lines 677, 712. Exact texts/IDs and all correct-row lines are in the JSON.

**Judgment — plausible protocol adaptation, not validated construct.** Short spans fit the requested shortest-answer prompt and avoid requiring question material to be repeated; this is a defensible rendering. It is stricter than semantic correctness, as the five condition-specific descriptive-prefix failures show. That small tested relaxation does not reverse D−A and cannot explain the low observed absolute scores by itself. It does not validate all rejected synonyms/partial-but-sufficient answers, justify “near floor,” or establish a capability ceiling. Use this scoring sensitivity as a limited robustness check, not proof that all other errors are substantive.

## 4.4 Disjointness and demo intended use

**CONFIRMED LCR versus demo disjointness.** Full LCR versus all demo rows has zero shared contexts, questions, titles, IDs, or scenario IDs. Demo `intended_use` is `demo/evaluation only; do not train on this exact suite` on all 256 rows. In the pilot, `phase1_run.py::target_items` feeds only target evaluation; training comes from `load_packs(LCR_SFT)` and `sample_curriculum(..., "train", ...)`, and dev feeds evaluation loss. `src/slcmi/training.py` uses only `train_set` for backward. No pilot code path trains on demo examples. Repository-wide prior checkpoint exposure is separately audited in `training_provenance.json`; this source-flow audit is not a claim about unknown external training history.

**CONFIRMED context/scenario split; REFUTED literal question disjointness within LCR.** Train/dev share zero contexts, titles, IDs, or scenario IDs, but **75 exact question strings** recur across different scenario contexts. This is template-level overlap, not identical training/evaluation examples. Do not claim complete question disjointness between LCR train and LCR dev.

## 4.5 Curriculum retention

**REFUTED — 2560 retains about 100% of the bounded curriculum.** The exact seeded samples rendered by `tokenize_example` using the canonical tokenizer and required interpreter yield:

| Set | Rows | Min / median / max tokens | Retained at 2048 | Retained at 2560 |
|---|---:|---:|---:|---:|
| Bounded train | 2400 | 1967 / 2243 / 3698 | 540 (22.5%) | 1759 (73.2917%) |
| Bounded dev | 160 | 2195 / 2365 / 2628 | 0 | 130 (81.25%) |
| Smoke train | 32 | 2001 / 2080.5 / 2482 | 8 (25%) | 32 (100%) |
| Smoke dev | 16 | 2202 / 2237 / 2308 | 0 | 16 (100%) |

The smoke did not expose the bounded-curriculum truncation defect. `tokenize_example` drops entire examples, so the retained curriculum can lose pack members and reweight task types. The JSON records each dropped example, variant, and task type. A fixed length sufficient for the measured maximum, within the user's 2–4K range, can repair this without a sweep; it must be disclosed and reverified before bounded training.

## Surviving interpretation

Survive: descriptive score differences for the tested inputs; query selection outperforming this random control; low observed LCR short-answer accuracy and limited sensitivity to the tested lenient scorer; exact LCR annotation preservation; demo exclusion from current training; deterministic short-target extraction. Do not survive: a causal interference verdict, perfect-retrieval/capability ceiling, pure context-length explanation, global preservation claim, exact-token matched controls, or the old curriculum-retention assertion. These corrections improve the experiment rather than constitute a failed audit.
