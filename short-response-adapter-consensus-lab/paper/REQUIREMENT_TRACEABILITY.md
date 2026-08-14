# Phase 2 Requirement Traceability

This matrix maps every numbered part of the governing Phase 2 specification to
authoritative current-state evidence. It is a completion aid, not a substitute
for the machine audit in `reports/phase2_completion_audit.json`. Status meanings:

- **COMPLETE**: the requested artifact or analysis exists at its intended scope.
- **BOUNDED**: evidence exists only at a deliberately limited exploratory scope.
- **DEFERRED / NOT CLAIMED**: a conditional or broader experiment was not run;
  the resulting claim remains explicitly unresolved.

| Part | Requirement | Authoritative evidence | Status / boundary |
|---:|---|---|---|
| I | Preserve and independently verify the 4,050-example cornerstone before interpretation | `reports/phase2_cornerstone_verification.json`; `reports/PHASE2_CORNERSTONE_VERIFICATION.md`; raw SHA256 `d3733391b53dc53e7d9ca899d7015ba6ff0b2faa7c28e52a2778a0827c45c32d` | COMPLETE; PASS 30/30, 32,400 unique condition rows |
| II | Preserve the n=32 pilot and explain the ranking reversal | `results/archive/sealed_n32/`; `paper/PAPER_ANALYSIS.md`; Figure 8 | COMPLETE; pilot is feasibility evidence, not method ranking |
| III | State the primary mechanistic question | `RESEARCH_QUESTIONS.md`; `paper/PAPER_ANALYSIS.md` | COMPLETE |
| IV | Formalize hard token voting | `paper/METHODS_AND_MATH.md`; Figure 1 | COMPLETE |
| V | Prove and test calibration invariance | `paper/METHODS_AND_MATH.md`; `teacher_forced_dev_analysis.json`; `hard_temperature_tie_sensitivity_dev.json`; voting tests | COMPLETE: proposals/vote counts/unique pluralities invariant; deployed unequal-temperature confidence tie-break explicitly not invariant |
| VI | Prestate competing hypotheses H1-H10 | `RESEARCH_QUESTIONS.md`; `paper/PAPER_ANALYSIS.md`; `reports/PHASE2_MECHANISM_REPORT.md` | COMPLETE; unresolved hypotheses remain labeled |
| VII | Respect TRAIN/DEV/TEST roles over 40,500 v2.2 examples | `paper/EXPERIMENTAL_PROTOCOL.md`; `paper/EVIDENCE_LEDGER.md`; split hashes | COMPLETE |
| VIII | Add cross-benchmark confirmation | cross gate; `cross_v212_analysis.json` | COMPLETE; fixed-method transfer is negative and is not a globally untouched holdout |
| IX | Paired rescue/harm analysis | `existing_results_analysis.json`; `rescue_harm_strata.csv`; `PHASE2_RESCUE_HARM.md` | COMPLETE on full TEST |
| X | Token-level mechanisms and divergence | full TEST traces in `existing_results_analysis.json`; divergence report/figure; `selected_dev_free_run_analysis.json` | COMPLETE descriptively; same-split evidence does not support a trajectory-correction claim |
| XI | Compare teacher-forced and free-running behavior on the same examples | teacher cache/analysis; `selected_dev_free_run_analysis.json`; Figure 9; Table 10 | COMPLETE on 4,050 paired DEV examples; ordering is preserved |
| XII | Expert calibration study | `teacher_forced_dev_analysis.json`; `confidence_conditioned_correctness_dev.json`; `robust_soft_controls_dev.json`; `v22_calibrated_test_analysis.json`; Table 5; Figure 4 | COMPLETE; calibrated soft beats hard on DEV but not v2.2 TEST, so calibration is insufficient |
| XIII | Rank-based and robust aggregators | all 1,143 committee-aggregator records; bounded robust controls; selected reciprocal-rank free run | COMPLETE; median is approximate at 2/18,760 uncertified positions; free-running rank sufficiency contradicted |
| XIV | Exhaust all nonempty committees | cached all-127 artifacts; `selected_committee_dev_analysis.json`; appendix CSVs | COMPLETE cached for all 127 subsets and nine aggregators; complete 48,600-row selected trajectory/systems panel |
| XV | Analyze committee-size scaling | cached scaling artifacts; `selected_committee_dev_analysis.json`; Table 7; Figure 3 | COMPLETE; cached and free-running selected curves are nonmonotonic and K=4 is best |
| XVI | Leave-one-out expert ablations | cached diagnostics; `selected_committee_dev_analysis.json`; Table 8 | COMPLETE local and free-running paired seven-removal family with systems/category deltas |
| XVII | Exact Shapley/marginal contributions | same analysis; appendix Shapley figure | COMPLETE locally |
| XVIII | Coalition analysis | full TEST coalition CSV/profiles; `coalition_strata_full_test.json`; coalition heatmap | COMPLETE across task, difficulty, answer class, domain, token position, EOS, and success/failure association; no causal interpretation |
| XIX | Error correlation and competence-diversity | `teacher_forced_expert_diversity.csv`; `pairwise_diversity_extended.json`; appendix diversity figure | COMPLETE for disagreement, double fault, Q, correctness phi/MI, success/failure Jaccard, complementary error, and TEST-path JS; maximum-disagreement hypothesis contradicted |
| XX | Controlled expert-temperature interventions | frozen 11-point grid in `teacher_forced_dev_analysis.json`; tie-boundary grid; Table 9 | COMPLETE locally for each expert, including strong/weak/reasoning/abstention roles |
| XXI | If technically safe, vary LoRA adapter dose | no Phase 2 dose artifact; excluded `v22-specialist-mean-dose-sweep` was not read or used | DEFERRED / NOT CLAIMED; conditional broad sweep, retained as future work |
| XXII | Distinguish weight-space soup from decision-space composition | `paper/METHODS_AND_MATH.md`; `RESULTS_ANALYSIS.md`; D_greedy provenance audit | COMPLETE conceptually/contextually; no matched-lineage causal superiority claim |
| XXIII | Fit and confirm adaptive consultation below full K=7 cost | `adaptive_teacher_forced_dev.json`; `adaptive_free_run_dev_analysis.json` | COMPLETE; overall DEV accuracy is statistically indistinguishable with 56.579% fewer forwards; class tradeoff reported |
| XXIV | Paired bootstrap, McNemar, Holm, effects, strata, clustered sensitivity | sealed statistics; rescue analysis; selected, v2.2-calibration, committee, adaptive, and cross analyzers; independence audit | COMPLETE for every declared family |
| XXV | Audit duplicates, templates, near-duplicates, and effective dependence | `data_independence_audit.json`; `PHASE2_DATA_INDEPENDENCE.md` | COMPLETE sensitivity audit; template families are deterministic heuristics |
| XXVI | Categorize and preserve failures | `qualitative_selection.json`; `qualitative_teacher_forced_selection.json`; failure and teacher-case reports | COMPLETE for full-TEST sequences and DEV gold-prefix local decisions |
| XXVII | Secondary semantic audit without replacing exact match | `semantic_audit_adjudicated.json`; `PHASE2_SEMANTIC_AUDIT.md` | COMPLETE, 100 cases; single-reviewer limitation |
| XXVIII | Build every named paper-package file | `paper/` inventory plus machine audit | COMPLETE |
| XXIX | Titles, abstract framework, introduction, hypotheses, contributions | `paper/PAPER_ANALYSIS.md` | COMPLETE as paper-ready drafting material |
| XXX | Primary-source related work and narrow novelty statement | `RELATED_WORK.md`; 25-entry `REFERENCES.md`/`references.bib` | COMPLETE; ELREA treated as closest prior art |
| XXXI | Methods, mathematical definitions, cost, justification | `paper/METHODS_AND_MATH.md` | COMPLETE |
| XXXII | Publication-quality architecture/pipeline figure specifications and renders | `FIGURE_PLAN.md`; `paper/figures/`; metadata hashes | COMPLETE including teacher/free Figure 9 and free-running Figure 3 panel |
| XXXIII | Datasets, models, hyperparameters, and infrastructure | `EXPERIMENTAL_PROTOCOL.md`; `PHASE2_ENVIRONMENT.md`; environment JSON | COMPLETE |
| XXXIV | Paper-ready comparative tables and required results analyses | `TABLES.md`; generated CSVs/metadata; mechanism report | COMPLETE for Tables 1–12 |
| XXXV | Causal ablation categories | full DEV subsets/LOO/temperature/tie/EOS/rank; `negative_controls_bounded_dev.json` for duplicate, weak injection, overconfidence, fixed-pool proxy | COMPLETE at full scope where promoted; BOUNDED for negative controls; adapter dose DEFERRED per Part XXI |
| XXXVI | Representative qualitative cases | `QUALITATIVE_ANALYSIS.md`; two deterministic selection JSON artifacts | COMPLETE for nonempty full-TEST and gold-prefix buckets |
| XXXVII | Discussion: what hard voting appears to do and does not prove | `RESULTS_ANALYSIS.md`; `PHASE2_MECHANISM_REPORT.md` | COMPLETE final bounded synthesis |
| XXXVIII | Explicit limitations | `LIMITATIONS_AND_ETHICS.md`; claims ledger | COMPLETE and conservative |
| XXXIX | Ethics statement | `LIMITATIONS_AND_ETHICS.md` | COMPLETE |
| XL | Future work | `FUTURE_WORK.md` | COMPLETE; dose and matched-seed causal controls retained rather than overstated |
| XLI | Verified references and appendix plan/content | bibliography files; appendix CSVs; figure/table plans | COMPLETE |
| XLII | Claim/evidence ledger | `CLAIMS_AND_SUPPORT.md` | COMPLETE/current, with supported, contradicted, and unresolved causal claims explicit |
| XLIII | Trace every number to raw evidence, hashes, and generators | `EVIDENCE_LEDGER.md`; table/figure metadata | COMPLETE for populated blocks |
| XLIV | Keep official repository read-only and ignore two mini-labs | before/after manifests; protected Git status; completion audit | COMPLETE |
| XLV | Durable, resumable, duplicate-safe live execution | append/fsync I/O, PID locks, runtime manifests, live reports, `run_remaining_pipeline.py` | COMPLETE; all strict runs closed, including explicit post-hoc reconciliation of the final 61 committee rows after an unclean worker exit |
| XLVI | Use staged compute-aware priority and stop unsupported grids | chronological manifests/reports; dose deferral; selected four-method DEV run | COMPLETE process discipline |
| XLVII | Explicitly answer 20 final mechanism questions | `reports/PHASE2_MECHANISM_REPORT.md` | COMPLETE; questions 1–20 finalized from long-run evidence |
| XLVIII | Prove the definition of done | `audit_phase2_completion.py`; `reports/phase2_completion_audit.json` | COMPLETE only when the machine audit status is PASS; narrative status never overrides it |

Final long-run bindings: `selected_dev_free_run_analysis.json`,
`adaptive_free_run_dev_analysis.json`, `selected_committee_dev_analysis.json`,
`v22_calibrated_test_analysis.json`, `cross_v212_analysis.json`, and
`final_synthesis_facts.json`.

## Deliberate non-claims

The package does not claim that specialist semantics, rather than training seed
or competence, causally produced the gain; no matched same-task multi-seed
family exists. It does not claim that a LoRA-dose threshold was measured, that
D_greedy is a matched weight-space control, that v2.1.2 is a globally untouched
holdout, or that full K=7 is deployment-efficient. These are scientific
boundaries, not missing values to be inferred from adjacent experiments.
