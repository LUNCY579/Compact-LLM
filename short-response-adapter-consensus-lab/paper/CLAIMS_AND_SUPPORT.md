# Claims and Support Ledger

No prose in the paper may state a stronger conclusion than the corresponding
row below. Machine artifacts, not this table, are the numeric source of truth.

| Proposed claim | Evidence | Dataset / role | N | Statistical support | Status |
|---|---|---|---:|---|---|
| The full artifact is complete, paired, unique, and hash-consistent | `reports/phase2_cornerstone_verification.json` | v2.2 TEST / audit | 4,050 | 30/30 integrity checks; 24 implementation tests | SUPPORTED |
| Hard K=7 beats the backbone | raw SHA `d373…c32d`; `sealed_statistics.json` | v2.2 TEST / confirmatory | 4,050 | +17.235 pp; exact McNemar (p=2.67\times10^{-208}); Holm (p=1.87\times10^{-207}) | SUPPORTED |
| Hard K=7 beats Track A on the evaluated finite corpus | same raw/statistics artifacts | v2.2 TEST / confirmatory | 4,050 | +2.667 pp; 95% paired bootstrap [1.531, 3.778] pp; Holm (p=1.36\times10^{-5}) | SUPPORTED |
| Hard K=7 beats the abstention individual | same raw/statistics artifacts | v2.2 TEST / confirmatory | 4,050 | +4.247 pp; Holm (p=5.42\times10^{-10}) | SUPPORTED |
| Hard K=7 beats raw probability K=7 on the evaluated finite corpus | same raw/statistics artifacts | v2.2 TEST / confirmatory | 4,050 | +3.160 pp; 95% paired bootstrap [2.321, 4.025] pp; Holm (p=4.36\times10^{-12}) | SUPPORTED |
| Hard K=7 generalizes over unseen reasoning families better than Track A | `data_independence_audit.json` | v2.2 TEST / cluster sensitivity | 15 reasoning families | reasoning-family cluster CI [−4.52, +13.70] pp; 114 structural-family CI [−2.62, +9.26] pp | UNRESOLVED |
| Hard K=7 generalizes over unseen structural template families better than probability mean | `data_independence_audit.json` | v2.2 TEST / cluster sensitivity | 114 heuristic families | structural-family cluster CI [+0.33, +5.80] pp; coarser reasoning-family CI [−1.63, +6.89] pp | PARTIALLY SUPPORTED |
| Hard's Track-A gain is rescue minus harm, not dominance | `existing_results_analysis.json` | v2.2 TEST / paired confirmatory global | 4,050 | 328 rescues, 220 harms, net +108 | SUPPORTED |
| Hard's probability-mean gain is rescue minus harm | `existing_results_analysis.json` | v2.2 TEST / paired confirmatory global | 4,050 | 226 rescues, 98 harms, net +128 | SUPPORTED |
| Answer/abstain policy composition materially affects the aggregate ranking | answer-class rescue/harm and behavior table | v2.2 TEST / descriptive stratification | 4,050 | Hard−Track A: −3.48 pp answerable, +42.59 pp abstention; Hard−Probability: +4.64 / −6.48 pp | PARTIALLY SUPPORTED |
| Hard/soft differences often begin before trajectory divergence | shared-prefix first-divergence audit | v2.2 TEST / descriptive | 4,050 | 1,804 divergent sequences; 1,502 first diverge at (t=0); zero shared-prefix top-token trace mismatches | SUPPORTED |
| Experts differ in directional confidence, entropy, and coalition participation | free-running trace profiles | v2.2 TEST / descriptive | 23,204 hard-token steps | Descriptive distributions; no gold-token calibration interpretation | SUPPORTED |
| Benign formatting/paraphrase errors explain the hard-versus-probability gap | `semantic_audit_adjudicated.json` | v2.2 TEST / secondary controlled error audit | 100 errors (50/method) | 0 semantic rescues; combined exact 95% CI [0.00%, 3.62%]; single reviewer | CONTRADICTED in audited sample |
| Hard top-1 proposals, vote counts, and unique-plurality winners are invariant to positive expert temperature scaling | monotonicity derivation, voting tests, `hard_temperature_tie_sensitivity_dev.json` | Mathematical / implementation + DEV boundary audit | 18,760 tokens | Exact property; all shared-T selected decisions unchanged | SUPPORTED |
| The deployed confidence-sum hard tie-break is invariant to unequal expert temperature scaling | `hard_temperature_tie_sensitivity_dev.json` | v2.2 DEV / local counterfactual | 1,097 tied positions | Fitted per-expert temperatures change 259 decisions; largest one-expert grid intervention changes 513 | CONTRADICTED |
| Calibration mismatch is sufficient to explain the free-running TEST gain | DEV temperature fit, same-split DEV free run, and frozen one-condition TEST confirmation | v2.2 DEV selection + TEST selected confirmation | 4,050 per split | Calibrated beats hard on DEV by +1.481 pp but trails hard on TEST by −3.432 pp | CONTRADICTED as a sufficient explanation |
| Calibrated probability fusion closes the hard-vote gap on v2.2 TEST | `v22_calibrated_test_analysis.json` | v2.2 TEST / selected confirmation with known-anchor and prior n=32 caveats | 4,050 | 13.827% calibrated versus 17.259% hard; −3.432 pp, 95% CI [−4.272, −2.617], Holm p=1.07e-15 | CONTRADICTED |
| Calibrated probability fusion closes the hard-vote gap on same-split DEV trajectories | `selected_dev_free_run_analysis.json` | v2.2 DEV / model selection | 4,050 | 26.790% calibrated versus 25.309% hard; +1.481 pp, 95% CI [+0.741, +2.222], Holm p=2.77e-4 | SUPPORTED on DEV |
| Robust reweighting, trimming, median pooling, or confidence clipping explains probability fusion's difference from hard voting | `robust_soft_controls_dev.json` | v2.2 DEV / post-selection exploratory robustness | 4,050; 18,760 tokens | Validation-weighted hard 24.77%, inverse-entropy 8.25%, trimmed mean 26.20%, median 13.73%, clip caps 0.5/0.7/0.9 26.44%/26.64%/26.62% versus raw 26.62% examples; median has 2 uncertified positions, all other winners certified | CONTRADICTED locally for the tested definitions |
| Rank information is sufficient | top-2 approval, Borda, reciprocal-rank cached/free-running results | v2.2 DEV / model selection | 4,050 | Free-running reciprocal rank 18.321% versus hard 25.309%; −6.988 pp, Holm p=2.19e-49 | CONTRADICTED |
| Hard suppresses overconfident mistakes | one-expert temperature perturbations plus divergence analysis | v2.2 DEV local + TEST descriptive | 18,760 tokens | Hard proposals/unique pluralities are invariant; unequal scaling can alter its confidence tie-break; soft fusion changes more broadly; no causal attribution to every corrected sequence | PARTIALLY SUPPORTED |
| Complementary specialist errors cause the gain | all-expert K1, 21-pair diversity/JS/MI/Jaccard, rescue, Shapley, and bounded negative controls | v2.2 DEV + TEST descriptive | 18,760 DEV tokens; 23,204 TEST steps; 8-example control packet | Dynamic token oracle 86.36%, +6.35 pp over best individual; maximum disagreement is negatively associated with utility at fixed K; duplicate General is exactly invariant, but fixed-pool proxy is not matched seed-only causal control | PARTIALLY SUPPORTED as potential, not cause |
| Specific specialist coalitions are causal | coalition frequencies plus cached and free-running leave-one-out/category deltas | v2.2 DEV / exploratory | 127 cached committees; 7 free-running removals | Track A/B/General/Abstention/Social are positive in deployed K7; Grounding is harmful; Procedure neutral; matched-training causality remains absent | PARTIALLY SUPPORTED for functional marginal contribution |
| Larger committees generally improve capability | all 127 cached subsets and selected free-running K panel | v2.2 DEV / exploratory/model selection | 127 cached; 7 selected free-running sizes | Free-running selected accuracy peaks at K4 27.333%, dips at K2/K5, and ends at K7 25.309% | PARTIALLY SUPPORTED, nonmonotonic |
| Teacher-forced selected best-at-K committees yield a favorable free-running quality/latency curve | `selected_committee_dev_analysis.json` | v2.2 DEV / model selection | 48,600 new rows plus reused K7 | Selected K4 is most accurate at 27.333%; size curve is nonmonotonic and latency/forwards increase with committee use | SUPPORTED as a selected-panel description |
| K=7 exhibits a threshold-like transition | committee-size trend tests | v2.2 DEV / exploratory | 127 committees | Best hard K4 27.31% exceeds full K7 25.26%; mean scaling is gradual/nonmonotonic | CONTRADICTED |
| The current confidence tie-break is a major source of the hard-vote gain | `tie_eos_counterfactuals_dev.json` | v2.2 DEV / local counterfactual | 18,760 tokens | Ties at 5.85% of positions; Track-A priority changes 675 tokens but shifts all-token example accuracy only +0.05 pp | CONTRADICTED locally |
| Restrictive EOS rules improve hard-vote local decisions | same counterfactual artifact | v2.2 DEV / local counterfactual | 18,760 tokens | Ordinary 25.26% examples; strict-majority 15.58%, confidence-threshold 12.47%, minimum-3 10.12% | CONTRADICTED locally |
| Hard corrects autoregressive trajectories | `selected_dev_free_run_analysis.json` teacher/free comparison | v2.2 DEV / model selection | 4,050 | Teacher and free ordering is unchanged; method accuracies move by at most 0.148 pp | NOT SUPPORTED |
| The abstention expert has positive marginal contribution | cached K7-minus-abstention, exact Shapley, and free-running removal | v2.2 DEV / exploratory/model selection | 4,050 | Free-running full-minus-K6 +2.370 pp overall and +28.889 pp on abstention; Holm p=1.97e-10 | SUPPORTED contextually |
| Each expert's sequence-level marginal effect and latency cost match its gold-prefix leave-one-out value | `selected_committee_dev_analysis.json` | v2.2 DEV / model selection | 7 paired K6 comparisons | Grounding harm is consistent, but Social changes sign and Procedure is neutral free-running; costs are measured rather than imputed | CONTRADICTED as a uniform correspondence |
| Adaptive consultation preserves K=7 overall quality at lower measured cost | `adaptive_free_run_dev_analysis.json` | v2.2 DEV / model selection | 4,050 | −0.198 pp, 95% CI [−0.765,+0.370], p=0.546; forwards −56.579%; latency 0.928 s versus 1.919 s | SUPPORTED on overall DEV; class tradeoff disclosed |
| The effect transfers under fixed methods to v2.1.2 TEST | `cross_v212_analysis.json` and provenance gate | v2.1.2 TEST / cross-corpus, not globally untouched | 1,320 | Hard 0.076% versus raw 3.030%; −2.955 pp, Holm p=1.53e-10 | CONTRADICTED |
| Hard K=7 is deployment-efficient | quality/latency/forward frontier including adaptive run | v2.2 TEST + DEV systems panel | 4,050 per split | Full K7 costs substantially more than K1; adaptive preserves overall DEV accuracy with 56.579% fewer forwards | CONTRADICTED for full K7 |
| Findings generalize across backbones, scales, or PEFT types | matched experiments | Not run | — | None | UNRESOLVED |

Final evidence bindings: `selected_dev_free_run_analysis.json`,
`adaptive_free_run_dev_analysis.json`, `selected_committee_dev_analysis.json`,
`v22_calibrated_test_analysis.json`, `cross_v212_analysis.json`, and
`final_synthesis_facts.json`.

## Status vocabulary

- **SUPPORTED:** direct evidence justifies the stated scope.
- **PARTIALLY SUPPORTED:** some predicted evidence exists, but a causal or
  sufficient-mechanism claim is not justified.
- **PRELIMINARY:** bounded pilot evidence only.
- **UNRESOLVED:** required evidence has not been measured.
- **CONTRADICTED:** current evidence runs against the proposed claim.
