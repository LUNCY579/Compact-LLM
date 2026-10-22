# Complete 288-case generation comparison: no supported adaptive benefit

Independent raw result acceptance is complete: review74a4aaa4 reports no material blocker. Native execution, corrected reader and author math recomputation are separately documented.

The same frozen Rineya checkpoint completed 288 cases / 144 related parents: 192 development cases and 96 exploratory validation cases across six shared authored generators. The original TAPS controller does not improve strict correctness over the DEV-selected fixed scale or the matched random control. Its small gains over the intact model coexist with losses of previously correct answers and more unsupported answers. No controller is promoted.

## Frozen quality comparison

| Policy | DEV correct /192 | Exploratory VAL correct /96 |
|---|---:|---:|
| Intact, no extra block | 63 | 33 |
| Unit extra block | 69 | 34 |
| Fixed 0.8 | 67 | 30 |
| Fixed 0.9 | 68 | 31 |
| Fixed 1.0, DEV winner | 69 | 34 |
| Fixed 1.1 | 68 | 34 |
| Fixed prior mean 1.1473425965 | 67 | 34 |
| Fixed 1.2 | 66 | 34 |
| Original TAPS | 67 | 34 |
| Matched random, seeds 17/29 averaged | 67 | 34 |
| Retrospective six-constant oracle |70|35|
| Own-bank query-blind last-entity solver |71|42|

Alpha 1.0 was selected from DEV and frozen before any validation. The retrospective validation fixed envelope is also 34/96; several larger scales tie it, but none replaces the frozen comparator. Random seeds are averaged within items, not counted as independent cases. The 513-entry A5 pool contains only generation-prefix observations from DEV TAPS, so longer responses carry more marginal weight; teacher-prefix diagnostics never enter it. Actual DEV scales span 1.1202288-1.1493279 with mean 1.1365496: the policy is a narrow, positive over-relaxation, with no bound saturation. This observed pool mean was not added to the fixed grid after outcomes.

The finite constant-selector oracle adds only one case in each split: headroom 0.521 percentage points on DEV and 1.042 points on VAL, below the preregistered five-point practical margin. TAPS capture is -2 on DEV and 0 on VAL. This oracle is not an upper bound on continuous scaling or token schedules. No confidence interval, confirmatory p-value or noninferiority claim is made: 96/48 parent units share six generators and the exposed surfaces cannot establish independent source confirmation. Equal-category/parent paired counts remain in REPORT.json.

## Patterns, paired gains and preservation

TAPS versus frozen fixed has zero gains / two losses on DEV, and one gain / one loss on VAL. The validation dialogue-memory gain is offset by a document-retrieval loss: dialogue_memory:20:v1 changes UNKNOWN to the correct 6455; document_retrieval_reconciliation:18:v1 changes a correct UNKNOWN on CONFLICT evidence to unsupported 312. TAPS and matched random have identical scored case endpoints in both splits. The detailed first-logit differences are secondary and do not establish useful reasoning.

TAPS versus intact has 12 gains / 8 losses on DEV, retaining 55/63 intact successes, and 7 gains / 6 losses on VAL, retaining 27/33. Unit also loses intact successes (57/63 retained on DEV). Required observed preservation screens fail; no rejected arm is described as preserving all original capabilities.

| Validation family (16 cases / 8 parents each) | Intact | Unit / frozen fixed | TAPS / random |
|---|---:|---:|---:|
| answerability source conflict | 8 | 10 | 10 |
| deduction composition | 9 | 8 | 8 |
| dialogue memory | 8 | 8 | 9 |
| document retrieval reconciliation | 8 | 8 | 7 |
| program state | 0 | 0 | 0 |
| signed ledgers | 0 | 0 | 0 |

Signed ledgers score 0 in both splits; program-state tracking is 1/32 for unit/TAPS on DEV and 0/16 on VAL. These dominant failures are not repaired by changing step size. Every best model score is below its own query-blind floor, so isolated corrections do not establish semantic reasoning beyond the structural shortcut. The positive-control solver verifies the bank; its success is not attributed to Rineya.

Unsupported canonical answers on unanswerable cases rise from 0/48 intact to 1/48 unit and 3/48 TAPS on DEV; on VAL they rise from 0/24 to 2/24 unit and 3/24 TAPS. TAPS false UNKNOWN responses on answerable cases are 42/144 on DEV and 19/72 on VAL. Format-valid outcomes are 191/192 for unit/TAPS on DEV, and 93/96 unit versus 91/96 TAPS on VAL; intact has 95/96 VAL. Separately, recorded EOS endings are 192/192 intact and 191/192 unit/TAPS on DEV, and 96/96 intact/unit versus 95/96 TAPS on VAL. TAPS has one validation cap stop; unit and intact have none. Format validity, EOS termination and correct answers are distinct endpoints. Literal UNKNOWN and the fixed canonical contract are measured; broader semantic abstention is not human-adjudicated.

## Mechanism and hypotheses

The separate 192-case CPU causal replay finds positive input-only coherence in every case; sign agreement 91/188 reliable directions equals constant increase, while constant decrease agrees with 97/188. Increasing alpha by .01 harms canonical UNKNOWN likelihood on all 48 unanswerable cases. Thirty-four of 96 paired parents prefer opposite local directions. These are local teacher-prefix measurements, not generated-output flips. The complete generation comparison now separately demonstrates no adaptive capture and actual abstention/format harms. EOS precision ordering in the CPU surrogate remains UNVERIFIED; the old null report is preserved.

| Hypothesis | Supported finding / limit |
|---|---|
| H1 scale calibration | No tested constant beats unit on DEV; fixed 1.0 is the selected control. |
| H2 adaptive headroom | Finite constant-grid headroom is one case per split; practical useful headroom unsupported. Continuous/schedule policies remain untested. |
| H3 causal prediction | Frozen coherence sign supplies no discrimination beyond constant increase on the qualified CPU surrogate. |
| H4 controller capture | No gain over the frozen fixed control; no gain over matched random; preservation fails. |
| H5 replication / preservation | Shared generators and prior exposure limit inference; retention and abstention regress. No candidate advances to confirmation. |
| H6 quality / cost | No acceptable quality-and-preservation gain; target-quality speedup is not established. |

## Qualified execution, actual cost and lineage

CPU and GPU results are not pooled into a quality or timing claim. The generation host is CUDA FP32/eager, TF32 off, full-prefix batch one, no KV cache, with the original first pass and at most one extra whole L12-L13 block. L14-L27 and the custom UNKNOWN readout stay intact. Six registered constants, original TAPS, A0/A1 and seeded A5 recompute their own trajectories. Scientific generation cap 8 and inherited SMOKE cap 24 are unchanged; max357 qualifies the mechanical envelope. No deeper native recurrence, fit, optimizer, download, protected/native TEST, teacher/cloud or deployment occurred.

All 3,296 physical evaluations comprise 3,168 scientific cells and 128 SMOKE cells. There are 9,200 actual API attempts: 9,198 full-forward attempts, 9,197 completed forwards and two precompute cache refusals. Generation contributes 9,180 tokens/calls; 1,144,030 completed prefix positions are measured. One expected forced partial exception has unknown partial positions, not zero. Diagnostic oracle/alias rows add no model calls.

Parent-charged native time is 743.390 seconds (12.39 minutes), worker parent-clock time 742.187 seconds. All three integrity boundaries preserve loaded state b23, ten assets, 107 admitted sources, custom code b77, frozen parameters and absent parameter gradients. Prefix byte/line/hash commitments place checks before fixed/pool freeze, before first VAL and after all cells. Checks themselves add zero model calls and cost 12.054 seconds. Peak sampled process-tree RSS is 2,898,788,352 bytes; peak allocated GPU memory 2,627,059,200 bytes. All resource floors and ceilings pass. Cumulative native use is 3,581.233/21,600 seconds; 18,018.767 remains.

Instrumentation and synchronized timing are charged. Sixteen inherited-reference rows have missing controller/logging timing, so complete totals for those fields remain null and known subtotals remain explicit. No production speed or CPU/GPU speed comparison is claimed. The single exact 2f42 allowance is consumed; session69295 is terminal and no worker/lock remains. The successful run does not establish a repaired driver or universal long-run stability; the earlier CUDA/TDR failure is preserved.

The first frozen reader9c95 emitted a primary-null report because four offline helpers were absent from the narrower native admission. Their exact bytes were recorded in the supervisor prelaunch manifest CDC. Separate post-terminal reader5bc binds only those four files there, retains all original native/scientific gates and changes no scientific helper/body. Independent source review6b766 and policy11 acceptance da490 precede the exclusive corrected analysis. The original reader, null report06b and native inputs stay unchanged in their original paths and a 22-input archive. No model retry followed this reporting error.

## Decision and reproducibility

This is a useful restricted negative outcome for the original coherence controller and this frozen operator/task bank. Stop the scale-estimator ladder here: no fitting, finer grid, deeper recurrence or operator/readout change is justified by these results. The evidence favors investigating source/query grounding and arithmetic/program execution with strong ordinary/stateless comparators rather than treating unsigned update energy as task progress. Those alternatives remain unimplemented and unmeasured; any new intervention needs its own scope, prospective controls, preservation gates and complete forecast. Keep the deployed model unchanged.

Controlling corrected artifact: analyses/bulk-restart-manifest-reviewed-001/REPORT.json SHA67c7d5d9fc128b148afc6d056cb26f9ecef4e05dde78a7405358125250cb9d21. Native execution audit: audits/full_generation_restart_native_execution_001.json. Author math recomputation5d966636 and independent adversarial acceptance74a4aaa4 are separate completed artifacts with explicit roles. Figures are exported from that exact report under analyses/bulk-restart-001-figures-002; the initial layout draft is retained.

Reproduce the source-bound offline analysis with the existing interpreter and a new exclusive output directory:

    & C:\Users\clclc\BusinessProject\Chat-Bubble\.venv\Scripts\python.exe -B -m experiments.rineya_taps_adaptive_recurrence_campaign_v2.bulk_restart_analysis_manifest_bound --run-root C:\Users\clclc\BusinessProject\Chat-Bubble-rineya-taps-adaptive-recurrence-v1\experiments\rineya_taps_adaptive_recurrence_campaign_v2\runs\native-bulk-restart-001 --out C:\Users\clclc\BusinessProject\Chat-Bubble-rineya-taps-adaptive-recurrence-v1\experiments\rineya_taps_adaptive_recurrence_campaign_v2\analyses\bulk-restart-manifest-reproduction-001

Do not rerun the consumed native grant. The actual run, source/test/review/manifest lineage, old failed runs, policies and case/journal tables remain in the campaign; STATE/CONTINUE and the final reproducibility index identify the last reviewed epoch.
