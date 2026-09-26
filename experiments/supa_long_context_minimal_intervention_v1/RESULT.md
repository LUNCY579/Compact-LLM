# RESULT — Supa Long-Context Minimal Intervention Pilot

Audit date: 2026-09-05. Execution update: 2026-09-06 UTC. **Phase 1 is COMPLETE on both cores: all four stages audited VALID/SMOKE_ONLY, every adapter fails both gates, winner FROZEN, Phase 2 not licensed. A follow-up single-variable ablation shows the target-metric failure is largely an abstention-gate artifact — see `ablations/gate_off_nora_v1/RESULT.md` and §"Gate-off ablation" below.**
The old report is preserved in `results/audit_20260905/RESULT_before_audit.md`.
Refuting and correcting a prior claim is a successful audit outcome.

## Conclusions that survived

The recorded Phase 0 predictions reproduce the stated published baselines.
The tested query selector scored below full context on answerable examples in
both suites and above the random control. Context reduction also increased
abstention on answerable examples. These are descriptive results for tested inputs.

**Withdrawn:** interference has been ruled out; D establishes a perfect-retrieval
ceiling; retrieval cannot help and intrinsic capability is the missing ingredient;
checkpoint choice has no effect; or context length explains the difference between
suites. A versus B/D changes formatting and content together. Answer-informed
selection is not the optimum over every input or retriever. Suite, tasks, scorer,
and localization also change along with length. No statistical significance is claimed.

**The same-LCR prior-training hypothesis is REFUTED.** The merged adapter's
cached Stage A training mix is CC0 long-context data plus short-response replay,
not the LCR SFT corpus selected for Phase 1. The bounded empirical adaptation
premise survives. A null would apply to this fixed recipe and budget, not disprove
late-layer adaptation as a method. Phase 0 does not establish that adaptation must work.

## Evidence and reproduction

Run these commands from repository root in PowerShell; P/N/D/V/T identify the
commands used in the checklist. All score arithmetic is recomputed from raw
predictions in `results/phase0_eval*/per_example.jsonl`, including rescoring against
accepted answers. Summaries are comparison targets, not inputs to the arithmetic.
Corpus metrics instead use raw corpus JSON and the real tokenizer; adapter metrics
come from the newly executed verifier. Unrecomputed old resource, power, and
merged-delta rounding estimates are omitted.

```powershell
# P — actual cached training mix and source identity
& '../Chat-Bubble/.venv/Scripts/python.exe' 'experiments/supa_long_context_minimal_intervention_v1/scripts/audit_training_provenance.py'
# N — raw prediction rescoring and independent arithmetic
& '../Chat-Bubble/.venv/Scripts/python.exe' 'experiments/supa_long_context_minimal_intervention_v1/scripts/audit_result_numbers.py'
# D — raw corpus, scorer sensitivity, and actual tokenizer
& '../Chat-Bubble/.venv/Scripts/python.exe' 'experiments/supa_long_context_minimal_intervention_v1/scripts/audit_data_design.py'
# V — fresh CPU adapter verification
$env:CUDA_VISIBLE_DEVICES = '-1'
& '../Chat-Bubble/.venv/Scripts/python.exe' 'experiments/supa_long_context_minimal_intervention_v1/scripts/phase1_verify_adapters.py'
Remove-Item Env:CUDA_VISIBLE_DEVICES
# T — mechanical checks
& '../Chat-Bubble/.venv/Scripts/python.exe' -m pytest 'experiments/supa_long_context_minimal_intervention_v1/tests' -q
```

Receipts and exact source paths: `results/audit_20260905/numbers.json`,
`data_design.json`, `data_design_before_fixes.json`, `training_provenance.json`,
and `*_notes.md`; fresh `results/phase1_adapter_verification/receipt.json`.
The old smoke is preserved under `results/audit_20260905/prior_phase1_smoke`.

## Audit checklist — HANDOFF section 4

| Claim | Status | Evidence / settling command |
|---|---|---|
| 4.4a: A1.1 lineage is Nimbus plus Stage A checkpoint-200 | CONFIRMED as recorded lineage | P; canonical `Rineya-Supa-A1.1-1.1.0-devfreeze/CORE.json::source`. No historical remerge was performed. |
| 4.4a: that adapter trained on this same LCR SFT corpus | REFUTED | P; actual `../Chat-Bubble/artifacts/halo_long_context_sft_v1/stage_a/mix/train_stage_a.jsonl`, exact source-message matches, mix builder/config, launch log. Details below. |
| 4.4b: no shared LCR/demo contexts, questions, titles | CONFIRMED | D; raw LCR and demo JSON have zero overlap in these fields, IDs, and scenario IDs. |
| 4.4c: demo evaluation-only restriction respected in pilot | CONFIRMED | D; raw `intended_use`, `phase1_run.py::target_items` and `training.py` data flow. Only LCR train rows feed backward. No assertion about unknown upstream history. |
| 4.2a: A versus B/D confounds format and content | CONFIRMED | D; `context_selection.py::chunk_context/assemble`. Even reassembling every chunk changes all evaluated raw contexts. |
| 4.2a: B/C/D comparisons repair the causal inference against A | REFUTED | Same code/data. B/C share assembly rules and chunk count, not exact token count; their contrast cannot isolate the simultaneous A-versus-B/D changes. |
| 4.2b: oracle is answer-informed | CONFIRMED | D; raw `supporting_facts` and gold-span overlap. Evaluated versus full-dev denominators differ; see below. |
| 4.2c: selector revision disclosed | CONFIRMED with omission | `preregistrations/phase0_context_refinement.md` discloses anchor and old ceiling changes; old 512 budget is absent. D records available source hashes. |
| 4.2c: exactly one revision, CPU-only, before model runs | UNCERTAIN | `git log --format='%H %s' -- experiments/supa_long_context_minimal_intervention_v1` has no committed lab history. Prose/digests cannot reconstruct exact chronology. Inspecting retained content is a material researcher choice. |
| 4.2d: preservation classes chosen by probing baseline | CONFIRMED as disclosed rationale; historical rates UNCERTAIN | `preservation.py` describes selection; D finds no class-probe receipt establishing quoted rates/timing. |
| 4.2d: actual classes match preregistration | REFUTED | D; preregistration names arithmetic_chain/counting_sets; source and original smoke use counting_sets/conditional_rules. Repository wins. |
| 4.2d: selected preservation gate proves general preservation | REFUTED | Selected non-floor classes make regression detectable but restrict coverage and condition it on baseline performance. This is a narrow tripwire. |
| 4.2e: multiple comparisons, shared-cell dependence acknowledged | CONFIRMED | N; `stats.py`, amendment and archived report. No multiplicity correction. Four cells reuse demo examples; LCR is a distinct sample. |
| 4.2e: old report made no significance claims | REFUTED | Archived report explicitly says “significantly worse/better.” Withdrawn here. |
| 4.2f: LCR has a published short-answer validity baseline | REFUTED for this harness; global absence UNCERTAIN | `suites.py`, `phase0_run_eval.py` have no LCR expected counts. Original corpus has a different `scripts/score_predictions.py`; absence everywhere is not established. |
| 4.3a: residual target unique on every answerable LCR row | CONFIRMED mechanically | D; `training_data.py::short_target`, all raw answerable rows. Mechanical uniqueness is not semantic validation. |
| 4.3b: A strict/lenient difference about two points | CONFIRMED | D/N; raw `phase0_eval_lcr_dev/per_example.jsonl`. Full condition sensitivity below. |
| 4.3c: fair short scoring establishes intrinsic ceiling | UNCERTAIN construct validity; REFUTED ceiling | Short spans fit the brief-answer prompt and occur in original answers. One lenient check does not validate synonyms, all semantic answers, annotation completeness, or retrieval optima. |
| 4.1a: primary A reproduces 45/125/170 | CONFIRMED | N; `phase0_eval/per_example.jsonl`, raw rescoring and split counts. |
| 4.1b: A1.1 ungated reproduces 41/121/162 | CONFIRMED | N; `phase0_eval_a11_ungated/per_example.jsonl`. |
| 4.1c: A1 gated reproduces 43/125/168 | CONFIRMED | N; `phase0_eval_a1_gated/per_example.jsonl`. |
| 4.1d: stored summary/analysis contrasts match raw arithmetic | CONFIRMED | N reports zero summary, analysis, or amendment arithmetic discrepancies; prose errors remain, corrected below. |
| 4.5a: PEFT targets L17/18/19 attention only | CONFIRMED at initialization | V; fresh receipt, `adapters.py::build_lora_config/verify_adapter`, installed PEFT checked in mechanics notes. |
| 4.5b: layers 0/1/2 refused | CONFIRMED | V plus separate calls for each protected layer, reproduced in `mechanics_notes.md`; `build_lora_config` raises. No pilot path removes or loops layers. |
| 4.5c: gate OFF training, ON every evaluation | REFUTED literally in audited code | `training.py` ran validation loss inside `gate_disabled`; generation had ON assertion. Canonical modeling computes loss before offset. Repair restores gate for validation too. |
| 4.5d: 2560 retains approximately all curriculum | REFUTED | D; real-tokenizer measurements below. Smoke hid bounded-curriculum row loss. |
| 4.5e: saved prior artifacts demonstrate nonzero finite gradients | UNCERTAIN | Old smoke has only A; no B/C trajectory. V does no backward. New runs must supply runtime evidence. |
| 4.5e: scaler skips counted | CONFIRMED in audited code; prior runtime UNCERTAIN | `training.py` counts skips/applied steps but advanced scheduler on skips and did not guard bf16 nonfinite updates. Repairs required before results. |

### Merged-adapter data

P recomputed **33,761** cached mix rows: **30,811** long-context and **2,950**
short-response replay. Every row matches its source by ID and complete message
hash. Sources are `rineya_long_context_cc0_v1_main/data/sft/train` and
`short-response-model-phase-0-1-2/data/train/train.jsonl` in the primary checkout.
The mix has zero LCR/demo ID matches and no complete LCR/demo context/question
strings inside its user messages. The launch log identifies this training file.

This resolves the same-LCR concern, but not exact historical row consumption:
there is no immutable historical mix hash or consumed-row trace tying individual
examples to checkpoint-200. Do not infer that every cached row was consumed,
that the model absorbed it, or that related tasks were never seen upstream.

### Phase 0 recomputation

| Demo core/gate | Showcase A | Stability A | A all | B all | C all | D eligible |
|---|---:|---:|---:|---:|---:|---:|
| A1.1/on | 45/64 | 125/192 | 170/256 | 147/256 | 97/256 | 111/160 |
| A1.1/off | 41/64 | 121/192 | 162/256 | 142/256 | 80/256 | 113/160 |
| A1/on | 43/64 | 125/192 | 168/256 | 147/256 | 96/256 | 111/160 |
| A1/off | 40/64 | 121/192 | 161/256 | 140/256 | 79/256 | 113/160 |

The first three baseline rows reproduce stated published counts; A1/off has no
supplied published gate. D contrasts always use its eligible frame.

| A1.1/on suite/frame | n | A correct | B correct | C correct | D correct |
|---|---:|---:|---:|---:|---:|
| Demo answerable | 192 | 148 | 118 | 67 | — |
| Demo unanswerable | 64 | 22 | 29 | 30 | — |
| LCR answerable | 150 | 49 | 45 | 15 | 57 |
| LCR unanswerable | 50 | 10 | 29 | 44 | — |

| Contrast | Frame | Difference, points | Original row-bootstrap 95% interval |
|---|---|---:|---|
| Demo B−A | all | −8.98 | [−14.06, −3.91] |
| Demo B−C | all | +19.53 | [+12.11, +26.95] |
| Demo D−A | eligible | −14.38 | [−22.50, −6.25] |
| Demo D−B | eligible | +3.75 | [−3.75, +11.25] |
| LCR B−A | answerable | −2.67 | [−10.67, +5.33] |
| LCR B−C | answerable | +20.00 | [+12.00, +28.00] |
| LCR D−A | answerable | +5.33 | [−2.67, +13.33] |
| LCR D−B | answerable | +8.00 | [0.00, +16.00] |

Intervals reproduce paired example resampling, 10,000 resamples, seed 20260905.
They are descriptive, uncorrected for multiplicity, and omit within-scenario
dependence. The separately labeled scenario-bootstrap sensitivity in `numbers.json`
is unregistered; it does not make this a confirmatory test. No significance claim.

Answerable demo A/B/C abstentions: **2/192, 21/192, 50/192**. Answerable LCR:
**18/150, 46/150, 109/150**. Unanswerable gains coincide with more wrong abstentions
on answerable rows, consistent with a response shift. This does not establish the
entire causal mechanism behind each wrong prediction.

### Format, oracle, and scorer corrections

Demo mean context tokens A/B/C/D: **1260.05 / 560.33 / 543.95 / 541.19**.
LCR: **2296.37 / 524.45 / 520.06 / 523.51**; the old LCR mean of 2381 was wrong.
D uses the eligible subset. Mean per-row B retention is **45.06%** demo and
**22.92%** LCR. The old demo table divided pooled means instead. C matches chunk
count, with unequal token lengths. These are joint presentation/content interventions.

Gold spans appear in supporting facts on **168/192** answerable demo rows and
**144/160** eligible demo rows. LCR full-dev is **553/600 (92.17%)**, but the
actually evaluated sample is **144/150 (96%)**. D retains all **507/507** annotated
facts on evaluated answerable LCR rows; that does not establish complete evidence
or the optimum input. Demo annotation localization remains approximate.

All **6,000** answerable LCR rows have one residual required span, present in the
original answer. On evaluated answerable outputs:

| Condition | Strict short match | Gold span anywhere |
|---|---:|---:|
| A | 49/150 (32.67%) | 52/150 (34.67%) |
| B | 45/150 (30.00%) | 45/150 (30.00%) |
| C | 15/150 (10.00%) | 15/150 (10.00%) |
| D | 57/150 (38.00%) | 59/150 (39.33%) |

D−A changes from **+5.33** to **+4.67** points. This relaxation explains little of
the observed error rate and does not reverse that contrast, but is not comprehensive
semantic adjudication. **38% is an observed score, not a capability ceiling.**

The core/gate follow-up preserves the sign of answerable B−A. It does not show
identical checkpoint effects: ungated all-row B−A is **−7.81** for A1.1 versus
**−8.20** for A1; eligible D−A is **−13.125** versus **−12.500**. Removing A1.1's gate
moves answerable B−A from **−15.625** to **−13.02**. These are reused examples,
not independent replication or proof of zero checkpoint effect. Similar inference
behavior does not guarantee similar training responses.

### Curriculum and mechanical corrections

| Seeded sample | Rows | Min / median / max sequence tokens | Retained at 2560 |
|---|---:|---|---:|
| Bounded train | 2400 | 1967 / 2243 / 3698 | 1759 (73.29%) |
| Bounded dev | 160 | 2195 / 2365 / 2628 | 130 (81.25%) |
| Smoke train | 32 | 2001 / 2080.5 / 2482 | 32 (100%) |
| Smoke dev | 16 | 2202 / 2237 / 2308 | 16 (100%) |

The old limit drops complete examples, reweighting tasks and breaking some packs.
A fixed **3840** limit retains the measured complete samples within the requested
short-context range. This mechanical repair is recorded before new adapter outcomes;
no length sweep is used. LCR versus demo is disjoint, but within LCR train/dev
**75** generic question strings recur across distinct contexts; contexts, titles,
IDs, and scenarios remain disjoint. Literal train/dev question disjointness is false.

D1 manifest collision and D3 gradient-bearing loss conversion are confirmed. D2's
unsafe rerun behavior is confirmed, but the assertion that the old smoke summary
predated current code is refuted by matching recorded source hashes at audit time.
D4 native-byte hashing is confirmed; exact historical chronology is uncertain.
Cross-stage model-state digest comparisons are not used; canonical file SHA-256
is a separate stable algorithm.

Additional blockers found: `--core a1` ignored the selected model and loaded A1.1;
validation loss ran gate-off; bf16 nonfinite gradients could reach the optimizer;
checkpointing could silently fall back; and partial combine could imply a winner.
These must be repaired before interpreting new results.

A further smoke exposed excessive allocation from full-sequence vocabulary logits.
The controller interrupted C, independently rehashed the canonical core unchanged,
and preserved that incomplete stage in `audit_20260905/pre_answer_logits_smoke`.
The final training path computes logits only at causal answer positions, retaining
the full decoder input and the same masked-answer objective. The preceding prompt
position and EOS supervision are retained; output references are released between
microbatches. The executable CPU equivalence check in
`results/audit_20260905/verify_answer_only_logits.py` passed all **15** cases across
fp32/fp16/bf16 and short/maximum lengths: adapter gradient differences were **0**
in every tested case, with maximum loss reduction-rounding difference **9.54e-7**.
This is an allocation repair, not an attention, architecture, or curriculum change.
Both canonical cores hashed unchanged; the full smoke was repeated after the repair.
The mechanical suite at that repair passed **47** tests (superseded by the integrated
resume suite below). These measurements establish only
the tested equivalence and mechanics, not statistical evidence of adaptation.

Fresh CPU V passes initialization/stem/attention targeting checks. Both adapters
have **122,880** trainable parameters. Zero-adapter reload/merge tests do not prove
nonzero learned serialization, gradient flow, or successful training.

## Phase 1 amendments and outcome

**Both cores' smoke runs and the bounded A1.1 comparison are complete; bounded A1 execution is underway.** The original
A-only smoke and interrupted memory-heavy smoke remain historical incomplete
artifacts, not adaptation nulls. No Phase 2 license follows from a smoke run.
The independent raw-record audit of this fresh smoke found complete matching
A/B/C data and valid receipts; both adapters applied all six attempts without skips.
Bounded processes use the fixed native allocator policy recorded in the amendment,
each arm manifest's `config.execution_environment`, and the controller's private
audit `execution_environment.json`; this governs allocation, not model behavior.

Fresh smoke scores, personally rescored from each arm's `per_example.jsonl`:

| Arm | Target correct / 32 | Answerable correct / 24 | Unanswerable correct / 8 | Preservation matched / 16 |
|---|---:|---:|---:|---:|
| A frozen | 20 | 18 | 2 | 6 |
| B LoRA | 22 | 18 | 4 | 6 |
| C NoRA-init | 10 | 4 | 6 | 7 |

B's two additional correct responses are entirely unanswerable items; answerable
correctness is unchanged. C abstains on **16/24** answerable items and loses **14**
answerable correct responses versus A. These six-attempt smoke observations are
not the bounded outcome and do not license Phase 2. Both adapters have six finite,
nonzero-gradient attempts, all applied, with no scaler or safety skips.
Evidence: `results/audit_20260905/root_resumable_smoke_audit.json` and
`root_resumable_smoke_frames.json`; all receipt/pair checks pass and every score,
summary and resource discrepancy array is empty. Saved tensor values separately
pass `root_resumable_smoke_saved_values.json`.

Pre-run corrections are recorded in `preregistrations/audit_20260905_phase1_repairs.md`:
fixed length 3840; retain and disclose code's counting_sets/conditional_rules
preservation classes; isolate/refuse stale arm directories; combine only compatible
complete records; fix requested core/dtype; mandatory checkpointing; finite-gradient
and skip accounting; gate restored for evaluation; version state hashing.

Keep the fixed rank/layers/LR/data order/optimizer and bounded budget of **300 optimizer attempts**; report applied and skipped updates separately.
The pre-result expectation is that A1 closely reproduces A1.1, which the comparison
can falsify. A preservation pass applies only to selected classes.

Phase 2 needs target gain at least **2 absolute points** OR **5%** relative error
reduction, and preservation regression at most **1 point**, on mechanically complete
runs. Null/negative Phase 1 stops the programme; it does not license a layer sweep.
The composition check remains unlicensed and will not run.

### Bounded A1.1 result — negative

All three bounded A1.1 arms and the combined stage are complete. The following
scores were independently rescored from each arm's raw `per_example.jsonl`:

| A1.1 arm | Target correct / 256 | Gain versus A (points) | Preservation matched / 200 | Preservation regression (points) | Individual gate |
|---|---:|---:|---:|---:|---|
| A frozen | 170 | — | 74 | — | Baseline |
| B LoRA | 110 | -23.44 | 69 | +2.50 | Fails target and preservation |
| C NoRA-init | 133 | -14.45 | 52 | +11.00 | Fails target and preservation |

The preregistered A1.1 decision is **keep FROZEN; Phase 2 not licensed**. LoRA loses
**60** target-correct responses; NoRA loses **37**. Both improve unanswerable-item
correctness, but those gains do not offset their answerable declines:

| Target frame | A frozen | B LoRA | C NoRA-init |
|---|---:|---:|---:|
| Answerable correct / 192 | 148 | 59 | 71 |
| Unanswerable correct / 64 | 22 | 51 | 62 |
| Answerable abstentions / 192 | 2 | 116 | 116 |

Both adapters show substantial over-abstention under the fixed gated protocol.
These results do not establish a general ceiling on late-layer adaptation, and
the cause of the behavior was not isolated by a separate intervention.

Preservation regresses by **2.5 points** for LoRA and **11 points** for NoRA, both
beyond the allowed limit. The class breakdown remains a selected two-class
tripwire, not a broad preservation claim:

| Preservation class, matched / 100 | A frozen | B LoRA | C NoRA-init |
|---|---:|---:|---:|
| counting_sets | 37 | 34 | 15 |
| conditional_rules | 37 | 35 | 37 |

B applied all **300** attempts. C used the same **300-attempt** budget but applied
**299** updates, with one scaler skip at attempt **6** and no safety skips. Every
applied update had finite nonzero gradients. Both consumed **2400** training
examples; **2392** contributed to C's applied updates. No attempts were added to
compensate. This is a comparison at matched attempt count and data exposure;
applied-update counts differ.

Both adapters used one physical model process and did not require a training
resume. Canonical and frozen-base hashes remain unchanged; every arm and pairing
check passes, with empty score, summary and resource discrepancy arrays. Saved
adapters contain only the expected finite attention LoRA tensors; this is not a
prediction-roundtrip test. Instrumented segment time is **8019.125 seconds** for B
and **7976.571 seconds** for C, not a measured time to a target-quality threshold.
A has two physical model loads because of its interrupted segment; its observed
time and peak-memory totals retain explicit lower-bound limitations.

Evidence in `results/audit_20260905/`: `root_completed_a11_audit.json`,
`root_completed_a11_frames.json`, `root_completed_a11_saved_values.json`, and the
independent `bounded_a11_B_independent_receipts.json` /
`bounded_a11_C_independent_receipts.json`. Data fingerprints and evaluation IDs
match across A/B/C. The required bounded A1 A/B/C comparison continues under
the pre-recorded settings and remains unfinished. No layer expansion or
composition check has been run.

### Bounded A1 result — negative, and a near-exact replication of A1.1

All three bounded A1 arms and the combined stage are complete. Independently
rescored from each arm's raw `per_example.jsonl`:

| A1 arm | Target /256 | Gain (pts) | RER | Preservation /200 | Regression (pts) | Individual gate |
|---|---:|---:|---:|---:|---:|---|
| A frozen | 168 | — | — | 74 | — | Baseline |
| B LoRA | 108 | −23.44 | −0.682 | 68 | +3.00 | Fails target and preservation |
| C NoRA-init | 128 | −15.62 | −0.455 | 53 | +10.50 | Fails target and preservation |

Preregistered A1 decision: **keep FROZEN; Phase 2 not licensed.** Frames:
answerable correct 147 / 57 / 65 out of 192; unanswerable 21 / 51 / 63 out of 64;
answerable abstentions 2 / 117 / 124 out of 192. Preservation by class:
counting_sets 37 / 33 / 16, conditional_rules 37 / 35 / 37 out of 100.

The pre-recorded expectation that A1 closely reproduces A1.1 **held**: every arm
lands within 5 target rows of its A1.1 counterpart (170→168, 110→108, 133→128),
with identical gate outcomes and the same over-abstention signature. The
checkpoint is not what drives the Phase 1 result. This is a comparison of the
two specified native-precision variants, not proof of zero checkpoint effect.

All four stages were then audited together with explicit stage arguments:
`phase1` and `phase1_a1` **VALID**, `phase1_smoke` and `phase1_smoke_a1`
**SMOKE_ONLY**, 17/17 pair checks per stage, every arm VALID with all receipt
checks true and **zero** entries in every discrepancy array. Saved-adapter
verification **PASS** on all eight adapter arms (24 tensors / 122,880 parameters
each, no GPU, no model loads). Canonical checkpoints unchanged throughout.
Evidence: `results/audit_20260905/final_all_stages_audit_20260906.json` and
`final_saved_values_20260906.json`.

### Gate-off ablation — the target-metric failure is largely a gate artifact

A follow-up single-variable ablation (`ablations/gate_off_nora_v1/`, own
`RESULT.md`) zeroed the cores' constant +5.156249985098839 abstention bias on
token 51378 (`UNKNOWN`), changing nothing else. It required no retraining, and
that is measured: the modeling `forward` computes the loss in `super().forward()`
and only then mutates the logits, and the lab consumes `out.loss` everywhere, so
gate-ON and gate-OFF dev loss are bit-identical (0.04797873464580445, 160/160
rows, max difference 0.0, reproducing arm C's recorded 0.047979). The saved
adapters already are the gate-off-trained adapters.

| core | NoRA gate ON | frozen gate ON | NoRA gate OFF | frozen gate OFF | gate-OFF gain |
|---|---:|---:|---:|---:|---:|
| A1.1 | 133/256 | 170/256 | **171/256** | 162/256 | +3.52 pts (RER +0.096) |
| A1 | 128/256 | 168/256 | **175/256** | 161/256 | +5.47 pts (RER +0.147) |

NoRA clears the preregistered *target* thresholds on both cores once the gate is
off, and `causal_disambiguation` returns from 9/32 to 30/32. Mechanism, from
first-token logits: the frozen model holds a ~13-nat margin against `UNKNOWN` on
answerable rows; training shifts it ~9 nats toward `UNKNOWN` (25% of the
curriculum has `UNKNOWN` as its entire target, and `UNKNOWN` is token 51378),
leaving less than the +5.156 bias, which then flips 50–81 answerable rows into
abstention.

**This does not overturn the Phase 1 decision.** Preservation is byte-identical
under both gate settings (0 of 200 predictions changed), so NoRA's ~11-point
regression is real adapter damage — confined to `counting_sets` arithmetic, with
`conditional_rules` untouched — and it still fails the gate on both cores. Nor is
it an efficacy claim: NoRA's gate-off gain decomposes as −34 answerable / +43
unanswerable (A1.1) and −28 / +42 (A1), so the entire net gain is better
abstention on unanswerable items, not better long-context reasoning.

Validity: every gate-OFF run has a matched gate-ON control from the same harness;
all six controls reproduce stored history with 0 mismatches across 456 rows; and
the gate-off frozen baselines independently reproduce the pre-existing
`phase0_eval_a11_ungated` (162/256) and `phase0_eval_a1_ungated` (161/256),
byte-identical on all 256 A1.1 predictions.

### Cross-core adapter initialization

The native-bfloat16 A1 smoke is complete: A/B/C score **20/21/8 out of 32** on the
target subset and **6/6/6 out of 16** on preservation. Both adapters apply all
**6** attempts without skips; every arm/pair check passes with empty discrepancy
arrays. These are mechanics-only observations. Evidence:
`results/audit_20260905/root_a1_smoke_audit.json`.

Bounded A1 frozen A is also complete: independent raw rescoring gives
**168/256** target correct and **74/200** preservation matched, with actual
`loaded_dtype=bfloat16`, unchanged canonical hashes, and all arm checks passing.
Evidence: `results/audit_20260905/root_bounded_a1_A_audit.json`. The A1 adapter arms
have since completed; the cross-core comparison is reported above.

The same seed/rank/initialization recipe does not yield identical initial LoRA B
tensor values across native cores. Actual initializer digests differ, while tensor
names, order, shapes and final float32 dtype metadata match. The installed PEFT
path initializes adapters, casts through the native base dtype, then promotes them
to float32; different rounding is a source-supported explanation. Elementwise
differences were not measured. Evidence: actual arm `identity.json` receipts and
`results/audit_20260905/cross_core_B_initialization_control_20260906.json`.
Interpret this as a comparison of the specified core/native-precision variants;
it does not isolate frozen weight differences while holding initial adapter bytes
identical. The qualitative prior about similar outcomes is now **borne out** by the
complete bounded A1 results: differing initial adapter bytes still produced target
scores within 5 rows and identical gate outcomes on every arm.


### Execution interruption and recovery

The first bounded A1.1 LoRA process and both controllers terminated without
completion receipts. Native process inventory confirmed that the jobs were gone;
the specific termination cause was not logged; the user subsequently confirmed
restarting Codex. The interrupted arm logged **153** applied attempts, but
had no saved optimizer state or final evaluation. Its records were archived under
`results/audit_20260905/interrupted_bounded_a11_arm_B_r1`. A post-interruption
hash check confirmed the canonical checkpoint and frozen Python sources unchanged.

That completed frozen A and subsequent failed retries are now archived; the
current sequence regenerates all arms under the adopted recovery source. Detached
controllers and their actual PID/start identities are recorded in `logs/*launch.json`.
Old interrupted work without recoverable optimizer state is historical resource
evidence only; it is neither added to the accepted arm's training budget nor
interpreted as a null result. Final bounded outcomes remain pending.


### GPU reset failure and synchronized retry

The first detached retry applied **4** updates before CUDA reported an illegal
memory access. Windows driver events at the same time record a GPU TDR reset
and restart; the underlying cause is not established. The failed stage and its
previous frozen baseline are preserved in `results/audit_20260905/pre_sync_phase1`.
A fresh CPU file-hash check again confirmed the canonical checkpoint unchanged.

A bounded diagnostic then completed **16** forward/backward microbatches with
`CUDA_LAUNCH_BLOCKING=1`, the exact original input fingerprint, and **0** optimizer
updates. Canonical and frozen source hashes remained unchanged. This tests that
short execution window, not full-run stability.

The synchronized retry regenerated A and began B before another process loss;
it is archived under `pre_resume_phase1`. The adopted sequence retains the same
synchronized launch setting for A/B/C on both cores. The later recovery code
change is documented below; data, optimizer, initialization and total training
budgets remain fixed. Earlier interrupted and failed attempts remain excluded
from scientific outcomes and separately identified when accounting for cost.


### Durable recovery adopted

The user confirmed restarting Codex. Windows job-lifetime probes showed that
hidden child processes can still belong to a kill-on-close job; full independence
was not established, and no job limit or permission was bypassed. The earlier
run losses are not scientific nulls. NVIDIA errors and a TDR event remain
separately recorded; their underlying cause is not established. No GPU clock or
power setting was changed.

The loop now commits the actual LoRA, optimizer, scaler, scheduler, RNG, cursor,
trajectory and validation state atomically. Explicit recovery continues the same
**300-total-attempt** course. Each arm has one active model process; physical
restart segments and their costs are recorded honestly, including unobserved
costs after hard termination. Checkpoints are selected only by latest committed
state, never validation quality. This cannot reconstruct old states that were
never saved.

The complete integrated CPU suite passed **151** tests on the required Python
runtime, including hard process deaths and independent resume/accounting audits.
The fresh CPU adapter verifier passed and canonical hashes remained unchanged.
Old active outputs are archived under `pre_resume_*` for source consistency.
The new smoke is complete. Separate GPU no-op restores of its B and C checkpoints
both passed: exact adapter parity with checkpoint and saved safetensors, exact
AdamW/scaler/scheduler/RNG values, GPU optimizer moments, and unchanged canonical,
frozen-base, source and arm-file hashes. Each verifier loaded one model and
performed zero new updates, forward passes, validation or checkpoint writes.
Receipts: `results/audit_20260905/gpu_resume_noop_phase1_smoke_B_20260906T021410660054Z.json`
and `gpu_resume_noop_phase1_smoke_C_20260906T021508318506Z.json`.
This verifies restoration of those completed GPU checkpoints; it does not prove
GPU prediction parity or uninterrupted-versus-resumed GPU update equivalence.

The full fixed controller was launched after these checks, with provenance in
`logs/resumable_full_controller.launch.json`. It runs bounded A1.1, then A1 smoke
and bounded A1, with one active model process at a time. Any recovery uses the
original total budget and latest valid state. Final scientific outcomes and
Phase 2 licensing remain pending.

That controller subsequently disappeared after frozen A wrote evaluation outputs
but before its completion and frozen-after receipts. The arm remains incomplete;
its raw predictions are retained as an interrupted physical segment. Independent
CPU checks found canonical files and frozen sources unchanged; no adapter updates
had occurred. The same controller was relaunched through explicit recovery, with
provenance in `logs/resumable_full_controller_r2.launch.json`. Frozen A retries
evaluation; the interrupted segment's unobserved final costs remain explicit.
Evidence: `results/audit_20260905/resumable_full_controller_exit_20260906T0223.json`
and `root_bounded_a11_after_controller_exit.json`.
Windows events subsequently establish a system restart during this interruption:
User32 recorded a restart request at **02:22:37 UTC**, followed by shutdown and a
new boot at **02:23:23.500 UTC** on 2026-09-06 (also confirmed by CIM boot time).
This explains the controller/model loss; the reason for the restart request is
not established. See `termination_diagnosis_20260906T022237Z.json` in the audit
directory. This evidence concerns the latest loss only, not the earlier CUDA/TDR
failure.
