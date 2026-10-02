# SYNTH-CRS-001 controlled study proposal

**Design only. Native implementation, complete preregistration, and launch approval
are pending.** Preserve this proposal; record later amendments and exact run IDs.
Do not replace CRS-001's procedure-memory study or any central E1 protocol.

## Primary hypothesis: task-condition diversity

Can independently verified, condition-diverse training examples improve atomic
skill acquisition and transfer at a fixed student-training budget beyond a strong
surface-varied verified baseline? This adapts a SYNTH-inspired data principle to
an existing checkpoint, not the paper's from-scratch single-stage claim.

Start with atomic R (reference resolution) and S (exact-size subset selection
with a target sum). In S-only prompts, values are already resolved: no hidden R
step. In R-only prompts, no subset selection is required. Reserve RS for transfer.
Use the established R/S definitions when integrating a specific corpus; do not
relabel data that already requires both operations as "atomic". U/P expansion,
bridge training, answerability training, and RL are separate future factors.

| Arm | Adaptation data | Target / role |
| --- | --- | --- |
| M0 | None | Unchanged checkpoint; enroll prior-correct behavior. |
| M1 | Independently verified atomic R/S with conventional templates and surface variation | Final-answer-only; primary control. |
| M2 | Same source-parent pool and authorized skills, with systematic task-condition diversity | Same final-answer-only format; primary treatment. |

Condition diversity may vary reference depth, distractors, target ranges, valid
solution positions, and constraints within the same atomic skill. It is a data
intervention, not a claim to isolate superficial wording. Predeclare condition
strata, answer distributions, task difficulty, rejected-candidate replacement,
and allocation rules so an accidental harder curriculum is not left unexplained.
The control must be a credible alternative, not unfiltered or unverifiable text.

## Shared controls and budgets

Pin identical checkpoint, tokenizer/chat template, trainable parameter set,
optimizer and schedule, loss masking, context cap, target format, skill allocation,
replay policy, decoding, evaluation interface, and selection rule for M1/M2.
Use the same canonical parent pool, split assignments, and acceptance verifier.
Do not change architecture, recurrence, teacher strength, or RL objective at once.

Use tokenizer-measured non-padding training tokens as the primary budget and
record supervised tokens, parent exposures, optimizer updates, wall time, peak
memory, padding, and generation/verification costs. Predeclare allowable matching
tolerances and residual imbalances; equal row count alone is not iso-budget.
If warmstart is needed, use the same separately authorized disjoint warmstart in
both arms, or neither. Do not choose it after seeing confirmation results.

## Sequence and gates

**G0 - CPU mechanics (future):** implement canonical task generation, two independent
reference paths, variant dependence checks, parent-family split isolation,
negative controls, export allowlists, and fail-closed record validation. Test
malformed inputs, duplicate/dependent worlds, incorrect intermediate states,
answer changes under counterfactuals, identity leakage, and overlength examples.
The currently supplied metadata tests are not these mechanics tests.

**G1 - Native qualification (future):** bind actual checkpoint/config/tokenizer and
numerical precision; verify unchanged-base scoring, token rendering, labels/masks,
EOS and padding, optimizer steps, save/reload, and memory admission on the actual
host. Native resource budgets are not inferred from model size or old free-VRAM
readings. Protected central CHECK/FINAL banks stay closed. On failure, record the
blocker instead of silently substituting another checkpoint or lowering checks.

**G2 - Development screen (future):** run M0/M1/M2 on fresh development parents,
track learning curves and failure categories. Select recipe and checkpoint using
DEV only. Test observed feasibility before increasing data or reasoning length.

**G3 - Independent confirmation (future):** freeze artifacts, selection rule,
statistical method, sample size/power rationale, training seeds, accuracy/retention
margins, and resource ceilings before obtaining confirmation labels. Run paired
parent-cluster evaluation with multiple training seeds. A single seed or exposed
suite is a development result, not confirmed general improvement.

**G4 - Representation experiment (only afterward):** hold data construction fixed;
compare final-answer-only, concise independently verified prose, and explicit
verified intermediate states. Match/report token budgets. Do not attribute an
M2 gain to reasoning traces when M1/M2 used the same answer-only target.

**G5 - Separate system/composition studies:** compare procedure memory under the
CRS-001 controls, or train independent skill donors and measure later composition.
For distillation, evaluate the student without teacher/scorer assistance. These
are separate permissions and experiments, not automatic follow-on launches.

## Evaluation and decision rules to freeze before confirmation

Primary endpoint: paired parent-level M2-minus-M1 first-attempt atomic accuracy,
with R and S also reported separately. Macro-average skill families using frozen
weights so a shift in task mixture cannot manufacture a gain. RS transfer is a
separate prespecified secondary endpoint, never a post-hoc replacement primary.

Report new correct cases and lost previously correct cases separately. Retention
uses enrolled source-correct rows in disjoint guards; replay is not a guard.
Report answerability precision AND coverage, erroneous abstention, evidence-use
errors, formatting errors, and counterfactual-pair success. Do not hide failures,
timeouts, or hardware aborts by shrinking the denominator. A whole-run abort is
an incomplete run, not evidence that a reduced cohort passed.

Confidence intervals must cluster by parent family; variants are not independent
samples. Preserve per-training-seed results and justify how seed variability is
included rather than pooling all outputs as independent observations. Define a
minimum useful gain and maximum acceptable retention loss before confirmation;
advance only when uncertainty bounds satisfy both and all qualification/resource
guards pass. These thresholds and sample size are intentionally unset here: no
confirmation launch is authorized until the run-specific registration fixes them.

If M2 only improves familiar renderings, record a template-specific result. If it
increases precision by refusing more questions, do not call it broader knowledge.
If retention worsens, no clean cumulative-learning claim. If results are null,
record the tested budget, uncertainty, failure analysis, and restart conditions;
do not delete the track. Do not test-tune a new endpoint to rescue the claim.
