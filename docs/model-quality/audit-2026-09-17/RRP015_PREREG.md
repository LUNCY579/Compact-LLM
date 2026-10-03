# RRP-015: a selection-clean transition/rehearsal pilot

Frozen before preparing either arm or observing its model outputs, 2026-09-17 UTC.
This is a new exploratory comparison, not a replication of the historical draws or
an authorization to release a model. The user's current goal explicitly authorizes
local experiments; the numeric limits below are agent-chosen bounds within that
authority, not a claim that the user supplied these numbers or reviewed the data.

## Question and fixed design

Does adding approximately one processed rehearsal token per transition token help
preserve measured behavior while learning three one-step transition families?

Both arms start independently from the unchanged Supa 1.7 source, weight SHA256
`962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b`.
Control C has 128 verified positive parents per family, one exposure, draw seed 17.
Treatment R uses exactly those same 384 transition rows plus the existing v3
preservation pool's abstention-class filter, at 1:1 processed-token budget (at most
one final-row overshoot). It is a mixture intervention: it changes total tokens,
update count, order, and each transition's schedule position; it does not isolate a
regularization mechanism at equal compute. No other recipe is changed:
assistant-response plus EOS, FP32, blocks 8–27, peak LR 1e-5, WSD20/60/20,
AdamW betas .9/.95, epsilon 1e-8, no decay, clipping 1, full rows, window 2048.

Both builders must exclude all parents of the historical process probe AND the
immutable SELECT-VIEWS-001 selection instrument (SHA256
`7dd53f2a751b54c349c2ef571e8f9c5153737f4a1ed3df5eafe25f95980e78a3`).
No unavailable-family draw may be silently shortened. Rehearsal must additionally
exclude panel IDs, panel reply classes, shared user-turn eight-word spans, and
digit-blinded last-question stems. This last guard catches short percentage tasks
misclassified as insufficient_information. Publish exact dropped and selected
counts, source hashes, rendered lengths, token totals, and zero-contact reports
before launch. Keep the source files unchanged. Human data review: NOT_PERFORMED.

## Outcomes and interpretation

1. Run frozen Supa and published Opallo on SELECT-VIEWS-001, then both new arms on
   that identical instrument. Every family is reported separately. Clearance means
   the two-sided Clopper–Pearson 95% lower bound strictly exceeds that instrument's
   stronger copy-before/constant-majority floor. No transcribed critical count.
2. Report the existing arithmetic panel's strict and permissive scores, source
   successes gained/lost, and the historical GATE v2 conjunction (panel >=62,
   source successes retained >=37/47, every selection family clears its floor).
   Passing this gate is NOT broad preservation or permission to release.
3. Measure the spent 256-row long-context suite and its 64 unanswerables ONLY as
   report-only diagnostics. Do not use these outputs for candidate selection,
   parameter changes, stopping on efficacy, or a confirmation claim. Include
   paired losses/gains, literal UNKNOWN, false abstention, and generation endings.
4. Publish both arms, every attempted launch, and row-level predictions. One seed
   cannot establish reliable retention. Fresh instances within these generators
   cannot establish independent cross-domain reasoning improvement. Nonsignificant
   differences do not establish equivalence or absence of forgetting.

Hypotheses, not promises: both arms clear acquisition floors; R retains more source
panel successes than C; no broad quality claim is expected to be supportable.
No adaptive follow-up is authorized by this registration. Further seeds or recipes
need a new prospective record, informed only by selection-permitted evidence.

## Execution and evidence

Use an isolated local sparse clone with independent refs on the controller's
already authorized branch name. Preserve all three pre-existing dirty items in
the original worktree. Keep source/model/dataset paths external and hash-locked;
do not duplicate or modify published assets. Commit only this experiment's code
and registration in the isolated checkout before training; do not push.

Use the MQ foreground watchdog, cooperative GPU lease and unchanged host/GPU/disk
guards. Requalify each process arm in-run before optimization. Train both arms
before legacy scoring writes any worktree evidence. Retain full resumable
checkpoints and all failures. Derived materializations may be removed after their
hash receipts and complete source checkpoints are retained; no checkpoint or raw
measurement may be deleted. No parallel model workers or paid jobs.

Session ceilings: 8,400 seconds owned model work; 2 planned training launches (1,200
seconds each), plus at most 1 explicitly recorded retry within the same session
ceiling; 8 evaluation launches (600 seconds each,
including startup). Per-arm train cap 600,000 processed tokens, 240 qualification/
microfit evaluation rows, 30,720 reserved generated tokens, 8 GiB new checkpoint
bytes. External scoring cap 2,560 rows and 245,760 generated tokens (96 per row
upper bound; panel/long-context use their own unchanged lower defaults). Total new
disk cap 18 GiB, and the 14 GiB training disk floor remains enforced. Stop and
preserve if any bound, source identity, exclusion check, qualification, numerical
check or resource guard fails. Do not weaken a guard to fit the second arm.

The model-training execution IDs remain RRP-008-PROCESS-DOSE and
RRP-009-PROCESS-MIX because their optimizer/data paths are reused; RRP-015 identifies
this prospective comparison and its newly bound data. Evaluation uses the existing
DenseReference conversion and greedy scoring path. Candidate FP32 checkpoint
tensors are rounded to source BF16 storage before FP32 evaluation, exactly as the
existing materializer does; this is exported-weight evaluation, not raw-FP32 or
native-BF16 serving qualification.

## Research rationale

Replay and conservative scheduling have external precedent, but neither a
universal rehearsal ratio nor guaranteed retention follows from it. Ibrahim et al.
study continual pretraining at 405M/10B with particular distribution shifts
([paper](https://arxiv.org/abs/2403.08763)); He et al. examine task-aware replay in
continual instruction tuning ([paper](https://aclanthology.org/2025.acl-long.1153/)).
These motivate controlled local comparisons, not transferring their efficacy
claims to Supa. Historical claims that one split mixture proved all broad replay
ineffective, or that a single failure proved a universal concentration threshold,
are stronger than the evidence supports.
