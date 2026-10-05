# RQSFT-001 implementation and execution sequence

This is the separately registered balanced-quality successor on
`codex/rineya-quality-sft-v1`, isolated at
`C:/Users/clclc/BusinessProject/ChatBubble-rineya-quality-sft-v1` from the DSD
worktree. Read PROTOCOL.md and AGENTS.md. Existing MQ/DSD/RRP source/evidence/
registrations and model source bytes remain read-only.

## Ownership and the current stop

Dataset construction/admission belongs to the existing **Plan Rineya dataset
pipeline** chat `01a109dd-a8bc-73d2-bf9c-5ebc09a00cde`. The actual request is
`C:/Users/clclc/BusinessProject/rineya-quality-sft-records/rqsft001-preparation-20261005/OWNER_REQUEST.json`.
Its owner acknowledged the200TRAIN/512DEV design, but actual construction and
admission remain pending. The consumer does not construct a competing bank,
read protected/excluded payloads, relabel fixtures or create an approval.

The requested bank is200TRAIN (80grounded,60solving,40preservation/dialogue,
20contracts) and512openDEV (four128-row cohorts). All80 grounded TRAIN cases are
40 intact present/missing evidence pairs. Boolean-tagged DEV answerability cases
also have complete pairs. Underlying parent/scenario/family/lexicon identities
and full histories survive. The owner must export actual independently
attributable rights/gold/partition/shortcut review receipts bound to the exact
files and ancestry. `owner-request` emits a pending schema, not admission.

## Safe preparation commands

Use the existing read-only MQ interpreter with Python3.11.15,torch2.10.0+cu126,
Transformers4.57.1,safetensors.7,tokenizers.22.2,psutil7.2.2. No installation or
environment upgrade is required. All new outputs/temp/caches must be under
BusinessProject outside every Git checkout and `.codex`.

```powershell
Set-Location C:/Users/clclc/BusinessProject/ChatBubble-rineya-quality-sft-v1
$python = 'C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe'
& $python -B -m experiments.rineya_quality_sft_v1 owner-request --output C:/Users/clclc/BusinessProject/rineya-quality-sft-records/new-owner-request
```

After receiving an actual admitted owner manifest, use its exact path (the
illustrative `owner-delivery` below does not exist yet):

```powershell
& $python -B -m experiments.rineya_quality_sft_v1 prepare --owner-manifest C:/Users/clclc/BusinessProject/rineya-quality-sft-records/owner-delivery/manifest.json --output C:/Users/clclc/BusinessProject/rineya-quality-sft-records/preparation-001
```

Preparation checks metadata and independent receipts BEFORE payload reads or
native tokenizer imports. It compiles ALL200 rows; there is no prefix selection,
padding/truncation, pair split or extra response-arm examples. Final-assistant+EOS
labels exclude previous assistant history; whole labels include all next-token
positions. Verified intentionally empty targets supervise EOS alone. Missing
captures are never empty-target success.

The output includes the exact sealed plan and `EXPOSURE_REVIEW_PENDING.json`.
An actual independent reviewer checks native token IDs, causal boundaries,
parent grouping, quotas, source/owner binding and costs before recording APPROVED
against that plan hash. Preparation does not fill this review. Commit the final
code/docs before qualification; code/dependency/checkout changes invalidate it.

## Conditional native commands — not executed in implementation

```powershell
& $python -B -m experiments.rineya_quality_sft_v1 qualify --plan C:/Users/clclc/BusinessProject/rineya-quality-sft-records/preparation-001/plan.json --review C:/Users/clclc/BusinessProject/rineya-quality-sft-records/preparation-001/approved-exposure-review.json --output C:/Users/clclc/BusinessProject/rineya-quality-sft-records/qualification-001 --execute-qualification
& $python -B -m experiments.rineya_quality_sft_v1 baseline --plan C:/Users/clclc/BusinessProject/rineya-quality-sft-records/preparation-001/plan.json --review C:/Users/clclc/BusinessProject/rineya-quality-sft-records/preparation-001/approved-exposure-review.json --qualification C:/Users/clclc/BusinessProject/rineya-quality-sft-records/qualification-001/qualification.json --output C:/Users/clclc/BusinessProject/rineya-quality-sft-records/source-001 --execute-evaluation
```

Zero-update qualification checks both objectives,512backward/2048forward/cache
shapes, full vocab loss/hidden-gradient parity, native template/history/EOS IDs,
exact220tail tensor inventory, actual math SDPA and frozen source/buffers. It
creates a0-step optimizer checkpoint and deliberately perturbs ONE SCALAR IN
EVERY selected parameter and both moment tensors, cursor and RNG, then restores
all state from physical safetensors. These mechanical sentinel writes are not
optimizer updates or learned candidates; failed restoration stays FAILED even
when independent cleanup restores memory. No-op/partial restores are tested.

Native stages run only through the foreground supervisor and one-use worker
context. Phase opt-ins are rechecked in the worker. Every qualifying witness,
raw prediction file and completed outer stage/supervisor receipt must agree;
an inner PASS or512-record file from a later failed stage cannot authorize work.
Source/data/code custody is checked before/after, including failed runs.

Baseline produces durable per-case predictions and a fixed-rubric blinded review
packet. Do not supply its private role-link file to reviewers. Two independent
blinded, prediction/rubric-hash-bound reviewers resolve natural abstention,
communication and any unresolved semantics. Missing/uncertain judgments stay
INCOMPLETE. Review results use the row-keyed adjudication schema in evaluation.py.

```powershell
& $python -B -m experiments.rineya_quality_sft_v1 review-source --plan C:/Users/clclc/BusinessProject/rineya-quality-sft-records/preparation-001/plan.json --source-predictions C:/Users/clclc/BusinessProject/rineya-quality-sft-records/source-001/predictions.json --source-reviews C:/Users/clclc/BusinessProject/rineya-quality-sft-records/source-reviews.json --output C:/Users/clclc/BusinessProject/rineya-quality-sft-records/source-report-001
& $python -B -m experiments.rineya_quality_sft_v1 train --plan C:/Users/clclc/BusinessProject/rineya-quality-sft-records/preparation-001/plan.json --review C:/Users/clclc/BusinessProject/rineya-quality-sft-records/preparation-001/approved-exposure-review.json --qualification C:/Users/clclc/BusinessProject/rineya-quality-sft-records/qualification-001/qualification.json --baseline C:/Users/clclc/BusinessProject/rineya-quality-sft-records/source-001/predictions.json --source-report C:/Users/clclc/BusinessProject/rineya-quality-sft-records/source-report-001/source-report.json --arm whole_1e-5 --output C:/Users/clclc/BusinessProject/rineya-quality-sft-records/whole-001 --execute-training
```

Then select `response_1e-5` once into a distinct new output. There is no sweep.
Training waits for the reviewed complete source/error report. Same source,
rows, atomic groups, WSD(.20/.60/.20→.1peak),FP32 CPU-moment AdamW(.9,.95),
eps1e-8,decay0,clip1,LR1e-5 and32updates/25ktokens. Every cleared-boundary
checkpoint remains; no automatic deletion. Update norms use bounded per-parameter
measurement, not another full-tail RAM snapshot. Final checkpoint only is evaluated.

## Resources, attempts and analysis

Training before-load6GiB host/6GiB GPU; qualification4.6/3GiB; post-load1.5/.5GiB.
These floors remain unchanged and are not fit certificates. Retain load/OS peak
working set/CUDA allocated+reserved peaks, actual logical token/call counters,
step norms, generation/EOS/caps and cost. Check disk for every expected3.52GiB
tail checkpoint before launch. Qualification wall1800s; baseline or whole
train+finaleval wall3600s under the owned-child watchdog. Cooperative memory
checks occur at operation boundaries, not continuously. The watchdog stops only
its own child tree; foreign processes/leases remain untouched.

The shared BusinessProject GPU lock and an existing legacy MQ temp-lock check
are conservative cooperation; foreign GPU compute processes or opaque inventory
refuse. Stale locks need owner resolution, never automatic removal. Logical
input counts do not measure checkpoint recomputation FLOPs.

Each plan gets one qualification attempt, one source measurement and one attempt
per SFT arm. Startup/failure consumes the reservation, no refund/refill/retry.
Controlled scientific resume is blocked until separately reviewed complete
attempted-cost/update-lineage accounting and nonzero-state restart evidence.
The physical recovery APIs/checkpoints remain preserved for that later work.

After independently adjudicating both final candidates, `analyze` takes exact
plan/source/candidate predictions and review paths, producing paired reports.
It binds the producing successful outer stages, all512IDs/prefixes/full official
EOS list151645/151643, actual plan/train exposure, code/runtime and source state.

Advancement requires +5pp acquisition, positive harder transfer, zero new
critical correctness losses OR safe-refusal→unsupported-assertion transitions,
preservation source-success retention>=98%, macro decline<=2pp, communication
nondecline and complete costs/finite updates/coverage. Zero source denominator is
undefined; missed adjudications are not false or zero. Literal UNKNOWN, semantic
abstention and format compliance remain separate. Numeric/JSON scorers reject
nonfinite/duplicate/type-coercion traps. Wrong strings do not prove an internal
execution, retrieval or knowledge mechanism without independent evidence.

Source reuse accepts only identity-matched public raw text/tokens and separately
attributes reused measurements. Old judgments/qualification/cost equivalence
are never reused. Changed prompt/history/runtime/generation needs new measurement.
All results remain DEV; no automatic promotion, serving change or broad efficacy
claim. Follow-up directions are frozen in PROTOCOL.md and the approved plan.
