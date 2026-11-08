# Amendment 002: prospective preservation data preparation

Date: 2026-09-28. Frozen before new retention generation or native outcomes.
This accepts **FAST data preparation only** after reviewing E0's host-RAM block.
It neither changes the E0 resource/numerical gates nor authorizes E1 training.

## Fixed deterministic banks

Create original retrieval, instruction-following and answerability tasks. Retrieval
has named evidence and distractors; instruction following uses precedence/format
constraints rather than the acquisition bridge's token-transform programs.
Answerable and missing-evidence twins share the same latent world and parent.
The role is assigned to the canonical world before surface rendering, solving or
scoring. Variants never select a different role. Selection seed: 20260928.

| Role | Retrieval parents/rows | Instruction parents/rows | Answerability parents/rows |
|---|---:|---:|---:|
| TRAIN | 256/256 | 256/256 | 128/256 |
| DEV | 64/64 | 64/64 | 64/128 |
| CHECK | 128/reserved only | 128/reserved only | 128/reserved only |
| TRAIN_SELECTION | 0 | 0 | 0 |
| FINAL | 0, closed | 0, closed | 0, closed |

Thus the deterministic replay candidate has 768 rows and 640 parent units. The
deterministic DEV panel has 256 rows and 192 parents. Neither is the whole replay
or preservation panel. Language still needs 256 replay rows and 64 independent DEV
parents; a genuinely longer-context sentinel is additional and separately gated.
The proposed E1 replay total remains 1024 rows, not 1024 independent parents.

Use a new role-allocation hash namespace with the same prospective 70/8/8/8/6
percent buckets as the bridge. Freeze four strata per domain and equal quotas
before generation. Generate by deterministic candidate order, with a fixed bounded
candidate ceiling. Reject a shortfall rather than borrowing another role. Export
TRAIN/DEV only; CHECK holds identifiers/hashes, not prompt/target/specification
payloads. FINAL stays unopened. Targets use strict exact JSON with independent
solvers, mutation controls, and no target truncation. Bind actual tokenizer/template
counts before any eligibility claim. No model or teacher is required for construction.

## Native baseline policy (still gated)

Retain all frozen TRAIN and DEV rows, including every failed baseline attempt.
There is **no adaptive filtering or replacement of TRAIN after seeing model
outcomes**. This supersedes the earlier proposal's training-side source-correct
selection: fixed-bank replay avoids an extra data-selection factor in E1.
After E0, measure baseline competence separately for each replay and DEV domain.
Replay is an intended preservation objective; a base-incorrect replay task is
reported as additional supervised acquisition, never as preserved competence.

Freeze the actual source-correct DEV membership before candidate evaluation;
report losses, repairs, false abstention and both-twins-correct separately. Parent
units, not twin rows, drive uncertainty. A zero source-correct denominator remains
undefined. No broad preservation claim or E1 promotion is allowed without the
unresolved language and long-context panels. This amendment does not lower any
promotion threshold and does not substitute deterministic tasks for language.

## Provenance and uncertainty

Language training sources are inspected only as TRAIN candidates. Recover original
parent/family membership through source bytes and generator lineage, not a version
label or a new random identifier. Selection rules and quality exclusions must be
frozen before native outcomes. Do not relabel old TRAIN or exposed DEV as fresh
evaluation. Existing protected banks retain all read/use exclusions.

Scope any historical overlap audit to the actual permitted fingerprints and source
lineage it checked. No exact-hash comparison proves semantic or pretraining
decontamination. Missing historical coverage remains explicit. No training launch
manifest is created by this FAST amendment.
