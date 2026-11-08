# SCP-001 — Sequential complementary skills

An isolated implementation of the next small-model research sequence, combining
**Titan-Sparse's measured FP32 update semantics** with **the retention-audit branch's
native serialization, training handoff, and evidence discipline**. This is not a
wholesale branch merge and does not replace any historical registration.

## Delivered versus still gated

Implemented and CPU tested: deterministic magnitude masking; rank truncation and
per-tensor norm controls; hash-locked tensor-by-tensor checkpoint/factor construction;
corpus identity/length/overlap/shortcut checks; independently verified teacher-record
interfaces; matched F/P/T training presentations; native RRP compiler handoff;
independent multi-reasoning and repeated-direct machinery; paired and factorial
analysis; immutable artifacts; source-bound stage receipts and CLI.

**Native integration is NOT complete.** No SCP pretrained model was loaded, no training
was run, no production corpus was accepted, and no confirmation bank was opened.
`reasoning.run_case(native=True)` explicitly refuses rather than mistaking an arbitrary
callback or a hash-shaped string for qualification. The checkpoint constructor emits
weight shards/factors, not a qualified serving artifact. A dataset mechanics receipt
never itself grants training. This commit provides runnable CPU mechanics/preparation
and analysis; it does not claim a completed capability experiment.

## Important correction to the earlier conversation

`k=0.25` is **coordinate retention**, NOT multiplication of all deltas by 0.25:

```
Delta = trained_FP32 - source_promoted_FP32
D     = source + M_0.25(Delta)
L16   = source + rank16(M_0.25(Delta))
N16   = source + M_0.25(Delta) * ||rank16(M_0.25(Delta))|| / ||M_0.25(Delta)||
```

Mask each trained tensor independently, retain `floor(numel*0.25)` entries, and break
magnitude ties by ascending flat index. The primary low-rank experiment compresses
**the incumbent's masked delta**. Raw full-delta compression is a different auxiliary
condition. Compression may destroy coordinate sparsity. Neither low rank nor stored
energy establishes skill composability. FP32 storage/computation is binding for this
incumbent; BF16 conversion is a different, unqualified intervention.

## Branch integration and provenance

Parent: `codex/dataset-retention-audit-v1@fbec90a0d0653aeb745d5223137ae3cfdd9c48dd`.
Complement: `single-layer-fine-tuning@954a0c2637e691b493c5ac3d488653a8c1adf675`.
These heads diverge: 20 commits on the single-layer side and 240 on the retention
side relative to their common ancestor. The retention parent contains the current
RRP/MQ machinery; the single-layer head does not contain that full experiment tree.

`SOURCE_PINS.json` distinguishes Git object SHA-1s from weight SHA-256s. Original
Titan CORE metadata and builder text are retained under `provenance/` byte-for-byte;
the builder copy is text for inspection, not a script to execute with its historical
absolute Windows paths. No published-core catalog is edited or promoted.

The 4B `270/180` and `142 seconds` headline was supplied by the user as a local result.
Its exact checkpoint/recipe were **not located in these pinned remote sources**. The
596M REP-001 data-selection draws are not 4B teacher training-seed replications.
Do not manufacture the missing teacher binding or assume BF16 LoRA must win.

## Sequential plan

| Stage | What the implementation supports | Gate for actual model work |
|---|---|---|
| Mechanics | CPU tests, immutable receipts, status dependency graph | Scoped PASS only; not a model receipt |
| Fresh instrument | Open TRAIN/CAL/CHECK identity, native rendered lengths, authorized fingerprints, shortcut adversary | Reviewed generator/independent solver, real tokenizer, rights/review, registered populations and floors |
| Teacher qualification/replication | Explicit blocked stages and identity requirements | Recover exact 4B recipe; qualify source; freeze two additional independent training seeds and resources |
| Compression | Source/full/sparse/rank/norm controls; factors and actual bytes | Exact FP32 assets, tensor allowlist, resource supervisor, then separate loader/behavioral parity |
| Matched training preparation | F final answers, P solver processes, T verified teacher processes; unchanged retention anchors | All teacher rows must verify, no success filtering; native compiler and exact row bindings |
| Student training | Handoff to existing RRP engine, not another optimizer | New exact-subject qualification and supervised session grant; no inherited permit reuse |
| Before/after reasoning | O, token-allowance T, REP and MR; real selector, raw output retention; 2x2 analysis | Qualified generation-only backend and supervisor, frozen bank/policy across old/new |
| Later work | Registered conditional queue for data efficiency/composition | Informative preceding result and separate bounded design |

Missing teacher assets do not block CPU work, compression preparation, or F/P
presentation construction. They do block the teacher-dependent comparisons. Do not
expand into a full combination tournament while these dependencies are unresolved.

## First commands (from repository root)

Python 3.10+; CPU tests require installed `pytest`, `torch`, and `safetensors`.
No test downloads a model. The native preparation handoff additionally uses the
repository's already pinned tokenizer/runtime; do not upgrade it implicitly.

```bash
python -m experiments.sequential_complementary_v1 selftest --out ../scp-records/mechanics-001
python -m experiments.sequential_complementary_v1 status \
  --binding ../scp-records/mechanics-001/BINDING.json \
  --receipts ../scp-records/mechanics-001/RECEIPTS.json
```

The first command creates a new output directory and refuses an existing one.
Only the mechanics stage should PASS. `status` without receipts intentionally reports
unverified stages as BLOCKED. `LOCAL_TEST_REPORT.json` records the implementation
session's scoped test run; it is not a receipt for the operator's native host.

Useful analysis commands, using scored JSONL with exact `id`, `world_id`, `family`,
and Boolean `correct` on every registered case:

```bash
python -m experiments.sequential_complementary_v1 paired \
  --before OLD_O.jsonl --after NEW_O.jsonl --out ../scp-records/paired-001.json
python -m experiments.sequential_complementary_v1 factorial \
  --a OLD_O.jsonl --b OLD_MR.jsonl --c NEW_O.jsonl --d NEW_MR.jsonl \
  --out ../scp-records/factorial-001.json
```

These commands reject population intersections, missing cases, duplicate IDs and
mismatched lineage. Report both repairs and damage. Unadjusted McNemar diagnostics
are omitted for repeated-world rows; clustered intervals are descriptive open-DEV
results, not equivalence or confirmation. Oracle candidate coverage is analysis-only.

## Data/API contract

`data.audit_rows` accepts explicitly open rows with `id`, `world_id`, `structure_id`,
`template_id`, `entity_pool_id`, `split` (`TRAIN`, `CAL`, `CHECK`), `family`, `question`,
`prompt` (system/user only), `reference`, and a separate `task` specification.
A verifier receives a label-free `PublicCase` plus the task, never the stored answer.
It must independently validate rendering as well as solve the task. The generic API
cannot certify the correctness of a caller-supplied verifier; pin and review it.

Supply **already authorized metadata-only fingerprints**, with explicit source
manifest coverage. The code never opens a protected bank to obtain them. Include
world/structure/template/entity-pool and normalized/digit-blinded text axes. Coverage
must encompass all relevant quarantines and prior exposures, not a convenient subset.
Repeated short questions/common 8-grams are cross-split diagnostics; full-prompt and
structural overlap are binding. External forbidden fingerprints are binding.

The shortcut screen fits question bag-of-words and all single-surface thresholds on
TRAIN and reports group-disjoint CAL performance, including family coverage. It is
not an exhaustive adversary or proof of no leakage. Full corpus acceptance still
requires registered answer heuristics, source rights, counterfactual/position checks,
and human review. Metadata hashes are identities, not signatures or proof of review.

`lcr_v2` is deliberately **not automatically accepted**: the inherited 6000-row,
eight-family/three-length-band corpus needs its complete position, mechanism, and
answer-shortcut receipts. The targeted pilot does not claim to replace that contract.

`data.matched_views` builds F/P, or F/P/T only when every required teacher row exists.
Teacher final answers and steps must independently verify; wrong/missing output is a
failure, not permission to drop a hard world. Preservation anchors are byte-identical
across arms. A new `FINAL_LINE_V1` acquisition output contract is prospective, not a
rewrite of historical exact-answer scores.

`training.compile_handoff` binds each presentation to audited TRAIN identities and
uses the inherited `serialization.compile_row` in direct mode. This preserves Qwen's
prefix/EOS/post-EOS masking. P/T processes are short text in that same channel, not an
unqualified hidden thinking channel. Equal worlds/presentations do **not** mean equal
tokens or compute; both processed and supervised tokens are recorded per arm.

## Exact next native integration task

Work from this branch in an isolated successor. Bind the source `962abc5b...`, trained
partial tensors `07156fb5...` and sparse incumbent `a28629c9...` to actual files;
verify tokenizer/config/template and all 220 tensor names. Recover the missing 4B
checkpoint, training recipe, data lineage and timing scope. Bind a fresh open bank,
all exclusion manifests, numeric budgets and the published protocol hash.

Reuse the existing MQ/RRP supervisor, loss, optimizer and checkpoints unchanged.
Add **new subject-specific qualification**, including an inference-only backend for
MR (the old RRP qualifier includes backward probes and is not an inference-only
permit). Prove zero-update identity, tokenizer/loader parity, cache correctness,
actual precision and resource reservation before permitting a bounded native job.
Do not set the source identity to Supa when loading Titan. Do not weaken inventory
or clean-checkout guards. Implement and test the host adapter before removing the
explicit native refusal in this release.

Write mutable records outside the checkout. The historical train-score-train loop
can dirty its own source tree; retain failed attempts and avoid interleaving legacy
scorers with additional training. No deletion of old evidence is authorized.

Fresh confirmation remains separately reserved and closed. Do not use the spent
256-row long-context bank or its embedded 64 unanswerables for training, selection,
thresholds or confirmation. Existing source metadata may be read; it is not new
experimental evidence. Improvements, preservation, coverage, abstention and cost
must remain separate endpoints.
