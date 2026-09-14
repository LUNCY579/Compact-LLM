# MQ-001 data stage: built, compiled, and stopped at the review gate

The branch shipped the training machinery and a **documented but unconstructed** 2,000-example
quality pool. That pool now exists, a frozen plan is compiled from it, and the sequence has
stopped exactly where it is designed to stop: a named human reviewer.

## What ran

```bash
MQ=C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe
$MQ -m experiments.model_quality_gpu_v1.build_pool --out C:/Users/clclc/BusinessProject/mq-data-002
$MQ -m experiments.model_quality_gpu_v1 prepare --source Supa \
   --train .../train.jsonl --dev .../dev.jsonl --excluded .../open-history-exclusions.jsonl \
   --selection-seed 17 --order-seed 17 --out C:/Users/clclc/BusinessProject/mq-runs/prepared-002
```

| artifact | rows | sha256 |
|---|---:|---|
| `mq-data-002/train.jsonl` | 2,000 | `da0142d8…` |
| `mq-data-002/dev.jsonl` | 64 | `a0f2accf…` |
| `mq-data-002/open-history-exclusions.jsonl` | 10,150 | `ef381a9a…` |
| `mq-runs/prepared-002/plan.json` | — | `13f017cc…` |

**Compiled plan** — `kind: native_plan`, internal `plan_sha256` **`81c4727b…`**, **24,910 of
25,000** processed tokens (above the 0.90 floor), **238** train rows selected from the 2,000
candidates, 64 dev rows, **13** update groups. `validate_plan` passes; every mask boundary
satisfies `0 < answer_start < answer_end < len(ids)`.

**Overlap audit: PASS** on id, `template_family`, normalised prompt and digit-blinded prompt
across train / dev / excluded. `semantic_and_generator_leakage` is
`REQUIRES_INDEPENDENT_REVIEW`, which is the gate below.

## How the pool was built

Source is v3.0, **single-turn rows only** — MQ-001 v1 accepts one user/assistant exchange
with an optional leading system turn, and multi-turn rows are excluded rather than
flattened, because flattening silently changes the task. 50,964 of the corpus qualify.

**Evaluation contracts come from the corpus's own answers.** A bare decimal gets
`scorer: number`; an answer of four words or fewer gets `scorer: exact` with that literal
answer; anything longer gets no `evaluation` and contributes response CE only. DEV is
35 exact / 6 number / 23 unscored.

**No `UNKNOWN` abstention rows were produced, deliberately.** MQ-001 requires such a task to
request that contract in its prompt, and v3's `insufficient_information` rows ask for a
natural knowledge-limit sentence instead. Relabelling them as epistemic abstention is exactly
the mislabelling the runbook warns against, so it was refused. An UNKNOWN screen needs
purpose-built prompts and its own review — and note that RAB-001 already measured this
behaviour directly on the long-context suite, where Supa emits `UNKNOWN` on 13 of 64
unanswerable rows and Opallo on 2.

**DEV is held out by `template_family`, not by row**, because different rows of one family
render the same stem with different fillers; a row-level holdout would leave paraphrases of
every DEV question in TRAIN and the id guard would not notice.

**DEV is deliberately enriched for scoreable answers.** A first build (`mq-data-001`, kept
and marked superseded) took the lowest-id row per family and left 31 of 64 rows unscored,
halving the primary metric's sample. Selection is on answer *shape* — a corpus property
fixed before any model runs — never on a model outcome, and it cannot bias the arm
comparison because every arm is scored on the same rows. It does make DEV a short-answer
instrument rather than a representative sample, which is recorded in `POOL_MANIFEST.json`.

**The exclusion bank is the contact point with RAB-001.** It carries the 150-row frozen
arithmetic panel and the 10,000-row `splits_large` pool so the audit can prove MQ touches
neither. Panel rows are multi-turn and are represented by their final user turn — the
question stem, which is the thing that would actually leak. **4,408 candidate rows were
dropped for normalised-prompt collision with that history**, so the guard is doing real work
rather than passing vacuously. No exclusion-bank row is ever trained or scored on.

## Two blockers, both real, neither a defect

**1. The review gate is a human decision.** `DATA_REVIEW_PENDING.json` binds plan
`81c4727b…` and requires `decision: APPROVED`, a named `reviewer`, a written `rationale`, and
four explicit boolean attestations — source rights, label/response contracts, scenario and
generator leakage, and dev-only-no-confirmation-outputs. **I have not filled any of these
in.** Self-approving a data review is the one thing this file exists to prevent.

**2. Native GPU qualification cannot run on this host.** `.venv-mq001` holds the exact pins
but `torch 2.10.0+cpu`; qualification needs a CUDA build, and the guards require **6 GiB
available host RAM and 6 GiB free GPU**. The host is at **~1 GiB free, 94% memory load,
23.5 GiB committed against 15.2 physical**. Even with a CUDA wheel installed it would refuse,
correctly.

## The operator sequence from here

1. Review and approve (or reject) `mq-runs/prepared-002/DATA_REVIEW_PENDING.json`, by name.
2. Free host memory — closing Chrome and idle Claude sessions recovered ~4 GiB earlier today.
3. Install an official Torch 2.10.0 **CUDA** wheel into `.venv-mq001` only.
4. `qualify` (zero updates), inspect its evidence, then run **one** explicitly chosen cell.

Nothing downstream of step 1 should start before step 1 is done by a person.
