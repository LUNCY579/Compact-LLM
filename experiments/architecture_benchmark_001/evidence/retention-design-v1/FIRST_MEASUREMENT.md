# The retention gate's first measurement, and the three defects it found

`G1_RETENTION` and `G2_RETENTION` are implemented, registered and executed. This records what
they measured, and — more usefully — what running them broke.

Every number below is from a schema-2 receipt at implementation digest `125d5481`, CPU FP32,
greedy, 24 new tokens, on the 150-row monitor probe. Open splits only: no sealed or DEV access,
no efficacy claim, and G3–G5 remain closed.

## 1. The frozen source's typed-grammar baseline, measured for the first time

Every retention threshold in the design is expressed relative to what the source can do, and
that number had never been measured under the grammar the gate actually uses. It is now:

| class | typed grammar | legacy headline | termination |
|---|---:|---:|---:|
| `percentage_reasoning` | **21/50 = 42.0%** | 65% | 49/50 |
| `ratio_proportion` | **16/50 = 32.0%** | 50% | 50/50 |
| `time_duration` | **10/50 = 20.0%** | 30% | 49/50 |
| **total** | **47/150 = 31.3%** | — | 148/150 |

**The legacy per-class baseline overstates the source in every class.** A retention floor pinned
to "65% / 50% / 30%" — the figures quoted throughout this programme's documents and in the
handoff that commissioned this gate — would have been calibrated against numbers roughly 1.5×
too high. This is the concrete reason the design refused to carry the old baseline over.

Recorded before the run, so this is a prediction rather than a rationalisation: the typed
grammar was expected to score the source *lower* than the legacy scorer's 33.3% on the 60-row
prefix, because it requires the whole answer to parse and rejects type and unit disagreement.
It does: 31.3%.

## 2. The unreviewed bare-value decision is load-bearing in exactly one class

`bare_policy` is the one axis the label audit left to human review — whether `54` is accepted
for a `54 sensor tags` target. Rather than argue it, both settings are scored on identical
predictions:

| class | permissive (registered) | strict | difference |
|---|---:|---:|---:|
| `percentage_reasoning` | 21/50 | 21/50 | 0 |
| `ratio_proportion` | **16/50** | **10/50** | **6** |
| `time_duration` | 10/50 | 10/50 | 0 |
| total | 47/150 | 41/150 | 6 |

The decision is worth six answers, **all of them in `ratio_proportion`**, and nothing at all in
the other two classes. That is a precise instruction for the pending review: it only has to
settle bare values for counts, money, distance and cup volume, and the answer cannot affect
`percentage_reasoning` or `time_duration`. Both settings agree on the verdict.

## 3. Defect found: the frozen grammar rejects a prefix the source actually uses

This is the one that matters, and aggregates hide it completely — it took reading all 150 raw
answers. Cross-tabulating the typed grammar against the legacy number-sequence scorer on the
source's own output:

| | legacy correct | legacy wrong |
|---|---:|---:|
| **typed correct** | 46 | 1 |
| **typed wrong** | **7** | 96 |

The single `typed+ legacy−` cell is the case the grammar was built for: `1 hour and 30 minutes`
for a `90 minutes.` target, which number-sequence matching rejects and exact duration arithmetic
accepts.

The seven `typed− legacy+` rows are **all one family**, at 100% concentration — a sentence
prefix the frozen contract did not anticipate:

    want '6:55 PM.'   got 'It ends at 6:55 PM.'
    want '4:10 PM.'   got 'It ends at 4:10 PM.'
    want '9:55 AM.'   got 'The coding sprint ends at 9:55 AM.'
    want '9:40 AM.'   got 'End time: 9:40 AM'
    want '10:15 AM.'  got 'It ends at 10:15 AM.'
    want '10:30 AM.'  got 'It ends at 10:30 AM.'
    want '10%.'       got 'The percent increase is 10%.'

The contract's allowed openers are `Answer:`, `The answer is`, `It is`, `It will be`, and `At`
for clocks. The source's dominant phrasing for a clock answer is `It ends at …`, which is none
of them. Six of the seven are `time_duration`, which is why that class reads 20%.

**This is not patched here.** The contract requires a scorer change to be versioned and reviewed
with its baselines re-measured, and widening a grammar immediately after looking at model output
is precisely the move this programme forbids elsewhere. It is the first item for the pending
grammar review, with the evidence attached.

Note the direction, because it is not the dangerous one: a too-narrow prefix list **shrinks the
reference set**, so there are fewer answers a candidate can be caught losing. The bias makes the
gate weaker, not more likely to certify something it should not.

## 4. The training-side gate runs — and demonstrates nothing about training yet

`G2_RETENTION_B28` PASSes 12/12 checks against the reference inherited from step zero, including
both drift controls: the frozen source reproduced its correct set exactly, and the reference
matched its parent's.

It is also **uninformative**, and the receipt says so. `arm_answers_changed` is **0 of 150**:
the registered engineering lane is 25 SGD steps at lr 1e-5 over 3,159 supervised tokens, and
that left every greedy answer byte-identical to the source. Nothing was retained because nothing
moved.

What this receipt does establish is mechanical: the stage executes, binds its two-parent DAG,
inherits rather than recomputes its reference, and recomputes every verdict from its own written
artifact. What it does not establish is any claim about capability under optimizer pressure.
**Read `arm_answers_changed` before `status`.**

## 5. Defect found: a sensitivity diagnostic that manufactured a failure

The first `G2_RETENTION_B28` run reported the strict policy as **FAIL with 6 losses** on a
checkpoint whose 150 answers were byte-identical to the source. No answer had changed, so no
answer could have been lost.

The cause was in the diagnostic, not the model: it scored the strict grammar against a reference
the *permissive* grammar had produced. Those two disagree by construction on exactly the six
bare-value rows in `ratio_proportion` from §2, so the "losses" were a grammar difference wearing
a capability failure's clothes. Each policy now carries its own reference (`reference_ids_by_policy`),
inherited per policy, with a regression test.

A sensitivity diagnostic that cannot distinguish a grammar difference from a capability loss is
worse than not having one, because it manufactures the exact finding the gate exists to detect.

## 6. Defect found: target validation that failed open

Found by the scorer's own specification tests before any model ran. Target validation was lazy —
performed inside the parse — so a `quantity` target with no declared `unit` raised only when the
prediction happened to carry a unit suffix. Given a bare-number prediction it scored as an
ordinary wrong answer, meaning **a malformed specification would have been recorded as a
capability loss**. Validation is now an unconditional whitelist: a typo like `bare_allow` is a
loud error rather than a silently missing permission, and a regression test crosses malformed
targets with six prediction shapes so no path can skip it.

## 7. A sixth digest reproduces everything to the element

Requalification from `8b47b395` through `125d5481`, across three intervening digests:

- `G0`, `G1_IDENTITY`, `G1_WIDTH`, `G1_CED_C28`, `G1_CED_C40`: **zero** non-timer metric
  differences, identical checks, identical statuses. Both CED arms still FAIL on the same three
  strict-parity checks.
- `G1_QUALITY`: B28 PASS 8/8; D40 FAIL on KL 1.4909, termination 0.750, distinct 4/24 — and the
  top-1 knife-edge is preserved unmoved at **0.5032** against its 0.5000 floor.
- `G2_CACHED_B28`: **bit-identical** — same `final_state_sha256`, same 25 losses. The only two
  differing fields are `teacher_cache_sha256` and `recipe_sha256`, both of which embed the
  implementation digest by construction; the `recipe` dict itself is identical.

## A claim of mine this falsifies

The first draft of HOTFIX-005, and the commit message that shipped it, said that across the 18
checkpoints of the 500k matched pair **no** recorded signal separates the capable arm from the
incapable one.

**That is false.** Seven of the eight non-task signals admit no separating threshold and four
invert, but top-1 agreement with the source does separate — weakest capable 0.32798 against
strongest incapable 0.32416. The gap is 0.0038, the registered floor of 0.5 fails both arms, and
the capable arm is still drifting toward the incapable arm's plateau at roughly 0.001 per
milestone, so the whole margin is worth about 3.8 milestones of its own drift. It is unusable as
a gate — but "unusable" and "does not exist" are different claims, and asserting the second
without checking is how HOTFIX-003 produced the claim HOTFIX-004 had to retract.
Artifact: `SIGNAL_SEPARATION.json`.

## What this does not establish

- **No retention claim under training.** The only trained arm measured did not change a single
  answer. The gate's discriminating power is untested against an optimizer that moves a model.
- **No claim about converted arms.** `G1_RETENTION` declares `G1_QUALITY` as a parent, and no
  conversion in this programme passes `G1_QUALITY`, so C28, C40 and D40 cannot hold a retention
  receipt at all. That is the wiring working as intended — a degenerate model has nothing to
  retain — but it means the registered gate has been demonstrated only on the frozen source.
- **B28's pass is largely tautological**, exactly as HOTFIX-004 recorded for `G1_QUALITY`.
- **Label review is pending** for all 150 references, and the grammar review is now open with a
  known defect in it.
- **A finite panel, not a population claim.** 150 prompts, three numeric classes, 24 generated
  tokens, one deterministic run, no seeds. Repeated checks do not accumulate confidence. The
  300-family confirmation certificate is not implemented and is not licensed.

## Reproduction

```bash
PY=C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe
cd C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification
$PY -m pytest experiments/architecture_benchmark_001/tests -q
$PY experiments/rab001_exploration/signal_separation.py
$PY experiments/rab001_exploration/scorer_bias_scan.py \
    experiments/architecture_benchmark_001/outputs/retention-125d5481/B28/retention.json B28_source
```

The gate itself is driven by `qualify-retention`; the exact invocations are in `RUNBOOK.md`.
Artifacts: `outputs/retention-125d5481/{B28,B28-trained}/`, `SIGNAL_SEPARATION.json`,
`SCORER_BIAS_B28_source.json`.
