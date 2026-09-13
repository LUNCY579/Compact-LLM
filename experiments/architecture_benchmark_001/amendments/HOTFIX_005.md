# HOTFIX-005 — G1_RETENTION / G2_RETENTION: the gate that scores answers

Date: September 13, 2026 (America/Chicago). This amendment **adds** two gates and tightens the
receipt DAG. It loosens nothing, changes no existing threshold, and opens no closed stage.
G3/G4/G5 remain closed.

## The defect this fixes

HOTFIX-003 closed the token-salad hole at initialization. It did not close this one: **every
check `G1_QUALITY` applies is likelihood- or teacher-agreement-shaped, and those are now
measured to decouple from capability.**

The sharpest measurement is the token-matched 500,099-target comparison
(`evidence/trainability/LONG_BUDGET_RESULT.md`, commit `ffc52a0`), CED C28 against an ordinary
matched control on identical rows, identical recipe, identical seed, 0.376 of an epoch so no
row is seen twice:

| | frozen source | C28 @ 500k | control @ 500k |
|---|---:|---:|---:|
| held-out CE | 5.5782 | **0.4727** | 0.3569 |
| reply-region CE | 2.6492 | 1.1039 | 0.6091 |
| KL to source | 0 | **8.5843** | 10.0334 |
| top-1 agreement | 1.000 | 0.3224 | 0.3280 |
| held-out task | **20/60** | **2/60** | **26/60** |

C28's final held-out CE is **5.1055 nats better than its own teacher**, from a model that
answers 2 of 60 questions the teacher answers 20. The 95% Wilson intervals are disjoint:
2/60 is [0.0092, 0.1136] against 26/60's [0.3157, 0.5590].

Apply the registered `G1_QUALITY` thresholds to those two trained checkpoints and the gate does
not merely miss the difference — it **fails the good model harder**. (These are DEV-split
quantities from the exploratory 500k run, against a DEV source CE of 5.5782; `G1_QUALITY` scores
on the 156-row calibration split, where the source CE is 5.430896. The comparison below applies
the registered *thresholds* to DEV measurements and is an illustration of where those thresholds
point after training, not a re-run of the stage.)

| check | limit | C28 @ 500k | control @ 500k |
|---|---|---|---|
| `ce_increase_within_margin` | ≤ 0.5 | −5.1055 PASS | −5.2213 PASS |
| `kl_to_source_within_margin` | ≤ 1.0 | 8.5843 FAIL | **10.0334 FAIL (worse)** |
| `top1_agreement_above_floor` | ≥ 0.5 | 0.3224 FAIL | 0.3280 FAIL |

Both arms fail on the same two checks for the same reason — training moves a model away from
its teacher — and the incapable arm sits *closer* to the teacher on both. Those thresholds are
calibrated for a freshly converted model, and they are not merely uninformative after training;
they are pointed the wrong way. The remaining two checks do not help: at 500k C28 terminates
60/60 and emits 28 distinct answers over 60 prompts against a floor of 15.

All eight non-task quantities the run records were checked for a separating threshold across
all 18 checkpoints of that matched pair, labelling capable as task ≥ 16/60 and incapable as
≤ 4/60 — the control spans 16..26 and C28 spans 0..4, so no checkpoint falls in the gap and the
labelling needs no tie-break. Artifact:
`evidence/retention-design-v1/SIGNAL_SEPARATION.json`.

**Seven of the eight admit no separating threshold at all, and all seven invert** — for each,
some incapable checkpoint reads *better* than the weakest capable one. Four of the sharpest:

| signal | inverted witness |
|---|---|
| held-out CE | C28 @500k (2/60) 0.4727 beats control @0 (20/60) 5.5782 |
| reply-region CE | C28 @500k (2/60) 1.1039 beats control @0 (20/60) 2.6492 |
| KL to source | C28 @25k (0/60) 2.9378 beats control @500k (26/60) 10.0334 |
| reply-region KL | C28 @125k (3/60) 1.6752 beats control @500k (26/60) 3.1319 |

The remaining three that invert are reply-region top-1 agreement, termination count and
distinct-output count. Reply-region CE is the signal most likely to be adopted as a capability
proxy, and it is one of the seven. It is therefore **not** adopted here and is recorded as a
diagnostic only. Counts are from `SIGNAL_SEPARATION.json`'s own `inverted_witness` fields; an
earlier draft of this table said "four of them invert", which undercounted its own artifact.

**One signal does separate, and saying otherwise would repeat HOTFIX-003's error.** Top-1
agreement with the source separates the two groups — but by **0.0038**: the weakest capable
checkpoint sits at 0.32798 and the strongest incapable one at 0.32416. Three things make that
unusable as a gate. The registered `G1_QUALITY` floor is 0.5, which **both** arms fail. No such
threshold was ever registered, so choosing one inside a 0.0038 window after seeing these 18
numbers is exactly the tuning `forbidden` lists. And the separation is an artifact of where the
run stopped: the capable arm's top-1 is still falling toward the incapable arm's plateau over
its last three milestones (−0.0080, −0.0016, −0.0010), so the entire gap is worth about 3.8
milestones of its own recent drift and closes at a larger budget. One seed, one arm pair.

The honest summary is therefore not "no likelihood-shaped quantity can ever separate these
models" but: **of every quantity this programme actually recorded, none supports a defensible
registered threshold, seven of the eight point the wrong way somewhere, and the single
exception is a 0.0038 knife-edge that is still closing.**

## What is added

Two stages, forward-only. Neither trains anything and neither can license training.

    G1_RETENTION_<arm>    parents [G1_QUALITY_<arm>]
    G2_RETENTION_<arm>    parents [G2_CACHED_<arm>, G1_RETENTION_<arm>]

`G1_RETENTION` measures retention at initialization and fixes the immutable reference.
`G2_RETENTION` measures the same quantity on a checkpoint that an optimizer has touched — the
training-side gate. Checks:

    source_probe_informative          every class has >= 1 source-correct terminated item
    no_reference_capability_lost      0 correct-to-incorrect transitions, per class
    generation_terminates_per_class   >= 0.8 terminated per class (reuses G0)
    probe_disjoint_from_training      zero family and id overlap with the training splits
    calibration_overlap_as_declared   observed overlap still equals the registered 2
    typed_scorer_recomputed           checks recomputed from the prediction artifact
    bare_policy_verdicts_agree        the strict and permissive policies reach the same verdict
    source_unchanged, probe_unchanged, calibration_unchanged, pilot_unchanged

`G2_RETENTION` adds two drift controls, `source_answers_reproduce_parent` and
`reference_matches_parent`, and both stages additionally bind the evaluated weights: the
`--checkpoint` supplied must hash to the `model.safetensors` its parent lineage declares, and
B28 at `G1_RETENTION` is the frozen source and may not be given one at all.

The reference is the **union of frozen-source-correct and arm-initialization-correct** items.
At step zero the arm's correct set is its initialization set, so the losses this reports are
exactly the source-correct items the conversion already dropped — which is why a broken
initialization blocks the retention lane rather than entering it. A broken initialization is a
recovery problem; passing a later recovery demonstration cannot retroactively claim
uninterrupted retention. **A gain never offsets a loss**, and the reference may not be
recomputed after training: `reference_reset_after_training` is in `forbidden`.

### The DAG change

`G1_QUALITY` was a **leaf**: nothing in the ladder consumed it. `G1_RETENTION_<arm>` now
declares it as a parent, and `G2_RETENTION_<arm>` declares both the trained checkpoint and the
step-zero retention receipt. Retention therefore cannot be claimed without quality lineage, and
quality cannot be claimed without construction lineage.

**This does not interlock training, and an earlier draft of this amendment overstated it.**
`G2_CACHED_<arm>` — the only optimizer path in the supported runner — has exactly the parents it
had before HOTFIX-005 and still validates only its construction receipt and matched control. An
arm whose `G1_QUALITY` FAILs can still run 25 engineering steps; what it cannot now do is hold a
retention receipt afterwards. These are terminal certifying stages, not a training guard. A real
interlock would require `G2_CACHED` to declare a quality or step-zero-retention parent, which
would invalidate the existing engineering lineage and is not done here.

## Scoring: why a new scorer was required

The exploration scorer compares the sequence of numbers in a string. That is format-tolerant in
the right way and type-blind in the wrong way. Measured on its own fixtures, it accepts
`10:25 PM` for `10:25 AM`, `5 hours` for `5 minutes` and `not 5` for `5`, and it rejects
`60 minutes` for `1 hour`. It also scored `54 sensor tags` as correct against a stored reference
reading `54 spare fuses` — for a question that asks about sensor tags. A retention verdict may
not rest on that.

`retention.py` implements the grammar frozen in `TYPED_SCORING_CONTRACT.md`
(sha256 `871d3bb8…`, bound in `protocol.json`). It parses the whole answer into a declared type
and compares exact rationals: no floats, no number scraped from surrounding prose, and EOS
termination required for correctness. All 51 counterexample rows of that document are test
cases written against the hashed specification.

**This is a scope reduction as well as a tightening.** The grammar accepts exact answers in a
bounded declared format and rejects valid free-form English that means the right thing. Every
source baseline in this programme was measured with the legacy scorer and **cannot be carried
over**: the 65% / 50% / 30% per-class figures do not transfer, and the source must be measured
anew under this grammar before any threshold is applied to any arm.

## Threshold provenance — stated plainly

The rules are transcribed unchanged from `RETENTION_GATE_DESIGN.md` at `f7bc290`, written
before this code existed. They are policy tolerances, not fitted values:

- **0 losses per class.** A gate permitting some forgetting must justify how much, and no
  measurement here supports a number. Zero needs no calibration.
- **Termination ≥ 0.8 per class** reuses `G0.min_termination_rate` rather than inventing a
  value, exactly as HOTFIX-003 did.
- **≥ 1 source-correct terminated item per class** is the weakest statement that rules out the
  `arithmetic_chain` failure, where the teacher scores 0/20 and the probe measures nothing.

**Disclosure, in the order HOTFIX-003 got wrong and HOTFIX-004 corrected.** Historical arm
scores under the *legacy* scorer were already known when these rules were written. Applying
these gates to C28, C40, D40 or B28 is therefore **retrospective demonstration, not
preregistered validation**, and this amendment claims only the former. What was *not* known is
any score under this grammar, for any model including the source.

No claim is made that no plausible setting of these thresholds changes any arm's verdict. That
is the claim HOTFIX-003 made and HOTFIX-004 had to retract.

## The one reviewable axis, and why it is measured rather than argued

`TYPED_SCORING_CONTRACT.md` leaves one decision to human review: whether a bare number is
accepted where the prompt arguably fixes the unit (`54` for `54 sensor tags`). The label audit
recorded this as pending and did not decide it.

`bare_policy` is registered as `permissive`. It is **not** presented as settled: every receipt
also reports the counts under `strict`, so the effect of the unreviewed decision on the verdict
is visible in the artifact rather than argued in prose. If the two policies ever disagree about
a verdict, that is a blocking review item, not a rounding error.

**Each policy carries its own reference set**, recorded as `reference_ids_by_policy` and
inherited per policy by `G2_RETENTION`. This is not a detail. The first implementation compared
the strict grammar against a reference the permissive grammar had produced; those disagree by
construction wherever a bare value is involved, so it reported six losses and a FAIL on a
checkpoint whose 150 answers were byte-identical to the source. A sensitivity diagnostic that
cannot distinguish a grammar difference from a capability loss is worse than none, because it
manufactures exactly the finding the gate exists to detect.

Duration rows are exempt from the policy: only 3 of the 19 duration prompts name the requested
unit, so for the other 16 a bare `90` cannot distinguish minutes from hours and is rejected
under both settings. Those three row ids are fixed in the contract.

## Known exposure, disclosed not gated

Two of the 150 probe families also appear in the open `calibration` split, which `G1_WIDTH`
uses for activation statistics. The count is registered as
`declared_calibration_family_overlap: 2` and the gate fails if the observed overlap stops
matching it — a drift detector, not a tuned threshold. The probe is fully family-disjoint and
id-disjoint from the training splits, which is the exposure that would matter during training.

`human_review_status` is `pending` for every corpus row, and the label audit is machine
arithmetic only. The 150 references were independently recomputed: 0 numeric mismatches, and 4
rows whose stored reference names a different counted object than its own question. Those 4
resolve to the prompt's entity in the target overlay; the historical labels are left unchanged.

## What this stage cannot do

Passing means an arm did not lose a reference-correct answer on **150 prompts of three numeric
classes, 24 generated tokens, one deterministic run, no seeds**, under a bounded answer grammar
whose label review is pending. It is not evidence of quality, recovery, efficacy or breadth of
retention, and it does not substitute for a registered DEV or sealed evaluation.

It is a **finite-panel measurement, not a population claim**. Repeated monitor checks do not
accumulate confidence. The separately designed confirmation certificate — a fresh
300-family-per-class panel with exact binomial bounds — is **not** implemented here and is not
licensed by this amendment: `evidence/retention-design-v1/CONFIRMATION_INVENTORY.json` records
that only 303 nominal families per class remain, that exposure completeness is `false`, and
that family-level statistical independence is `established: false`.

Note also that B28's pass is largely tautological: for that arm the implementation compares the
source to itself, so the reference is its own correct set and losses are 0 by construction. All
of this gate's discriminating power is on converted and trained arms — the same caveat
HOTFIX-004 recorded for `G1_QUALITY`.

**A pass is uninformative when the arm did not move, and the receipt must say which case it
is.** `arm_answers_changed` records how many of the probe answers differ from the source's. The
first `G2_RETENTION_B28` run returned **0 of 150**: the registered engineering lane is 25 SGD
steps at lr 1e-5 over 3,159 tokens, which left every greedy answer byte-identical. That receipt
therefore demonstrates that the gate runs, binds its DAG and reproduces its parent — and
demonstrates nothing whatever about retention under optimizer pressure. Read `arm_answers_changed`
before reading `status`. The training-side gate has not yet been exercised on a checkpoint that
actually changed, and no claim here says otherwise.

**The gate's own first measurement found a defect in its own scorer.** Reading the 150 raw
source answers — not the aggregate — showed 7 rows the legacy scorer accepts and this grammar
rejects, and all 7 are one family: a sentence prefix the frozen contract did not anticipate
(`It ends at 6:55 PM.`, `End time: 9:40 AM`, `The percent increase is 10%.`). Six of the seven
are `time_duration`, which is why that class reads 10/50 here. Evidence:
`evidence/retention-design-v1/SCORER_BIAS_B28_source.json`.

That list is **not** patched in this amendment. The contract requires a scorer change to be
versioned and reviewed with its baselines re-measured, and widening a grammar immediately after
looking at model output is the move this programme forbids everywhere else. It is recorded as
the first item for the pending grammar review. Note the direction: a too-narrow prefix list
shrinks the reference set, which makes the gate **weaker** — fewer answers a candidate can be
caught losing — so the bias does not inflate any retention claim.
