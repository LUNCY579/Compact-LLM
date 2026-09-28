# The best panel model forgets how to stop, and the panel reports the opposite

The 150-row arithmetic panel is the only instrument this programme has used. `rineya_reasoning_forge_v1`
is a second one: 16 procedurally generated reasoning families (deduction, scheduling, relational,
truth-puzzles) with long multi-record contexts, an independent solver verifying every reference
answer, and dedicated transfer and length-generalization splits.

It is clean and it is usable:

* **No overlap with the panel.** 0/150 panel stems appear verbatim in forge `train` or
  `test_iid`, and 0/150 with digits blinded. Procedural generation delivered what
  `template_family` disjointness did not.
* **The source is off the floor and far from saturated**: 32/200 exact (16.0%), 168 rows of
  headroom, 199/200 terminating.
* **Almost no format confound**: 32 exact against 33 containment, so one row. The panel has 17 of
  103 failures that are presentation rejections rather than reasoning errors.

Scored on GPU deliberately. A panel probe must run on the CPU because the panel's reference is a
frozen CPU measurement and a ~1e-4 FP32 difference can flip a greedy argmax. Forge has no frozen
reference, so the constraint does not apply and GPU prefill on 2k-token prompts is far faster.
Long rows are skipped rather than truncated: cutting a record-collection prompt can delete the
record the question depends on, turning a reasoning test into a retrieval-failure test.

## The result

Two checkpoints, each trained on `splits_large` (36 reply classes, none of them arithmetic and
none of them forge), probed on the identical 200 forge rows with the identical scorer. Paired
McNemar, exact, two-sided.

| model | panel | forge exact | forge terminated |
|---|---:|---:|---:|
| source | 47/150 | 32/200 | 199/200 |
| B28 full, seed 17 | **73/150** | 27/200 | **184/200** |
| B28last20, seed 17 | 67/150 | 29/200 | **183/200** |

**Accuracy did not move.** B28 full: 10 source-only, 5 model-only, 15 discordant, p = 0.3018.
B28last20: 6 and 3, 9 discordant, p = 0.5078. The apparent 32 -> 27 drop is sampling, and reading
it as a regression would have been wrong.

**Termination collapsed, and it is not close.**

    B28 full     lost 15, gained  0, 15 discordant, p = 0.00006
    B28last20    lost 16, gained  0, 16 discordant, p = 0.00003

Fifteen and sixteen rows lost the ability to emit a stop token, **zero recovered**, in two
independently trained models. A perfectly one-sided effect at p ~ 5e-5 twice over.

## Why this matters more than the accuracy number

On the panel, the same metric moved the other way: termination went **148/150 -> 150/150** for
both models. Finetuning made stopping *perfect* in distribution while breaking it out of
distribution. The panel does not merely miss this regression; it reports improvement on the
identical quantity.

The registered gate could not catch it either. `G1_RETENTION` requires
`min_termination_rate_per_class = 0.8`, and 184/200 is 92%. Nothing fires.

A model that fails to stop on 7.5% more out-of-distribution inputs produces visible garbage in
deployment regardless of what its accuracy says. This is the concrete form of "skill gains
without forgetting catastrophically" failing — and it was invisible until a second instrument
existed.

## What this does and does not establish

* It does **not** show the panel gains are fake. 47 -> 73 is real on the panel and survives the
  noise floor against shallower arms.
* It does **not** show forge accuracy degraded. That comparison is null, twice.
* It **does** show the recipe that maximises the panel score carries an off-distribution cost
  that the panel is structurally unable to see, reproduced across two architectures and two
  trainable-parameter budgets.
* n = 1 per checkpoint, and forge's own seed-noise floor has not been measured. The accuracy
  nulls are consistent with noise; the termination effect is far too one-sided to be.

## Next

Measure forge's noise floor the way the panel's was measured (`NOISE_FLOOR.md`): one arm, four
seeds. Until then, treat forge accuracy differences under ~15 discordant rows as unread. The
termination finding does not need that floor -- 15-0 and 16-0 admit no noise explanation -- but
any accuracy claim on this instrument does.
