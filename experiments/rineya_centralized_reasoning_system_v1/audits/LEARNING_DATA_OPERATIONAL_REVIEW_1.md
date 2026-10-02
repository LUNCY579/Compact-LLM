# Learning-data operational review 1

Reviewer `/root/runtime_audit`; builder `/root/data_audit`, 2026-10-02.
**REPAIR_REQUIRED: fixed-bank provenance and actual grouping are not enforced at
the TRAIN target-exposure boundary.** This is the learning-data milestone's first
operations review, not an exception or another repair of a quarantined pipeline.
No model/training/reader/probe/source-candidate gate is opened by this report.

I did not build this dataset, oracle, renderer or tests. I read the complete frozen
402-line source,202-line tests,597-line contract and retained initial shortcut
counterexample directly. Shared model ancestry, workspace privileges, supplied
concept rules and earlier pipeline work limit independence. This is fresh source/
CPU interface scrutiny, not independent neural or real-world truth.

## Exact subjects

| Artifact | SHA256 |
| --- | --- |
| learning_data.py | 59373cbbb235d2e561ef062e8f21d6fe8b151f423d2d03e8b1e10e59cc17eb8a |
| tests/test_learning_data.py | f9cc3a268c1f0fae6b97223469232a1134e0ddbd8eb2547169e0bcbfcadce805 |
| contracts/LEARNING_DATA_V1.json | 1114ac25f67046e5e2695557ffdc5682e0360883498154463acbe795797142b6 |
| initial identifier-leak counterexample | 83ae5598f0f184baa251f4922d0461062b45037814f399494ada36466a29da96 |

Pins were checked before inspection and again before report publication. The
initial draft is preserved in history/learning_data_draft001; the builder also
reports preserving this first repair in history/learning_data_repair1. No builder,
core, scorer, runtime, dataset or registration source was changed by this review.
Only this audit report is written.

## Executed validation and supported scope

All17 supplied tests passed independently in0.148 seconds, with no skips/errors.
They cross-check all32 sources against the supplied CPU core and independent typed
latent oracle,32 consistent renamings, hand semantic controls, role restrictions,
metadata omission and the registered weak shortcut probes. No Torch/Transformers/
tokenizers/safetensors module, native/model/tokenizer call, current host telemetry,
canonical host database, Gold bank, confirmation or protected CHECK/FINAL material
was used. Additional controls ran only against in-memory copies.

The deterministic builder's actual public/gold/latent/lineage hashes, parent
membership and order match the frozen contract. It produces32 distinct parents,
one variant per parent,8 cases per registered family and2 of each status per
family. Source UTF-8 sizes are450..576 bytes; canonical target maximum is39 bytes.
The named TRAIN composition groups are disjoint from the declared V1 DEV family
names, with shared grammar/primitive/chronology semantics explicitly disclosed.
That grouping claim is not semantic, pretraining or natural-language isolation.

Public model_input returns only deep-copied source/query/rules, omitting case,
parent, family, split and private target/latent metadata. Extra public gold fields,
CONFIRM split, bool query time, oversized multibyte source, duplicate case IDs,
bool lineage variant, extra hostile source and native_eligible=True reject in
independent controls. Private gold exposure is explicitly TRAIN-only; rejected
DEV/CONFIRM/CHECK/FINAL role paths produce no new bank. Targets are exactly the
canonical two-field status/value oracle answer, not invented explanations or
validated-procedure distillation. Source-only code imports no runtime/parser/
scorer/model and calls no shell or filesystem generator API.

Native/training eligibility flags remainFalse, target token/mask fields areNone,
and no tokenizer fit, loss-mask ownership, learning step, efficiency frontier,
calibration or retention claim is made. Byte counts are not token counts. Future
qualified tokenizer/template, actual fit/masks, source retention enrollment and
separately reviewed learning recipe remain required.

## Preserved initial shortcut and limits

The initial draft's query entity numeric suffix modulo4 predicted32/32 statuses
without source semantics. Its13 tests established engineering/oracle consistency
and missed that shortcut. The negative evidence and draft sourcea85e/tests274a
remain preserved; this review does not replace that result with later successes.

The current bank removes numeric public ordinals, uses label-independent opaque
per-field names and randomized within-family status assignment, matches every FACT
count to4, and checks semantic renaming. All10 frozen shortcut measurements were
recomputed and match the contract. Status scores are8/32 for identifier modulo,
7/32 identifier hash,10/32 draw position,5/32 order position,8/32 family majority
and8/32 FACT-count majority. The last two are correctly marked TRAIN-label-fitted.
These probes and renaming controls refute the identified perfect shortcut; they
do not establish absence of all shortcuts, out-of-sample generalization or actual
learning. Supplied code and a deterministic32-parent draw can be memorized.

## LD-OPS-001 — P1: declared family/cell counts do not bind actual records

validate_bundle checks manifest.family_counts against a static8-per-family dict
and global four-status balance. It does not recompute actual case-family counts,
status cells per family, or semantic composition consistency with each declared
family. Lineage.family need only repeat the supplied case.family.

Independent production-API counterexamples, using deep copies and no source edit:

1. Set every case.family and lineage.family to empty_joint_gates. Recompute public
   and lineage hashes. validate_bundle returnsTrue and training_targets returns32
   targets, while actual counts are32/0/0/0 and the manifest still claims8/8/8/8.
2. Assign each case's family according to its private gold status, and mirror that
   family in lineage; recompute the same hashes. Actual family counts are8each,
   but each family contains8 of one status rather than2 of each. Validation still
   returnsTrue and exposes32 TRAIN targets.

Neither counterexample claims generated bank bytes are wrong. They prove a
consumer can accept a corrupt, coherently rehashed grouping under the same
registration labels. Later per-parent/family learning dose or disjoint-family
claims could then be based on declared rather than actual membership.

Verification: derive actual family counts and per-family status cells from the
records, compare them to the exact registered structure, and bind the claimed
composition/family to the registered latent/source identity. Reject both controls
before target exposure. Preserve all scheduled32 denominators; do not repair
corruption by resampling or rewriting expected registration bytes.

## LD-OPS-002 — P1: self-consistent hashes permit false fixed-bank provenance

The validator compares supplied manifest hashes to supplied records, but does
not consume an independently fixed registration or regenerate/verify the exact
registered101/211 bank. Parent and case IDs are bounded/unique strings, not checked
against the registered ancestry/identity formula or contract membership. Native
flags are still closed, but training_targets treats this same validator as enough
for intentional gold exposure.

Independent counterexamples:

- Replace all parent IDs with unique forged-parent-i strings, mirror them in
  lineage, and recompute public/lineage hashes. Validation returnsTrue and32 TRAIN
  targets are exposed under claimed registered101/211 provenance.
- Replace all case IDs with unique forged-case-i strings; consistently rekey gold,
  latent and lineage, recompute hashes and shuffle the new IDs with order211.
  Validation again returnsTrue and32 TRAIN targets are exposed.

These are exact production predicates, not a rewritten checker. Hash equality
shows consistency, not authority to redefine the frozen draw, parents or ordering.
The existing unresealed-corruption tests reject obvious modifications but miss
these coherently resealed controls. Contract metadata alone does not repair an API
that never checks that fixed contract/bank identity.

Verification: bind every target-exposure/eligibility consumer to an immutable
registered bank identity outside the caller's mutable bundle, or verify the exact
deterministic registered draw and its frozen hashes/ordered parent/case/lineage
membership. Preserve source/target byte associations. Add coherently resealed
parent/case/source/query/world/gold/order/family corruption tests; semantically
self-consistent alternate banks must not masquerade as this registration. A newly
intended bank requires its own explicit versioned registration, not forged hashes.
The protection is cooperative artifact integrity, not an OS security claim against
same-user source/registration forgery.

## LD-OPS-003 — P2: loaded bundle seeds accept wrong types

The builder enforces exact integer draw/order arguments, but validate_bundle only
uses equality for manifest.draw_seed/order_seed. A copy with101.0/211.0 passes and
training_targets still returns32. This misstates the typed registration boundary.
Require exact integer seed types at the consuming boundary and reject bool/float/
string/null variants alongside the fixed-bank controls. No new draw or data is
needed to close this finding.

## Reproducer outline

The independent controls used this resealing operation after the mutations above:

```python
for key, value in [('public_sha256', b['cases']),
                   ('gold_sha256', b['private']['gold']),
                   ('latent_sha256', b['private']['latent']),
                   ('lineage_sha256', b['private']['lineage'])]:
    b['manifest'][key] = data.digest(value)
order = [c['id'] for c in b['cases']]
random.Random(211).shuffle(order)
b['manifest']['ordered_case_ids'] = order
assert data.validate_bundle(b)
assert len(data.training_targets(b)) == 32
```

Mutating all families requires only case/lineage family changes; parent forgery
requires only their parent fields; case forgery rekeys all private maps and case
membership. Float seeds require no resealing because they are not currently
included in the four immutable-content digests. Source and targets stayed within
bounds and native flags stayedFalse in every accepted control.

## Disposition and continuation

The generated corrected bank itself has useful reproducible CPU semantics and
honest limited shortcut measurements. Approval is withheld for this frozen
consumer/provenance implementation until the three findings close on fresh bytes.
This remains a bounded learning-data review/repair milestone; no pipeline owner
exception, new holdout, inference, training or actual probe action follows.
Retain the initial perfect-leak history, current17-test success scope and all
negative controls. Root's separate scientific review must also assess the declared
hypothesis and transfer/shortcut limitations. No aggregate CPU pass establishes
native model eligibility or independently confirmed learning.
