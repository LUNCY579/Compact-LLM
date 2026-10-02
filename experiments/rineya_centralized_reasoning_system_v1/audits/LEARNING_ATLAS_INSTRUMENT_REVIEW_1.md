# CRS learning-atlas instrument review 1

Client date 2026-10-01; investigator `/root/data_audit`. Scope: read-only review
of the preliminary learning contract, data/scorer/atlas implementation and theory,
followed only by the separately assigned versioned CPU instrument repair below.
No new task bank, native input, confirmation material, model, training step or
process/registry change is authorized by this report. Protected central banks
remain closed. Shared workspace/model ancestry is disclosed. The original
scorer/atlas and epoch001 receipts must remain unchanged.

Decision on the preliminary microstudy: **ACTUAL_DATA_AND_TRAINING_INSTRUMENT_NOT
YET_QUALIFIED**. The closed contract correctly does not grant native training.
The next useful correction is a frozen data/mask/evaluation contract; it is not
another prose declaration that the native gates have passed.

## Inspected bindings and existing material

| Artifact | SHA256 |
| --- | --- |
| `contracts/LEARNING_ATLAS_MICROSTUDY_V1.json` | `9c0113afe749cb375917df78937e6ab9125afa9e52f081a7169c6e8d8a642d73` |
| `data.py` | `6a1a051e9f2da0411760f66c42f77de42ea402050c54e8c184c9fbb9a4730091` |
| original `scoring.py` | `4ea996b7c3eca875c4691e2274b3beb492ba5b3e704ec77d3f200f7697b54c21` |
| original `atlas.py` | `ab6fc3745f84c5f338e204d1ad7fb7f88bb3a8241267c2e2a66c69c6a000e785` |
| `theory/learning_atlas_v1.json` | `b6ce52e987b682398f40e0b1fac0904df4025fe7a3d6a21b20f04a96412a5b59` |
| `theory/hypotheses_v1.json` | `0401ea92278191e3aea500f72096f4239af7c9286181c9ed1591342249b98163` |

Read the existing epoch001 result/registration metadata without rerunning its
generator or opening a holdout. Existing banks are already exposed CPU material:

| Bank | Seed | Cases | Unique parent groups | Scope |
| --- | --- | --- | --- | --- |
| TRAIN fixture | 20261001 | 32 | 16 | Four assignment families; all32 SUPPORTED |
| DEV fixture | 20261001 | 64 | 32 | Eight held-composition groups, four parents/group; SUPPORTED36, UNKNOWN16, CONTRADICTORY8, REFUTED4 |
| Reader DEV diagnostic | 20261001016 | 16 | 8 | One parent/group, two variants; status counts11/3/1/1 |

The LA contract instead names training draw101 and order211. That future
32-parent draw has no frozen bank, optimizer IDs, tokenizer/masks or source
competency enrollment in the inspected package. The atlas is honestly empty;
native curves/efficiency/frontier/mechanisms remain unmeasured.

## Findings

### LA-I1-001: cases, unique parents and optimizer presentations are conflated

**Evidence:** `data.py` groups indexes in pairs under `index // 2`. Existing
TRAIN32 metadata contains16 parents. The microstudy proposes32 parents,
eight updates and accumulation4/batch1, which permits32 optimizer presentations
in one pass, not64 sibling variants.

**Impact:** Feeding existing32 rows while reporting32 independent examples
doubles the parent count. Feeding two variants for32 parents requires64
presentations/16 updates and violates the registered eight-update dose.

**Counterexample:** The two case IDs with one parent hash have the same sampled
names/times and variant ancestry; they cannot count as two unique acquisitions.

**Verification:** Freeze exactly32 optimizer parent IDs, one selected variant
per parent, ordered case IDs and accumulation groups; count parent IDs from
actual consumed rows. Record discarded/unselected siblings and failure exposure.
Checkpoints4/8 then correspond to16/32 presentations and at most16/32 unique
parents. Do not silently substitute epoch001 seed20261001 for draw101.

### LA-I1-002: there is no within-family acquisition bank in the current DEV split

**Evidence:** DATA_V1 assigns each of four TRAIN families wholly to TRAIN;
DEV has eight different registered composition groups. The hypothesis names
within-family acquisition versus held-composition transfer, but no separate
unused-parent within-family probe is frozen.

**Impact:** Training-set accuracy can look like acquisition through memorization;
calling current DEV “within-family” contradicts the registered family split.
New seeds or family names do not establish unfamiliar-template transfer.

**Verification:** Register an optimizer bank and disjoint unused-parent
TRAIN-role probe within the same open development domain, explicitly labeled
instance-disjoint/template-shared. Keep it TRAIN-role rather than falsely
relabeling shared generators family-disjoint DEV. Use the existing DEV groups
only as held-composition development transfer. All siblings/renames/counterfactuals
of a parent stay with its bank intent and never enter both optimization and
evaluation. Fresh confirmation remains deferred.

### LA-I1-003: current semantic scores require perfect format

**Evidence/reproducer:** `score_one({'status':'SUPPORTED','value':'Alice',
'extra':'irrelevant'}, {'status':'SUPPORTED','value':'Alice'})` returns false
for semantic, format, status and value correctness. `score` defines semantic
and strict accuracy from the same counter.

**Impact:** Removing an extra field can appear as a reasoning gain even though
status/value were already correct. The current instrument cannot test the
hypothesis's competing explanation “format learning.”

**Verification:** Version a bounded semantic projection independent of strict
format. Preserve strict unique JSON/type/finite-value parsing; uniquely readable
correct status/value with extra fields may be semantically correct while strict
format is false. No prose guess/extraction from gold. Operational failure must
remain failed and cannot become UNKNOWN/abstention. Retain old epoch scores.

### LA-I1-004: atlas schema accepts inconsistent dose metadata

**Evidence/reproducer:** A synthetic native-shaped row with32 unique parents,
eight presentations, undefined family/depth/draw/order/evaluation seeds, valid
hash-shaped strings and a finite acquisition estimate passes `validate_row`.
No synthetic row was saved into the atlas.

**Impact:** Efficiency denominators can be inconsistent or unreproducible.
Hash syntax and caller-written `NATIVE_EXPLORATORY` are not execution attestations.

**Verification:** Version strict explicit family/depth/seed and consistent dose
validation (`unique_parents <= presentations`, updates compatible with exposure),
require exact bindings and a measurement receipt association before calculating
a native efficiency. Receipt-byte association is still not truth/qualification;
the independently reviewed registrar must supply actual executed evidence.

### LA-I1-005: duplicate variants can change the parent estimate

**Evidence/reproducer:** Two parent records with deltas+1 and−1 yield0.
Duplicating the +1 record changes `paired_parent_interval` to1/3. Records lack
case/variant identity, so duplicate delivery cannot be detected.

**Impact:** Parent clustering fixes correlated resampling but does not fix
duplicated or missing observations. A retry can bias a point estimate without
increasing parent count.

**Verification:** Require unique case IDs and unique variant IDs within parents,
reject duplication and changed parent/family bindings; compare against the
frozen expected case bank before estimating. Missing/failed scheduled cases
must stay accounted. Bootstrap whole parent vectors, conditional on the declared
family mix; never call variant/checkpoint/order repetitions independent draws.

### LA-I1-006: absent incumbent is silently replaced with source

**Evidence/reproducer:** `skill_matrices(checkpoints, source)` defaults incumbent
to source and emits a numerical `incumbent_losses`, despite no approved
incumbent identity or responses having been supplied.

**Impact:** Source-only retention can be misreported as preservation against
both source and approved incumbent.

**Verification:** Missing incumbent remains UNAVAILABLE with undefined incumbent
denominators/losses. Explicit incumbent panels must bind matching fixed item
membership plus actual checkpoint identity externally. Do not invent an
approved CRS incumbent from a source model name.

### LA-I1-007: native serialization, response masks and original-skill panels are missing

**Evidence:** LA contract specifies caps/optimizer/qv rank8, but not actual
prompt/target token IDs, chat template, response boundary/masks, EOS counting,
loss normalization, initializer seed, native decoding/parser, evaluation IDs or
legacy source-correct competency banks. DATA source is CPU grammar and raw
public JSON, not a measured native prompt. Native D2 qualifies none of these.

**Impact:** Prompt learning, leaked metadata, truncation or response-only
supervision can change the intervention. A state-tracking panel alone cannot
measure broad original model skills or calibrated uncertainty.

**Counterexamples:** Independently tokenized prompt/target may disagree at a
BPE boundary; unmasked prompt tokens change completion SFT into whole-message
training. Serializing public family `incomplete_coverage` exposes a status cue.
An eight-case status subset cannot certify a2pp retention margin. Model output
without probability/confidence data cannot establish calibration.

**Verification:** Freeze byte-exact native renderer, joint token streams and
verified response masks; model policy receives only source/query/rules, not
case/parent/family/split/oracle metadata. Enroll actual unchanged-source responses
on independently solved legacy strata before updates. Keep calibration,
language and long-context UNMEASURED until their own qualified instruments.

## Concrete next instrument correction

This is a prospective contract specification, not a generated dataset or a
launch permit. The smallest consistent adaptation is ordinary final-answer
SFT, external procedures/executors disabled during model-only measurements.
Origin is supplied independent-oracle supervision, not Rineya-discovered or
validated-procedure distillation.

1. Freeze a TRAIN-role draw101 pool with at least64 parent groups across the
four existing assignment subfamilies. Before any model output, choose32 optimizer
parents (eight/family) and32 disjoint unused-parent probes (eight/family); choose
one variant per parent by a declared ID/seed rule and preserve both siblings
in lineage. Do not use probe labels for updates. Existing TRAIN32 is not this
bank. If a smaller pilot is chosen, register its actual parent count/dose before
execution rather than filling the missing parents with repetitions.
2. Freeze existing exposed DEV64 as the held-composition/status development
instrument: all32 parents and all64 scheduled variants, four parents/group.
Use the four-status confusion matrix and positive-owner accuracy separately
from determinate/abstention outcomes. This is shared-grammar transfer, not broad
natural-language/family/population confirmation. Reader16 is a diagnostic
screen, not a substitute for the DEV64 instrument.
3. Define a separate original-skill panel before baseline, for example bounded
integer arithmetic, literal single-query retrieval and constrained instruction
following under new independently checked CRS source contracts. Those panels
do not exist in this registration yet; no central CHECK/FINAL reuse. Source-correct
membership is frozen from actual update0 responses and never replenished after
a weak result. If resource ceilings require fewer development items, preregister
that smaller bank; retention/broad-language claims then remain underpowered or
unmeasured. Do not call novel state tasks the original model's complete skills.
4. Use a compact fixed model renderer: task rules plus SOURCE text and the
entity/time QUERY. Omit IDs/family/split from the neural prompt while retaining
them in evaluator receipts. Canonical target is only `status,value`, with no
explanation/procedure or evidence-ID target. Compile using the exact pinned
tokenizer/chat template with thinking disabled and explicit EOS. Measure every
whole prompt at<=512 and whole response at<=64, including header/EOS conventions;
the complete native sequence ceiling is576. No truncation or silent exclusion.
Exact token fit is UNMEASURED here because the actual renderer/targets are not
frozen; short CPU source size alone does not establish512-token fit.
5. Jointly tokenize complete messages and independently verify the assistant
boundary. Prompt/header labels are−100; every eligible assistant target token
and declared termination token is supervised, with the causal shift checked
independently. Log actual input IDs, label/mask hashes, processed versus
supervised token counts, and complete failed attempts. For32 presentations at
the declared576/64 maxima, training ceilings are18,432 processed and2,048
supervised tokens before any qualification/retry accounting; current32,768/8,192
caps are ceilings, not a quota to fill. Register sequence-mean completion CE
then mean over each four-case accumulation group, or declare a different
normalization prospectively; never silently change it between treatments.
6. Bind adapter initialization/RNG seed separately from draw101/order211.
Freeze named q/v tensors and zero-adapter parity, rank8/alpha16/dropout0,
FP32 adapter/optimizer states and BF16 base under exact native qualification.
Checkpoint0/4/8 after zero/sixteen/thirty-two consumed presentations; actual
steps/exposures override planned values on failure. Retain full optimizer,
scheduler/scaler if present, every RNG/cursor and accumulation boundary.
7. Source versus one ordinary adapter estimates that bounded adaptation effect;
it is not a dose-matched retention/selective-update comparison or frontier.
Any later replay/update-scope treatment must start from the identical checkpoint
and initializer, use equal32 presentations, masks, processed/supervised-token
dose and update count, fixed evaluation IDs, and account different real compute.
Replay must replace a declared part of the dose rather than add uncounted rows.
Declare treatment/control composition and sampling before outputs. The current
microstudy can proceed without adding every ablation, once its own gates pass.

## Precision, predictions and interpretation

The existing DEV64 has only32 parent groups and four per composition stratum;
reader16 has eight parents. One train draw/order cannot estimate training-draw
or unseen-family uncertainty. Even under an optimistic independent zero-loss
Bernoulli approximation,32 source-correct parents imply an8.94% one-sided95%
upper loss bound, not2%. Four independent zero-loss parents imply52.71%.
Rough zero-loss thresholds are149 independent units for a2% upper bound and59
for5%; these are illustrations, not this study's noninferiority/power design,
and absolute paired-accuracy margins require their own analysis. Correlated
variants cannot supply the missing independent units. No-forgetting remains
unestablished even with no observed losses; family worst strata and all raw
losses/repairs remain visible.

Freeze CRS-LA-H01 as a prediction of **gain differences**, not final accuracies:
`100 * ((acquisition8-acquisition0) - (transfer8-transfer0))`, conditional on
the declared panels, with its stated heuristic0..20pp range. Update4 is an
intermediate diagnostic and update8 the predeclared endpoint; do not select
the best checkpoint after looking. Raw accuracy differences can simply reflect
baseline task difficulty. Keep floor/ceiling effects and format/abstention
alternatives explicit. An observed association is not an internal neural
mechanism; one recipe point does not establish a universal efficient frontier.

All processing/evaluation/qualification and failure costs must be registered
within the one-worker/two-hour epoch/four-hour rolling budgets. Training token
caps do not include uncharged evaluation or qualification compute. Native
maximum576 decoding, finite confined gradients, zero/no-op parity, served
export precision, complete resume equivalence and sustained host fit remain
separate gates. The current unknown GPU identity refusal is preserved.

## Versioned repair assignment

The coordinator authorized new `scoring_v2.py`, `atlas_v2.py`,
`tests/test_instruments_v2.py` and `contracts/INSTRUMENTS_V2.json` only, preserving
original modules and epoch001 evidence. V2 will address LA-I1-003–006 with
independent counterexample tests; it does not construct the LA-I1-001/002/007
data/tokenization/native instruments or qualify a learning outcome. Fresh
scientific review must inspect the frozen V2 artifacts before future-runner
integration. Implementation/test bindings will be appended below.

## Versioned CPU implementation freeze

Implemented the assigned new instrument files only; the original scorer/atlas
hashes above are unchanged. The initial contract file was mistakenly created
in the repository's outside `contracts/` directory, then moved into the CRS
package before testing. The confirmed empty outside directory was removed;
no unrelated file was moved or deleted. All final owned files are inside CRS.

| Frozen V2 artifact | SHA256 |
| --- | --- |
| `scoring_v2.py` | `4233a0841824191955c523a2d09d0bdb1fa9d2fc5b4ab430af39594fbf9c2f54` |
| `atlas_v2.py` | `403cec6b49c37b9b623ccf03c39d19c695cbe288ce50f9062e6e62644f078a57` |
| `tests/test_instruments_v2.py` | `227627953e77cb571174067d357dcd4c641ab7c08ed036243536c22a0f9b3105` |
| `contracts/INSTRUMENTS_V2.json` | `6c3b2bb9bb4d37db7507e497afa224266e2e1e93d649970c548a71aa91f97bf0` |

Executed `python -B -m unittest
experiments.rineya_centralized_reasoning_system_v1.tests.test_instruments_v2 -v`:
**20 passed, 0 failures, 0 errors**. Counterexamples independently test correct
semantic fields with bad format, operational failures never becoming abstention,
missing/extra denominators, boolean/float/nonfinite type confusion, duplicate
JSON keys, refutation versus unknown, inconsistent parent/exposure counts,
missing native metadata, unbound/changed receipt bytes/gain/dose, duplicate
case/variant deliveries, frozen bank coverage, missing incumbent and undefined
source-correct denominators. No task bank or native result was generated.
`git diff --check` passed.

Useful APIs:

- `scoring_v2.score_one(answer, gold)` / `score(predictions, gold)` accept bounded
  unique JSON strings or already parsed response objects; they preserve semantic
  projection, strict format, operational errors and all scheduled cases.
- `atlas_v2.validate_row(row)` validates the version2 declared schema, explicit
  metadata and consistent dose; it does not attest execution.
- `atlas_v2.efficiency(gain, row, measurement_receipt=bytes)` requires exact
  receipt-byte/row/dose/gain association before native numeric ratios. Its output
  explicitly states truth is not attested and promotion is not allowed.
- `atlas_v2.paired_parent_interval(records, expected_cases=bank)` requires case
  and parent-local variant IDs and rejects duplicates. Future native integration
  must supply the expected bank to verify no observations were omitted.
- `atlas_v2.skill_matrices(checkpoints, source, incumbent=None)` preserves source
  losses/repairs and leaves missing incumbent fields unavailable, with no
  noninferiority claim.

These instrument repairs address LA-I1-003–006 under their declared contracts.
They do not close LA-I1-001/002/007: actual optimizer/probe/legacy banks,
tokenization/masks/native decoding, original-skill enrollment and native
training/export/resume are still missing or closed. No model/system/distilled
gain or forgetting frontier is inferred from passing these tests. The
independent scientific reviewer owns fresh approval of these frozen bytes;
the coordinator owns any separately reviewed future-runner integration.
