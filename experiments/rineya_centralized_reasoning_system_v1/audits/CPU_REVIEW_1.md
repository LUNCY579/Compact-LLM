# CRS-001 CPU scientific review, round 1

Client date: 2026-10-01, America/Chicago. Reviewer: `/root/scientific_audit`.
Disposition on the reviewed bytes: **REPAIRS REQUIRED** before a complete CPU
vertical-slice qualification. Findings do not establish any native/model failure
or efficacy. Root has begun repairs; their closure belongs to a fresh review.

## Evidence and independence

Read raw `core.py`, `data.py`, `scoring.py`, `atlas.py`, their tests, CPU/data/atlas
contracts and hypothesis ledger before interpreting builder commentary. The
reviewer did not author these modules and shares the coordinator's model lineage,
workspace and privileges. Logical separation is useful; there is no independent
OS referee or secret gold boundary among same-account engineering peers. No
protected bank or native model was opened. Probes use new hand CPU worlds only.

The independent latent oracle does not import the runtime parser or state solver;
its manual controls provide meaningful complementary checking. They still share
the declared domain semantics and author environment. Registered TRAIN/DEV
composition families share grammar, renderer and primitives; the manifests
correctly avoid claiming new-language transfer or pretraining exclusion.

The sources were mutable working files, not a committed complete freeze. The
following hashes bind the scientific code examined and reproduced before repairs:

| Artifact | SHA256 |
| --- | --- |
| `core.py` | `2b7d62705827fbd744181e9baba33149f09266c03949a3a9a768ef0ad7d42054` |
| `data.py` | `6a1a051e9f2da0411760f66c42f77de42ea402050c54e8c184c9fbb9a4730091` |
| `scoring.py` | `ac3f180b67b954d38422841ec3937438d780a00497cc95b6d67b0a7ed7bfd644` |
| `atlas.py` | `04fb131c616fb04535e01aea5bd28c5ec606966738809d2f4b9d4b623ebe8b70` |
| `contracts/CPU_V1.json` | `93c7f2305aca1fe8a01f0b001e568f0413558c5edfb9158fadfe0c9d9648a6a4` |
| `contracts/DATA_V1.json` | `2c244bce88e15d958ad911e6a6dac307c169c0218ccaa81821ec556fc230bf9c` |
| `contracts/LEARNING_ATLAS_MICROSTUDY_V1.json` | `9c0113afe749cb375917df78937e6ab9125afa9e52f081a7169c6e8d8a642d73` |
| `theory/hypotheses_v1.json` | `0401ea92278191e3aea500f72096f4239af7c9286181c9ed1591342249b98163` |

Paths resolve under `experiments/rineya_centralized_reasoning_system_v1/`.
Native builder/operations changes were outside this scientific review. Source
and primary-paper audit artifacts are separate outputs.

Independent invocation with bundled Python and bytecode writes disabled:

```text
python -B -m unittest discover -s experiments/rineya_centralized_reasoning_system_v1/tests -v
```

Observed: **55 tests passed** in the reported 0.666-second test-suite interval.
The command completed normally. This suite count includes operations tests;
operations correctness is reviewed separately. Six additional adversarial probes
below expose contracts that those passing tests did not cover.

Common reproducer setup:

```python
from experiments.rineya_centralized_reasoning_system_v1 import core, data, scoring, atlas

def case(source, complete=True):
    return {"id": "audit", "parent_id": "audit-parent", "family": "hand_adversarial",
            "split": "DEV", "source": source, "query": {"entity": "parcel", "time": 9},
            "rules": {"cancel_mode": "remove_event", "equal_time": "contradictory",
                      "complete": complete}}
```

## CPU-SCI-01 — Reader substitution bypasses source-reference validation

**Priority P1. Evidence:** reviewed `core.py:417` accepts arbitrary injected
reader output; `_validate_events` is called only within FixtureReader. CHECK
validates IDs against that same reader's events and claims source-reference
consistency, without independently binding custom-reader events to source spans.

**Impact:** the native/alternate-reader interface could invent an event and pass
the declared reference verifier. The supplied fixture reader's own correctness
does not qualify that boundary.

**Reproducer:** inject a reader with origin `MODEL_PROPOSED` returning one INIT
event `parcel Mallory`, ID `invented`, and reference hash `"0"*64`, offsets 0:1,
UTF8_BYTE. Give it a source containing only actual Alice/Bob events. Observed
runtime answer: SUPPORTED/Mallory; stop ANSWER; error null; evidence `invented`.

**Verification:** revalidate every reader's exact event schema, allowed kinds,
IDs/count limits, reference hash/offsets and source-span event signature before
inserting state. Foreign/invented/altered events and missing references must stop
as FAILED, never supported answers. Preserve a valid alternate-reader positive
control so the negative test does not merely forbid every substitution. This
provides source consistency, not general natural-language truth verification.

## CPU-SCI-02 — Unknown prerequisite plus simultaneous owner misclassified

**Priority P1. Evidence:** `core.py` trace_state inserts owner names even when
that transfer has an UNKNOWN guard, then gives multiple owners precedence over
the unknown condition. Independent `data.py` oracle records only definite
supported owner bindings when deciding conflict.

**Reproducer source:**

```text
e0 @0 INIT parcel Alice
e1 @1 TRANSFER parcel Bob CONFIRMED
g0 @0 REQUIRE e1 permit
e2 @1 TRANSFER parcel Carol CONFIRMED
```

Query parcel at 9 with complete coverage. There is no permit fact. Independently
construct the corresponding latent INIT, guarded Bob TRANSFER, REQUIRE and
unguarded Carol TRANSFER events. Observed: runtime CONTRADICTORY/null; latent
oracle UNKNOWN/null.

**Impact:** isolated family success does not qualify compositions of uncertainty
and conflict; these statuses are distinct scorer labels. A later status-repair
claim could be driven by instrument disagreement.

**Verification:** keep conditionally applicable/unknown candidates separate from
definitely applicable conflicting assignments. Freeze one rule for this exact
case, reconcile oracle/runtime/contract, and add paired manual controls combining
unknown and contradictory gates with known/empty/simultaneous assignments. The
oracle should remain independent rather than copying the repaired runtime code.

## CPU-SCI-03 — Supported aggregate values cannot be scored

**Priority P1 for the declared six-operation pipeline. Evidence:**
`core.py:433` produces integer AGGREGATE values; its aggregate test expects `2`.
`scoring.py:9` accepts only nonempty strings for SUPPORTED values.

**Reproducer:** use source `e0 @0 INIT parcel Alice`, `e1 @0 INIT box Alice`,
`e2 @1 TRANSFER parcel Bob CONFIRMED`; graph LOOKUP → AGGREGATE(field=owner) →
CHECK. Observed supported integer value 2 with no runtime error, but both the
runtime answer and an equivalent `{"status":"SUPPORTED","value":2}` gold fail
scorer validity.

**Impact:** a declared primitive works in isolation but cannot complete the
independent scoring path. Silently stringifying counts would alter the answer
contract and permit type confusion unless explicitly registered.

**Verification:** register owner and integer aggregate answer types, reject bool
as integer, and add an end-to-end numeric gold/prediction scoring test. Preserve
strict string ownership checks and malformed/extra-field rejection. Define
aggregate oracle scope separately from owner-query gold; do not derive gold by
calling the runtime aggregate implementation.

## CPU-SCI-04 — Fallback UNKNOWN can erase failed execution in integration

**Priority P1 for the new runner; not an allegation of already inflated results.**
Evidence: `run_case` returns a fallback UNKNOWN answer alongside operational
error; `scoring.score_one` sees no failure if given only that answer.

**Reproducer:**

```python
r = core.run_case(case("e0 @0 INIT parcel Alice", complete=False),
                  limits=core.Limits(max_operations=1))
scoring.score_one(r["answer"], {"status": "UNKNOWN", "value": None})
```

Observed runtime BUDGET_EXHAUSTED with `BudgetExhausted: operation_limit`; the
answer-only score reports semantic_exact=true and operational_error=false.

**Impact:** dropping the envelope counts a failed case as correct insufficiency.
Four-state correctness and calibrated abstention then cease to represent an
executed decision.

**Verification:** the runner must carry failed/aborted/unavailable envelopes to
scoring as errors or missingness; preserve every scheduled case. Test failure on
gold UNKNOWN and CONTRADICTORY specifically, because answerable cases alone do
not reveal laundering. Receipt status/case ledger and scorer totals must agree.
Legitimate successful UNKNOWN remains scoreable.

## CPU-SCI-05 — D does not enforce its declared treatment identity

**Priority P2; required before reporting a B/C/D runner comparison. Evidence:**
reviewed `core.py:408` checks D's snapshot type only and labels it size-matched
irrelevant without validating either property.

**Reproducer:** `core.run_case(case("e0 @0 INIT parcel Alice"), arm="D",
snapshot=core.supplied_snapshot())` succeeds with the same relevant C library
hash and 339 bytes. Thus D currently accepts the treatment library itself.

**Impact:** a labeled irrelevant-content control can contain applicable content;
mere format or class membership does not identify that experimental arm.

**Verification:** bind D against a specific frozen C snapshot, enforce the
registered byte/format size criterion and nonapplicability/content identity,
reject relevant C content and wrong-size libraries, and include a valid matched
D positive control. Native token matching is separate from CPU byte matching.
Keep C versus B primary; D is diagnostic. Initial supplied graphs are identical
and their parity establishes infrastructure only.

## CPU-SCI-06 — Native atlas rows accept missing provenance and NaN outcome

**Priority P1 for native atlas admission; no native row was actually produced.**
Evidence: reviewed `atlas.py:17` checks field presence and dose/resource counters,
but not identity bindings or acquisition/transfer/retention/calibration validity;
`efficiency` trusts an externally supplied gain.

**Reproducer:** initialize every REQUIRED_ROW_FIELDS entry to None, then set
evidence_class=NATIVE_CONFIRMATION, unique_parents/presentations/processed_tokens/
supervised_tokens/updates/worker_seconds=1, failures=0, acquisition=float('nan').
Observed validate_row accepts the row and efficiency(.1,row) emits .1 per every
listed denominator despite missing checkpoint/data/scorer/qualification/commit
bindings and invalid acquisition.

**Impact:** a native label can turn incomplete or nonfinite evidence into
apparently measured efficiency. A filled schema is not a complete learning atlas.

**Verification:** validate native hash/provenance/seed/recipe/exposure bindings,
finite typed measurement and uncertainty fields and consistency of measured
dose. UNMEASURED outcomes stay undefined and cannot yield measured learning
efficiency. Define which missing panels are allowed versus blocking a specific
claim; do not fabricate transfer/retention/calibration. Preserve null/zero
denominators as undefined and do not claim a native qualification from validation.

## Positive scope and carry-forward limits

The reviewed graph validator enforces prior-node dependencies and operation
arity/input types; WorkingState replacement rejects dependency cycles and
invalidates transitive dependents. Snapshot reads return detached copies and
evaluation-memory writes are rejected. The owner-query parser spans use UTF-8
bytes; invalid references, aliases, cancellation, malicious source instructions,
budget exhaustion and four semantic states have meaningful hand controls.
The supplied B/C default graph and primitives are identical. These observations
support a useful CPU foundation, conditional on the above repairs.

The sampler's parent grouping and registered composition-family split are
appropriately narrower than independent languages/domains. Complete coverage
is supplied explicitly, not inferred from retrieval. Shortcut baselines are
descriptive and not an exhaustive leakage certificate. Before native efficacy,
freeze additional metadata/query-only attacks and confirm model prompts do not
reveal family/scoring metadata unnecessarily.

Atlas source/approved-incumbent retention and parent-cluster intervals correctly
retain undefined denominators and label uncertainty conditional on the observed
family mix. They do not yet certify statistical noninferiority, retention of
each previously acquired skill relative to its earlier best checkpoint, or
new-family generalization. The preliminary microstudy leaves native training
closed and admits its one-draw limitation. Exact acquired/transfer panels, actual
32-parent dose and masks must be frozen before launch; the hypothesis ledger's
empty observation arrays and uncalibrated prospective range are proposals.

Round 2 should inspect repaired bytes and the new runner/public subprocess
boundary, replay these six controls plus the original suite, and bind the
resulting freeze. Preserve this report and earlier failures. Three review/repair
rounds are the milestone ceiling; agreement does not replace independent truth.

## Round 2 addendum — repairs replayed against fresh scientific freeze

The coordinator declared core/data/scoring/atlas/runner/runtime-worker/contracts/
hypotheses/scientific tests frozen for this pass; operations and native work were
separate. These are working-file hashes, still awaiting a committed source freeze:

| Artifact | Round-2 SHA256 |
| --- | --- |
| `core.py` | `5d681943beeaa89ca961d2f618cb440574981caaa1e23299af7b896f4203c65d` |
| `scoring.py` | `4ea996b7c3eca875c4691e2274b3beb492ba5b3e704ec77d3f200f7697b54c21` |
| `atlas.py` | `ab6fc3745f84c5f338e204d1ad7fb7f88bb3a8241267c2e2a66c69c6a000e785` |
| `runner.py` | `0e676febafc19979dcdb0a76f159f20d0b31b7054f834fce09f2c3d6e11bc7a7` |
| `runtime_worker.py` | `811f8057c24588ad140851978ba112d69e84f1048103eb157e6e61acd554538f` |
| `tests/test_core.py` | `f0f80ac7ba1f1862c4e60fb037a6ac99d16da07ffd610c4ed220cbd20cf16a44` |
| `tests/test_data.py` | `e0f1fbeacbef9f6b7c17b39de9040fd8a0306a04ff8dd53651558d6814c83b45` |
| `tests/test_atlas.py` | `b4b4aeea0f472665cb9d60c2e4a503585fb3e0c827de49be2d7f144418b16b32` |
| `tests/test_boundary.py` | `50877048064c984e628ab5c179e6b751f4641fadbfe16a33066917d5989a2ab8` |

Fresh focused suite execution: 17 core, 16 data, 4 atlas and 3 public-boundary
tests, **40/40 passed**, with no native tests/models invoked. The six original
adversarial probes were replayed and the public worker was exercised with a
valid public case and a forbidden gold-field case.

| Finding | Observed repair verification | Status at round 2 |
| --- | --- | --- |
| CPU-SCI-01 | Foreign span FAILED; valid independently supplied copying reader still answers Alice; altered source signatures rejected by regression | Original foreign/invented-span bug repaired; typed validation gap below remains |
| CPU-SCI-02 | Unknown-gated Bob plus definite Carol now UNKNOWN in both runtime and latent oracle | Closed for recorded counterexample |
| CPU-SCI-03 | Runtime integer 2 scores correctly against integer gold; bool prediction rejected | Closed for CPU structured aggregate answer typing |
| CPU-SCI-04 | Error projection scores false on gold UNKNOWN; runner selects answer only when operational error is absent | Closed in inspected runner projection; lifecycle accounting separately reviewed |
| CPU-SCI-05 | Relevant C library FAILED as D; wrong-size D FAILED; registered matched D answers normally | Closed for supplied CPU unused-content fixture; native semantic-control limitation below |
| CPU-SCI-06 | Prior native row rejected; UNMEASURED numeric efficiency rejected; undefined gain returns null denominators | Closed for demonstrated malformed/undefined admission |

Public JSON worker rejects unknown/gold fields, duplicate keys and nonfinite
constants. A subprocess over a valid public ownership fixture returns Alice;
the same case with a `gold` field returns FAILED. Its imports exclude the scorer
and latent oracle. This is logical/process interface separation; same-account
filesystem privileges remain shared and are disclosed.

### CPU-SCI-01b — Exact spans still bypass typed event argument semantics

**Priority P1. Evidence:** repaired `_validate_events` binds schema and source
signature but does not check event-kind argument arity, modality enum or each
identifier's value. These remain checked only inside FixtureReader.

**Concrete round-2 reproducer:** source is:

```text
e0 @0 INIT parcel Alice
e1 @1 TRANSFER parcel Bob INVALID
```

Inject a reader returning the two literal INIT/TRANSFER events and their actual
whole-line Evidence spans; keep IDs, times, arguments and source hash identical
to the source. Observed: FixtureReader returns FAILED, but the injected exact-span
reader returns ANSWER/SUPPORTED/Alice with error null. The invalid transfer
modality is silently treated as inapplicable. Thus the common reader boundary is
not yet the claimed typed extraction validator. Malformed arity can likewise
reach downstream indexing instead of a registered deterministic rejection.

**Verification:** factor or reapply the complete typed event validator for every
reader before state insertion: allowed kind, exact arity, identifier constraints,
time bounds, modality/fact-state enums, duplicate/reference behavior and configured
event count. Align fact-state acceptance with the declared DATA grammar. Retain
exact-span invalid-modality/invalid-arity negatives and valid alternate-reader
positives. Then repeat relevant source/semantics/boundary tests for the third and
final review round. Disposition remains **REPAIRS REQUIRED** for this boundary.

### D interpretation stays limited to the current CPU fixture

The repaired D builder produces the same operation graph as C while renaming
ID/applicability/validation and padding to identical canonical UTF-8 byte size.
The runtime never selects or executes D's graph; the paired supplied B/C graph
also remains identical. This is adequate to test the present CPU unused-library
plumbing, and the runner explicitly claims infrastructure/parity only. It is
not a semantically irrelevant procedure-information intervention for an LLM:
the graph still encodes ownership solving. Before a native procedure-memory
experiment, supply genuinely irrelevant operation content and tokenize/measure
the actual library prompt; do not upgrade this CPU D receipt into that control.

No new outcome or learning claim is permitted by closing these fixture findings.
The preliminary atlas remains a schema/analysis tool; qualified native curves,
previously acquired-skill retention, fresh confirmation and independent
noninferiority evidence are still absent.

## Round 3 final disposition — SCIENTIFIC_CPU_MECHANICS_PASS

The final round closes CPU-SCI-01b and the demonstrated scientific/scoring
interface findings **within the registered CPU fixture scope**. No consequential
scientific blocker remains for executing the supplied CPU vertical slice after
the separate operations review and source/receipt freeze pass. This is scientific
review acceptance, not native-launch authority, native qualification or efficacy.

Common reader validation now checks exact event schema, arity, identifier values,
modality/fact-state enums, reference schema/source signature, complete extraction
coverage and conflicting duplicate source identities. Exact-span INVALID and
omitted-cancellation injectors return FAILED. DATA FACT syntax is explicitly
aligned with all four source statuses. The unknown-gate/equal-time hand
counterexample remains UNKNOWN under both implementations. Numeric aggregate
scoring and operational failure projection retain strict non-bool typing and
failed-case denominators.

The duplicate FixtureReader verifier call and redundant final reference resolution
were removed. The remaining common verifier charges each successfully resolved
event span and full-source coverage span as it proceeds, retaining incurred bytes
on later failure. Independently observed on a 75-byte source: successful validation
charges 150 verification bytes; omission of the cancellation rejects after 135
verification bytes. This is the declared source-verification counter, separate
from reader bytes and measured worker wall time; it is not a hardware bandwidth
or process-memory measurement.

Fresh final round: **40/40 focused scientific tests passed** (17 core, 16 data,
4 atlas, 3 boundary). After the last redundant-line removal, three targeted
reader/status/scorer regression tests passed again. Independently generated
TRAIN 32 and DEV 64 cases at seed 20261001 produced **zero runtime/latent-oracle
mismatches**. DEV gold included 36 SUPPORTED, 8 CONTRADICTORY, 16 UNKNOWN and
4 REFUTED cases. These are supplied controlled-grammar CPU implementation
observations, with **zero model calls**; none is a model-quality result.

Final scientific source bindings:

| Artifact | Accepted SHA256 |
| --- | --- |
| `core.py` | `18d3d7de48a52e341f2e43dd12ddebcae771d48d02eb4af527929caa5d935016` |
| `data.py` | `6a1a051e9f2da0411760f66c42f77de42ea402050c54e8c184c9fbb9a4730091` |
| `scoring.py` | `4ea996b7c3eca875c4691e2274b3beb492ba5b3e704ec77d3f200f7697b54c21` |
| `atlas.py` | `ab6fc3745f84c5f338e204d1ad7fb7f88bb3a8241267c2e2a66c69c6a000e785` |
| `runner.py` | `3b137883a58ee11b100db930196c0ef7d025fec2ecafc073b08064a1ae361b53` |
| `runtime_worker.py` | `811f8057c24588ad140851978ba112d69e84f1048103eb157e6e61acd554538f` |
| `tests/test_core.py` | `54860e576923b866229d2e378fe06dc7593f3ee7fc0a94af13300d1208ff2e4a` |
| `tests/test_data.py` | `e0f1fbeacbef9f6b7c17b39de9040fd8a0306a04ff8dd53651558d6814c83b45` |
| `tests/test_atlas.py` | `b4b4aeea0f472665cb9d60c2e4a503585fb3e0c827de49be2d7f144418b16b32` |
| `tests/test_boundary.py` | `50877048064c984e628ab5c179e6b751f4641fadbfe16a33066917d5989a2ab8` |
| `contracts/CPU_V1.json` | `93c7f2305aca1fe8a01f0b001e568f0413558c5edfb9158fadfe0c9d9648a6a4` |
| `contracts/DATA_V1.json` | `53e589f2dc29af19e5ba91353b865ae6faefaf63679e5d759388c0f80cc191d9` |
| `contracts/LEARNING_ATLAS_MICROSTUDY_V1.json` | `9c0113afe749cb375917df78937e6ab9125afa9e52f081a7169c6e8d8a642d73` |
| `theory/hypotheses_v1.json` | `0401ea92278191e3aea500f72096f4239af7c9286181c9ed1591342249b98163` |

Retain all earlier round findings and hashes above. This milestone used three
review/repair rounds; no additional full review round is claimed. Reopening a
consequential question requires a new bounded resolving milestone/experiment.
The source hashes cover only the inspected scientific surface; operations/native
qualification has separate reviewers and contracts.

Quarantined/unsupported claims remain explicit: native efficacy or discovery,
native semantic irrelevance of the unused CPU D graph, actual neural learning
efficiency, preservation/noninferiority, calibration, new-language transfer and
internal mechanisms. Existing central CHECK/FINAL stay closed. The next native
candidate requires a new independent review of exact geometry, numerical/source/
resource/containment contracts; this CPU acceptance does not transfer to it.
