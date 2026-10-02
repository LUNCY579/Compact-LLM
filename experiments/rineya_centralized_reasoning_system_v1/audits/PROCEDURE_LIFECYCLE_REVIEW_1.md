# CRS-001 supplied-procedure lifecycle scientific review

Client date: 2026-10-01, America/Chicago. Reviewer `/root/scientific_audit` did not
build these lifecycle or instrument modules. Separate milestone from the prior
CPU and native diagnostic reviews. Initial disposition: **REPAIR REQUIRED** for
failure provenance; the completed development run's finite fixture observations
remain usable within their recorded supplied-scaffold scope.

Reviewer privileges, workspace and model lineage are shared with the coordinator.
This is a distinct artifact/code inspection, not independent ground truth or an
OS-secret evaluator. No Rineya/native model, external GPU process termination,
central protected gold, or confirmation bank was used. Existing native refusal
and approvals are preserved; this CPU milestone does not alter native admission.

## Frozen artifacts and inspected raw evidence

All package-relative paths below resolve under
`experiments/rineya_centralized_reasoning_system_v1/`.

| Artifact | Initial reviewed SHA256 |
| --- | --- |
| `procedure_development.py` | `5991ebe5e35ad39eb3c01792842e803b59bf0758d57de54254684bb8d1335e0d` |
| `tests/test_procedure_development.py` | `74dbe95c32600e38db1b2bb06ffda96ac62db1ac26d6b3f7a2d0859862c16c73` |
| `contracts/PROCEDURE_DEVELOPMENT_V1.json` | `990d21dca7de909c20e33b7ac9677b242e639c56e1ff8a811725c209bae211c2` |
| `outputs/procedure_development_v1/receipts/procedure-development-result-99334ff59b7f26a7aade93513021850d22f8782dc7719d1b7e395d2100570154.json` | `8774abf2bdcba92c3e310f863bf41a8892f850d5725094999be101f2fe75af74` |

Read the raw implementation, contract and tests before relying on the builder's
summary. Independently traversed and hash-verified **77 referenced receipts**
from the actual 32-case × two-candidate run, including all 64 public-worker
case receipts, proposals, trials, counterexamples, five linked history events,
registration and frozen snapshot. This verification used read-only JSON/files;
it did not rerun the saved experiment or modify its SQLite database.

Verified job identity:
`procedure-development-8ad2f82b2622b4daaf9d1c4779fedf243c88a4b811850c6f71be680876f1a56a`.
Every history index/previous digest and the final history tip join correctly.
The frozen entries reproduce their snapshot digest, retain version 2 and
CODEX_SUPPLIED origin, and their graph equals the supplied B DEFAULT_GRAPH.

Independent bundled-interpreter lifecycle suite execution after raw inspection:
**6/6 tests passed**, reported suite interval 9.344 seconds. These are temporary
CPU public-worker and injected failure controls, with no models. The separate
two-distinct-interruptions probe below exposes a gap beyond those passing tests.

## Measured fixture observations and their meaning

| Candidate | Semantic correct | Format valid | Valid semantic counterexamples | Operational errors | Development status |
| --- | --- | --- | --- | --- | --- |
| Supplied v1, premature FILTER(until=0) | 18/32 | 32/32 | 14 | 0 | REJECTED_DEV |
| Supplied v2, remove that filter | 32/32 | 32/32 | 0 | 0 | DEV_VALIDATED_CPU_SUPPLIED |

The 32 DEV cases include 18 SUPPORTED, four CONTRADICTORY, eight UNKNOWN and
two REFUTED labels. All 64 scheduled case receipts are retained. Raw costs:
10.370999999988271 known CPU worker seconds, 448 primitive operations, zero
unknown CPU attempts, and zero model loads/calls, native seconds or training
updates. The worker-time field is deliberately a lower bound on total campaign
cost, not complete coordinator/engineering/review cost or a learning-efficiency
measurement. Scanning/verification costs remain separated per trial.

These observations demonstrate rejection of a deliberately supplied bad graph
and testing/freezing a predetermined Codex revision. Both graphs are supplied
before the outcomes; the provenance and limitations disclose that fact. The
revision's receipt links prior counterexamples, but the revision was not generated
from those failures by Rineya, or dynamically discovered by the controller.

Version 2 is the already supplied strong B baseline graph. Therefore v2 versus
v1 is a scaffold correction, **not C versus B procedure-memory efficacy**,
autonomous discovery, weight acquisition, calibrated abstention, confirmed
transfer or retention. DEV is already exposed/consumed. Passing every recorded
fixture supports that finite development scope; it is neither universal
correctness nor fresh confirmation. All such stronger claims are false in the
actual result receipt, which is appropriate.

## Lifecycle checks that are scientifically sound

The bounded registration permits 16 or 32 distinct DEV cases, exactly two
supplied candidates and at most 64 public-worker attempts, with explicit
coordinator/per-worker ceilings. Parent variants stay grouped and TRAIN/DEV
composition roles are audited; the shared grammar/renderer limitations remain
visible. Actual proposals bind exact parent candidate digest, monotonic version,
identified proposer/reviser, restricted typed graph and applicability.

Gold/scorer feedback and revision history remain coordinator-side; the worker
receives a public case and immutable snapshot. Immutable test-input FROZEN is
explicitly distinguished from accepted-library validation. The external latent
oracle/scorer supplies the development decision. Operational failures alone
quarantine the trial and cannot falsify a graph; a successful second candidate
does not demonstrate repair if the first failed only operationally.

Native-proposed/revised origins remain closed even when an engineering caller
supplies hash-valid status receipts. The supplied CPU controller has no registered
native proposal adapter. Snapshot writes remain denied during evaluation.
Terminal cached results verify nested evidence instead of regenerating missing
or corrupt children. Completed per-case receipts provide useful restart
idempotency, subject to the repeated-controller-failure finding below.

## PL-SCI-01 — Repeated controller failures lose distinct failure evidence

**Priority P1 for recovery/provenance. Evidence:** `run_development`'s exception
handler publishes each controller interruption under the same immutable
`procedure-development-failure` kind/study key. Distinct exception/progress
payloads conflict with the first receipt.

**Independent reproducer:** in a temporary OperationsStore, use count 16 and
seed 202610034. Keep real public CPU workers. Patch the coordinator's execute
wrapper to complete one case and then raise RuntimeError("first controller
interruption"). Resume that same registration, complete another execute call,
and raise RuntimeError("second distinct interruption"). No native model runs.

Observed first attempt: RuntimeError with one failure receipt containing the
first message. Observed second attempt: OperationsError("receipt
conflict/corruption; existing bytes preserved"), still only one failure receipt
and only the first RuntimeError payload. The second original failure is masked
by failure publication. Completed case evidence remains present; this finding
does not accuse the run of duplicating those workers or falsifying its scores.

**Impact:** preserving one failure and rejecting overwrite is preferable to
deletion, but the promised full interruption history is incomplete. Repeated
restart diagnosis can lose its actual new failure reason and progress/cost
context. An uninterrupted 64-case success does not exercise this contract.

**Verification:** assign each coordinator attempt a durable distinct identity
and publication key while preserving stable study/candidate/case keys. Retain
all original exceptions, available costs and history/progress references.
Publication failure must not erase or replace the original exception. Add a
two-distinct-interruptions regression that verifies two durable failure receipts,
then resumes to completion without any duplicate case worker. Keep earlier
failure receipts unchanged and do not rerun old completed work just to produce
a new failure record.

## Pending instrument-v2 review

The parent requested a separate raw review of frozen scorer/atlas v2 instruments
for semantic-versus-format separation, duplicate observation identities and
missing incumbent metrics. Those artifact hashes and raw validation are pending
from the data/instrument owner. This report does not certify those uninspected
changes or retroactively regrade the preserved initial lifecycle run. New
instrument results need their own version/source bindings.

Maximum three review/repair rounds applies to this new milestone. Preserve this
initial finding/result record and append narrowly bound repair/instrument
closures; prior CPU/native reviews retain their original scope and evidence.

## Targeted recovery closure, round 2

PL-SCI-01 is closed on `procedure_development.py` SHA256
`33aedea527acad5d97efe23568d742a1f164920ba402288fd7f35145c62be841`.
The new targeted test `tests/test_procedure_recovery_v2.py` is SHA256
`1e896b27fff3ba68ad4ef617231c954089ab11155c193cfb5f416eb6f5882fa2`.
Failure receipts now bind study key plus the distinct coordinator attempt UUID,
while case/candidate keys stay stable. If failure publication itself fails, the
original exception is re-raised with a publication-failure note.

The added focused test passed. Independently repeated the original two distinct
interruptions with actual public CPU workers and then resumed the same study:
two failure receipts preserve both RuntimeError messages and distinct attempt
IDs; completion uses exactly 32 unique worker keys for 16 cases × two candidates,
with no duplicated completed worker. A separate injected disk-publication failure
re-raises `RuntimeError: original controller error`, retaining a note about the
OSError instead of masking the original with it. These are temporary CPU tests,
not a new native result or a regrade of the saved 64-case run.

The initial acceptance run and its original code/contract/result hashes remain
unchanged. A new acceptance freeze must name the repaired source; it cannot
retroactively replace the old execution's identity. No graph, corpus or scorer
was changed by this recovery repair.

## Frozen instrument-v2 independent review

Read complete v2 scorer/atlas source, contract and tests before executing
controls. Their implementation owner remains distinct from this reviewer.

| Instrument artifact | Reviewed SHA256 |
| --- | --- |
| `scoring_v2.py` | `4233a0841824191955c523a2d09d0bdb1fa9d2fc5b4ab430af39594fbf9c2f54` |
| `atlas_v2.py` | `403cec6b49c37b9b623ccf03c39d19c695cbe288ce50f9062e6e62644f078a57` |
| `tests/test_instruments_v2.py` | `227627953e77cb571174067d357dcd4c641ab7c08ed036243536c22a0f9b3105` |
| `contracts/INSTRUMENTS_V2.json` | `6c3b2bb9bb4d37db7507e497afa224266e2e1e93d649970c548a71aa91f97bf0` |

Independent suite: **20/20 passed**, reported interval 0.003 seconds. Additional
raw probes confirmed that correct explicit status/value fields plus an extra
field yield semantic=true/strict=false; an explicit failed UNKNOWN yields no
semantic success; duplicate raw JSON keys fail; duplicate paired observation or
missing bound case rejects. A missing incumbent remains UNAVAILABLE/null while
the source-correct loss is still recorded.

The semantic projection is appropriately narrower than freeform language
meaning: it reads unique bounded typed status/value JSON without prose/gold
repair. Strict-format and semantic outcomes differ by design. REFUTED and
UNKNOWN/CONTRADICTORY remain distinct; positive answerable and determinate
denominators are named separately. Evidence-reference truth and calibration
remain unmeasured by this scorer.

Atlas v2 requires explicit schema/version/dose, seeds and source bindings for
native-shaped rows and actual measurement receipt bytes for a numeric efficiency
calculation. It explicitly returns truth/qualification/promotion=false; a
hash-valid invented fixture cannot become proof that a model executed. Parent
intervals reject duplicate case IDs/parent variants, retain whole-parent means,
and report family-conditional descriptive scope and small-cluster warnings.
`expected_cases` supplies complete identity/membership checks; a frozen external
bank binding remains mandatory before future native integration. Missing
incumbent and zero source-correct denominators stay undefined. The original
scorer, atlas and old result receipts are preserved; v2 is a separate instrument.

### PL-SCI-02 — Native gain/baseline semantics must be bound before integration

**Priority P2 integration condition; not a fabricated-native-result finding.**
Current v2 efficiency byte-binds a declared gain but does not define/validate
its relationship to the row's acquisition estimate or a starting baseline.

**Concrete probe:** use a clearly invented schema fixture with acquisition
estimate 0.1 and a receipt declaring gain 0.9. The helper accepts the receipt
and computes ratios, while explicitly returning native_truth_attested=false.
If gain were same-panel acquisition improvement, its implied baseline would be
−0.8, an impossible accuracy. The instrument owner confirms the current helper
does not establish that gain is that same acquisition endpoint.

**Verification before native efficiency/atlas use:** register endpoint/scope,
baseline and candidate proportions from the same panel and dose comparison;
compute gain from those bound values; reject impossible/inconsistent pairs.
If gain denotes another endpoint, bind that endpoint explicitly instead of
silently treating it as acquisition. Independent run truth, qualification and
precision/power remain separate obligations. Do not infer a learning frontier
from declared arithmetic or byte association alone.

## Scientific disposition for this milestone

**SCIENTIFIC_CPU_LIFECYCLE_AND_INSTRUMENT_PASS**, limited to the repaired
supplied-graph lifecycle and versioned CPU instrument/schema behavior above.
There is no remaining consequential blocker for those narrow CPU uses. New
registered runs must bind their actual repaired source/contract/instrument
version, and completed old receipts keep their old identities.

PL-SCI-02 and future native integration are explicitly deferred/quarantined:
same-panel gain/baseline binding, actual qualification/execution, untouched
confirmation, acquisition/retention/calibration/efficiency and mechanism claims
remain unestablished. The successful lifecycle remains a predetermined Codex
scaffold correction; version 2 equals the supplied strong B graph and establishes
neither B/C procedure-memory gain nor autonomous Rineya discovery. No native
permission, GPU-process policy, model weights or protected banks change here.
