# OBS-001 — source observability before claim admission

**Repository:** LUNCY579/ChatBubble  
**Audited parent:** `research/delta-multi-reasoning-factorial-v1@2e5db22d0964fff3ca7a709bfe16fcffa9d1a1b5`  
**Successor:** `research/dmrf-source-observability-v1`  
**Scope:** targeted inspection and additive engineering guards. No experiment rerun or outcome revision.

## Decision

An observation that cannot reach the claimed source must not count as evidence against the claim. Equally, a check that can see a file cannot necessarily see tensor identity, a check that can see a literal cannot prove absence of equivalent code, and a check that reads only some rows cannot certify the whole bank.

The correction at the audited parent is preserved. It reports a remote inventory of 22 MPOS-002 files and byte-identical restore/hash comparisons for three named files. This audit read the correction through GitHub. It did **not** independently access the private vault or repeat those comparisons. That limitation is not a claim that the backup failed, is absent, or cannot be verified by its owner.

## Inspected sources and findings

The exact Git blob IDs for seven inspected files are in `audit/observability_findings.json`. Other source and directory listings informed navigation, but this is not an exhaustive certification of every conclusion on the branch.

| ID | Observable gap | Disposition |
|---|---|---|
| OBS-01 | Local disk search versus remote vault; presence versus sample/full integrity | Keep the correction. Require medium, locator, immutable revision, explicit subjects, access/completion, and a capable method. |
| OBS-02 | Universal wording following literal searches; execution claims inferred from declared dataset overlap | Scope the negative to the actual corpus/query. The specific parser inspection and score corrections are not invalidated by this wording issue. Training-history conclusions require run-bound evidence. |
| OBS-03 | `prose.scan([])` or missing/empty context can return `clean=True` | Refuse an informative-clean interpretation unless actual nonempty text and unique row identities were observed. |
| OBS-04 | Locator hit rates and candidate floors can use different populations; missing candidates fall back silently | Expose common scored IDs and skipped IDs; do not certify a whole bank or candidate floor from an inapplicable/partial view. |
| OBS-05 | `can_manufacture()` maps inapplicability to `[]`; `worst_swing()` maps it to `0.0`; empty pools can yield apparent zero swing | Separate observed zero, insufficient sensitivity, and unavailable measurement. Require actual row/pool coverage and prior input binding. |
| OBS-06 | `score_mpos.py` still emits an earliest-person residual-cue interpretation after the candidate-identity rationale was withdrawn | Leave numbers and historical receipts untouched. The old secondary is not admissible as causal-exclusion evidence without independent, source-bound sensitivity qualification. |
| OBS-07 | Prediction and pair dictionaries can overwrite duplicates; missing prediction fields become empty strings | Add read-only completeness, duplicate, emitted-empty, same-cell and pair-band preflights. |
| OBS-08 | File identity, tensor identity and determinism have different observation requirements; the triage example loops only over A's tensor keys | Retain existing corrections. Require symmetric tensor inventories and the appropriate property-level collector; byte mismatch alone is not meaningful model change. |

These are source-level findings and interpretation limits, not proof that any additional historical experiment was numerically wrong. Existing primary values, arm selection, significance decisions, model files and permissions are not changed.

## What was added

`audit/observability.py` is a standard-library-only, read-only admission layer. It does not import model code, legacy scorers, or experiment banks.

The claim contract binds:

- A source medium, exact locator and immutable revision. Unknown identities remain unobservable.
- A narrow predicate and nonempty, unique, explicit subject IDs. Wildcard/global scopes are rejected.
- An exact measurement contract: query, normalization, rule universe, threshold, population or other relevant parameters must be supplied by the collector and match the claim.
- A centrally allowed method/property pairing. A local inventory cannot be relabelled as a remote observation, and an inventory cannot become a byte comparison.
- Access/completion state, DIRECT versus REPORTED/SYNTHETIC basis, method digest, raw-evidence digest, and per-subject results.
- A successful source-, measurement- and method-bound positive control for negative searches. A failed or absent control blocks absence/clean claims.

The positive control is a necessary qualification, not a universal proof of sensitivity. Collector honesty and an appropriate control remain explicit trust boundaries.

### States that must not be collapsed

| State | Meaning |
|---|---|
| `SUPPORTED` | The scoped inference is admissible **given the direct collector attestation** and its declared evidence. |
| `CONTRADICTED` | A capable, correctly scoped direct observation contains a counterexample to the stated universal claim. |
| `INCONCLUSIVE` | Some required subjects remain unobserved, including incomplete or paginated negative inventories. |
| `UNOBSERVABLE` | Wrong/unbound source, incapable method, unavailable/inapplicable/unrun check, missing qualification, or unsupported inference. |
| `UNVERIFIED` | Reported/synthetic evidence, or missing evidence/method digests; not independent verification. |
| `INVALID` | Malformed input or invalid/ambiguous scope. It cannot be approved by declaring it INVALID. |

Partial listings can still establish a returned object's presence or refute its absence. They cannot establish absence from unreturned pages. Three verified objects can support a three-object identity claim, not a 22-object claim.

Unsupported predicates deliberately include general causal exclusion, statistical equivalence/no effect, global implementation absence, unobserved training history, model behaviour change from a file hash, and build determinism from agreeing builds. They require a different qualified observation, not a stronger adjective.

### Read-only preflights

`row_coverage()` checks nonempty rendered text and unique IDs. `locator_coverage()` returns a single population for candidate-floor interpretation and records every skipped row. It does not replace the legacy metric.

`checked_rule_sweep()` is the preferred sweep entry point when the input rows are available. A qualified collector must bind `input_sha256` **at measurement time** to the canonical digest of `{rows, pool, split_field}` using `digest()`. The adapter verifies that binding, actual text/pool coverage, level counts, finite consistent measurements and the prospectively supplied rule universe. Do not attach a newly computed hash to an old unbound receipt and call that retrospective provenance. Missing historical binding means the stronger claim stays closed.

`sweep_coverage()` is a lower-level structural validator for imported results; its `pool_observed` input is an attestation, not an independent observation. Prefer `checked_rule_sweep()` whenever source rows exist. An observed zero-swing rule is explicitly insensitive to a cue differential; a zero or empty result from an inapplicable check is never treated as a clean instrument. A finite sweep only describes its declared rules and threshold, not all possible confounds.

`prediction_coverage()` rejects missing fields and duplicate IDs while accepting a genuinely emitted empty string. Extra rows require explicit admission. `pair_coverage()` rejects duplicate pair-band slots, missing bands/cells and cross-cell pairs. These preflights do not change any scorer, prediction or endpoint.

## Running the review

From a checkout of the successor:

```bash
python3 -m unittest discover \
  -s experiments/delta_multi_reasoning_v1/tests \
  -p test_observability.py -v

python3 experiments/delta_multi_reasoning_v1/audit/observability.py \
  --registry experiments/delta_multi_reasoning_v1/audit/observability_claims.json \
  --bindings experiments/delta_multi_reasoning_v1/audit/observability_findings.json \
  --repo-root .
```

An optional `--out NEW_PATH.json` writes a create-only sidecar. An existing output is refused. Input and historical receipts are never overwritten.

Every registry entry includes a proposed narrow claim, an observation and `declared_status`. The CLI exits nonzero when those disagree, the registry is empty/duplicated/malformed, or a pinned inspected source changes. A source change calls for a scoped re-audit; do not update a pin merely to silence it.

The initial registry deliberately records that **this audit did not independently verify the vault round trip**. Its consistent review has zero supported claims. That is expected, and it does not reverse the parent correction. Review consistency is not a scientific PASS.

## Enforcement and boundaries

The added GitHub workflow runs only on relevant pull requests or manual dispatch. It runs the new CPU fixtures, source-blob checks and structured registry review with read-only repository permission. It has no push trigger, training step, native model integration or vault credential access. It was not remotely executed during preparation.

No existing legacy function is changed or transparently monkey-patched. This preserves exact historical calculations and all existing artifact bytes. The new preflights and registry are the supported prospective audit/publication path. A legacy caller that ignores them can still bypass them; arbitrary unregistered prose is not mechanically understood. Branch protection was not changed, so the workflow is not claimed to be a mandatory merge gate.

`DIRECT`, digests and positive-control records are collector attestations. The generic validator checks their structure and consistency; it does not authenticate signatures, fetch raw external evidence, prove an arbitrary supplied digest corresponds to that evidence, or independently verify a collector's execution. The source-binding CLI does independently recompute the seven checked-out Git blob IDs. No qualified remote-vault collector is supplied or claimed here.

The tests use controlled CPU fixtures. They are evidence that these guards catch the exercised failure modes, not that a model improved or that the branch's entire historical research record is verified. The complete legacy test suite, native runtime, official weights, private vault and remote CI were not executed in this environment.

## Outcome preservation

The intended publication is additive only, parented at the audited commit. No source file listed in the bindings, existing result/receipt, dataset, model, permit or experiment script is edited. Remote publication should be checked by an exact-parent comparison: all changed paths must be newly added OBS-001 files or its scoped workflow. Preserve this distinction between source inspection, local fixture qualification and remote publication verification.
