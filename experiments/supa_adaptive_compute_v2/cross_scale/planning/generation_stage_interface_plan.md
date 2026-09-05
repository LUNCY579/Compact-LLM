# Draft engineering plan: Track B generation-stage interface

Status: planning only, 2026-09-05. This document is **not a preregistration,
scientific selection, or authorization to execute**. It does not select a depth
pair, bridge objective, gate dose, control scale, threshold or runtime. It was
prepared by reading implementation and protocol files; no partial matrix metrics,
efficacy outputs, model weights or answer-bank contents were inspected for this
plan. No implementation is supplied here.

The current CPU 16-pair, six-objective mechanism matrix must finish under its
existing frozen source map. The sole entry condition for later preparation is
the existing complete-matrix `MECHANISM_GO` and
`small_generation_eligible_for_separate_freeze=true`, independently verified.
`MECHANISM_NO_GO`, an unfinished attempt, missing pair/control, ambiguous duplicate
success, or failed provenance leaves generation closed. Even a mechanism GO only
permits preparing the separately frozen 128-case architecture-development stage;
it does not license the 1,024-example confirmation bank.

## Isolate the later stage without changing existing source commitments

Proposed implementation root: `cross_scale/generation_stage/v1/`, with its own
`src/`, `scripts/`, `configs/`, `protocol/` and `tests/` beneath that root.
Use a distinct stage identifier such as `cross_scale_generation_v1`. Store new
receipts under a distinct `cross_scale/evidence/generation_v1/` subtree and any
ignored bulk artifacts under `cross_scale/local/generation_v1/`. Exact names and
receipt schema will be frozen before execution.

Do not add Python under `shared/`, programme `scripts/`, or existing
`cross_scale/src/` and `cross_scale/scripts/`; do not add documents under the
existing `cross_scale/protocol/`. `shared.hashing.source_files("cross_scale")`
recursively scans those locations. Adding later files there would invalidate the
current matrix's source map. Native calibration also binds shared/programme
sources, so those roots must remain untouched. The proposed generation subtree
and this `cross_scale/planning/` directory are outside both current inventories.

Implement a **stage-owned registration and admission adapter**, not a change to
the old registration schema. It must bind an explicit transitive file map: all
new stage implementation, tests and protocol files; the exact unchanged shared,
legacy and Stage 1 modules actually imported; and any explicitly imported frozen
native operator dependencies. Record the complete Stage 1 registration separately
as upstream provenance. Additional versions can use sibling directories with
their own inventories. No version may silently replace another version's code.

Keep old source maps exact. Stage 2 verifies the old committed dependencies and
receipts against their old map, then verifies its own new map independently. Do
not reinterpret an additive Stage 2 map as the Stage 1 registration. Freeze and
commit the new stage source, configuration, scorer/data bindings and registration
before generation or runtime qualification. A source change after freeze fails
admission and needs a new prospective version.

## Verify the selected bridge without refitting it

The stage's export/verification command should consume one explicitly identified
completed Stage 1 aggregate and its upstream registration/configuration. It must:

1. Verify the original committed registration, source map, dataset manifest,
   capture receipt and checkpoint identities. Reuse
   `cross_scale/scripts/run_alignment.py`'s `validate_receipt`,
   `read_capture(..., metadata_only=True)` and `collect_completed_pairs` where
   their current contracts apply. Metadata-only capture verification still hashes
   the activation artifact; it does not allocate the multi-gigabyte tensor cache.
2. Reconstruct exact 16-pair/six-objective coverage, with exactly one complete
   attempt per pair and no unclosed attempt. Retain the hashes and status of every
   failed/abandoned attempt. Verify every successful `metrics.json` and
   `bridges.pt` against its pair receipt and shared capture identity.
3. Re-run the unchanged `select_matrix` on the full verified metrics and require
   equality with `selection.json`, the aggregate receipt's embedded `selection`,
   and the hashed `matrix.json`. Verify the aggregate's terminal status and
   started contract too. Stage 1 receipts use a digest of their embedded verified
   registration object in `registration_sha256`; distinguish that legacy value
   from the registration file's internal contract digest and from its file hash.
4. Accept only the single selected `(supa_layer, donor_layer, objective)` produced
   by that frozen rule. Load the selected pair's hash-verified `bridges.pt` with
   `weights_only=True` on CPU. Extract that objective and the **same pair's** BS
   and BR state dictionaries. Do not rerun fitting, reseed BR, reshuffle BS, use a
   different pair for a null control, or choose an optimizer checkpoint.
5. Validate the exact `up`, `up_bias`, `down`, `down_bias` keys, dimensions,
   float32 dtype, finiteness and nontrainable state. Compute individual state
   digests as well as the entire upstream artifact's file hash. Preserve exact
   normalization and affine biases; the downward bias cancels from the residual.
6. Create an exclusive small selection/export manifest binding the selected and
   null artifacts, all upstream receipt/config/source hashes, Supa checkpoint and
   tokenizer sidecars, donor revision/shards/config, insertion and donor indices,
   and calibration fit/held-out partitions. If making smaller tensor exports,
   prove tensor-by-tensor equality to the upstream dictionaries and bind both
   original and exported hashes. Missing local bulk artifacts block admission;
   they must not be recreated by an unregistered refit.

The exporter may verify metadata and copy exact selected tensors after the full
gate; it must not run a model or access architecture/confirmatory answers. Once
exported, downstream inference verifies this provenance before loading models.

## Candidate/control interface and fixed gate grid

The only gate grid already specified by Stage 1 is **0.03, 0.10, 0.20, 0.40**.
Carry it forward exactly. A cell should have a distinct `cell_id`, scientific
`arm`, gate where applicable, operator artifact digest and insertion site. Keep
one global frozen stage configuration containing the full matrix, so baseline
reuse across gate cells does not hide configuration changes.

| Scientific arm | Engineering treatment to prepare |
|---|---|
| X0 | Intact frozen canonical Supa baseline, evaluated on all 128 same cases. |
| XG | Single Stage 1 selected bridge and donor layer at each of the four gates. |
| XS | That selected pair's already-fitted BS bridge with the same donor, insertion and four gates. |
| XR | That selected pair's already-fitted BR bridge with the same donor, insertion and four gates. |
| XM | Mandatory generic matched-compute control, with exact design/initialization/strength rule still requiring prospective specification. |
| XN, if available | One independently frozen native operator whose identity and eligibility are fixed before this generation stage. |

Use a unique cell key, for example an arm plus an exact canonical gate string,
without pretending four doses are four independent scenario samples. X0 can be
reused within this stage only through a matching complete baseline shard and
stage/registration/data/runtime identity. XS/XR are never removed because they
perform well. XN availability is a pre-freeze condition; do not substitute a native
candidate after viewing Track B generation or import Native efficacy rows into
the Track B denominator. If unavailable, disclose that condition rather than
manufacture an XN result.

XM needs new engineering: Native AM's width matches two native layers, not the
selected donor-plus-bridge work. Do not simply rename AM or reuse Native's fitted
alpha. Compute the target from the selected donor geometry plus both bridge
matmuls, and prospectively specify an independent untrained residual transform,
its rounding convention for approximate work matching, seed, parameter inventory
and strength policy. Any scale fitting must use Track B calibration only under
its own frozen rule, never architecture efficacy, and must retain held-out
reporting. Whether XM has one fixed treatment or paired dose cells is an open
pre-freeze decision; this document does not resolve it.

The shared evaluator expects a context-managed wrapper with `records`, `calls`,
`capture`, RMS extrema and an extra-work inventory. The current
`cross_scale.src.graft.CrossScaleOperator` is an adapter exposing `context(model)`,
not that wrapper itself. Its returned historical `Graft` provides the required
hook lifecycle. The stage factory should return this wrapper and attach explicit
added/active parameter counts and `extra_matmul_flops_per_token` from
`legacy_graft().extra_flops(expert)`. The inherited recurrence count alone would
otherwise understate graft work. Preserve `h + g[(L(project(h))-project(h)) down]`,
ordinary positions and causal masks, donor-specific rotary computation, no cache,
frozen parameters/buffers, and exact disabled/zero-gate/restored identities.

## Reuse and the missing execution layer

| Existing file/function | Reuse and boundary |
|---|---|
| `shared/model_loading.py`, `runtime.py`, `integrity.py`, `timing.py` | Canonical model loading, pinned runtime/weights, state hashes, memory and single heavy-worker lease. Verify donor/bridge state and resource footprint additionally. |
| `cross_scale/src/donor_layers.py` | Pinned donor identity and actual HF single-layer loading; pass a separately validated stage runtime adapter without changing this source. |
| `experiments/cross_scale_layer_graft_v1/graft.py` | `Bridges`, `ResidualExpert`, `Graft`, donor/bridge FLOP accounting. Preserve the historical file. |
| `shared/datasets.py` | Existing frozen Track B `architecture_dev` role, exact 128 original scenarios and manifest checks. No Track A rows, no calibration rows in the denominator. |
| `shared/generation.py::evaluate_case` | Greedy full-prefix generation, canonical-answer NLL, first-token logit artifacts, hidden diagnostics, strict malformed-output guards, synchronized latency and work estimates. |
| `shared/evaluation.py::shard_cases` | Deterministic scenario ownership. Keep linked data in one shard if later separately allowed. |
| `shared/scoring.py`, historical `output_flags` | Frozen strict exact/JSON scoring and independently reconstructed validity. Incorrect but well-formed responses remain outcomes. |
| `shared/statistics.py`, `reporting.py` | Paired wins/losses, exact McNemar, scenario bootstrap intervals, NLL deltas, family summaries, identity rate, multiplicity tools and quality/cost frontier. |

Do **not** call `shared.evaluation.run_shard` or `aggregate_runs` as though a new
stage registration were a track registration. They call the old track-wide
`verify_registration` and `source_files` internally. Passing a new config alone
would leave new stage code outside that source inventory. The later thin
stage-owned runner must reuse the lower-level machinery above and supply its own
registration, baseline compatibility, cell matrix, immutable receipts and audits.
Do not monkeypatch shared registration globals to bypass this boundary. Native
analysis also hardcodes Track A arms/roles and cannot serve as Track B's decision
function; reuse its audit concepts and shared statistics, not its scientific gates.

Proposed new files, all beneath `cross_scale/generation_stage/v1/`:

- `src/protocol.py`: exact stage configuration, closed confirmation gate, fixed
  permitted cells, selected-artifact and runtime admission contracts.
- `src/provenance.py`: stage source map, immutable upstream verification/export,
  registration verification and digest semantics.
- `src/operators.py`: X0/XG/XS/XR/XM/conditional-XN factories, work/state inventories.
- `src/execution.py`: shared-evaluator shard lifecycle, baseline prerequisites,
  failure receipts, separate stage evidence paths, exact resumption contracts.
- `src/analysis.py`: independent raw-evidence reconciliation, complete cell coverage,
  paired summaries and the later preregistered mechanical/dose selection rule.
- `scripts/export_selected.py`, `freeze_stage.py`, `run_shard.py`,
  `run_campaign.py`, `audit_stage.py`: narrow commands with explicit stage identity.
- `configs/generation_v1.json` and `protocol/preregistration.md`: created only
  after full mechanism GO and completion of outstanding prospective decisions.
- `scripts/qualify_runtime.py` and its stage-owned receipt verifier, if GPU is
  selected; the existing alignment qualifier is not a generation qualifier.
- `tests/test_provenance.py`, `test_operators.py`, `test_execution.py`,
  `test_analysis.py`: tiny fixtures and synthetic receipts, never capability claims.

## Data, runtime, measurements and decision boundaries

Before freezing, establish by metadata/provenance that the 128 Track B architecture
scenarios remain unused for efficacy; freeze their existing manifest and exact
IDs. Do not edit a question after an answer. Record that this stage is an
engineering/dose-selection bank, not independent capability confirmation. The
1,024-case confirmation and its diagnostics stay closed throughout this draft.

CPU FP32/eager/four-thread/no-cache remains the reference. GPU use needs a
separate generation-stage CPU/GPU qualification for the actual full Supa plus
selected donor/bridges and every distinct control/dose treatment, committed before
GPU generation. Preserve the Stage 1 CPU evidence as CPU evidence. Freeze numeric
tolerances, fixtures, resource admission and full-run budgets before qualification;
do not loosen them after failure. Sequential execution must hold the existing
machine lease. Include combined Supa/donor/bridge RSS and VRAM; a single streamed
layer does not imply that full-generation memory automatically fits.

Require matching X0 before candidates, exclusive attempts and fail-closed
nonfinite/cache/hash/gradient/deadline/RMS/malformed-output handling. Record all
attempts, including failed doses. Never truncate failed evidence into a successful
paired denominator. The prospective policy must specify whether and how a
predefined destructive-dose rejection permits completing other cells; it cannot
be improvised from partial scores. Audit only complete eligible cells, with all
rejections disclosed and no new seeds/doses appended.

For every complete cell report raw accuracy, paired wins/losses, exact McNemar,
scenario bootstrap confidence intervals, absolute/relative error reduction,
canonical-answer NLL delta, full-output identity, first-token cosine/KL, ordinary
hidden/residual RMS, invalid outputs and safety stops, prefill/decode/wall time,
RSS/VRAM, parameter inventories and dominant FLOPs with exclusions. Reconstruct
scores/validity from raw responses and verify first-token artifacts, not just
saved summary fields. Compare XG with X0 and same-dose XS/XR, and with XM/XN under
the treatment matching fixed in the later protocol. Label dose and family
comparisons exploratory or selection analyses; preregister multiplicity and
resampling seeds/counts before reading generation results. A 128-case screen
cannot by itself establish a small general capability gain.

Still unresolved, and required before any execution: numeric inertness and
destructiveness rules; preservation/validity eligibility; deterministic dose
selection and tie-breaking; which comparisons constitute the selection family;
XM specification; XN availability/identity; runtime qualification and resource
limits; shard allocation; and confirmation-entry requirements. These decisions
must follow the full mechanism gate and a separate prospective generation freeze.
No value is inferred here from the partially executing matrix.

## Required tests before that freeze

- Reject incomplete/duplicate matrices, absent nulls, open attempts, altered
  selection, wrong registration-digest interpretation and edited bridge bytes;
  verify tensor-exact export from the selected and same-pair BS/BR dictionaries.
- Verify actual tiny HF graft ordering, original masks/positions, donor RoPE,
  affine downward-bias cancellation, every fixed gate, zero-gate/disabled/restored
  identity, seed isolation, frozen tensors and accurate extra-work accounting.
- Exercise baseline-before-candidate, cell-aware coverage/resumption, failure
  preservation, no overwrite, current-stage versus old-stage separation and
  rejection of arbitrary gate/bridge/control overrides.
- Prove the new subtree does not alter either old Stage 1 or Native calibration
  source maps. Verify its own registration does include every new stage file and
  transitive imported dependency. Subsequent stage versions must remain isolated.
- Reconcile synthetic raw answers/logits and validity; reject partial denominators
  and scenario duplication; test paired analyses and the eventual frozen
  dose-selection rule with controls that win, lose, stay inert or fail safety.
- Verify runtime admission fails without the stage's own qualification and test
  actual CPU/GPU agreement only in the separately authorized qualification run.

Nothing in this draft permits implementation changes to the live matrix's frozen
sources, execution of a generation model, or opening confirmation early.
