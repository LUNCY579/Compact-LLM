# GD-001 producer attachment contract — October 5, 2026

Status: offline integration preparation. No eligible completed trajectory was found.
Native export compatibility, scoring, checkpoint-selection efficacy and confirmation
remain unqualified. The existing README, protocol, runbook and October 3 receipts
remain in force. This contract authorizes no training or native inference.

## Exact producer handoff

The producer's existing owner must approve a prospective plan before wiring this
observer. Reuse that owner's chat; do not start a competing campaign. Prefer CE8
only after independently approved research scope, export format and retention
policy qualify. Its six discarded qualification updates are excluded. The present
CE8 custom parameter/Adam/RNG checkpoints and rolling retirement do not qualify.
RDD is a possible later producer, subject to its own authorization and these checks.

Give the owner these files: `producer_hook_example.py`, `integration.py`,
`attachment.py`, this contract, and `export_parity.py`. This handoff has not been
sent to another chat. No live or frozen trainer was edited.

1. Bind one exact original 500M-700M source and matching tokenizer, including
   configuration, all weight shards, tokenizer/template files and full file hashes.
   A warm initialization containing a prior delta needs its own exact source identity;
   its bare donor base is not interchangeable. Custom readouts, unrelated adapters,
   merged/derived cores and missing weight payloads cannot stand in for snapshots.
2. Approve the actual checkpoint schedule and stable absolute step directories
   before training outcomes. Retain 2-19 original same-format HF or standard LoRA
   saves in strict step order, with unchanged architecture/adapter configuration.
   Preserve every planned payload and all provenance; disable prospective pruning
   for this approved trajectory. GD does not change the producer's retention policy.
3. After each existing producer save has fully returned, call
   `integration.py::record_snapshot`. The example `after_completed_save` checks
   the approved path/step and returns the exact contract entry and receipt file hash.
   Run after all export shards/metadata are finished, outside an optimizer step.
   Never save into the base directory or reuse a rolling directory. The hook records
   and hashes; it performs no conversion, save, optimizer update or evaluation.
4. Preserve actual source, filtering, rendering and consumed token/text payloads
   as described below. Include each training and replay consumption occurrence,
   including repetitions, packing/selection history, and actual truncation decisions.
   Template hashes or a raw source recipe alone do not establish what was consumed.
   Retain other qualification/DEV exposures in separately identified producer
   provenance; those exposures are not checkpoints or substitutes for training rows.
5. Supply the completed training receipt, exact base reference and last planned save
   as the final control. Bind counts, times, steps, provenance and hashes. An export
   receipt or interrupted run cannot substitute for completed research training.
6. Have GD validate the complete retained attachment offline. A compatible-looking
   inventory is still metadata evidence. Original custom-to-PEFT exports require
   independent export parity and actual strict loading before compatibility claims.

For example, owner code inserted only under its approved prospective plan:

```python
from experiments.rineya_generalization_dynamics_v1.producer_hook_example import after_completed_save

# The producer owns this existing save and finishes it before the observer runs.
existing_export_to_unique_step_directory(checkpoint_path)
entry = after_completed_save(
    approval_plan={"path": approved_plan_path, "sha256": approved_plan_file_hash},
    receipt_directory=external_gd_receipt_directory,
    checkpoint_directory=checkpoint_path,
    step=actual_optimizer_step,
    kind="peft",  # "hf" only for original standard full-model exports
)
# Preserve entry and the original directory; do not replace weights with the receipt.
```

## Retained files and schema

`attachment.py` is the executable schema reference. Contract JSON has
`schema: "gd001_attachment_v1"`, `evidence_kind: "producer_metadata"`,
`trajectory_id`, `expected_base_parameters`, and:

| Field | Retained binding |
|---|---|
| `producer` | `owner`, `study_id`, `independently_approved: true`, `artifact_role: "original_training_trajectory"` |
| `approval_plan` / `completion_receipt` | Absolute original JSON file `path` and full-file `sha256` |
| `base` / `tokenizer` | Absolute export `path` and registry `content_sha256` over complete admitted file inventory |
| `checkpoints` | Ordered `{step, kind, path, content_sha256, snapshot_receipt: {path, sha256}}`; last is final |
| `provenance` | `events` and `training_manifest`, each with retained absolute `path` and full-file `sha256` |

The prospective plan repeats producer/source/tokenizer/count/trajectory identity,
`checkpoint_schedule: [{step, kind, path}, ...]`,
`status: "APPROVED_PROSPECTIVE"`, `purpose: "research_training"`,
`approved_at_utc`, and
`retention: "all_original_snapshots_and_provenance"`.

The completion JSON repeats those identities and schedule, `status: "COMPLETED"`,
`purpose: "research_training"`, `training_started_at_utc`, `completed_at_utc`,
`final_step`, the provenance references, `source_count`, `rejected_count`,
`unused_accepted_count`, `consumed_count`, and `consumed_per_step` keyed by actual
optimizer step. It also binds the full `checkpoints` entries and `final_checkpoint`
equal to the last entry, including saved content hashes and post-save receipt references.
Approval must precede training start. All timestamps have timezones.

Events JSONL contains globally unique `event_id` and the following `event_type`:

| Type | Required payload |
|---|---|
| `source` | Parent/source IDs, one-based `source_row_number`, retained JSONL `source_ref`, exact `raw_record` |
| `filter` | Prior source event ID, accepted/rejected disposition and reason, complete `filtering_history` operations/decisions/reasons |
| `consume` | Prior source/filter event IDs, matching parent/source IDs, actual optimizer `step`, pre-render `prompt`, `raw_rendered_prompt`, `consumed_prompt`, actual `input_ids`, tokenizer content hash, `rendering_history`, exact `truncation` object |
| `unused` | Source/filter event IDs and reason an accepted row was not consumed before completion |

Every retained source JSONL row has exactly one source/filter disposition, including
rejections and accepted rows not used before the final step. Repeated consumed rows
remain separate occurrences, in step order; every completed optimizer step has
consumption and agrees with the completion counts. Retain packing, sampling,
rendering/template options and fragment boundaries in the history operation payloads.

`truncation` contains `policy: "none" | "left" | "right"`, `original_input_ids`,
`max_tokens` (null for none), and `history`. A real truncation history is exactly
`[{operation: "truncate", side, removed_tokens, max_tokens}]`; retained consumed
IDs must be the declared token slice. GD's own DEV inference never truncates.
The validator checks recorded token slices without executing the tokenizer;
native prompt/token parity remains a separate obligation.

`training_manifest` is the exact JSONL projection of consume events as
`{prompt: consumed_prompt, parent_id}`, including repetitions and order. No raw-data
summary, omitted rejection history, missing original file, metadata-only weight
inventory or conveniently small prompt subset passes this contract.

DEV preparation additionally excludes normalized exact matches against each
consume event's raw question, raw rendered text and consumed text. Rendered-only
projection can otherwise hide a raw question overlap when templates or parents differ.
Bind every relevant open qualification/DEV/teacher/replay exposure inventory through
the reviewed proposal's `public_exposure_manifests: [{path, sha256}, ...]`. Each JSONL
row has `scope: "public_training" | "public_qualification" | "public_dev"`,
`public: true`, actual `prompt` and `rendered_prompt`, `parent_id`, and
`parent_namespace`. Both prompt views and parent IDs are exclusions; namespaces
remain preserved provenance. This is an exact exclusion audit, not semantic
decontamination. The producer owner must attest that the list covers its relevant
open exposures; no existing sealed/confirmation bank is read to build it.

All receipts use content digests, not digital signatures. Owner assertions and counts
cannot independently prove that upstream events were never omitted or fabricated.

## Offline commands and registered DEV proposal

```powershell
python -m experiments.rineya_generalization_dynamics_v1.attachment_cli check --contract C:/Users/clclc/BusinessProject/gd001-records/PRODUCER/attachment.json
python -m experiments.rineya_generalization_dynamics_v1.attachment_cli prepare-dev --contract C:/Users/clclc/BusinessProject/gd001-records/PRODUCER/attachment.json --proposal C:/Users/clclc/BusinessProject/gd001-records/PRODUCER/reviewed-dev-proposal.json --output C:/Users/clclc/BusinessProject/gd001-records/PRODUCER/dev-preparation-001
```

These commands inspect complete supplied files and prepare DEV; neither loads a model.
Use them only with the identified producer's open, authorized training provenance.
Do not point them at sealed or confirmation banks. `mechanical_fixture` attachments
cannot produce native specifications. This preparation CLI and producer hook require
fresh output directories under external BusinessProject/gd001-records, avoiding sibling
source checkouts too. Put TEMP/environments/caches under BusinessProject outside source.

`configs/dev.proposal.example.json` deliberately fails until exact identities,
step numbers and review digest are supplied. The proposed first milestone fixes
DEV seed 17, 24 pairs per family, existing default selection policy, CPU FP32,
two threads, plain non-thinking prompts, 512 input tokens and 16 output tokens.
It uses the exact base plus one prospectively selected intermediate and the verified
final snapshot: 864 prompt scores, at most 15,552 DEV forwards. Two golden probes
per checkpoint for each producer/GD trace add at most 216 forwards, for a conservative
combined 15,768. Aggregate limits are 1,800 seconds and 16,000 actual forwards across
DEV, both parity traces, all three checkpoints and resumed attempts. Existing
7.4 GiB RAM/VRAM and 1 GiB disk guards remain unchanged. CUDA or
another runtime would require its own exact reviewed preregistration before outcomes.

The reviewed proposal has status `REVIEWED_PROPOSAL_NOT_EXECUTION_AUTHORIZATION`,
stage `DEV_ONLY_NATIVE_LOADING_SCORING`, `separate_native_authorization_required: true`,
`confirmation_authorized: false`, current `code_sha256`, exact contract file reference,
three numeric `checkpoint_steps`, policy/runtime/DEV settings, and canonical body digest
via `common.signed`. Preparation preserves the validated attachment, exact spec, policy,
DEV payload and suite hash together. Missing identities remain blockers, not placeholders
that a native runner may infer. This delivery's unresolved proposal is in the external
receipt directory.

Before future `init`, revalidate the attachment, proposal, dependency/runtime and
all retained files against that preparation receipt. Existing pipeline registration
does not bind extra attachment keys: preserve and verify the external sidecar as
part of the reviewed handoff. No producer wiring or native execution was added here.

## Export parity and loader qualification to perform later

`export_parity.py::compare_export_parity` compares supplied traces using a signed
pre-outcome policy, expected source/schedule/DEV probe registration and two signed
producer/GD trace receipts. Freeze exact runtime versions, tokenizer, format and
greedy decoding. Proposed CPU FP32 tolerances are absolute/relative 0.00001; numerical
failure is a blocked qualification, not permission to increase tolerances afterward.
Register a bounded DEV probe inventory with multi-token answer and distractor cases,
and fixed vocabulary IDs for logit probes before model outcomes.

Require original base plus two saved snapshot identities; strict empty missing/
unexpected/mismatched/error-key load records; matching actual base parameter count;
exact LoRA tensor keys/shapes/values; actual rendered text and prompt IDs; unchanged
generation IDs/EOS/stop reasons; both complete candidate token inventories; each
teacher-forced candidate token's logit probe vector and target log probability; and
answer-only sums shifted once. All trace payloads and strict-load/scoring receipts
remain available. The comparator imports no native libraries and establishes only
`TRACE_PARITY_MATCHED`, never independent proof of execution or native quality.

The first native milestone, after separate authorization and host availability, is
strict loading and full DEV scoring of the exact base plus two saved snapshots,
sequentially with complete load/unload and cumulative budget receipts. Export-parity
probe calls are included in the combined 15,768-forward upper bound; no additional
conversion, probe or model call is an unmetered operation.
The later supervisor plan must meter producer parity, GD parity and DEV together;
the existing DEV split ledger alone does not account for external producer calls.
Use an owned-worker supervisor for the shared 1,800-second hard boundary; cooperative checks
cannot interrupt an in-flight load/forward. Never terminate another owner's process.

Tiny loader fixtures and supplied trace comparisons do not establish checkpoint
selection efficacy. Adequate DEV support, qualified scoring and a locked DEV selection
are prerequisites. Confirmation stays unopened until an eligible completed trajectory,
qualified native scoring, locked DEV selection and separate confirmation authorization
all exist. No model promotion or serving-policy change follows from this contract.
