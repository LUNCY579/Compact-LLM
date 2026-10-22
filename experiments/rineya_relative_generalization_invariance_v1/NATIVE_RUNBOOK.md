# Native and existing-evidence runbook

## 1. Environment and qualification

Work in a separate checkout of the dedicated research branch. On the operator's research
machine, use an isolated Python environment and enter this experiment directory. Do not start
concurrent GPU jobs or dispatch Codex tasks from this program.

```sh
python -m pip install -r requirements-native.txt
python -m pytest -q
python -m pytest -m optional_hf -q
```

The two optional integration tests must actually **pass**, not merely be collected and skipped.
They use locally constructed tiny random Qwen3 weights and tiny PEFT adapters; they still do
not qualify Rineya accuracy, training quality, sparse-weight preservation or CUDA behavior.
Record installed versions and then run a bounded native parity/preflight check. The delivery
machine could not install Transformers/PEFT; its Torch tests use an explicitly labelled causal
mechanism fixture, not a Qwen3 substitute.

## 2. Pin exact local inputs

The runner accepts local directories only. Source artifacts remain read-only. Copy
`configs/native.example.json` to a new run-specific spec and replace all placeholders with
absolute local paths and measured fingerprints. The hash is a directory-manifest digest over
file names, sizes and individual SHA-256 values, **not** the hash of a single weight file.

```sh
python -m rgi fingerprint --path "/absolute/local/Titan-Sparse" --kind model
python -m rgi fingerprint --path "/absolute/local/tokenizer" --kind tokenizer
python -m rgi fingerprint --path "/absolute/local/registered-adapter" --kind adapter
```

FP32 is required. The loader refuses non-Qwen3 architectures, custom remote-code mappings,
quantization, missing/unexpected state-dict keys and unregistered adapter fingerprints. It
supports ordinary bias-free LoRA without `modules_to_save` or DoRA. Use the originating
research program's exact sparse host and adapter combination. A path called Titan-Sparse is
not evidence of identity. Safetensors must already exist locally; no weights are downloaded.

A comparison may cross host identities, but a composition interaction must share one host.
For collection, use `adapters: []` for off, one entry for A or B, and both entries for A+B:

```json
{
  "path": "/absolute/local/adapter_A",
  "sha256": "MEASURED_ADAPTER_DIRECTORY_FINGERPRINT",
  "coefficient": 0.25
}
```

The entries belong in the spec's `adapters` list. A+B must use the same coefficients as the
corresponding single-component arms. Low-rank concatenation represents the weighted sum
without modifying base weights. Sequentially trained adapters require their own provenance;
do not pass them off as independently trained fixed deltas.

The conservative default free-memory checks are 8 GiB host RAM and 7 GiB CUDA memory. They
are refusal thresholds, not a guarantee against OOM. Native floors cannot be set below 4 GiB.
There is no automatic quantization, CPU fallback, precision downgrade or smaller-context retry.
Collection releases its own model and unused CUDA cache between arms. Failed work must remain
recorded; resource changes require an explicit new attempt consistent with the protocol.

## 3. Export an explicit bank

Existing RCWM/BWM generators should export their own exact prompts/answers and semantic IDs.
Do not replace those generators with `mini-bank` for native scientific claims. Raw bank format:

```python
from rgi.io import seal, write_new

raw = {
    "schema_version": 1,
    "kind": "raw_bank",
    "evidence_class": "native_model",
    "items": [{
        "item_id": "example-0001",
        "parent_id": "parent-0001",
        "lineage_id": "semantic-lineage-0001",
        "family": "ledger",
        "operation": "transfer",
        "rule_case": "posted",
        "suite": "skill",
        "format_id": "ORIGINATING_RUNNER_FRAME_ID",
        "split": "development",
        "horizon": 1,
        "target_rule": False,
        "semantic_payload": {"state": [10, 20], "op": "transfer", "amount": 3, "posted": True},
        "prompt": "EXACT_ORIGINATING_PROMPT_ENDING_WITH_A_VALID_ANSWER_DELIMITER\n",
        "target": "[7,23]",
        "content_spans": [[1, 2], [3, 5]]
    }]
}
write_new("runs/raw-development.json", seal(raw))
```

The prompt above is a schema illustration, not a usable benchmark prompt. Export the exact
registered frame, including chat-template rendering where applicable, from the source runner.
The collector does not invent a chat template or alter thinking-mode flags. Content spans are
zero-based, half-open character offsets **inside the target**. Use explicit semantic annotation;
the independently authored mini-world fixture's numeric heuristic is not a universal mask.

```sh
python -m rgi encode --raw runs/raw-development.json --spec runs/native.json --out runs/development.json
python -m rgi preflight --spec runs/native.json --bank runs/development.json --out runs/preflight.json
python -m rgi collect --spec runs/native.json --bank runs/development.json --out runs/off.json --execute
```

Encode once and share that exact bank across models. No tokens are silently truncated. A token
crossing the prompt/target boundary is a refusal, not a request to tokenize the two pieces
independently. Register a compatible delimiter and rebuild development data first.

## 4. Reuse complete cached token evidence

The source producer must export one JSONL record per scored token. Required keys are
`item_id`, `position` (absolute position in the encoded input), `target_id`, `context_sha256`
(SHA-256 of canonical JSON of all preceding token IDs), `role`, and finite nonnegative `nll`.
Positions are not target-relative. The record order may differ; complete keys must match.

Create a sealed producer receipt containing:

```python
from rgi.io import digest, file_hash, load, seal, write_new

bank = load("runs/development.json")
spec = load("runs/native.json")
receipt = {
    "kind": "external_token_loss_receipt",
    "records_sha256": file_hash("runs/source-token-losses.jsonl"),
    "bank_sha256": bank["content_sha256"],
    "tokenizer_sha256": bank["tokenizer_sha256"],
    "evidence_class": bank["evidence_class"],
    "scoring_mode": "teacher_forced",
    "context_policy": "shared_reference",
    "model_id": digest(spec),
    "host_sha256": spec["model_sha256"],
    "components": {},
    "execution_contract": {"collector": "NAME_AND_VERSION_OF_QUALIFIED_PRODUCER", "precision": "float32"}
}
# Populate components from real registered adapter hashes and coefficients when present.
# Record the producer's actual runtime contract; do not erase mismatches to force alignment.
write_new("runs/source-receipt.json", seal(receipt))
```

```sh
python -m rgi import-losses --bank runs/development.json --rows runs/source-token-losses.jsonl --receipt runs/source-receipt.json --out runs/imported.json
```

Different runtime contracts are rejected during comparison. First-token-only evidence cannot
be expanded to all-token losses; create a separately declared first-token bank or recollect a
complete development bank. Accuracy and greedy generations cannot be converted to NLL.

## 5. Build matched data plans and train only after review

Export and encode training, development and confirmation separately. Deduplicate semantic
payloads and lineages across splits. Use the full original training pool as the contamination
exclusion set, not only the selected rows.

```sh
python -m rgi match-controls --bank runs/train.json --pairs 64 --seed 17 --out runs/controls
python -m rgi control-plan-matrix --targeted runs/controls/targeted.json --control runs/controls/control.json --spec runs/native.json --seeds 17 23 29 --out runs/plans
python -m rgi train --bank runs/controls/targeted.json --spec runs/plans/targeted_s17.spec.json --plan runs/plans/targeted_s17.plan.json --out runs/train-targeted-s17 --execute
python -m rgi train --bank runs/controls/control.json --spec runs/plans/control_s17.spec.json --plan runs/plans/control_s17.plan.json --out runs/train-control-s17 --execute
```

Review every plan's token counts, input identities, optimizer settings and update limits before
execution. Run remaining seeds serially only when authorized. The training kernel normalizes
gradients by total supervised tokens per accumulated update, masks prompt labels, clips only
LoRA gradients and saves only the adapter plus receipts. It never saves over a published core.
These commands perform real training when dependencies and local inputs are present; none were
executed against Rineya during delivery.

Create fresh evaluation specs naming each saved adapter fingerprint. Development losses and
behavior determine whether a later fresh confirmation is warranted. Training completion is
not an acquisition or retention result.

## 6. Bridge existing world-model and protected-behavior runners

Export outcomes as a JSON object with:

```
kind: behavior_outcomes
complete: true
invalid_policy: count_as_failure
bank_sha256: content hash of the registered behavioral bank
model_id: digest of the exact registered model spec
evidence_class: native_model
freeze_sha256: content hash of the confirmation freeze (confirmation only)
rows: [...]
```

Every row contains `item_id`, `parent_id`, `family`, `split`, `payload_sha256`, `endpoint`, and
finite numeric `value`. Use distinct registered endpoint names for one-step accuracy,
self-rollout final-state accuracy, legality, exact copying, planning regret, joint-use success
and protected behavior as needed. An endpoint has one value per registered item. All variants
of one parent retain the same parent ID. Failures remain present and count according to the
registered metric. A normalized-regret endpoint is lower-is-better; accuracy is higher-is-better.

For multi-goal evaluations, either export the preregistered per-parent/per-plan summary or
represent the goal-specific rows as distinct registered items within the same parent cluster.
Never create independent parent IDs for dependent goal readouts. The behavioral bank may be
separate from the teacher-forced bank but must be frozen and disjoint from training/development.
The package does not reverse-engineer undocumented RCWM/BWM record formats.

## 7. Fresh confirmation

Prepare an `inputs.json` mapping `training_bank`, `development_bank`, `confirmation_bank`,
`reference_spec`, `candidate_spec`, and optionally `behavior_bank` to local file paths. Start
from `configs/confirmation.example.json`, replace illustrative endpoints/counts/margins with
the approved study, and change `approved` only after review.

```sh
python -m rgi freeze --inputs runs/inputs.json --settings runs/approved-protocol.json --ledger runs/confirm-ledger --out runs/freeze.json
python -m rgi collect-frozen --freeze runs/freeze.json --out runs/frozen-token-outputs --execute
python -m rgi confirm --freeze runs/freeze.json --reference runs/frozen-token-outputs/reference.json --candidate runs/frozen-token-outputs/candidate.json --reference-outcomes runs/frozen-behavior-reference.json --candidate-outcomes runs/frozen-behavior-candidate.json --out runs/confirmation-result.json
```

The behavioral exports in the last command must come from the registered external behavior
runner, not from the loss profiler. Missing files or unregistered items fail closed. A consumed
failed analysis cannot be silently rerun. `RESULT.json` inside the ledger is the durable source
of truth; the requested output is a companion copy. `render_report` creates a human-readable
view without changing the evidence.
