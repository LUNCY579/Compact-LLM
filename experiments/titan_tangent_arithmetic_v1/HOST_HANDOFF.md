# Exact-host handoff

## Prerequisites and first action

No native checkpoint or Transformers runtime was present in the implementation environment.
The Qwen3 adapter is written but **not yet exercised on Hugging Face Qwen3 or Titan**. CPU
fixture success is not native adapter qualification. `qualify` is the first engineering run.

Use a clean checkout of the new branch's exact published commit. Keep model assets, tokenized
banks and all session outputs **outside** the checkout. Do not auto-download assets or modify
another session's workloads/lease to fit this run. Install/pin a host-compatible environment
separately; do not assume the CPU-test Torch version alone qualifies a Transformers version.
The adapter checks exact installed `torch`, `transformers`, `safetensors`, `psutil` versions.

Each invocation requires a new external output directory. Preflight is stdlib-only and loads
no models. A missing local file is reported as LOCAL_ASSET_NOT_OBSERVED, never as failed remote
backup verification. The wrapper has no remote backup observer.

```bash
python -c "from experiments.titan_tangent_arithmetic_v1.binding import package_digest; print(package_digest())"
git rev-parse HEAD
python -m experiments.titan_tangent_arithmetic_v1.host --manifest /absolute/qualify.json --preflight-only
python -m experiments.titan_tangent_arithmetic_v1.host --manifest /absolute/qualify.json --out /absolute/runs/tta-q001
```

Do not use the example below without replacing all placeholders and completing the declared
human/data checks. The manifest binds one requested action; the code does not manufacture
source/data qualification from a model display name.

## Manifest contract

```json
{
  "schema": "TTA-001-HOST-1",
  "action": "qualify",
  "device": "cpu",
  "source_commit": "<40-character exact checkout commit>",
  "package_sha256": "<package_digest()>",
  "runtime": {
    "torch": "<exact installed version>", "transformers": "<exact installed version>",
    "safetensors": "<exact installed version>", "psutil": "<exact installed version>"
  },
  "base": {
    "family": "Supa-1.7.0",
    "directory": "/absolute/exact-supa",
    "files": {
      "config.json": "<sha256>", "model.safetensors": "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b",
      "tokenizer.json": "<sha256>", "tokenizer_config.json": "<sha256>"
    }
  },
  "scope_prefixes": ["model.layers.8."],
  "qualification_probe": {
    "path": "/absolute/q-probe.jsonl", "sha256": "<sha256>",
    "status": "developmental", "human_reviewed": true,
    "tokenizer_sha256": "<same tokenizer.json hash>", "eos_token_id": 151645
  },
  "data": {
    "path": "/absolute/q-probe.jsonl", "sha256": "<sha256>",
    "status": "developmental", "human_reviewed": true,
    "tokenizer_sha256": "<same tokenizer.json hash>", "eos_token_id": 151645
  },
  "forbidden_parents": {"path": "/absolute/excluded-parents.json", "sha256": "<sha256>"},
  "limits": {
    "max_wall_seconds": 600, "max_logical_calls": 16, "max_length": 128,
    "max_rows": 1, "max_optimizer_steps": 0, "min_available_ram_gib": 12
  }
}
```

The EOS number above is an example, not auto-inserted: it must agree with the bound config
and reviewed serialization. **List every file in the model directory**, not only the four
shown. Unbound files are rejected. The current loader expects a single `model.safetensors`.
Use actual model-tokenizer compatibility, not matching architecture names. No remote code.

The single-layer example is a **restricted-scope qualification**. For the historical full
Titan update, bind all prefixes `model.layers.8.` through `model.layers.27.` and requalify that
exact scope. A receipt for one layer cannot authorize the tail-20 run. A full-model scope is
a different study and has much larger resource needs.

`qualification_probe` remains identical across the later manifests; `data` may change to a new
reviewed developmental bank. Qualification identity binds the base files, runtime, scope,
probe and implementation package. Modifying any of these invalidates the receipt.

## Tokenized input schema

Each JSONL row has exactly `id`, `parent_id`, `task`, `split`, `input_ids`, `labels`.
IDs and parents are nonempty strings. Use `diagnostic` for qualification/diagnostics and
`train`, `dev`, `test` for a fresh developmental study. Prompt labels are -100; supervised
response labels equal the corresponding input token IDs and include the final EOS. The
first label is -100, and at least one shifted response token must be supervised.

Render/tokenize with the exact model tokenizer outside this runner and review the artifact;
no hidden chat-template, truncation or EOS guessing occurs here. The parent exclusion file is
a JSON list of all prohibited historical parent IDs supplied by the research records. An empty
file cannot prove historical disjointness. Human review and exclusion completeness are
operator attestations; the code can verify their bindings, not their substantive truth.

## Follow-on diagnostic

After an actual successful qualification, change `action` to `diagnostic`, retain zero
optimizer budget, supply the diagnostic bank and add:

```json
{
  "qualification_receipt": {"path": "/absolute/runs/tta-q001/RESULT.json", "sha256": "<sha256>"},
  "historical_endpoints": [{
    "name": "Titan-seed17", "representation": "trained_partial_fp32",
    "path": "/absolute/delta_T_seed17.safetensors",
    "sha256": "07156fb5aa4a8c510c63e8af2118a08a8b6bf004549532d497c2e89e2deec24d",
    "base_sha256": "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b"
  }]
}
```

Despite its filename, this bound artifact is the trained partial tensors; the worker subtracts
the source to construct the delta. Scope keys must match exactly. The BF16 Titan export and
Titan-Sparse weights are NOT substitutes for this original endpoint. A second independent
trained endpoint must bind the same base. The first diagnostic uses full endpoint deltas;
rank-truncated composition is available through `controls.py` as a distinct, recorded variant.

## Follow-on study

Set action `study`, bind fresh A/B train/dev/test rows, include `tasks: ["A", "B"]` and:

```json
{"training": {"steps":28,"batch_size":14,"lr":0.00001,"schedule":"wsd",
 "warmup_fraction":0.2,"decay_fraction":0.2,"end_fraction":0.1,"weight_decay":0.0,
 "sampling":"one_pass","max_processed_tokens":100000,"seed":17}}
```

For 384 rows/task this uses 28 batches, with a smaller final batch, and one exposure. That is
an explicit port recipe, not an assertion that the original trainer used this batching. A
study needs 112 optimizer steps and an independently costed wall/logical-call budget, not the
qualification example's budget. Four standalone delta safetensors and value hashes are saved;
post-hoc reuses the ordinary deltas. These are NOT self-contained released models. The 21-alpha
development sweep and final test use teacher-forced scoring, so costs grow with rows and length.
The optional `control` dev/test rows open the negation comparison, never a control training arm.

## Stops and receipts

The supervisor enforces a wall timeout; the engine counts logical nonlinear/JVP invocations
separately. JVP calls are not counted as one equal-cost ordinary forward. CPU threads=1,
deterministic algorithms, eager attention, no KV cache, no autocast, no CUDA.

After imports the free-RAM gate is at least 12 GiB. After loading, an additional analytical
allowance accounts for the anchor copy, deltas/moments/transients and output headroom; it can
block full-scope studies even on a machine that cleared 12 GiB initially. These are conservative
planning allowances, **not measured peaks**. There is no OOM fallback to quantization or a
smaller unregistered scope. First qualify a modest bound on a suitable host; record actual cost.

`PREFLIGHT_PASS_NOT_EXECUTED` is not native success. `QUALIFICATION_PASS` is backend mechanics
on the bound prefix, including backward compatibility, not training efficacy. `DIAGNOSTIC_COMPLETE`
is a no-optimizer measurement. `STUDY_COMPLETE_DEVELOPMENTAL` is completion, not improvement;
individual methods may still have undefined normalization. A timeout, exception, absent result
or partial output is never success. Failed sessions are not reused. A source/package change
requires a fresh binding; no artifact from a completed historical run is overwritten.
