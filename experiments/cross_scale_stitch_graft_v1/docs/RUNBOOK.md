# Offline runbook

## Environment

Run from the repository root with Python 3.10 or newer and the scoped mechanics dependencies. The delivered verification records the exact actual versions. There is no global repository dependency edit or automatic environment installation.

`requirements-mechanics.txt` pins the tested Python libraries; select the CPU Torch distribution in your environment. The optional `native_qwen.py` candidate requires Transformers 4.51.3 but that package is deliberately not included as a default dependency or silently installed.

## Commands

```bash
python -m experiments.cross_scale_stitch_graft_v1 plan
python -m experiments.cross_scale_stitch_graft_v1 verify-source
python -m experiments.cross_scale_stitch_graft_v1 qualify --output ../xsg001-check-001
python -m experiments.cross_scale_stitch_graft_v1 demo --output ../xsg001-demo-001
```

Each output path must be new, its parent must exist, and it must be outside the experiment source directory. This prevents accidental overwrite of source or old evidence. A partial output directory after a crash is not resumed or silently overwritten; retain it and choose another name.

`qualify` verifies the source inventory, launches a separate single-thread CPU test process, enforces a 120-second timeout, checks the exact expected test IDs/count, and rejects all failures/errors/skips. It saves full worker output and a summary. `PASS_MECHANICS_ONLY` does not open any native gate.

`demo` creates two random causal models, captures matching token grids, fits two stitch maps and separate graft-site maps, exercises both stitch directions and a nonzero FFN graft, checks exact bypass, runs four interface-only fixture updates and verifies unchanged source tensors. Its fictional outcome panel exists only to exercise scoring.

## Fit existing approved captures

```bash
python -m experiments.cross_scale_stitch_graft_v1 fit --request ../xsg001-demo-001/fit_request.json --output ../xsg001-refit-001
```

The request has exactly four keys: `source`, `target`, `check_source`, `check_target`. Each reference contains `directory` relative to the request file and `manifest_sha256`. The demo writes a portable example with real fixture hashes.

Each capture directory contains `manifest.json` and `values.safetensors`. Its contract includes:

- Full source identity, explicit raw capture site, role and registration digest.
- One row key per activation: example, independent group, canonical prompt digest, complete input-ID-sequence digest, physical token position and token ID.
- A detached finite CPU floating-point matrix with the exact registered width.

Fitting requires pairwise row equality and FIT/MAP_CHECK isolation by group, example and prompt digest. CHECK and RETENTION cannot be passed as fitting roles. Hashes detect changes to the declared files; they do not establish that an untrusted producer gave honest group labels or chose genuinely open data.

The bridge artifact stores FP64 weights in PyTorch `[out,in]` layout, its intercept, source/target identities/sites, registration, fitting policy and metrics. The report binds the values and contracts of all four capture banks. `load_bridge()` additionally requires expected source and target identities, sites and registration. The `Bridge.from_fit(..., delta=True)` return-map constructor deliberately drops the intercept.

The bounded CLI checks total materialized payload size and available host/cgroup RAM before loading all four banks. It is not an 8B streaming capture runner. Large maps that exceed its envelope remain blocked rather than silently switching to a different rank or precision.

## Score already-collected outcomes

```bash
python -m experiments.cross_scale_stitch_graft_v1 score --request /path/to/paired-panels.json --output ../xsg001-score-001
```

The request has `receiver` and `candidate` objects matching the `Panel` dataclass. Each contains a full model identity, registration/scorer/decoding digests, an open evaluation role, complete expected example IDs and `Outcome` records. Outcomes include actual prompt and input-ID digests, group/family, task correctness, format correctness, UNKNOWN expectation, literal-UNKNOWN correctness and status.

Outcome booleans must be JSON `true`/`false`, not strings or integer proxies. The registered external task scorer, not this aggregation module, must establish those values. The aggregator does not perform model generation or treat free-form references as a universal semantic scorer. Missing, duplicate, error and unmatched rows fail the complete panel. Precision, input IDs, template and decoding changes require a separately declared contrast.

The result reports group-resampled paired intervals, per-family deltas and explicit behavior regressions. A row-wise exact test is labeled descriptive because independence, arm selection and multiple comparisons are not automatically resolved. Neither the scoring output nor its p-value can promote a candidate.

## Using composition helpers

Supply already frozen/eval source modules to `Decoder`. Fit appropriate bridges using typed captures and reload the artifacts against expected identities/sites. Construct `StitchedDecoder` or `GraftedDecoder` without changing the source module lists. For an FFN graft, supply the explicit donor identity and block index as well as the donor norm/FFN. Source weights remain shared read-only by contract; external aliases must not mutate them.

`OrdinaryAdapter` is a no-donor control with the same fixed-gain bridge parameter count. `interface_parameters()` checks the precise trainable set before constructing an optimizer. It does not authorize native training. `masked_next_token_loss()` uses target-position masks and refuses an unpredicted position-zero target or an empty supervised sequence.

`composition_descriptor()` is the candidate identifier and must be recomputed after interface changes. The output-policy source `.identity` is intentionally not a complete candidate identity. Attach the descriptor and all fitted artifact hashes to future prediction manifests.

## Reading failure states

A source hash mismatch, bad input contract, unsupported cache, custom host class or insufficient memory is an engineering refusal. An incomplete run is not a negative model result. A low reconstruction error is not a task gain. A zero/bypass identity pass is not useful donor transfer. A nonzero fixture change proves only that the path is active.

No command here runs an old sealed evaluator, starts a native optimizer, downloads donors or dispatches CI. Full-model recovery requires a later distinct design and authorization, not a command-line override to this package.
