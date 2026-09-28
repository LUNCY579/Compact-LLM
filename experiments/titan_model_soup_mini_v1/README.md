# Titan model-soup mini experiment — SOUP-MINI-001

**Status: implementation and tiny-fixture mechanics tested; native weights and model behavior NOT tested.**

This is a self-contained, inference-only research preparation slice. It adds no optimizer,
dataset, scoring-policy replacement, release, or experiment grant. Its native collector is
fail-closed: it loads no model unless a later SOUP-MINI-001 authorization receipt and protocol
amendment bind the runtime, decoder, tokenizer, checkpoint, suite and review identities. It
cannot spend GEN2-001, INT-001, or another experiment's permits. Existing research branches
and their records are not modified.

## Decision

Proceed with a small shared-base soup screen. Prioritize **Titan-Sparse + Opallo**, then
**stored Titan + Opallo** as a precision-stratified secondary comparison. Both donors must
first demonstrate the relevant skill on the new calibration data. Opallo is not automatically
a useful donor merely because it was fine-tuned: it has measured retention regressions.

Do not describe Titan + Supa as a newly discovered procedure: INT-001 already tested the
unrounded FP32 Titan delta at 0.25, 0.50 and 0.75. Its 0.50 arm scored acquisition 219/384,
retained 42/47, versus stored Titan's 234/384 and 39/47. This is a trade-off, not lossless
acquisition. Long-context numbers from that screen are report-only, not selection evidence.

Do not describe same-skill seed averaging as untested either: TT-001's seed-averaging
measurements are already discussed in the September 18 attenuation correction. The new
question is the exact Titan/Opallo artifact pairing and conditional local-child pairings.
The audit was targeted, not an exhaustive proof that no other historical soup exists.

## What the five new primary artifacts answer

| Arm | Formula | Role |
|---|---|---|
| `supa_titan_50` | 0.5 Supa + 0.5 stored Titan | Titan-component attenuation control; NOT INT-001's unrounded parent |
| `supa_sparse_50` | 0.5 Supa + 0.5 Titan-Sparse | Sparse-component attenuation control |
| `supa_opallo_50` | 0.5 Supa + 0.5 Opallo | Opallo-component attenuation control |
| `titan_opallo_50` | 0.5 stored Titan + 0.5 Opallo | Secondary artifact-level composition |
| `sparse_opallo_50` | 0.5 Titan-Sparse + 0.5 Opallo | Primary artifact-level composition |

These plus the four immutable endpoints are **nine model configurations, not nine training
runs**. Evaluate one configuration at a time. No quarter-point sweep, TIES, DARE, SLERP,
per-layer search, rank truncation, or three-way soup is included in this mini screen.
Do not call a nonsignificant difference noninferiority, equivalence, or no forgetting.

All output weights and interpolation arithmetic are FP32. Stored Titan and Opallo already
contain BF16 rounding: conversion to FP32 does NOT recover their original trained deltas.
Native endpoints must use the same FP32 evaluation path as the soups. A BF16/FP16/GGUF
serving artifact would be a separately qualified model, not a cheaper copy of this result.

## Local fine-tuned Titan — conditional extension, one checkpoint only

The local machine and its weights were not accessible during this implementation. The
example manifest intentionally contains **no guessed local path or local-checkpoint hash**.
The template includes an unbound `local_ft` entry. Bind it only after its exact parent,
ancestry, full-weight hash, and complete fine-tuning parent ledgers have been independently
reviewed. Set `parent_id` to `titan` or `titan_sparse` and bind the path and SHA-256 of the
completed `local-lineage.example.json` copy in `lineage_receipt`. V1 supports an exact published
Titan or Titan-Sparse parent; an unrounded FP32 parent or different parent needs a dated
registration amendment, not an alias. Raw `trained_fp32.safetensors` files may be partial
absolute trained tensors, despite `delta_*` backup names. Do not treat them as full models,
subtract the base twice, or add them directly as deltas. Raw LoRA A/B matrices must not be
averaged; materialize effective full weights with the existing validated exporter first.

Conditional recipe IDs are `local_parent_50`, `local_supa_50`, and `local_opallo_50`.
For L = T + dL and T = S + dT:

- (L + T)/2 = S + dT + 0.5 dL: retain the inherited Titan update, shrink only the new update.
- (L + S)/2 = S + 0.5 dT + 0.5 dL: shrink both the inherited and the new update.
- (L + O)/2 = S + 0.5 dT + 0.5 dL + 0.5 dO: a different composition question.

Do not add dT again to a child that already contains it. A local-vs-Opallo experiment also
requires the `local_supa_50` and `supa_opallo_50` component controls. The local parent can be
Titan-Sparse; replace T accordingly. No additional training is proposed.

## Local commands

Use a separate environment or the already-qualified research environment; do not upgrade a
live training environment. `requirements-mechanics.txt` records the versions exercised here,
not a qualified native Rineya runtime. No dependencies are installed automatically.

```powershell
cd experiments/titan_model_soup_mini_v1
python -m pytest -q
Copy-Item manifest.example.json manifest.local.json
# Fill absolute input directories in manifest.local.json; do not change published hashes.
python soup.py preflight --manifest manifest.local.json
# Choose a NEW output directory outside every source directory; its parent must exist.
python soup.py merge --manifest manifest.local.json --arm sparse_opallo_50 --output C:/Research/SoupRuns/sparse_opallo_50
```

`preflight` without `--arm` checks the four published endpoints. `preflight --arm ID` checks
Supa and that arm's dependencies, including a local parent when applicable. It hashes weights
and sidecars, checks canonical config and full tensor inventory, and rejects changed token
IDs/templates, custom gated Supa variants, incompatible keys/shapes, quantization configs,
raw adapters, partial checkpoints, and sharded inputs. Equivalent explicit/omitted tied
`lm_head.weight` is supported only after value equality is checked.

A successful merge creates a standard FP32 `model.safetensors`, tokenizer/config sidecars,
and `BUILD.json`. `INCOMPLETE.json` or `FAILED.json` means **do not evaluate**. Sources are
rehashed after writing. Outputs are never overwritten. There is no automatic retry or cleanup.
The code is a bounded-row streaming writer using the documented safetensors format; it does
not hold multiple complete state dictionaries or instantiate a language model.

A 596,049,920-parameter FP32 output contains 2,384,199,680 tensor-data bytes (about 2.22 GiB),
plus headers and sidecars. The builder requires output space plus a 1 GiB margin. Building
all five primary outputs at once would need about 11.1 GiB for tensor data alone; this is
why the CLI builds **one explicitly selected arm** and not an automatic all-arm batch.
Actual Windows native peak memory/wall time remain unmeasured. File-backed mappings and the
OS page cache are not bounded by the numerical chunk size. Keep competing jobs paused only
through their own owner-approved controls; do not kill sessions, reclaim leases, lower guards,
or spend a different experiment's grant.

## Evaluation handoff

`collect_native.py` is the prepared native handoff. It emits raw predictions for one arm and
one split at a time; it does not score correctness, select candidates, or grant execution.
It requires a later authorization receipt with `native_authorized: true`, while the registered
`protocol.json` currently remains false, so the command below is expected to refuse today.
The collector binds exact checkpoint, tokenizer, suite, runtime, decoder, qualification and
ancestry hashes; it enforces FP32, greedy, batch-one, 4,096-total-token and 64-new-token
limits, and refuses to overwrite an existing output.

```powershell
python collect_native.py --arm sparse_opallo_50 --split CAL `
  --model C:/Research/SoupRuns/sparse_opallo_50 `
  --suite C:/Research/SoupRuns/CAL.jsonl `
  --authorization C:/Research/SoupRuns/SOUP-MINI-001-AUTH.json `
  --protocol protocol.json `
  --output C:/Research/SoupRuns/raw/sparse_opallo_50.CAL.json
```

Use the existing separately qualified dense reference collector only after binding its exact
code, model/checkpoint, tokenizer, runtime, decoding and test-bank receipts. `score.py`
consumes externally scored per-row results; it does not independently validate that external
scoring was computed correctly.

```powershell
python score.py audit-splits --cal CAL.jsonl --check CHECK.jsonl --train donor1_parents.jsonl donor2_parents.jsonl
python score.py compare --reference supa_run.json --candidate sparse_opallo_50_run.json
python score.py compare --reference supa_sparse_50_run.json --candidate sparse_opallo_50_run.json
python score.py compare --reference supa_opallo_50_run.json --candidate sparse_opallo_50_run.json
```

Every fine-tuned ancestor's training ledgers are required, not just the newest child's rows.
The split auditor checks the **explicit `parent_id`**, never a string parsed from `id`. It
cannot verify that omitted ancestor ledgers are complete. A positive audit is only supplied-ledger
disjointness, not qualification. Keep historical quarantined/SPENT banks report-only. Do not
relabel a random split of already-inspected rows as a new holdout; see `EVIDENCE.md`.

Run JSON schema (all hashes must be real 64-character SHA-256 strings):

```json
{
  "metadata": {
    "experiment": "SOUP-MINI-001", "suite_id": "YOUR_FROZEN_SUITE", "split": "CAL",
    "dataset_sha256": "...", "scorer_sha256": "...", "runtime_sha256": "...",
    "decoding_sha256": "...", "tokenizer_sha256": "...", "checkpoint_sha256": "..."
  },
  "rows": [
    {"id": "row-001", "parent_id": "parent-001", "domain": "transition", "correct": true}
  ]
}
```

Missing/extra/duplicate rows, non-boolean correctness, metadata mismatch and mismatched
parent/domain identity refuse. Reports separate source-correct losses from new gains in
each domain. Repeated parent views suppress row-level McNemar p-values and add a labelled
parent-all-views-correct descriptive endpoint. P-values are raw and are not evidence of
noninferiority; use parent-aware intervals and preregistered multiplicity handling for any
future formal claim. Every report keeps quality/release claims false.

## Remaining boundary

Required before any native calibration: local files bound; endpoint loader parity; actual
runtime and deterministic decoding locked; whole ancestry contamination audit; fresh admissible
CAL/CHECK banks, verifiers and shortcut floors frozen; a separate SOUP-MINI-001 resource/lease
and native-read grant; and an immutable candidate-selection record before CHECK is opened.
`protocol.json` registers a proposed mini-screen, NOT an executable authorization mechanism.
Missing these items does not justify relaxing another experiment's gates.

The highest-value first action is **compatibility preflight on exact local endpoints**, followed
by one primary `sparse_opallo_50` artifact when compatible. No claim about its intelligence is
possible from these mechanics tests alone.
