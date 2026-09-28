# Runbook — D28-001

Use the new branch in an isolated full checkout/worktree. Do not change main,
FWT/RAB files, source assets, old receipts, branches, tags or GitHub workflows.
Run commands from repository root. No command downloads or substitutes weights.

## 1. Inspect and scoped random-weight tests

```bash
python -m experiments.dual28_ced_v1 inspect
python -m pytest -q experiments/dual28_ced_v1/tests
python -m experiments.dual28_ced_v1 tiny-smoke \
  --out experiments/dual28_ced_v1/outputs/tiny-001
```

The tests require torch, numpy, safetensors and pytest. They do not import native
Transformers execution. `tiny-smoke` uses a two-layer donor copied into two stacks;
its evidence can never satisfy an official P1 parent. Analytic full-shape meta
construction does not materialize pretrained weights or demonstrate runtime fit.

Each output path must be a *new* child of this package's `outputs/`. Existing paths
and symlink paths are rejected. Do not reuse a failed/interrupted attempt.

## 2. Exact official P1 qualification

The existing approved local installation is `Supa/` at repository root. Every asset
must match `experiments/architecture_benchmark_001/source_lock.json`. Missing weights,
LFS pointers, changed tokenizer files, alternate family cores and source symlinks
are not valid substitutions. No generic hub fallback is supported.

Use torch `2.10.0+cpu`, Transformers `4.51.3`, numpy `2.3.5`, safetensors `0.7.0`.
The command sets four CPU intra-op threads and binds the remaining runtime details.
Execution is FP32/eager; source storage is the locked BF16 safetensors file. A full
checkout descended from the pinned parent and committed, unchanged tracked package
and dependency files are required. Evidence-only commit changes also change the
checkout binding and require a new P1 for that head.

```bash
python -m experiments.dual28_ced_v1 qualify \
  --source Supa \
  --out experiments/dual28_ced_v1/outputs/official-p1-001
```

This executes native B28 parity and new B28/H56/P56/D56/C28/C56 mechanical checks.
It does **not** score capability, open recovery, or train any arm. H/P/C initial
checkpoints are retained under the attempt directory and reloaded sequentially to
avoid holding both original and reloaded variants simultaneously. About 13.2 GB
(decimal) of checkpoint weights plus a 1-GiB reserve are required on disk by the
preflight. This is a disk guard, not a RAM/GPU-fit certificate.

A successful `result.json` must have experiment `D28-001`, stage `P1`, status `PASS`,
all required checks, correct current bindings and no campaign permissions. Failure
or interruption is recorded in `blocked.json`; it is not a substitute PASS. Preserve
its checkpoint fragments and diagnose the first failed gate without changing
thresholds to fit an observed outcome. Historical FWT/RAB receipts are invalid here.

## 3. Separately requested P2 engineering smoke

Only after a fresh official P1 passes on the *same* source, code, runtime and exact
checkout, and the bounded smoke is explicitly requested:

```bash
python -m experiments.dual28_ced_v1 smoke \
  --source Supa \
  --parent experiments/dual28_ced_v1/outputs/official-p1-001/result.json \
  --out experiments/dual28_ced_v1/outputs/official-p2-001 \
  --ack-engineering-only
```

The engine runs H then P sequentially, 25 steps each, on the new paragraph fixtures.
It freezes both original stacks and trains only the shared bank and new reads.
The parent's H/P/C mechanical checkpoint hashes are rechecked before the smoke.
It records data order/hash, losses, all-tensor gradient coverage, frozen-state
checks, post-update mechanics and state digests. It does not export a recovered
model or authorize another stage. A P2 PASS cannot serve as another P1 parent.

## 4. Stop boundary

There are no `train`, `recover`, `evaluate` or `deploy` commands. A pure-CED migration
or efficacy campaign needs the registrations and reviewed implementation described
in RESEARCH_PLAN.md. Do not enlarge the engineering fixtures or step count and call
that recovery. Do not treat the local report as pretrained qualification or change
the pinned CPU recipe into a GPU/quantized recipe without new qualification.

The local delivery verified only the scoped package and nine exact imported files,
not every test in the repository or a full-checkout official run. Real weights and
Transformers were absent; the missing-weight preflight was intentionally preserved.
