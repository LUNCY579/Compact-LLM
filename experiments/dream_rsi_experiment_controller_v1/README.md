# Rineya Dream-RSI experiment controller v1

**RINEYA-DREAM-001: implemented offline research machinery, not a demonstrated
Rineya improvement.** Native execution and DREAM-002 are closed.

Branch: `research/dream-rsi-experiment-controller-v1`.
Base: `b0ca54bdc1b69e2b19d7baeca2e3fdde3ff0a7b2`.
Read [PROTOCOL.md](PROTOCOL.md) before admitting history or changing a metric.
Read [HISTORY_AUDIT.json](HISTORY_AUDIT.json) for the targeted source audit:
**zero real historical worlds are admitted in this commit.**

## What is executable

- Strict world/catalog loading, immutable records, original-receipt field binding,
  checkpoint/data/protocol identities, mandatory leakage keys and disjoint splits.
- Prefix-only branch/continuation replay with atomic worker-bounded batches,
  support misses, opaque handles, deterministic resets and complete decision logs.
- Fixed, greedy, random and adaptive controllers; bounded data-only policy evolution
  on training worlds, followed by one validation selection and hash-bound freeze.
- Gated capability/retention Pareto frontier, exact hypervolume/attempt AUC,
  retained raw cost fields, paper-style diagnostic and paired cluster bootstrap.
- One-shot heldout evaluation, exclusive artifacts, source fingerprints and an
  executable local qualification runner. No arbitrary generated policy code runs.

This is an independently authored **constrained-policy adaptation**, not the
unreleased authors' code, a full LLM policy-development agent, or an online RSI
system. Parameter count, latency and memory are hard gates, not extra Pareto axes.
The fixed arm is an explicit new baseline, not a reconstruction of prior human
research scheduling. Full assumptions and differences are in PROTOCOL.md.

## Run from the ChatBubble repository root

The engine uses Python's standard library. Tests require `pytest`; do not install
the repository's model-training requirements just to run this package.
The checked interpreter and test runtime are recorded in the qualification receipt.
Use an output directory that does not already exist. These commands work without
model weights, CUDA, network credentials or API keys.

```bash
python -m pytest -q experiments/dream_rsi_experiment_controller_v1/tests

python -m experiments.dream_rsi_experiment_controller_v1 demo --out dream_demo_001

python -m experiments.dream_rsi_experiment_controller_v1.qualify --out dream_qualification_001

python -m experiments.dream_rsi_experiment_controller_v1 history-status
```

`demo` creates ten explicitly synthetic worlds (3 train, 2 validation, 5 test),
evolves/finalizes a policy and performs 60 heldout replay episodes: four arms x
five worlds x three fixed seeds. These fixtures include plateaus, delayed gains,
implementation failures and high-capability/low-retention traps. The synthetic
numbers are neither imported Rineya results nor a power analysis.

`qualify` runs the isolated test suite, saves raw stdout and JUnit, compiles the
package, runs the synthetic pipeline, and binds source/artifact hashes in an
exclusive receipt. It does not run the full ChatBubble suite or remote CI.

The run directory includes:

```text
catalog_commitment.json   # metadata only; no hidden outcomes
plan.json                 # complete objective, constraints, seeds and search budget
search.json               # all candidate specs, train/validation feedback, ancestry
freeze.json               # selected policy and integrity envelope
holdout_opened.json       # written before test loading; consumed even on failure
episodes.json             # every arm/world/seed decision, reveal, curve and cost
report.json               # aggregates and primary paired cluster comparison
```

Cost values are `null` when missing or when replay requests unsupported actions.
Do not replace those with zero, infer GPU savings from probe counts, or tune the
controller after inspecting heldout outcomes. Source/protocol changes invalidate
an old freeze; a new revision needs an unexposed test set.

## Admit actual history deliberately

The executable schemas are `parse_world`, `Outcome.parse`, `identity_keys`, and
`validate_catalog` in `schema.py`; `fixtures.make_world` is a complete, explicitly
synthetic example of the record structure. The provenance-binding tests contain
fabricated receipt examples ONLY for mechanical testing. Do not relabel these
fixtures as historical data.

An actual world must include pinned checkpoint, protocol and dataset hashes;
study/task/lineage identities; a fully measured baseline; ordered attempted nodes;
complete proposal-context references; separate execution/qualification statuses;
measured quality/cost/resource fields; and accountable historical admission.
Historical admission requires original JSON receipt bytes and one binding for
every identity, baseline and node field. A binding is `{target, source, pointer}`:
`target` is a JSON pointer in the world, `source` is the source-array index, and
`pointer` selects the exact original value. Sources include `path`, `sha256`,
`git_blob`, `git_commit`, and `repository_path`. Units/normalizers cannot be guessed.

The catalog has `schema="rineya-dream-catalog-v1"`, `kind`, and `worlds`. Each
entry has `id`, relative `path`, canonical `sha256`, `split` and `leakage_keys`.
Use the module's `digest(raw_world)` for canonical world commitments; source
receipt SHA256/Git blob hashes instead bind the original bytes. Both are checked.

```bash
python -m experiments.dream_rsi_experiment_controller_v1 check-world --world history/world_001.json --source-root history

python -m experiments.dream_rsi_experiment_controller_v1 fit --catalog history/catalog.json --plan historical_plan.json --out historical_run_001

python -m experiments.dream_rsi_experiment_controller_v1 evaluate --catalog history/catalog.json --run historical_run_001
```

These historical paths are instructions, not files or completed results supplied
by this branch. `DEFAULT_PLAN.json` contains the synthetic example configuration;
freeze justified historical thresholds before selection. The historical path
will remain blocked until enough genuinely disjoint, source-admitted worlds exist.
The loader detects undeclared required identity keys, not undisclosed real-world
relationships; curator review remains essential.

## Evidence in this branch

`evidence/QUALIFICATION_RECEIPT.json` identifies the exact local source and raw
artifacts tested. Captured pytest output, JUnit, the synthetic report/freeze and
an explicitly labeled sample trace are included. Full synthetic episodes and
worlds are reproducibly generated by `qualify`; their hashes are recorded.
No source report or inherited experiment is edited by this package.

The first synthetic qualification did **not** establish adaptive-over-fixed
superiority. The frozen default pipeline is kept rather than tuned to that
exposed fixture comparison. Regardless of its sign, a fixture comparison cannot
establish native quality, GPU savings or historical research efficiency.

## Next gate

Reconstruct and independently review original closed histories, including failed
attempts and proposal dependencies. Preserve corrections and inconclusive states.
Then freeze a disjoint historical catalog and run the registered offline study.
A positive historical result would justify designing, not automatically launching,
a separately authorized live experiment. `native` intentionally exits with a
BLOCKED status; no live dispatch capability is present.
