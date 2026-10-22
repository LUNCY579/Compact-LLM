# RGI-001 — Rineya Relative Generalization Diagnostics

**An isolated implementation of the October 9, 2026 RGI research proposal.**
Implementation repository: `LUNCY579/ChatBubble`. Parent: `post-training-Mimo` at
`03db218ceebc61a2bc04f2482cf7e094045e01c1`. The world-model and retention repositories
are read-only evidence sources, not deployment destinations.

## What this experiment answers

Does a change to the same evaluation problem primarily shift mean prediction loss,
selectively repair particular rule cases, or interfere with an existing capability?
RGI-001 separates those measurements. It does **not** treat agreement of loss profiles,
weight orthogonality, or a passed software test as proof of a skill or no forgetting.

**Delivery status:** runnable analysis, explicit token-evidence importer, exact-dose control
selection, three-seed plan generation, unmerged LoRA backend, confirmation bookkeeping,
and tests. See `VALIDATION.md` for what actually ran. Native Qwen3/Rineya loader, PEFT
integration, CUDA execution and scientific efficacy remain unqualified here.

## Quick start (no model or GPU)

Run these commands from this directory, in an environment with `requirements.txt` installed:

```sh
python -m pytest -q
python -m rgi demo --out runs/demo-001
python -m rgi compare --reference runs/demo-001/base.json --candidate runs/demo-001/scaled.json --out runs/scale-diagnostic.json
python -m rgi interaction --base runs/demo-001/base.json --a runs/demo-001/A.json --b runs/demo-001/B.json --ab runs/demo-001/AB.json --out runs/composition-diagnostic.json
```

Every output is new-only. Choose a new run directory instead of overwriting evidence.
`demo` emits constructed losses, not sampled LLM results. Its canaries include a uniform
shift, perfect correlation with non-unit slope, selective changes and injected interference.

## Implementation map

| Module | Implemented responsibilities |
|---|---|
| `rgi/io.py` | Canonical SHA-256 receipts; atomic new-only JSON; semantic split checks; exact token/context alignment |
| `rgi/data.py` | Fast-tokenizer boundary checks; content/format/mixed masks; executable miniature test domains; exact-dose matched controls |
| `rgi/importer.py` | Import existing normalized token losses with source checksums, complete coverage and explicit producer provenance |
| `rgi/metrics.py` | Unit-slope offset fit; centered CCC; single-run discrepancy; parent bootstrap; global-offset slices; composition interaction; independent behavioral gates |
| `rgi/native.py` | Local FP32 dense-Qwen3 loader contract; answer-position projection chunks; frozen-host LoRA-only optimization; exact budget plans; three-seed control matrix |
| `rgi/protocol.py` | Frozen input/spec checksums; separate one-attempt collection and one-read analysis ledgers; behavioral gates and report rendering |
| `rgi/__main__.py` | All command-line entry points; no implicit native execution |

## What is measured

For aligned observed-target losses `x` (reference) and `y` (candidate), weights give equal
mass to each family, then each parent, then each observed token inside that parent:

```
c = weighted_mean(y - x)
r = y - x - c
residual_rms = sqrt(weighted_mean(r**2))
centered_ccc = 2*cov(x,y)/(var(x)+var(y))
interaction = loss_AB - loss_A - loss_B + loss_base
```

The implementation also reports Pearson correlation and a freely fitted slope **only as
diagnostics**. Perfect correlation is not the offset-only hypothesis. For one fixed run,
`0.5*var(r)*(1/var(x)+1/var(y))` is the weighted counterpart of the paper's normalized
pair-discrepancy expression; it is not the paper's expectation over independent training runs.
Degenerate variance yields `null`, not a fabricated perfect correlation.

Report both token and whole-answer profiles. Content, format and mixed tokens have separate
profiles. Slice residuals use the **global** offset; they are not individually recentered.
Negative residual means better than the average change, not necessarily better than the base.
RGI departure may reflect learning or damage. Neither large nor small departure is a promotion gate.

## Real-data contract

Raw banks and encoded banks are sealed JSON objects. Each example must identify its parent,
semantic lineage, payload, split, family, operation, rule case, horizon, suite and output frame.
Raw examples include the exact prompt, target, semantic payload and target-relative character
spans defining content. `encode` tokenizes the full prompt plus answer once, refuses any token
that crosses the answer boundary, and never silently truncates. A token spanning content and
syntax is marked `mixed`, not arbitrarily reassigned.

`native.collect` scores teacher-forced targets using the same reference contexts across models.
Self-rollout errors and plan quality belong to the **separate behavioral-outcome schema** in
`NATIVE_RUNBOOK.md`. Do not align different models' self-generated contexts as RGI observations.

To reuse cached evidence, normalize the existing producer's records to the explicit importer
schema. Per-row accuracy, first-token-only logs or greedy text do not imply full token NLL.
The importer rejects missing tokens rather than inventing losses. Existing confirmation banks
that have been read are discovery context, not fresh confirmation for this program.

## Matched-control experiment

`match-controls` reads **training data only**. It selects targeted rule cases and controls
from buckets matching family, frame, horizon, processed tokens and supervised tokens exactly
for every pair. Controls are random **conditional on those matching buckets**, not unbiased
samples of the whole corpus. The receipt records the surviving coverage contrast. Infeasible
matching fails instead of padding targets or manufacturing examples.

```sh
python -m rgi match-controls --bank runs/train.json --pairs 64 --seed 17 --out runs/controls
python -m rgi control-plan-matrix --targeted runs/controls/targeted.json --control runs/controls/control.json --spec runs/native.json --seeds 17 23 29 --out runs/plans
```

This creates six explicit specs and plans, but starts **zero training jobs**. Within each
seed, example-index order, updates, processed-token budget and supervised-token budget match.
The default pilot recipe presents 64 examples over 16 updates with accumulation 4. These are
new proposed pilot settings, not a recreation of RCWM's historical 1,501-row adapter.

## Native execution and limits

See `NATIVE_RUNBOOK.md`. Native execution requires installed optional dependencies, actual local
weights/tokenizer, matching hashes, resource gates and an explicit `--execute`. Model loading
refuses hub IDs, remote code, quantization and state-dict mismatches. Base parameters are frozen;
training permits only LoRA parameters. Adapters are loaded unmerged; weighted combinations use
low-rank concatenation, never a dense merge of Titan-Sparse. No published core is modified.

A bounded AdamW LoRA runner is included. A Muon/LoRA optimizer substitution is deliberately
not assumed valid. Evidence from other separately qualified optimizers can use the same importer
and diagnostics; that is an additional study, not a result of this implementation.

The miniature ledger, prerequisite and register generators are independently authored **software
fixtures**, not replacements for RCWM, RegisterWorld or PlanBench. This package does not recreate
those researchers' label budgets, frame conventions, data distributions or published numbers.
It also does not implement a new planner: existing preregistered RCWM/BWM runners supply rollout
and planning outcomes through the explicit bridge schema.

## Research governance

`configs/confirmation.example.json` is a draft and refuses to freeze until explicitly approved.
Freeze training/development/confirmation banks, native model specs, scoring settings, bounded
claim wording and behavioral endpoints before looking at new confirmation predictions.
Collection and analysis have separate, consumed-on-attempt ledgers. A failure is preserved;
never delete a ledger to disguise an additional read. Filesystem bookkeeping cannot prove that
nobody inspected a copied bank elsewhere: this is an auditable workflow, not a security boundary.

Behavioral gates use parent-cluster bootstrap intervals with Bonferroni correction. Degenerate
constant parent differences cannot yield a zero-width-bootstrap certificate; they require a
separately preregistered exact/bounded test. Missing behavior, inadequate parent counts, wrong
provenance, changed inputs or synthetic evidence cannot establish a native research result.
Even `CONFIRMED_FIXED_ARTIFACT` does not set product-quality or release flags.

Further rationale and the staged experiment plan: `PROTOCOL.md`. Primary references and
relationship to existing Rineya records: `SOURCES.md`.
