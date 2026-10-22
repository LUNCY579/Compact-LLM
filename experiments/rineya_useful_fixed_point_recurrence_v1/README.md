# FP-RINEYA-001 — Useful Fixed-Point Recurrence

**Status: implemented CPU/reference apparatus; native qualification not run.**

Research question: can a compact Rineya model become more correct before it
becomes stationary, and only then make safe use of that stationarity?

Dedicated branch: `research/rineya-useful-fixed-point-recurrence-v1`.
Base: integrated `post-training-Mimo` at
`03db218ceebc61a2bc04f2482cf7e094045e01c1`.
Paper: *Towards Looped Models Done Right — Part II: Rethinking at Fixed Points*.
Sources, exact adaptation differences and audit limits are in SOURCE_AUDIT.md.

## What is implemented

| Component | Implementation and boundary |
|---|---|
| Injection comparison | Matched additive/orthogonal per-token interfaces with low-rank anchor adaptation, shared decay/scaling parameterization and zero-start interpolation. |
| Useful-convergence diagnostics | Separate raw/applied updates, actual operator residual, injection change, suppression, masked state changes, anchor alignment, RMS and logit KL. Never infer correctness from stationarity. |
| Finite-depth training mechanics | Live-anchor truncated gradient reference, full-window comparator, external-interface-only optimizer, split/time/token/gradient checks. Tested on synthetic regression, not native training. |
| Depth prior | Separate CPU REINFORCE/entropy/mean-penalty reference with resume state and >=3 support points. NOT inference stopping or an exact upstream optimizer reproduction. |
| Native bridge | Scoped, restorable, tensor-only 28-layer custom-Supa L12-L13 application; CPU FP32; first pass intact; at most one extra pass; no cache; exact per-layer call contexts. |
| Native Stage 0/1 runner | Local-only checkpoint/source/runtime/receipt/data validation; reuses inherited ARMR scorer and bounded greedy generation. Unwrapped/intact/raw/zero-additive/zero-orthogonal/restored arms. |
| Evidence and analysis | Exclusive run directories, hashes, explicit paired outcomes, correction/damage counts and paired cluster bootstrap; seeds/splits cannot be silently pooled. |

There is **no native training CLI**, no checkpoint download, no GPU dispatch,
no automatic schedule, no terminal-KV implementation and no hidden training run.
`native-screen` deliberately evaluates *zero-start* interfaces: both must reproduce
raw recurrence before trained interfaces are considered.

## Run CPU checks from the repository root

Use a compatible CPU PyTorch/pytest environment; do not replace the canonical
native runtime just to run these tests. See VALIDATION.md for tested versions.

```bash
python -m pytest -q experiments/rineya_useful_fixed_point_recurrence_v1/tests
python -m experiments.rineya_useful_fixed_point_recurrence_v1 fixtures --out /tmp/fp-rineya-new-fixture-run
python -m experiments.rineya_useful_fixed_point_recurrence_v1 preflight --permit experiments/rineya_useful_fixed_point_recurrence_v1/configs/native_screen.example.json
```

The final command **intentionally exits 2 / BLOCKED**. It does not import a model
loader or run inference. Choose a new output directory for every run.

## Prepare a future native screen — do not launch during the laptop pause

The historical host is custom `Qwen3AbstainForCausalLM`, 28 layers, approximately
0.6B total parameters, with canonical `model.safetensors` SHA-256:
`a99f079d32212bbcbb703c2e2bad801ea32d43afea52004f38bf94eec5dc4026`.
This is NOT automatically the current warmed MiMo training anchor.

After separate approval and requalification, copy the example permit outside
canonical model directories and bind:

- `allow_native_screen` and `laptop_pause_cleared` explicitly to true;
  `alpha` to a frozen DEV-selected scale (the example's 1.0 is not a selection result).
- `checkpoint_files`: **every** model-directory file's relative path -> SHA-256,
  including tokenizer/config/custom Python. No symlinks or pickle/bin checkpoints.
- `source_files`: repository-relative paths -> SHA-256 for all Python files in
  `experiments/anchor_routed_middle_recurrence_v1/src` and
  `Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src`.
- `runtime_versions`: exact `python`, `torch`, `transformers` version strings;
  `host_qualification`: `path`, `sha256` of passing `qualification.json`, and
  `manifest_path`, `manifest_sha256` of its passing canonical `run_manifest.json`.
- `dataset`: `split: DEV`, the exact file `sha256`, and ordered `case_ids` for
  at most 16 explicitly approved cases in the inherited ARMR schema.

```bash
python -m experiments.rineya_useful_fixed_point_recurrence_v1 native-screen \
  --permit /path/to/approved-permit.json --repo /path/to/ChatBubble \
  --checkpoint /path/to/exact-canonical-supa \
  --cases /path/to/approved-dev.jsonl --out /path/to/new-fp-native-run
```

Guardrails: pre-load free RAM >=7.4 GiB, runtime free RAM >=6 GiB, process-tree
RSS <=8 GiB, batch 1, <=4096 input tokens, <=24 generated tokens, inherited
90-second per-case checks, one-hour screen budget. Time/RAM checks are cooperative,
not OS-level preemption. The full inherited baseline and new bridge must qualify;
this command cannot substitute a different checkpoint or open deeper-depth gates.

A successful run yields `NATIVE_APPARATUS_PASS_NOT_EFFICACY`. It does not promote
a model or authorize native training. Current published evidence has zero native
loads, zero native optimizer steps and no native quality/memory/speed estimates.

## Analyze paired outcomes

`report --rows outcomes.jsonl --out NEW_DIRECTORY` consumes one arm/seed/split.
Each record requires `case_id`, `cluster_id`, `arm`, `seed`, `split` (DEV or
RETENTION), and actual boolean `baseline_correct` / `candidate_correct`. Use the
frozen task scorer to define correctness; never coerce fractional scores to bool.
The cluster bootstrap is descriptive on DEV, not independent confirmation.

See PROTOCOL.md for the staged experiment and validation gates, and AGENTS.md
for preservation, authorization and branch-scope rules.
