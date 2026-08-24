# Deferred GPU confirmation

Status: `NOT_RUN`. The target is the published Qwen3-0.6B Halo core. GPU work is
blocked while the Halo confirmatory protocol remains `DRAFT_NOT_REGISTERED`.
Rineya/R2 and Aegis are not active candidates.

CPU compatibility smokes passed on five published family cores, but each was a
single 192-token no-decode gate. They do not authorize a family GPU sweep;
Halo remains the primary GPU target.

CPU development has advanced beyond the three-case mini evaluation, but it
still does not clear a broad confirmatory GPU launch. Endpoint v1 has one
narrow, locally registered prospective validation: after a first B0 stop at
9/24 and difficulty-calibrated B0 18/24, B3-B2 retention was `+0.0401786`
(`p=0.01953125`) on 12 clusters, while answers were B3 5/12 vs B2 6/12
(`p=1.0`). This is not a public preregistration or broad protocol.
The 88-cell count study found age-derived/head-averaged exact equivalence to
true C only under one fixed schedule. The 110-cell schedule study found a
positive frozen-endpoint interaction (`+0.00936074`/round, `p=0.00390625`) but
no detected answer interaction (`p=0.4375`). Both reuse 11 B0-selected
development clusters.

The open-loop adaptive-budget screen also does not advance: shuffled-statistic
allocation exceeded entropy-proportional allocation, so the specificity gate
failed (`NO_SIGNIFICANCE_SPECIFIC_ADVANTAGE`). It has no answers, closed-loop
state propagation, measured sparse-storage overhead, or GPU candidate status.

Its closed-loop padded successor also fails promotion: entropy answers were
8/12 versus uniform 5/12 (`p=0.25`), shuffled retained more, and physical
padding exceeded logical payload. Experiment C had all five attention-prior
arms at 5/12 and normalized retention equal to raw/shuffled. Experiment D had
all six memory arms at 5/12 and normalized retention below M0/random.
Experiment E's risk AUC was 0.50. Experiment F's logistic controller was
perfect on only three held-out clusters/two positive-negative pairs and remains
`UNDERPOWERED_OVERFIT_SCREEN_ONLY` with high overfit risk. None is a GPU candidate.

The controlled R1-R8 ladder is construction-only. In its bounded R5-R8
full-cache screen, R7 v1 is invalid/uninformative; corrected R7 v2 is balanced
but Halo answered all YES and stopped at 4/8 with 0/4 NO. R8 scored 0/2 and
stops before compressed or GPU conditions. Earlier R1-R4 calibration
likewise had benchmark-family R2 at 0/8. These are benchmark labels, not
Rineya/R2 model results.

## Launch gates

All must be linked from a frozen candidate manifest:

- a materially larger, diverse endpoint replication and cluster-aware broad
  analysis are registered beyond the 12-cluster local validation;
- invariant, leakage, identity, rollback, zero-bias, zero-slot, and byte checks
  pass on Halo;
- model, tokenizer, config, benchmark, and resolved-config hashes are frozen;
- the CPU screen is complete under the broad registered method, including
  prospective resolution of incapable R2/R7/R8 task-family slices;
- no unresolved implementation error affects the candidate;
- candidate order is frozen before GPU outcomes;
- the GPU has enough memory for the declared retained-history, active-chunk,
  transient-query, and full-cache paths.

Runtime status: local recorder implementing audited semantics; 4/4 audited
primitives match exactly and end-to-end source-runtime parity is `NOT_RUN`.
Source parity is not silently implied by a
successful GPU run.

## Candidate order

1. `B0` full cache where the context fits;
2. `B2` raw accumulation;
3. `B3` normalized accumulation;
4. only one pre-frozen exploratory architecture that survives the CPU screen.

The current CPU evidence prioritizes B0/B2/B3 only. It does not justify adding
adaptive allocation, an attention prior, memory slot, controller, or broad
family sweep. B3's local retention contrast makes it the only compressed
candidate with positive prospective retained-state evidence, but the absent
answer gain and runtime-version gap prevent promotion. The local validation ran
Torch 2.13.0/Transformers 5.15.0; the current lock and B-D screens use
2.12.1/5.14.1. Reproduce the endpoint under the frozen candidate runtime before
GPU work. The exact
age-derived equivalence is a reason to include an irregular-exposure schedule
on CPU before GPU, not a reason to substitute age for C without another gate.
Eligible-pool shuffled/circular v2 now provide clean process-local count
controls, but their nulls and separate-process reproducibility caveat do not
justify a GPU branch.

Use successive halving: one case/context first, then the shortest registered
matrix, then longer contexts for surviving variants. Do not launch a broad
architecture or family sweep.

## Source-checkout command template

This CLI depends on checked-out `scripts/` and `schemas/`; it is not a
standalone-wheel interface. From the lab directory after every gate is true:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
$modelPath = (Resolve-Path `
  "..\Rineya_Published_Cores\Rineya-Halo-1.2.0-devfreeze" `
).Path
$env:CUDA_VISIBLE_DEVICES = "0"

& ..\.venv\Scripts\python.exe -m rineya_significance_lab.cli run `
  --config configs\deferred_gpu.yaml `
  --model-path $modelPath
```

The generic runner is not yet implemented for this config, so the template
correctly remains non-launchable until implementation and gates are complete.
Save the resolved config, not just this template.

## Provisional matrix

The following values are planning inputs, not frozen confirmatory settings:

- contexts: `8192`, `16384`, `32768` tokens;
- evidence depths: `0.10`, `0.50`, `0.90`;
- seeds: `101`, `211`, `307`, `401`, grouped by latent-case cluster;
- greedy decode, batch size one, bf16;
- 2,048 retained-history entries, 1,024 active-chunk tokens, and at most 64
  transient query tokens;
- both KV-only and total-persistent-state matched across compressed controls.

Halo's native maximum is 40,960 positions. Any RoPE/YaRN modification is a
different architecture condition and requires its own capability reference.

## Measurements and stop rules

Report exact and capability-conditioned answer accuracy only after the exact
evidence metric is frozen. Also report evidence outcomes, cluster-aware
uncertainty, prefill/decode/scoring/gather time with CUDA synchronization, peak
allocated/reserved VRAM, process RSS, CPU offload bytes, and trace overhead.

Stop a candidate on invariant failure, hash mismatch, unmatched storage, clear
registered-quality regression, or overhead inconsistent with its purpose.
Preserve negative results. A promising CPU result remains
`NEEDS_GPU_CONFIRMATION` until a registered GPU matrix completes.
