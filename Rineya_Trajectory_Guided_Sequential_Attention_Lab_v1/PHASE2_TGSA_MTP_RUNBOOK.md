# TGSA v5.1 Staged Runbook

This is the operator path for `research/tgsa-v5-1-unified-programme`. It replaces
older wording that treated a v4 competence `GO` as possible. V4 is frozen as a
rejected regime boundary; all new apparatus work begins with v5.1.

## 0. Non-negotiable operating rules

- Keep the checkout clean before every v5.1 wrapper invocation.
- Place evidence outside the repository. Untracked evidence inside the checkout
  intentionally fails the clean-worktree gate.
- Never edit `configs/phase1_v5_plan.json` in place. Its exact Git blob is part
  of the prospective registration.
- A scientific `NO_GO` exits zero after a complete valid run. Read the verified
  verdict; do not interpret the process exit code as scientific success.
- Do not open a sealed mechanism, Rineya, or wall-clock bank during apparatus
  qualification.
- Phase 2A and Phase 2B use different MTP semantics and evidence roots.

## 1. Checkout and install

```powershell
git fetch origin --prune --tags
git switch research/tgsa-v5-1-unified-programme
git pull --ff-only origin research/tgsa-v5-1-unified-programme

git status --short
cd Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

`git status --short` must print nothing.

## 2. Verify the frozen programme before any run

```powershell
python scripts/verify_v5_1_programme_contract.py `
  --contract configs/v5_1_programme_contract.json `
  --repo-root .. `
  --output "$env:TEMP\tgsa_v5_1_programme_verification.json"
```

Required result: `passed: true`. This verifies the exact source-plan blob, clean
checkout, complete pinned lineage, and the distinct Phase 2A/Phase 2B semantic
identities. It does not evaluate model quality.

## 3. Optional development smoke

The smoke plan is development-only and can certify only software/evidence
closure.

```powershell
$SmokeRoot = Join-Path $env:TEMP "tgsa_v5_1_smoke"
Remove-Item -Recurse -Force $SmokeRoot -ErrorAction SilentlyContinue

python scripts/phase1_v5_1_qualify.py `
  --contract configs/v5_1_programme_contract.json `
  --plan configs/phase1_v5_smoke.json `
  --stage probe `
  --rung S1 `
  --output-dir "$SmokeRoot\probe" `
  --repo-root ..

python scripts/verify_phase1_v5.py `
  --plan configs/phase1_v5_smoke.json `
  --report "$SmokeRoot\probe\probe_report.json"

python scripts/phase1_v5_1_qualify.py `
  --contract configs/v5_1_programme_contract.json `
  --plan configs/phase1_v5_smoke.json `
  --stage qualification `
  --rung S1 `
  --exposure SE1 `
  --output-dir "$SmokeRoot\qualification" `
  --repo-root ..

python scripts/verify_phase1_v5.py `
  --plan configs/phase1_v5_smoke.json `
  --report "$SmokeRoot\qualification\qualification_report.json"
```

Expected scope: development-only reports, no mechanism or Rineya authorization.

## 4. Prospective operator attestation

The nondevelopment wrapper requires:

```text
--attest-no-prior-outcome-inspection
```

Use the flag only when this statement is true:

> No nondevelopment outcome produced under the locked v5 plan was inspected
> before the prospective v5.1 lock.

If it is not true, stop. Create a newly reviewed amendment with fresh decisions
and evidence roots rather than retroactively calling the existing plan
preregistered.

## 5. Capacity prerequisite

Choose an evidence root outside the checkout:

```powershell
$Root = "D:\TGSA_Evidence\v5_1"
$Plan = "configs/phase1_v5_plan.json"
$Contract = "configs/v5_1_programme_contract.json"
```

Run `C1` first:

```powershell
python scripts/phase1_v5_1_qualify.py `
  --contract $Contract `
  --plan $Plan `
  --stage probe `
  --rung C1 `
  --output-dir "$Root\probe_C1" `
  --repo-root .. `
  --attest-no-prior-outcome-inspection

python scripts/verify_phase1_v5.py `
  --plan $Plan `
  --report "$Root\probe_C1\probe_report.json"
```

Read `probe_report.json` and `verification.json` together.

- If `C1` is verified `GO`, do not run `C2` or `C3`.
- If `C1` is verified `NO_GO`, run `C2` with otherwise identical commands.
- Run `C3` only after verified `NO_GO` at both `C1` and `C2`.
- If `C3` is also `NO_GO`, stop the registered ladder. Do not add `C4`, more
  steps, another seed, or a new family under this registration.

Freeze the first passing rung using every probe report up to that rung, in
registered order. Example for a `C2` pass:

```powershell
python scripts/select_phase1_v5_1.py `
  --contract $Contract `
  --repo-root .. `
  capacity `
  --plan $Plan `
  --report "$Root\probe_C1\probe_report.json" `
  --report "$Root\probe_C2\probe_report.json" `
  --output "$Root\frozen_apparatus_band.json"
```

The frozen band cannot authorize a mechanism test by itself.

## 6. Complete exposure qualification

After a passing capacity rung is frozen, run **all** four exposures for that
same rung. The example below assumes `C2`.

```powershell
foreach ($Exposure in @("E1", "E2", "E3", "E4")) {
  $Out = "$Root\qualification_C2_$Exposure"

  python scripts/phase1_v5_1_qualify.py `
    --contract $Contract `
    --plan $Plan `
    --stage qualification `
    --rung C2 `
    --exposure $Exposure `
    --frozen-band "$Root\frozen_apparatus_band.json" `
    --output-dir $Out `
    --repo-root .. `
    --attest-no-prior-outcome-inspection

  python scripts/verify_phase1_v5.py `
    --plan $Plan `
    --report "$Out\qualification_report.json"
}
```

Do not select the highest score. Apply the registered resource-order rule only
after the complete verified grid exists:

```powershell
python scripts/select_phase1_v5_1.py `
  --contract $Contract `
  --repo-root .. `
  exposure `
  --plan $Plan `
  --frozen-band "$Root\frozen_apparatus_band.json" `
  --report "$Root\qualification_C2_E1\qualification_report.json" `
  --report "$Root\qualification_C2_E2\qualification_report.json" `
  --report "$Root\qualification_C2_E3\qualification_report.json" `
  --report "$Root\qualification_C2_E4\qualification_report.json" `
  --output "$Root\selected_apparatus_exposure.json"
```

A selected apparatus exposure still supports no TGSA mechanism claim.

## 7. MTP-free mechanism stage

A verified apparatus `GO` licenses preparation—not execution—of a separate
MTP-free mechanism preregistration. Freeze before dispatch:

- selected capacity and exposure;
- `cascade`, `history_control`, `learned_delta`, `replay_control`, and
  `deranged_control`;
- the primary, serial-depth, and alignment estimands;
- new seeds and a sealed bank;
- paired initialization and matched training dose;
- stop rules and multiplicity handling.

No MTP loss may shape this stage. A mechanism `NO_GO` is carried forward
honestly; it does not prevent the independent system-utility lane, but it blocks
mechanism interpretation downstream.

## 8. Phase 2A trajectory-MTP bridge

Use a separate branch/evidence root and semantic identity:

```text
study_kind = mechanism_bridge
mtp_semantics = sequential_teacher_forced_recursive_v1
```

Required conditions are `dormant`, `backbone`, and `trajectory` with identical
MTP capacity. Run gradient modes in the frozen order:

1. `frozen_probe` — primary;
2. `stop_gradient` — secondary;
3. `joint` — tertiary co-training utility.

Only one registered TGSA block may expose `control_only` telemetry. The cascade
control may not receive a trajectory tensor through a side channel. Phase 2A
cannot support speculative-decoding or wall-clock claims.

## 9. Phase 2B system factorial

Use the independently pinned Phase 2B implementation and a different evidence
root:

```text
study_kind = system_factorial
mtp_semantics = parallel_independent_offsets_v1
```

The mandatory paired block is:

```text
parallel_ar
parallel_mtp
tgsa_ar
tgsa_mtp
```

Do not omit a cell. All cells retain the same MTP structure; AR cells keep it
dormant and frozen. Preserve:

- within-family exact state;
- cross-family compatible-tensor identity;
- real-path main-logit drift tolerance;
- exact token IDs and termination reason;
- accepted-prefix order and first-mismatch rollback;
- registered quality non-inferiority;
- registered minimum TGSA serial-round reduction;
- closed artifact verification.

A Phase 2B `GO` is a system result even when the mechanism lane is `NO_GO`.
It cannot be described as proof that trajectory guidance caused the effect.

## 10. First real Rineya checkpoint

The first real run must use a full source-tree hash and the same four Phase 2B
cells. Start with a small distributed converted-layer set and adapter-only
training. Advance only after the previous stage preserves the registered
baseline:

```text
adapters -> attention -> converted_layers -> all
```

A combined `GO` requires initialization drift, exact decode, closed evidence,
within-scope competence, quality non-inferiority, and serial-round reduction.
It supports only a checkpoint-scoped conclusion.

## 11. Wall-clock benchmark

Do not time opportunistically and promote the result. Freeze a separate hardware
protocol after checkpoint qualification. Include warm-up, compilation,
synchronization, cache behavior, batch/context/generation shape, TTFT, TPOT,
decode throughput, memory, utilization, repeated trials, and uncertainty.

A serial-round reduction with slower wall-clock is not a speed success. A
wall-clock improvement with quality outside the registered margin is not a
combined success.

## 12. Minimum evidence to preserve

For every stage retain:

- exact config and programme-contract hashes;
- Git commit and clean-tree state;
- source checkpoint/data/tokenizer hashes where applicable;
- environment and device identity;
- paired-initialization audits;
- fixed final checkpoints and periodic resume state;
- raw per-example predictions;
- exact-decoding traces where applicable;
- competence, analysis, gate-input, gate, and verification reports;
- the stage-specific claim ceiling.
