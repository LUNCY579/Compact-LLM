# MRS-1 — current-candidate results

**Superseded by an executed run. This file previously reported a preload resource refusal;
that is no longer the blocking event.** The earlier refusal receipts remain in
`evidence/native-supa-001/` and `evidence/native-d1-001/` and are not rewritten.

| Claim | Status |
|---|---|
| SOFTWARE PASS | **TRUE** — 96 scout tests + 19 supervisor tests |
| NATIVE EXECUTION PASS | **PARTIAL** — qualification passed; generation gated |
| OPEN DEV RESULT | **FALSE** — cells A and B NOT PRODUCED |
| MECHANISM SCOUT RESULT | **FALSE** |
| CONFIRMATORY CLAIM | **FALSE** |
| RELEASE CLAIM | **FALSE** |

## What actually happened

On 2026-09-17 host memory recovered to 8.29 GiB, clearing the 6 GiB preload guard that had
refused both earlier attempts. Those refusals spent no model launch ("a blocked preload
spends no model launch"), so the new runs are a first execution, not a retry.

| Attempt | Outcome | Evidence |
|---|---|---|
| `native-supa-001` | preload refusal, 2.099 GiB < 6 GiB, torch never imported | `evidence/native-supa-001/failure.json` |
| `native-d1-001` | preload refusal, 2.265 GiB < 6 GiB, torch never imported | `evidence/native-d1-001/failure.json` |
| `native-supa-002` | qualification PASS → **SCHEMA_SMOKE FAIL** | `evidence/native-supa-002/` |
| `native-d1-002` | qualification PASS → **SCHEMA_SMOKE FAIL** | `evidence/native-d1-002/` |

Both `-002` runs loaded their checkpoints and generated. `NATIVE_QUALIFICATION` passed on
both: strict loading with zero missing, unexpected or mismatched keys; cache-versus-full
agreement at max abs 2.10e-05 (D1) and 4.20e-05 (Supa); exact causality; identical
deterministic repeat; tied embedding storage intact; correct 596,049,920-parameter inventory;
state digest unchanged.

| Run | Model loads | Generation calls | Reserved tokens | Peak GPU | Peak host RSS | Worker wall |
|---|---|---|---|---|---|---|
| `native-d1-002` | 1 | 90 | 7,712 | 2.90 GiB | 2.45 GiB | 52.84 s |
| `native-supa-002` | 1 | 90 | 7,712 | 2.90 GiB | 2.42 GiB | 32.23 s |

## Why generation stopped

The prospectively frozen `SCHEMA_SMOKE` gate requires ≥ 6 of 8 valid, terminated outputs per
role. Both models returned **0/8 on every one of the 11 role keys** — `R0`, `R1`, `R2`, `R3`,
`A1`, `D1`, `D2`, `D3`, `A2_selector`, `A3_selector`, `S1_diagnostic`. Every output
terminated; none parsed.

The models emit a bare in-vocabulary answer token and stop after two tokens:

```
R0 raw_text: '3'      (d1)        R0 raw_text: '0'      (supa)
R0 raw_text: '2'      (d1)        R0 raw_text: 'UNKNOWN' (supa)
```

The frozen contract requires one JSON object, so `decode('3')` yields an integer, fails
`isinstance(data, dict)`, and is correctly rejected as `invalid answer schema`.

**This is not an instrument defect.** The rendered chat template was verified to deliver the
frozen system prompt verbatim — including its "Return one JSON object" instruction — with
thinking disabled. The models simply do not follow it.

The gate did its job: it stopped both runs before any CAL or CHECK generation, so no scored
population was spent on an unusable output contract. `correctness_inspected` is false in both
smoke receipts.

## Consequence for MRS-1

Cells **A and B are NOT PRODUCED**, and both `MODEL_LAUNCH` tokens are now spent against
`max_models = 2`. MRS-1 forbids prompt repair on smoke failure, and its posttrain contract
requires that a necessary incompatibility create a new experiment rather than a silent
amendment. **MRS-1 is therefore closed at this result.** Its freeze
(`278ce0962bd10864fba7d8ce2e7248579f18b22495d176c4f0f40db903a68837`), bank, prompts and
evidence stand unchanged and are not reopened.

## What followed

Two successor artifacts were built on new isolated branches, neither of which modifies MRS-1:

* **MRS-FP-1** (`research/mrs-format-probe-v1`) measured five output contracts on the 8
  permanently-open SMOKE items. `C1_BARE` — a bare answer token — was the only eligible
  contract, at 8/8 on every gated role for both models. The probe also established the
  limitation precisely: this family emits **exactly one field** and then stops. A prefilled
  opening can choose the container, but nothing observed makes it emit a second field.
* **MRS-2** (`research/multi-reasoning-scout-v2`) is the successor instrument on that
  contract, reusing this bank byte-for-byte by hash. It completed full 120-item generation on
  both candidates and **produced cells A and B**. See its
  `research/multi_reasoning_scout_v2/CURRENT_CANDIDATE_RESULTS.md`.

The MRS-1 negative result is retained in full. It is the reason the successor exists.
