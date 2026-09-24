# FUNCTIONAL-ADDITIVITY-001 — do dSparse and dOpallo add in function space? (preregistration)

Written 2026-09-23 **before any arm was scored by this instrument.** The scoring and analysis code
(`score_tf.py`, `analyse.py`) is committed together with this file, and no output from it existed
at commit time. Authorised in-session by the operator on 2026-09-23 ("Additivity on S-bank"):
local teacher-forced scoring of existing arms, no generation, under the shared GPU lease.

**This is not SOUP-MINI-001 CAL/CHECK.** No quality, retention or release claim is made. One
instrument, one skill, n = 192.

## Question

The weight-space interference between the two deltas is small: global cos 0.0025, I_abs 0.0038,
and |cos| ≈ 0.012 inside the shared subspace (`delta_subspace_geometry_v1`). Behaviourally,
ADD-100 keeps Titan-Sparse's transition skill once one trailing period is forgiven: 97 vs 102,
p = 0.33. But exact-match counts are coarse at n = 192, and weight-space statistics have failed to
predict function in this programme (`orthogonality-is-not-the-lever`).

So the question is asked directly in function space, per row and with a continuous readout. Is the
effect of adding dOpallo to Titan-Sparse the same as the effect of adding dOpallo to the base?

```
I(r) = f_ADD(r) − f_SPARSE(r) − f_OPALLO(r) + f_BASE(r)        (second-order mixed difference)
```

`I ≡ 0` is exact functional additivity of the two deltas on row r.

## Instrument

- Bank: `../fresh_dual_skill_instrument_v1/T_BANK_V2.json`, sha256 `fc094136a5b581cb…`, 192 rows,
  admitted as attempt 2 of 2. Rows are system + user. The gold `reference` is oracle-verified.
- Arms: existing FP32 arms in `../titan_opallo_skill_vector_v1/arms/` — **BASE** (0,0),
  **SPARSE** (1,0), **OPALLO** (0,1), **ADD-O75** (1,0.75), **ADD-100** (1,1). No arm is built.
- Rendering: `score_arms.render`, verbatim, with thinking disabled. The template self-test from
  `score_arms.py` must pass at startup.
- Tokens: `tok(render(system,user), add_special_tokens=False)`, then
  `tok(reference, add_special_tokens=False)`, then `<|im_end|>` (151645). One forward pass per row,
  `use_cache=False`, FP32, CUDA, TF32 forced off and recorded. The logits at prompt_len−1 …
  prompt_len+G−1 predict the G gold tokens and then EOS.
- The same runtime as the recorded generation runs: the interpreter that ran `V2_ARMS.json`
  (python 3.11.15, torch 2.6.0+cu124, transformers 5.14.1). `sys.executable` is recorded.

## Segments, fixed now, and why

The period convention does **not** act only at EOS. `gold + "."` re-tokenizes the final gold token
on **128/192** rows: `}` becomes `}.` (id 7810) and `"]` becomes `"].` (id 5521). On the 64 integer
rows the period is a separate token (`.`, id 13) at the EOS position. So:

- **content** = gold tokens 0 … G−2 (all but the final gold token)
- **closing** = the final gold token + EOS

The primary question is on **content**, where the skill lives. **Closing** carries the surface
convention and is reported separately.

## Readout: clipped gold-token log-odds, not log-probability

Log-probability is capped at 0. If SPARSE already assigns p ≈ 1 to a token, any gain dOpallo gives
the base cannot appear on top of SPARSE, so `I < 0` would be **manufactured by the ceiling**. The
primary readout is therefore the gold token's log-odds, `z = logit_gold − logsumexp(logits_other)`,
which is numerically stable and has no ceiling. It is clipped to [−10, +10] so the far tails
(|p − 0 or 1| < 4.5e-5), where variation is irrelevant, cannot dominate sums:

```
Z_content(arm, r) = Σ_{t ∈ content(r)} clip(z_t, −10, 10)
Z_closing(arm, r) = Σ_{t ∈ closing(r)} clip(z_t, −10, 10)
```

**Secondary** readouts, registered now and reported regardless:

- (a) the same sums of **unclipped log-probability**
- (b) `Z_content` restricted to **decision positions**: those where at least one of the five arms
  has p(gold) < 0.9. The rule is symmetric across arms and blind to I.

## Gates — each checked before the primary is read. A failed gate stops the analysis.

| gate | requirement | why it can fail |
|---|---|---|
| **G-A template** | `score_arms.template_selftest()` passes (384 recorded prompts byte-exact, negative control rejects a missing think block) | wrong render ⇒ a different instrument |
| **G-B strict canary, by identity** | per arm, the rows where teacher-forced argmax equals gold at **every** gold position and at EOS must equal the recorded strict-correct set (`CALIBRATION_V2.json` BASE/SPARSE; `V2_ARMS.json` OPALLO/ADD-O75/ADD-100), symmetric difference **≤ 2 rows** per arm | an indexing off-by-one, a tokenization mismatch, or a different model: any of these makes the readout measure something other than what generation measured. Expected sets: BASE 0, SPARSE 102, OPALLO 1, ADD-O75 63, ADD-100 6 |
| **G-C determinism** | SPARSE is scored a second time in a separate model load; the per-row max \|ΔZ_content\| and \|ΔZ_closing\| must be ≤ 1e-3 | nondeterministic kernels make row-level differences noise |
| **G-D denominator** | mean over rows of `g_S = Z_content(SPARSE) − Z_content(BASE)` has a bootstrap 95% CI entirely > 0 | if SPARSE's skill does not register on content, κ is undefined |

## Primary estimand and outcomes (content, clipped log-odds)

```
κ = mean_r I(r) / mean_r g_S(r)
```

The 95% CI is a percentile bootstrap over rows: 10,000 resamples, seed 20260923, ratio of means
per resample.

| outcome | rule |
|---|---|
| **ADDITIVE** | CI ⊂ (−0.10, +0.10): the interaction is within ±10% of Titan-Sparse's own content effect |
| **SUB-ADDITIVE** (functional interference) | CI upper < 0 **and** point ≤ −0.10 |
| **SUPER-ADDITIVE** | CI lower > 0 **and** point ≥ +0.10 |
| **UNRESOLVED** | anything else, including a CI too wide to decide. **An unresolved result is not additivity.** |

The outcomes are mutually exclusive by construction. ADDITIVE is an *equivalence* claim, and it
requires the CI to be narrow, so an underpowered run cannot produce it.

Registered expectation (stated as a belief; it is not a gate): ADDITIVE, at roughly 60%;
SUB-ADDITIVE is the main alternative. The basis is ADD-100's period-tolerant 97 vs SPARSE's 102,
plus OPALLO's small tolerant gain over BASE (8 vs 0).

## Secondary, registered now, reported regardless

1. The same κ and outcome rule for readouts (a) and (b).
2. **Closing:** mean I_closing with CI, in log-odds units. Also κ_closing if the closing
   denominator `mean(Z_closing(SPARSE) − Z_closing(BASE))` has a CI > 0; otherwise the raw mean
   only. Consistency expectations taken from the recorded generations (not tests of the question):
   `Z_closing(ADD-100) < Z_closing(SPARSE)` on ≥ 150/192 rows, and
   `mean(Z_closing(OPALLO) − Z_closing(BASE)) < 0`.
3. **P_S on content**: `P_S = mean(Z_ADD − Z_BASE) / mean(Z_SPARSE − Z_BASE)` with CI, next to its
   additive prediction `1 + mean(Z_OPALLO − Z_BASE)/mean g_S`.
4. **β-curvature at α = 1**, descriptive: `ρ(r) = Z(ADD-O75) − [0.25·Z(SPARSE) + 0.75·Z(ADD-100)]`,
   mean with CI, for content and closing.
5. Per-family breakdown (`ledger_replay`, `prerequisite_plan`, `register_machine`) of κ and I.
6. Row-level: mean I_content on the 11 rows SPARSE solves and ADD-100 loses under the
   period-tolerant rule, against all other rows. Descriptive.
7. Period propensity at the convention site, per arm. For merged rows this is the log-prob of the
   merged token (`}.` / `"].`) at the final-gold position; for integer rows it is the log-prob of
   `.` at the EOS position.
8. A canary with no tolerance, reported only: the rows where content argmax matches gold and the
   final position's argmax is either the gold token or its merged-period variant, against the
   recorded period-tolerant set (`gen == ref` or `gen == ref + "."`).

## Limits, stated now

- This is one instrument on one skill (transition), and it measures Titan-Sparse's skill
  surviving dOpallo. **There is no O-side counterpart.** `O_SKILL_DEFINABILITY.md` shows there is no
  Opallo-specific skill on any existing instrument, so "does Sparse damage Opallo's skill" has no
  object.
- Additivity is measured at one scale (α = β = 1), plus a single β-curvature point. It is not
  local linearity; no gradient or JVP is computed.
- A teacher-forced readout conditions on the gold prefix. It measures the model's belief about the
  correct answer, not what greedy decoding would reach. G-B ties the two together on the strict
  set only.
- The ±0.10 equivalence margin is a convention, fixed here. It has no external derivation.

---

## Provenance erratum (2026-09-23), by this file's author

- **The opening claim is false.** It says the scoring and analysis code were committed together
  with this file before any output existed. This session was interrupted before committing
  anything. Another session then wrote `score_tf.py` (saved 15:25, eleven minutes after this
  file) and ran it (`TF_RAW.json`, 15:27). It wrote `analyse.py` (last saved 15:28:50) **62
  seconds after `TF_RAW` existed**. Nothing was committed before the run.
  - **Mitigation.** This file's mtime (15:14) precedes all outputs, and its text is unchanged
    since then. A cold re-implementation written from this text alone, before reading
    `analyse.py`, reproduces the primary to 5 decimals. That shows the primary is computed
    correctly. It does **not** restore preregistered status to the analysis code.
  - **What the late analyser got wrong:** secondary (b) was implemented per row (see RESULTS.md
    Erratum 1).
- **The "Segments" section cites `"]` → `"].` (5521). No row in this bank ends in `"]`.** All 128
  merged rows end in `}`.
- **The clip premise was false.** "It is clipped to [−10, +10] so the far tails … cannot dominate
  sums" assumed the clip would bind only on irrelevant tails. It binds on 21.2% of SPARSE's
  content positions against 7.0% of ADD-100's, and it biases I toward 0. The ADDITIVE label is
  correct under this registration and **not robust to it**; see
  `amendment_1/AMENDMENT_1.md` §1.
