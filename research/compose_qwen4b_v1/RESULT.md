# COMPOSE4B-001 — P1 HELD. A second skill is free at matched magnitude on a base with headroom.

> **PROVISIONAL.** One integrity check is outstanding and is named in §5. The result is reported
> with it open rather than held back, because the check is cheap and the direction of its risk
> is stated.

**2026-09-21 · 3 arms × 384 rows · one training run · registered in `PREREG.md` with seven
amendments, every one committed with `scored/` verifiably empty.**

`quality_claim` false. `release_permitted` false. **No arm promoted.**

## 1. The result

| arm | construction | acquisition /384 | registered band |
|---|---|---:|---|
| `TH_s` | `s_t·(δ_T + δ_H)` | **262** | 200–280 ✓ |
| `SCALED_T` | `s_t·δ_T` | **254** | 200–280 ✓ |
| `H_only` | `δ_H` alone | **2** | 0–200 ✓ |

**Primary contrast, `TH_s − SCALED_T`, exact paired McNemar on `select-views-001`
(`views_sha256 7dd53f2a…`, endpoint `canonical`, within-draw):**

| gains | losses | net | nd | exact p | MDE at nd | min attainable p |
|---:|---:|---:|---:|---:|---:|---:|
| 14 | 6 | **+8** | 20 | **0.1153** | **10** | **1.9e-06** |

**Verdict: P1 — a POWERED null.** `|net| = 8 ≤ MDE = 10`, and significance *was* attainable at
this discordance (min p 1.9e-06), so this is a null the instrument could have broken and did
not. It is **not** an underpowered null and is **not** an equivalence certification.

## 2. What makes this stronger than the 596M result, not weaker

INDEP-001 showed on the 596M that a second skill costs zero acquisition **when the allocation is
fixed** — independent per-skill coefficients, +3.4 % magnitude, 235 = T16's 235.

**COMPOSE4B-001 finds no cost without fixing the allocation**, and against four measured
handicaps registered *before* the data:

| handicap | 596M | here |
|---|---:|---:|
| δ_T attenuation inside the composite | 0.675 | **0.5869** (δ_T pays 41.3 %) |
| δ_H vs its own operating point (~0.25) | ~2.7× | **2.35×** |
| `cosine(δ_T, δ_H)` | 0.0027 | **+0.0923** (~34× less orthogonal) |
| `‖δ_H‖ / ‖δ_T‖` | 1.053 | **1.3163** |

**The gate skill's directions cost nothing even when the composite is built the wrong way.**

`H_only = 2/384` is the load-bearing control here: δ_H cannot do the transition contract *at
all*. So the +8 is not H contributing task skill — it is H's directions failing to displace T's,
which is what near-orthogonality predicts and what the shared-scale penalty was being blamed on.

## 3. What the data bound, and what they do not

`equivalence_upper_bound_rows = 0.31` — one-sided 95 % CP upper bound on the loss-side excess.
Verified by two independent implementations (direct binomial inversion and incomplete-beta),
agreeing to 5 decimal places, with the peer-confirmed perfect-tie sanity value 7.92 reproduced.

**That 0.31 is tight only because the observed direction favours `TH_s`.** Had the 20 discordant
pairs split the other way it would have been ~9.7 rows. **No tolerance margin was registered**
(Amendment 5 established that none below ~8 rows is satisfiable at this n), so this bound is
reported **descriptively** and is not a pre-registered equivalence verdict.

## 4. What this does NOT license

Registered before the data and binding now:

- **Not "composition is free" in general.** One pairing, one base, one draw, one bank.
- **Not a statement about damage.** Only acquisition was measured. Retention, the arithmetic
  panel, long context and literal `UNKNOWN` are **NOT MEASURED** for any arm here — the same gap
  INDEP-001 identified across every prior composition arm.
- **Not an equivalence claim.** §3.
- **Not a resolution of the five confounds.** Headroom, attenuation, H's absolute strength,
  allocation and orthogonality all differ between the two bases and **this design separates
  none of them.** A P2 would have been confounded; a P1 is *robust to* all five, which is why
  the null is the informative outcome here and the alternative would not have been.

## 5. OUTSTANDING: a determinism check on `TH_s`

`TH_s` was generated under GPU contention from another lane. Wall clock attributes it exactly:

| arm | wall | s/row | excess vs clean |
|---|---:|---:|---:|
| `TH_s` | 2412.9 s | 6.28 | **+1764 s (~29 min)** |
| `SCALED_T` | 649.3 s | **1.69** | 0 — clean |
| `H_only` | 821.6 s | 2.14 | +173 s |

**The contended arm is the one that scored higher**, so the risk is not neutral: if contention
perturbed generation, it perturbed the arm carrying the +8. Greedy decoding with TF32 off is
deterministic and this programme has measured probe determinism at exactly 0/256 — but that is
an argument, not a measurement on these bytes.

**A spot-check regenerating a `TH_s` subset and comparing against the stored rows is queued for
the next free GPU window. If it diverges, this contrast is BLOCKED, not caveated.**

(An earlier note of mine said the check was queued on `SCALED_T`. That was wrong — `SCALED_T` is
the clean arm. Corrected here.)

## 6. Provenance

One training run: δ_H, LoRA r=16, peak 1e-4, 28 groups, 648 rows, 54,217 processed / 6,274
supervised tokens, loss 6.854 → 1.584. δ_T reused from QWENFT001 (`run/adapter`, unmodified).
Build gate PASS: worst relative error 6.131e-08 against the 2e-06 tolerance, `‖TH_s‖ = ‖δ_T‖` to
5.756e-08, 252 modules, 0 fallbacks. `s_t` computed once and read thereafter, never recomputed.
Train/eval disjointness established on **content** with a positive control after id-namespace
matching was shown vacuous: 0 exact overlap, 0 of 648 H prompts sharing a 20-gram with any view,
0 digit-blinded template collisions.
