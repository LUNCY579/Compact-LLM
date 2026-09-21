# COMPOSE4B-002 — the gate skill SURVIVES composition, and the registered control FAILED

**2026-09-21 · 5 arms × 224 rows · zero training · registered in `PREREG_002.md` + Amendment 1,
both committed with `scored/` verifiably empty.**

`quality_claim` false. `release_permitted` false. **No arm promoted.**

## 1. The five arms

| arm | sensitivity /112 | attempts /112 | competence /112 | J_decision |
|---|---:|---:|---:|---:|
| `base` (4-bit, no adapter) | 77 | 111 | **3** | +0.6786 |
| `delta_t` | 67 | 112 | 57 | +0.5982 |
| **`th_s`** = s_t·(δ_T+δ_H) | **93** | 112 | **66** | **+0.8304** |
| `scaled_t` = s_t·δ_T | 74 | 112 | 51 | +0.6607 |
| `h_only` | 79 | **89** | 54 | +0.5000 |

## 2. Registered outcomes

| | contrast | net | nd | p | MDE | ceiling | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| **P1** | `th_s − scaled_t` sensitivity | **+19** | 31 | **0.00088** | 13 | 38 | **FIRES** |
| **P2** | `th_s − scaled_t` competence | **+15** | 35 | **0.01667** | 13 | 61 | **FIRES** |
| **P3** | `h_only − base` sensitivity | +2 | 48 | 0.885 | 16 | 35 | **DOES NOT FIRE** |

**P1 fires decisively.** Adding δ_H's directions at matched δ_T magnitude buys **+19 boundary
rows**, well inside the ceiling-aware band `[13, min(30, 38)]` registered in Amendment 1.
**The gate skill survives composition on a base with headroom.**

**P2 fires in the UNREGISTERED direction.** I registered P2 as *"a powered null or nothing"* for
damage. It is not a null: composition **improves** arithmetic competence by +15 rows
(p = 0.0167). The direction was not predicted, so per the programme's convention this is
reported and **claimed as nothing** — but it is the opposite of the damage the experiment was
built to look for.

## 3. THE REGISTERED CONTROL FAILED — reported first, not buried

**P3 was the retracting outcome and it did not fire.** `h_only − base` on sensitivity is
**+2 (nd 48, p = 0.885)** against a registered band of **≥ +20**. Not ceiling-limited
(ceiling 35, MDE 16) — δ_H alone genuinely does not raise boundary abstention over this base.

What I registered: *"if P3 does not fire, δ_H is not demonstrably functional on this base and
COMPOSE4B-001's null is UNINTERPRETABLE."*

**I do not get to swap a passing measure in for a failed registered one.** So, precisely:

- **P3 failed as written, and the clause stands as written.**
- **But the clause's PREMISE — that δ_H might be ≈ noise — is refuted by the registered
  PRIMARY**, not by anything post-hoc. P1 shows δ_H's directions deliver +19 boundary rows in
  exactly the composed configuration COMPOSE4B-001 tested, at p = 0.00088.
- Independently, δ_H alone lifts arithmetic competence from the base's **3/112 to 54/112**
  (+51, nd 53, p < 1e-5). δ_H is not noise by any available measure.

**So COMPOSE4B-001's null is interpretable — rescued by its registered primary, not by
substitution.** What is retracted is my *choice of control*, not the result.

**Why P3 was the wrong operationalisation, stated so the error is reusable.** I anchored on the
596M, where the frozen source abstains 71/112 and δ_H raises it to 88. **This 4B base already
abstains 77/112** — it is a different animal, exactly as `qwen3-4b-vs-rineya-cores` records
(scale beats every core on abstention). A marginal gain over an already-abstaining base was the
wrong question. **The right control was the one I made primary.**

## 4. Why `h_only` looks weak and `th_s` does not — the magnitude story, confirmed

`h_only` is δ_H at **full trained strength**. Its attempts collapse to **89/112** against the
base's 111 — it **over-abstains on 22 answerable rows**. That is precisely H-ALPHA-001's finding
that *δ_H at full strength is unusable*, now reproduced on a different base.

Inside `th_s`, δ_H sits at **s_t = 0.5869**, within the saturated-benefit band. Same delta, two
magnitudes, opposite verdicts:

> **The gate skill is not weak on the 4B. It is over-dosed when applied alone at α = 1, and
> works when attenuated — which is [[delta-magnitude-saturates-early]] reproduced on a second
> base and a second instrument.**

## 5. The composite beats the shipped single skill on the gate axis

Descriptive, not registered:

| contrast | net | nd | p |
|---|---:|---:|---:|
| `th_s − delta_t` sensitivity | **+26** | 38 | **0.00002** |
| `th_s − delta_t` competence | +9 | 21 | 0.078 (n.s.) |

`th_s` carries **+26 boundary rows over δ_T alone** and is not detectably worse on competence.
It also holds the highest `J_decision` of any arm here (+0.8304).

## 6. What this does NOT license

- **Not a release or promotion.** One instrument, one base, one draw.
- **Not a general "composition is free".** It is one δ_T/δ_H pair at one `s_t`.
- **Not a damage result on the endpoints INDEP-001 named.** `references_retained` and the
  150-row panel remain structurally unrunnable on a 4B; long context remains SPENT/REPORT-ONLY.
  This closes the gap only on the instrument that was admissible.
- **Not a claim that P2's +15 is a real improvement** — unregistered direction, reported only.
- **P3's failure is a real registered miss** and is recorded as such, not explained away.

## 7. Provenance

Instrument `HSKILL_VIEWS_V2.jsonl` sha256 `ba1f2982…`, rule `boundary_rule.py` sha256
`416cfb6a…`, both verified at run time; the rule is **imported**, never reimplemented.
`HSKILL-CONFIRM-002` untouched. Render gate: 224/224 rows agree between the Supa and Qwen
tokenizers, 0 mismatches, mutation control fires; `prompts_sha256 25d3b50f…`;
`enable_thinking=False`; `max_new_tokens=48`. All contrasts paired on identical 112-row id sets.
Statistics are the canary-validated `exact_mcnemar` / `mde_at` from COMPOSE4B-001, which
reproduce six recorded values from two prior experiments exactly.
