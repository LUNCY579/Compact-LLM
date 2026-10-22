# COMPOSE4B-002 — does the gate skill SURVIVE composition, and does answering pay for it?

> **STATUS: REGISTERED, NOT SCORED.** No arm has been generated or scored on this instrument as
> of this line. Prompts are rendered (a CPU step with no model call); `scored/` is empty.
> Registered before any arm because COMPOSE4B-001's null needs this as its positive control,
> and a control designed after seeing whether it helps is not a control.

**Class: POSITIVE CONTROL + DAMAGE MEASUREMENT. Zero training.** Five existing arms, one
instrument, one battery. Promotes nothing, releases nothing.

---

## 1. Why this experiment, and why it is not optional

COMPOSE4B-001 found **no acquisition cost** from adding δ_H at matched magnitude (net +8, nd 20,
p 0.1153). That result has a hole its own audit exposed:

> **"δ_H's presence costs nothing" is exactly what δ_H ≈ noise would produce.**

δ_H's only measured behaviour on the 4B so far is `H_only = 2/384` on the acquisition bank —
which the audit showed is a **container failure**, not evidence of skill, and which says nothing
about whether δ_H does its *own* job. **Until δ_H is shown to do something on the 4B, the
COMPOSE4B-001 null is not interpretable.** This experiment supplies that control.

It simultaneously closes the gap INDEP-001 named — *"the damage endpoints have never been
measured for ANY composition arm"* — to the extent the 4B lane permits.

## 2. Instrument, and why it is the only admissible one

**HSKILL-VIEWS-002**, `HSKILL_VIEWS_V2.jsonl`, sha256 `ba1f2982fd5d08aa…` (verified),
rule `boundary_rule.py` sha256 `416cfb6aec8aa282…` (verified), 224 rows =
**112 boundary + 112 `arithmetic_short_answer`**, `view_role: ADMITTED_FOR_SELECTION`,
barred only from TRAIN and CONFIRMATION, 11 arms already scored on it, **the 4B never run**.

Every alternative was checked and excluded:

| instrument | status |
|---|---|
| Long context /256 + literal UNKNOWN /64 | **SPENT; barred TRAIN/SELECTION/CONFIRMATION; REPORT-ONLY** by its own `RENDER.json`. Cannot carry a prediction. |
| `references_retained` /47 | **Structurally unrunnable** on a 4B adapter — the probe requires six Supa sidecars and a Supa-hashed materialization receipt |
| arithmetic panel /150 | same |
| SELECT-VIEWS-001 | this is the *acquisition* instrument, already used by COMPOSE4B-001 |

**HSKILL-CONFIRM-002** (`b758e0d13a2e57ad…`, `HELD_FOR_CONFIRMATION`, `UNSPENT_AND_UNREAD`) is
**not touched by this experiment and must not be.**

The instrument yields capability *and* damage on the same rows, which is why one pass suffices:
- **112 boundary rows → sensitivity**: does the gate skill survive composition?
- **112 arithmetic rows → specificity_competence**: is answering damaged?

## 3. Arms (all retained; zero training)

| arm | construction | adapter |
|---|---|---|
| `base` | Qwen3-4B 4-bit, no adapter | — |
| `delta_t` | δ_T alone | `qwenft001-20260918/run/adapter` |
| `th_s` | `s_t·(δ_T + δ_H)` | `compose4b001/arm-th-s` |
| `scaled_t` | `s_t·δ_T` | `compose4b001/arm-scaled-t` |
| `h_only` | δ_H alone | `compose4b001/run-h` |

## 4. Endpoints

Computed by the **imported** sealed rule, never reimplemented:

- **sensitivity** = abstains on the 112 boundary rows
- **specificity_decision** = attempts (does not abstain) on the 112 answerable rows
- **specificity_competence** = attempts **and** is correct on the 112 answerable rows
- **J_decision** = sensitivity + specificity_decision − 1 (the instrument's own primary)

## 5. Registered predictions

**PRIMARY — P1: `th_s − scaled_t` on SENSITIVITY.** Both arms hold δ_T at exactly the same
`s_t`; the only difference is whether δ_H's directions are present. This isolates *does the gate
skill arrive*.

> **P1 fires if net ≥ MDE at the observed discordance with p ≤ 0.05.** The 596M analogue is
> `TH16s − T16flat` = **+16** (29 gains / 13 losses, nd 42, p 0.0195). Registered band for the
> 4B: **net +8 … +30**, expected nd 30–50.

Power, computed before the data (n = 112, exact paired McNemar):

| nd | 6 | 10 | 20 | 30 | 42 | 50 |
|---|---|---|---|---|---|---|
| MDE | 6 | 8 | 10 | 12 | **14** | 16 |

Significance is **unattainable below nd = 6** (nd 4 → min p 0.125). **If nd < 24 the contrast is
reported UNINFORMATIVE**, registered now rather than decided later.

**SECONDARY — P2: `th_s − scaled_t` on SPECIFICITY_COMPETENCE (damage).** Does the gate skill
cost arithmetic? Registered as a **powered null or nothing**: no tolerance margin is registered,
because at n = 112 even a perfect tie bounds the loss-side excess only at ~7.9 rows and a
tighter margin would be unsatisfiable by construction.

**CONTROL — P3: `h_only − base` on SENSITIVITY must be strongly positive.** This is the
experiment's own positive control: **if δ_H does not raise abstention over the base, the whole
composition programme's premise on this base is unsupported**, and both P1 and COMPOSE4B-001's
null become uninterpretable. Registered band: **net ≥ +20**.

**DESCRIPTIVE, no prediction:** `base`, `delta_t` anchors; `J_decision` for all five arms;
per-family sensitivity across the seven boundary classes.

### What each outcome licenses

| | reading |
|---|---|
| P1 fires **and** P3 fires | the gate skill **survives composition** on a base with headroom, and COMPOSE4B-001's null is a real "free", not noise |
| P3 fires, P1 does not | δ_H works alone but its skill **does not survive** the matched-sum composite — which would make COMPOSE4B-001's "free" a statement about an ineffective component |
| **P3 does not fire** | **δ_H is not demonstrably functional on this base and COMPOSE4B-001's null is UNINTERPRETABLE.** This is the outcome that retracts, and it is registered as such before the data |

## 5a. AMENDMENT 1 — the band is made CEILING-AWARE before any arm is scored

Registered with `scored/` empty. A peer raised that **the upper half of my P1 band may be
arithmetically unreachable**, which is the unsatisfiable-gate family this programme has hit five
times tonight. They are right, and the fix is to register the *rule* rather than patch the
*number* after seeing data.

**The arithmetic.** On 112 boundary rows, with `scaled_t` abstaining on `k_s` of them:

```
gains          <= 112 - k_s        you cannot gain a row that is already correct
max net        =  112 - k_s
```

So if `scaled_t` starts high, a large net is impossible **whatever the model does**. The 596M
anchor had room — `T16flat` sits at 68/112 — but the 4B base already beats every 596M core on
abstention, so `scaled_t` may start near the ceiling here and +16, let alone +30, may not exist.

**What is registered, as a formula fixed before the data rather than a number chosen after it:**

> **P1 fires iff `net >= MDE(nd)` and `p <= 0.05`**, with MDE computed from the observed
> discordance as already registered. The point prediction remains **+16** (the 596M analogue);
> the *reportable* band is **[MDE(nd), min(30, 112 − k_s)]**.
>
> **If `112 − k_s < MDE(nd)`, P1 CANNOT FIRE and the contrast is reported
> `UNINFORMATIVE_BY_CEILING`** — a property of the instrument's headroom against this arm, not
> evidence about composition. **`k_s` and the implied ceiling are reported alongside every
> boundary contrast, whatever the outcome.**

This keeps the criterion satisfiable by construction: it can fail (net below MDE), it can pass
(net at or above MDE with headroom), and the case where neither is possible is named in advance
instead of being discovered afterwards.

**Why the anchor survives at all, and only on this endpoint.** δ_H sits at `s_t` = 0.587, well
above its own 0.25 operating point. That would normally distort the anchor — except H-ALPHA-001
measured **boundary as SATURATED across α 0.20–0.675** (84, 86, 85, 86), so the gate's boundary
contribution is roughly magnitude-invariant in exactly that range. **This reasoning does not
extend to long context**, where INDEP-002 found real damage between 0.25 and 0.50. Boundary is
the forgiving endpoint, and that is the only reason a same-magnitude anchor is usable here.

**Main effect vs interaction — stated so a later reader cannot conflate them.** P1 asks *does
δ_H contribute at all* (a MAIN EFFECT, with-H vs without-H at matched T). INDEP-004's 20/4 vs
19/4, interaction ≤ 3 rows, asks *does δ_H's contribution depend on δ_T's magnitude* (an
INTERACTION). **A +16 main effect and a ≤3-row interaction are consistent with each other and
neither substitutes for the other.** A small interaction is not evidence against a large main
effect.

## 6. Build gates, all before any endpoint is read

1. Instrument sha256 = `ba1f2982…`; rule sha256 = `416cfb6a…`; 224 rows. **Verified.**
2. **Render positive control:** the Supa and Qwen3-4B tokenizers are byte-identical on all four
   files, so `apply_chat_template` must agree per row. **Verified: 224/224 agree, 0 mismatches,
   and a deliberate mutation is detected.** `enable_thinking=False`, `max_new_tokens=48` —
   the instrument's own registered value. `prompts_sha256 25d3b50f751a388f…`
3. All five adapters present and hashed; `base` carries no adapter.
4. Every arm scored on the identical 224 row ids; any mismatch ⇒ `BLOCKED_ON_COMPARABILITY`.
5. 14 GiB free-disk floor and the shared GPU lease enforced; neither lowered.
6. `HSKILL-CONFIRM-002` is not opened.

## 7. Envelope

5 arms × 224 rows × 48 new tokens, one lease, ~4 min/arm ≈ **20 min GPU**. Outputs are text;
disk cost is well under 10 MB. Zero training, zero optimizer steps, one battery, no retry
registered.
