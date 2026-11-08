# Magnitude is the master variable — the state of the delta programme, 2026-09-20

Written to answer four operator questions: what is worth keeping since Titan-Experimental and
Titan-Sparse; would **more** delta help; would **orthogonal** deltas help; what next.

**Class: SYNTHESIS of recorded results.** No new measurement. Every number below is cited to the
arm that produced it. Nothing here is a release claim.

---

## 1. The one finding that unifies the programme

> **In every lane measured, a delta's BENEFIT saturates well below its trained magnitude while
> its COLLATERAL COST keeps accruing. The trained magnitude is past the knee on every instrument
> tested so far.**

Two independent lanes, two instruments, same shape:

| lane | benefit limb | at the knee | at full strength | cost limb | at the knee | at full strength |
|---|---|---:|---:|---|---:|---:|
| **generator** δ_T *(INT-001)* | acquisition /384 | **234** @ α 0.75 | 235 @ α 1.00 | retention | higher | keeps falling |
| **gate** δ_H *(H-ALPHA-001)* | sensitivity /112 | **86** @ α 0.25 | 88 @ α 1.00 | specificity /112 | **112** | **93** |

**The benefit limb is flat across the top: 1 row for the generator, 2 rows for the gate.** The
cost limb is not flat. Training to convergence and deploying the result at α = 1 pays the full
cost for the last ~1% of the benefit.

### The gate case, in full, because it is the cleanest thing the programme has produced

| α | sens /112 | spec /112 | `J_decision` | vs the frozen source |
|---:|---:|---:|---:|---|
| 0.00 *(source)* | 71 | 111 | 0.6250 | — |
| **0.25** | **86** | **112** | **0.7679** | **+15 sens, +1 spec** |
| **0.50** | 85 | 112 | 0.7589 | +14 sens, +1 spec |
| **0.675** *(`SCALED_H`)* | 86 | 111 | 0.7589 | +15 sens, ±0 spec |
| 0.85 | 85 | 104 | 0.6875 | +14 sens, **−7 spec** |
| 1.00 *(`H16`, as trained)* | 88 | **93** | 0.6161 | +17 sens, **−18 spec** |

*(α = 0.05 / 0.10 / 0.15 were also built and establish there is no further gain below 0.25.)*

> **The trained delta buys 2 extra boundary rows and pays 19 answerable ones for them.**
> Quarter strength keeps 15 of the 17 and costs nothing.

**Correction to how this is being reported.** The peer session's headline is *"α = 0.25 is the
optimum."* That over-reads it. **0.25, 0.50 and 0.675 are within ONE row of each other on both
halves** (86/112, 85/112, 86/111). At n = 112 that is not a resolvable ordering — it is the same
defect as reading a J difference off fewer than 6 discordant pairs, which this programme has
already been bitten by twice.

> **The defensible finding is a PLATEAU from ~0.25 to ~0.675 and a COLLAPSE above it**
> (spec 112 → 104 → 93). The plateau's width is the result; its argmax is not.
> **You do not need to find a precise α. You need to stay off the top end.**

That is a much more useful engineering statement, and it is the one the data supports.

---

## 2. What is worth keeping since Titan-Experimental / Titan-Sparse

Ordered by how much they would change what you do.

| # | result | class | why it matters |
|---|---|---|---|
| **1** | **α ∈ [0.25, 0.675] on δ_H recovers +15 of 17 sensitivity rows at zero specificity cost** | OPEN DEV RESULT | a **free, deployable** quality gain — no training, ~80 s of weight arithmetic |
| **2** | **Data dose has NOT saturated: 239 → 304** | OPEN DEV RESULT | the 596M is **data**-limited, not capacity-limited — the one axis where *more* still pays |
| **3** | **Composition at matched magnitude is free** — SCALED-T (−12, p 0.0428, T's own shrinkage) + SCALED-H (+18 spec) | MECHANISM SCOUT | two skills **coexist**; the 60-row "interference" was attenuation, not conflict |
| **4** | **Composition is a REPAIR** — composing recovers 19/19 of the gate skill's cost | MECHANISM SCOUT | magnitude is an **acquisition-only** variable; gates compose, generators do not |
| **5** | **Qwen3-4B + LoRA scores 270/384 and 180/256** on the same rows | OPEN DEV RESULT | scale beats **every** 596M arm on both axes — the honest oracle for what is being given up |
| **6** | **The task delta is low-rank** — P₁₆ retains 0.812 of the norm and scores 236 vs 235 | MECHANISM SCOUT | this is **why** all of the above is cheap; rank-16 arithmetic costs seconds |
| **7** | **δ_MIX lies inside span{δ_T, δ_H}** — cos 0.7429, above the replication ceiling | MECHANISM SCOUT | a **jointly trained** delta is largely a linear combination of separately trained ones |
| **8** | **Train/eval disjointness verified on the native lane** — 384 vs 384, 0 shared parents | PREMISE CHECK | the acquisition numbers are **generalisation**, not recall |

**Counterweight, stated because it is the one that cuts against the programme's own promotions:**
the SPARSE-001 k = 0.25 core was promoted, and then **RCI-001 — the sealed confirmation
instrument, now spent — found it LOSES abstention against the frozen source, 0 gains / 15
losses.** That is the only *confirmed* head-to-head the programme has, and it went against a
promoted artifact. Everything in the table above is **development-grade** and none of it has been
through a confirmation instrument.

---

## 3. "Would applying MORE delta weight changes be beneficial?"

The question splits three ways and the evidence answers each differently.

| reading | answer | evidence |
|---|---|---|
| **more magnitude** (larger α, train longer/harder) | **NO — decisively** | every swept lane's optimum is strictly below trained strength; gate 0.25–0.675, generator ~0.75, LWF 0.50, LoRA 0.50; **high dose abolishes abstention entirely, 13 → 0** |
| **more deltas** (stack more skills) | **SAFE, NOT PROFITABLE** | at matched magnitude composition costs nothing measurable — but `SCALED_H → TH16s` has nd = 4 and 1, so it also **buys** nothing measurable. That is a **BOUND, not a null** |
| **more DATA into one delta** | **YES — the only "more" that pays** | the dose ladder runs 239 → 304 **with no saturation**; every transform arm re-slices ONE delta trained at the **lowest** rung |

> **More data, yes. More magnitude, no. More deltas, neutral.**

The practical consequence is uncomfortable and worth stating plainly: **most of the programme's
arms are transforms of a single under-trained delta.** The magnitude axis is now well mapped and
has been shown to be a *calibration* problem with a free fix. The dose axis is barely explored and
is the only one still returning gains.

---

## 4. "Or apply delta weight changes at different dimensional orthogonality?"

**This is already refuted, and the refutation runs in the surprising direction.**

| pair | mean cosine | composed result |
|---|---:|---|
| **same skill** (s23, s29 — two seeds) | **0.60** | **247**, *above* the 236 one-delta ceiling |
| **different skill** (δ_T, δ_H) | **0.151** *(unrelated baseline 0.0885)* | **lost 60 rows** raw |

> **The MORE overlapping pair composed BETTER. Overlap does not order interference.**

So deliberately orthogonalising deltas — projecting one out of the other's subspace, training in
disjoint directions — is **not supported by the one measurement that bears on it.** The variable
that predicted composition outcome was magnitude, and once magnitude was matched the angle
explained nothing left over.

Two further reasons the orthogonality framing is weaker than it sounds here:

1. **δ_MIX sits inside span{δ_T, δ_H} at cos 0.7429**, with coefficients a = +0.457, b = +0.817.
   A delta trained on the *mixture* is substantially recoverable as a linear combination of the
   two trained separately. The skills are not occupying cleanly separable subspaces — and the
   joint solution does not leave the span of the parts.
2. **Rank-16 truncation is already a severe dimensional restriction and it is nearly free**
   (0.812 of the norm, 236 vs 235). The delta lives in a small subspace to begin with. Two
   low-rank deltas in a 596M-parameter space are *generically* near-orthogonal without anyone
   arranging it — which is why the different-skill cosine is 0.151 and not something larger.
   **Engineering orthogonality would be arranging something that is nearly true by default.**

**What would change this verdict:** a same-skill / different-skill pair at *matched magnitude and
matched dose*, with the angle deliberately manipulated. Nothing on record does that — the 0.60 and
0.151 pairs differ in skill, seed, dose and corpus simultaneously. **So the honest status is:
orthogonality is unsupported as a lever and untested as a variable,** and the evidence that does
bear on it points the opposite way from the intuition.

---

## 5. What to do next — and the blocker that outranks it

### The blocker: every α optimum on record was SELECTED on the bank that reports it

H-ALPHA-001 swept 7 values and reported the best on the same 224 rows. INT-001 swept 4 and
reported the best on the same 384. **That is selection on the test set.** The programme's own
recorded lesson — *a maximand chosen without checking its MDE is a tie-break wearing a criterion* —
applies directly, and the plateau analysis in §1 shows the top three points differ by one row,
which is exactly what selection noise looks like.

**This does not threaten the large effects.** The 112 → 93 specificity collapse and the +15
sensitivity gain over the source are far outside any plausible selection artifact. **It threatens
the argmax only** — and §1 already withdraws the argmax in favour of the plateau.

**Both confirmation instruments are SPENT.** RCI-001 (1/1) and RCI-002 (1/1). So the correct next
move is not another sweep.

### Recommended next, in order

1. **Build a confirmation instrument for the magnitude claim, and register the PLATEAU — not the
   argmax — as the hypothesis.** The claim to confirm is *"α in [0.25, 0.675] beats α = 1.00 on
   specificity by ≥ 10 rows with sensitivity within 4"*, which is large, pre-registered, and has an
   MDE the instrument can actually meet. A sweep cannot confirm itself.
2. **Push the dose axis, because it is the only unsaturated one.** Every transform result is a
   re-slice of a delta trained at the lowest rung. Retraining δ_T at the top of the dose ladder and
   re-running the α sweep answers the question the whole programme keeps circling: *does the knee
   move with dose, or is ~0.75 a property of the architecture?* **This needs a budget grant — it is
   training, not arithmetic.**
3. **Do NOT build the orthogonality experiment as currently imagined.** §4 gives the reason. If it
   is built anyway it must manipulate angle at matched magnitude and matched dose, or it will
   re-measure magnitude a fourth time.
4. **Mirror the α sweep on the 4B lane** — it is free arithmetic, and the 4B is the only artifact
   beating the 596M on both axes. If the knee is at ~0.25–0.75 there too, magnitude calibration is
   architecture-independent and becomes a standing deployment step rather than a finding.

---

## 6. Standing cautions carried forward

- **Never quote a pooled net.** Decompose to gains/losses; three of this programme's reversals were
  hiding inside a net.
- **`views_sha256` is the only identifier of a bank.** `7dd53f2a…` is the acquisition bank and is
  *also* recorded as `selection_sha256` on training preparations — verified disjoint, see
  `analysis/TRAIN_EVAL_DISJOINTNESS_20260920.md`.
- **Cross-draw differences need |net| ≥ 20; within-draw transforms of one delta do not.**
- **A check whose two sides cannot match is indistinguishable from a passing check.** Assert a
  shared namespace and run a positive control before reading a 0.
- **A statistic correct for one object is not correct for a sum of two such objects**
  (`P₁₆(δ_T) + P₁₆(δ_H)` is rank-32; its norm is Frobenius, not top-16).
- **Six extractor-class bugs this week.** Five were caught by collision with an already-known
  number; only one by a gate. Build the gate.

---

# Amendment 1 — INDEP-001 closes the composition question, and it is a MAGNITUDE account end to end

**2026-09-21.** The paired session built the arm this document's §3 said was missing: **each skill
with its OWN coefficient**, rather than one scale applied to the sum.

    INDEP-001 = theta_src + 1.00 * P16(delta_T) + 0.25 * P16(delta_H)

| arm | allocation | acquisition /384 | vs `T16` |
|---|---|---:|---:|
| `T16` | δ_T alone at 1.0 | 235 | — |
| `TH16` | raw sum, both at 1.0 | 175 | **−60** |
| `TH16s` | ONE scale on the SUM | 219 | −16 |
| `SCALED_T` | δ_T alone at 0.675 | 223 | −12 *(p = 0.0428)* |
| **`INDEP-001`** | **δ_T at 1.0, δ_H at 0.25** | **235** | **0** |

> **Adding an entire second skill costs ZERO acquisition rows when it carries its own
> coefficient.** `INDEP-001` lands exactly on `T16`. Its registered band was ≥ 228.

**The mechanism, measured:** `‖P₁₆(δ_T)‖ = 0.674`, `‖P₁₆(δ_H)‖ = 0.710`. Adding δ_H at **1.00**
raises total magnitude **+45.2%**; at **0.25** it raises it **+3.4%**. The shared-scale rule paid
for that 45% by **stripping 31.1% off everything, δ_T included** — funding a gate skill carried at
**four times** its useful strength by shrinking the generator by a third.

> **"Composition is expensive" was never about interference. It was about an allocation rule
> nobody had questioned.** SCALED-T showed 12 of the −16 was δ_T's own shrinkage; INDEP-001 shows
> the shrinkage was **never necessary**.

## §3 of this document is superseded on one row

| reading | was | **now** |
|---|---|---|
| **more deltas** (stack more skills) | SAFE, NOT PROFITABLE | **SAFE AND FREE — at independent per-skill coefficients.** The cost was the shared scale, not the second skill |

`more data → YES` and `more magnitude → NO` are unchanged. **The unifying statement strengthens:
magnitude is the master variable, and ALLOCATION across skills is part of magnitude.**

## §4 needs a correction: I was quoting the wrong overlap number

INDEP-001's build gate originally required the composed update's norm to equal the **quadrature**
of the two scaled norms — which *asserts* `⟨δ_T, δ_H⟩ = 0`, a premise COMPOSE-001 had already
measured false. The 4.68e-04 it flagged was not an error; **it was the cross term.** Replaced with
a triangle-inequality check and the cross term reported:

| quantity | value |
|---|---:|
| `⟨P₁₆δ_T, P₁₆δ_H⟩` | **+1.306e-03** |
| **flattened-delta cosine** | **+0.00273** |
| rank-16 **subspace** overlap | 0.151 |

> **These are different objects and I was quoting the second as if it were the first.**
> As *vectors* the two deltas are essentially **orthogonal (0.0027)**. The 0.151 is a *subspace*
> overlap. §4 above reports "0.151 against an unrelated baseline of 0.0885" as the operative
> number for interference; **it is not even the right quantity.**

**This makes §4's conclusion STRONGER, not weaker.** The two deltas are as close to orthogonal as
makes no practical difference, **and composition still cost 60 rows unrescaled.** So:

> **Orthogonality was already available and it bought nothing.** The variable that governed the
> outcome was allocation. Engineering more orthogonality cannot help a pair that is already at
> cosine 0.0027 — which is the strongest form the §4 refutation can take.

## Consequence for the recommendation

§5's item 2 (push the dose axis) is unchanged and still needs a grant. But **item 3 —
"do NOT build the orthogonality experiment" — is now settled rather than merely unsupported**, and
a new item outranks item 4:

> **Re-run every composition on record with independent per-skill coefficients.** It is pure
> weight arithmetic over deltas already on disk, and on the one pair tested it converted a 60-row
> loss into a 0-row loss. Every composed arm in this programme was built under the allocation rule
> that INDEP-001 has now falsified.

---

# Amendment 2 — I UNDERSTATED Amendment 1. The second skill is free on every endpoint measured.

Amendment 1 was written from a partial report and said *"costs zero acquisition rows."* The
completed INDEP-001 result (`BATTERY_COMPLETE`, 2 arms × 4 endpoints, all five predictions
resolved) is stronger on two counts.

**First, this is the first composition arm in the programme whose DAMAGE endpoints were actually
measured at all.** COMPOSE-001 and MAG-001 were both refused at `exit=2` on the host-RAM guard, so
every composition claim on record until tonight rested on acquisition and `J_decision` alone.

**Second, the control is `T16` rebuilt and re-measured under tonight's probe hashes**, not quoted
from history — and it independently reproduced four figures never used to define its canary
(panel 68, retained 39, long context 159, literal `UNKNOWN` 10).

| endpoint | composed | control `T16` | paired | exact p |
|---|---:|---:|---:|---:|
| acquisition /384 | **235** | 235 | — | — |
| family floors | 3/3 | 3/3 | — | — |
| arithmetic panel /150 | 73 | 68 | 8 / 3 | 0.2266 *(not claimed; 10-row floor)* |
| `references_retained` /47 | 39 | 39 | — | — |
| **long context /256** | **164** | 159 | 6 / 1 | 0.1250 |
| literal `UNKNOWN` /64 | 11 | 10 | — | — |
| **boundary /112** | **85** | 69 | **20 / 4** | **0.0015** *(powered to 1.2e-07)* |
| answerable /112 | 112 | 112 | **0 / 0** | 1.0 |

> **A whole second skill for +3.4% total magnitude, at no measurable cost on any of four
> endpoints — and the gate skill arrives at FULL strength: +16 boundary rows at p = 0.0015.**

## The long-context number is the one that settles it

| arm | allocation | magnitude | acquisition | **long context** |
|---|---|---:|---:|---:|
| `TH16` | raw sum, both at 1.0 | +45.2% | 175 | **127** |
| `TH16s` | ONE scale on the SUM | +0.0% | 219 | *never measured* |
| **`INDEP-001`** | **each skill its own scale** | **+3.4%** | **235** | **164** |

**`TH16`'s long context of 127 sits THIRTY-TWO ROWS BELOW THE FROZEN SOURCE.** That was the single
worst composition result in the programme, and it was read for days as evidence that two skills
cannot coexist.

> **It was never skill conflict. It was a gate skill carried at four times its useful strength,
> funded by a shared rescale that stripped a third off the generator. Fix the allocation and the
> damage does not shrink — it disappears, and long context goes ABOVE the single-delta control.**

## §3's answer changes from "neutral" to "yes"

| reading | §3 said | Amendment 1 said | **now** |
|---|---|---|---|
| **more deltas** | SAFE, NOT PROFITABLE | safe and free on acquisition | **YES — free on all four measured endpoints, and the added skill delivers its FULL effect** |

`more data → YES` and `more magnitude → NO` stand. The corrected three-way answer is:

> **More data: yes. More magnitude: no. More SKILLS: yes — provided each carries its own
> coefficient.** The thing that was never free was the *shared scale*, and no one had tested
> composition without it.

## What this does NOT license

- **One pair, one point.** `(1.00, 0.25)` is a single sample of a two-dimensional surface;
  neither coefficient was swept *in composition*.
- **Panel +5 and long context +5 are NOT claimed** — p = 0.2266 and 0.1250, and the panel's
  measured noise floor is 10 rows. What is claimed is the **absence** of damage, and the boundary
  gain.
- **δ_H's 0.25 was chosen from H-ALPHA-001's sweep on a bank it was selected on**, which is the
  same selection-on-the-test-set problem HPLAT-001 exists to address. The composition result does
  not inherit a confirmation it never had.
