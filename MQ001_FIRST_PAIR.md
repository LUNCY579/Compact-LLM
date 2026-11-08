# MQ-001 first pair: what actually happened, and why the headline was wrong

**Verdict: no objective claim. The apparent response-only advantage is substantially a
degenerate polarity shift, and its p-value does not survive dependence-aware analysis.**

This is the result my own pre-registration predicted would not survive
(`MQ001_CELL_PREREG.md`: *"Treat single-cell DEV deltas as diagnostics, not results"*). I
reported it as a finding anyway before verifying it. The correction below came from an
adversarial review that I then re-verified by hand.

---

## The engineering result, which does stand

Both cells ran clean at commit `131f0af`, bound to plan `81c4727b…`, qualification
`qualification-002`.

| | `whole_1e-5` | `response_1e-5` |
|---|---:|---:|
| updates / processed tokens | 13 / 24,910 | 13 / **24,910** |
| supervised targets | 24,672 | **1,944** |
| objective CE | 5.496 → 2.400 | 2.619 → 2.195 |
| gradients finite | all 13 | all 13 |
| frozen weights changed | no | no |
| DEV terminated | 64/64 | 64/64 |

The control held exactly: identical rows, order, update boundaries and processed tokens;
only the supervised-target set differs. Both artifacts self-report
`quality_improvement_established: false`.

## The headline, and why it dies

Reported: DEV exact 14/41 vs 7/41, paired 8-vs-1, **p=0.0391**.

**1. The response arm stopped saying "No" entirely.** Across all 64 DEV rows, first-token
polarity:

| | YES | NO | non-polar |
|---|---:|---:|---:|
| baseline (Supa) | 0 | 4 | 60 |
| `whole_1e-5` | 1 | 5 | 58 |
| **`response_1e-5`** | **7** | **0** | 57 |

Scored rows with a polar gold are **5 YES / 2 NO**, so a degenerate *always-yes* policy
scores 5/7 on that slice. This is not a subtle confound; it is most of the effect.

**2. Three of the eight wins are one byte-identical string.** The discordant golds:

| gold | count | won by |
|---|---:|---|
| `'Yes, they conflict.'` | **3** | response |
| `'2000 meters.'`, `'18.'`, `'32.'`, `'65 minutes.'`, `'Yes.'` | 1 each | response |
| `'No.'` | 1 | whole |

McNemar assumes independent discordant pairs. Three rows answered by one learned surface
prior are one success, not three.

**3. My DEV set makes the confound unfalsifiable.** All three
`contradiction_detection` rows in DEV have gold `'Yes, they conflict.'`. There is no
negative case, so the instrument **cannot distinguish** "learned contradiction detection"
from "learned to say *Yes, they conflict.*". That is a defect in the corpus I built, not in
the training code.

**4. The p-value is not robust to any of it.**

| treatment | split | p |
|---|---|---:|
| as reported | 8 v 1 | 0.0391 |
| collapse the three identical golds | 6 v 1 | **0.1250** |
| collapse the whole polarity cluster | 5 v 0 | **0.0625** |
| move one row | 7 v 2 | **0.1797** |

Cluster-level sign-flip permutation tests done during review land at 0.1875 (by task type)
and 0.2500 (by mechanism). **Nothing lands below 0.05.**

**5. It is weaker than this programme's own measured seed noise.** RAB-001 recorded a
*150*-row panel producing p=0.0352 between two runs differing only by seed. A 41-row
instrument at n=1 producing p=0.0391 is not evidence.

**6. The package forbids this analysis.** `PLAN.md:65` — *"Do not turn repeated measurements
of the same prompts into independent test examples."* `analyze.py` computes no p-value at
all, so 0.0391 was produced entirely outside the sanctioned pipeline.

## My "9 of 13 genuine content corrections" was also overstated

A fair count of response-arm gains that are neither formatting nor the polarity cluster:
**`'200 meters'→'2000 meters.'`, `'12'→'18.'`, `'16'→'32.'`, `'105 minutes'→'65 minutes.'`** —
four numeric corrections, not nine. My substring classifier counted
`'The liquid spills out.'→'It spills.'` as content when it is a terseness rewrite, and
counted each polarity row separately.

The CE margin (2.1668 vs 2.4940) is close to definitional and is not independent evidence:
response CE *is* the response arm's training objective, while whole-sequence supervision
spends gradient on prompt tokens.

## What is actually worth keeping

* The **harness works**: masks, WSD, frozen-backbone enforcement, checkpointing, create-only
  evidence, honest `QUALITY_UNDECIDED` status. Two cells trained end to end with no faults.
* **Response-only reaches a lower response CE with 12.7× fewer supervised targets.** That is
  a real efficiency observation and the one thing worth a properly powered test.
* A concrete, cheap DEV defect to fix before any further cell.

## Required before any objective claim

1. **Rebalance DEV polarity.** Ensure polar-gold rows are roughly balanced YES/NO and that
   `contradiction_detection` has negative cases. Until then any polarity shift scores as
   skill. This needs a new pool build, a new plan, and a new operator review.
2. **Replicate.** Both arms at three order seeds minimum, analysed with dependence-aware
   clustering by task family — not row-level McNemar.
3. **Drop trailing punctuation from exact golds**, or score a normalised form. A baseline of
   1/41 means the instrument is measuring style adoption first and task skill second.

## Process note

The adversarial review that caught this ran 52 agents; **26 failed on a session usage limit**
and their findings were never verified. The titles that died unverified include an
effective-learning-rate confound between the arms and several scorer-blindness claims. They
are not counted here, for or against.
