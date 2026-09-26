# RESULT — Supa Long-Context Minimal Intervention Pilot

Lab `supa_long_context_minimal_intervention_v1`, protocol `1.0.0-pilot`.
Target checkpoint **Rineya-Supa-A1.1-1.1.0-devfreeze**, read-only throughout.

Status: **Phase 0 complete on two suites. Phase 1 in progress.**

---

## Validity gate

Condition A over all 256 rows of the Rineya Long-Context Demonstration Suite
reproduced the published Supa-A1.1 result **exactly**:

| | Showcase64 | Stability192 | all256 |
|---|---:|---:|---:|
| published (gated) | 45 | 125 | 170 |
| **reproduced here** | **45** | **125** | **170** |

Model state hash was identical before and after inference, and all ten canonical
checkpoint files hashed identically before and after every stage. The canonical
checkpoint was never modified.

---

## PHASE 0 — zero-training context refinement

### Primary suite — Rineya Long-Context Demonstration Suite (n = 256)

Contexts 1048–1481 tokens (median 1264). All four conditions share model,
checkpoint, decoding, output length, scorer, prompt and seed policy.

| condition | score | accuracy | mean context tokens | retention |
|---|---:|---:|---:|---:|
| **A** full | 170/256 | **66.4%** | 1260 | 100% |
| **B** query-selected | 147/256 | **57.4%** | 560 | 44.5% |
| **C** random equal-token | 97/256 | **37.9%** | 544 | 43.2% |
| **D** oracle evidence | 111/160 † | **69.4%** † | 541 | 43.0% |

† Oracle is defined on 160 of 256 rows only. See *Oracle availability* below;
D is never compared against an A computed on a different row set.

**Contrasts (paired bootstrap, 10,000 resamples, seed 20260905):**

| contrast | frame | delta | 95% CI | McNemar p |
|---|---|---:|---|---:|
| B − A | all (256) | **−8.98 pts** | [−14.06, −3.91] | 0.000824 |
| B − C | all (256) | **+19.53 pts** | [+12.11, +26.95] | 1e-06 |
| C − A | all (256) | −28.52 pts | [−35.94, −21.09] | <1e-06 |
| B − A | oracle-eligible (160) | −18.12 pts | [−24.38, −11.88] | <1e-06 |
| **D − A** | oracle-eligible (160) | **−14.38 pts** | [−22.50, −6.25] | 0.000824 |
| D − B | oracle-eligible (160) | +3.75 pts | [−3.75, +11.25] | 0.405 |

**Peak memory** 1.86 GB GPU allocated / 2.65 GB RSS. **Wall-clock** 385 s for 928
generations, one model load. Malformed-output rate 0.0% in every condition.

### The selector is not the failure

The mandated random control settles this. At a matched budget
(B/C token ratio 1.030), query selection beats random selection by **+19.5
points** with an interval far from zero. The selector is finding relevant
material. It is simply that removing the rest costs more than the relevance
gains.

Condition D makes the point harder. Its localisation key is answer-bearing: in
this suite the gold answer string sits inside `supporting_facts` on 87.5% of
answerable rows and **90% of the oracle-eligible rows**. D therefore selects with
privileged information — and still loses to full context by 14.4 points.

**An evidence-selection condition that is allowed to cheat cannot beat simply
keeping the whole document.** That closes the question for this family of
interventions far more firmly than B − A alone could.

### The apparent gain on unanswerable rows is a response-bias shift

Splitting by answerability shows what the aggregate hides:

| frame | n | A full | B query | C random | B − A |
|---|---:|---:|---:|---:|---:|
| all | 256 | 66.4% | 57.4% | 37.9% | −8.98 |
| answerable only | 192 | 77.1% | 61.5% | 34.9% | **−15.62** |
| unanswerable only | 64 | 34.4% | 45.3% | 46.9% | +10.94 |

Abstention rate **on answerable rows** — where saying UNKNOWN is always wrong:

| condition | abstention rate on answerable rows |
|---|---:|
| A full | **1.0%** |
| B query | 10.9% |
| C random | 26.0% |

Cutting context makes the model abstain on questions it can answer, roughly ten
times more often under B and twenty-six times more often under C. The +10.9
points on unanswerable rows is that same bias scoring as correct. The honest
read is the answerable-only frame: **−15.6 points**.

### Which task families break, and why

| task type | A full | B query | B − A |
|---|---:|---:|---:|
| temporal_correction | 96.9% | 59.4% | **−37.5** |
| count_with_exclusions | 50.0% | 37.5% | −12.5 |
| alias_resolution | 12.5% | 3.1% | −9.4 |
| causal_disambiguation | 100.0% | 93.8% | −6.2 |
| latest_verified_record | 84.4% | 81.2% | −3.1 |
| multi_condition_identity | 59.4% | 56.2% | −3.1 |
| rule_exception | 46.9% | 43.8% | −3.1 |
| object_state_tracking | 81.2% | 84.4% | +3.1 |

`temporal_correction` collapses, and the mechanism is legible. These items turn
on a *later record superseding an earlier one*. The superseding correction
usually shares few terms with the question, so a query-relevance selector ranks
it low and drops it — leaving the superseded record looking authoritative. The
same logic covers `count_with_exclusions` and `latest_verified_record`.

On this suite the context that a relevance selector calls irrelevant is
frequently the context that determines which answer is current. That is not
interference; it is the task.

### Oracle availability

| | count |
|---|---:|
| rows | 256 |
| answerable | 192 |
| **oracle-eligible** | **160** (83.3% of answerable) |
| excluded — unanswerable | 64 |
| excluded — a fact failed to localise | 32 |

Unanswerable rows are excluded by construction: a question with no determined
answer has no supporting evidence to select, and this suite gives them a constant
boilerplate `supporting_facts` string with no localisation signal. Rows where any
fact failed the 0.34 threshold are excluded rather than approximated — a two-hop
oracle missing one hop is not an oracle.

This suite states supporting facts in abstracted form: **0 of 424 appear
verbatim** in their context, so localisation here is lexical and approximate.
That limitation is what the confirmatory suite below removes.

### Confirmatory suite — LCR dev (n = 200) — THE DIRECTION FLIPS

This suite exists to fix the primary suite's two weaknesses: contexts are
2112–2554 tokens (mean 2381 against 1260), and all 2688 supporting-facts entries
are **exact substrings** of their context, so condition D is localised exactly
rather than approximately — oracle-eligible on **150/150** answerable rows, with
zero localisation failures.

| frame | n | A full | B query | C random | D oracle |
|---|---:|---:|---:|---:|---:|
| all | 200 | 29.5% | 37.0% | 29.5% | — |
| **answerable only** | 150 | **32.7%** | **30.0%** | **10.0%** | **38.0%** |
| unanswerable only | 50 | 20.0% | 58.0% | 88.0% | — |

Contrasts on the answerable frame:

| contrast | delta | 95% CI | excludes 0 |
|---|---:|---|---|
| B − A | −2.67 pts | [−10.67, +5.33] | no |
| **D − A** | **+5.33 pts** | [−2.67, +13.33] | **no** |
| D − B | +8.00 pts | [+0.00, +16.00] | no |
| B − C | +20.00 pts | [+12.00, +28.00] | **yes** |

**This partially contradicts the primary suite and the contradiction is real.**
On the primary suite the oracle was 14.4 points *below* full context; here it is
5.3 points *above*. Same selector, same protocol, same model — different context
length and a much weaker starting point.

Three things must be said about it at once:

1. **The headline +7.5 on the "all" frame is an abstention artifact.** On the 50
   unanswerable rows B − A is +38.0 points (20% → 58%), because refinement drives
   abstention up: 20% → 58% → 88% across A/B/C. Abstention on *answerable* rows
   rises the same way, 12.0% → 30.7% → 72.7%. The honest answerable-only frame
   shows B − A = **−2.67**, still the same sign as the primary suite.
2. **No contrast that matters excludes zero.** D − A = +5.33 with CI
   [−2.67, +13.33]. This is suggestive, not established. The one interval that
   does exclude zero is B − C = +20.0, which again says only that the selector
   beats random at a matched budget.
3. **The absolute level is near-floor and that is genuine, not a scoring
   artifact.** A_full is 32.7% here against 83.8% on the primary suite's
   oracle-eligible frame. Checked: strict exact match 32.7% vs a lenient
   "gold span appears anywhere in the output" 34.7% — a 2-point gap. The errors
   are substantive wrong-entity answers (`survey plaque CI-000-203` where the
   gold is `CI-000-202`), not format mismatches. LCR items carry 3–4 supporting
   facts against the demo suite's 1–2.

**The number that matters most for the programme is 38%.** Condition D hands the
model an exactly localised gold-evidence subset — the best any retriever could
possibly do — and accuracy reaches only 38.0%. Perfect retrieval leaves 62% of
these items wrong. Whatever refinement is worth at this length, it is worth a
few points on top of a ceiling that is itself low. Retrieval is not what is
missing; capability is. That argues *for* Phase 1, and it is the one conclusion
both suites reach by different routes.

**Reading.** At ~1.3K tokens, where the model is strong (66–84%), refinement has
no ceiling to reach. At ~2.4K tokens, where the model is near floor (33%), an
*exactly localised* oracle sits slightly above full context while the cheap
lexical selector sits slightly below it. That is the programme's third
interpretation case — *the model can reason over the correct evidence, but the
cheap selector is failing to expose it* — and it must not be reported as model
failure. It is also not established: the interval spans zero.

**Peak memory** 2.68 GB GPU allocated / 3.17 GB RSS. **Wall-clock** 2844 s for
750 generations — about 8× the primary suite's per-generation cost, from GPU
memory oversubscription while a second experiment held the card.

### Phase 0 conclusion

**Length-dependent, and weaker than the primary suite alone suggests.**

* **At ~1.3K tokens (primary, validity-gated): NOT SUPPORTED**, decisively. B − A
  and D − A are both negative with intervals excluding zero, while B − C = +19.5
  confirms the selector works. An answer-informed oracle could not beat keeping
  the whole document.
* **At ~2.4K tokens (confirmatory, exact oracle): INCONCLUSIVE**, leaning toward
  a small retrievable ceiling the cheap selector cannot reach (D − A = +5.3,
  interval spanning zero; B − A = −2.7).

The claim that survives both suites is narrower than "interference is not the
bottleneck": **a cheap lexical selector never beat full context on answerable
rows on either suite**, and the abstention shift is present and large on both.
Whether a *better* selector could win at longer contexts is open, and the LCR
oracle is the first evidence that it might.

### Superseded statement

An earlier draft of this report concluded a flat **NOT SUPPORTED** from the
primary suite alone. The confirmatory suite does not support that generality.
The conclusion above replaces it.

### Original primary-suite conclusion, for the record

**CONTEXT INTERFERENCE: NOT SUPPORTED** *(on the primary suite only)*.

Query selection is significantly *worse* than full context (−9.0 points overall,
−15.6 on answerable rows) while being significantly *better* than random at a
matched budget (+19.5 points). An answer-informed oracle is also worse than full
context (−14.4 points). Refinement has no ceiling to reach on this checkpoint and
this suite; the bottleneck is not irrelevant context competing for capacity.

Per the programme's own interpretation guide, this is the case that reads as:
*the checkpoint likely requires intrinsic capability improvement.* Phase 1 tests
exactly that.

---

## AMENDMENT 01 — does the answer depend on the checkpoint or the gate?

Follow-up, **not preregistered** (Phase 0's primary result was already in hand).
Prompted by the question "does Supa-A1, the model without the baked gate, behave
differently?"

**Premise correction.** Supa-A1 is *not* an ungated model. Its `config.json`
carries the same `abstain_token_id: 51378` and the same
`abstain_logit_bias: 5.156249985098839`, and its modeling file is byte-identical
to A1.1's. A1 is the **bfloat16 store** of the same recipe. "Different
checkpoint" and "no gate" are therefore two separate axes, and both were run as
a full 2x2 on the primary suite, reusing the cached A/B/C/D contexts so the
inputs are byte-identical across cells.

Gate-off is `abstain_logit_bias = 0.0` on the in-memory config only. Verified
before running, on both cores: toggling it moves the logit of token 51378 by
+5.164 (fp16) / +5.125 (bf16) — 5.15625 rounded in each format — and moves
**every other logit by exactly 0.0**.

### Validity gates

| cell | published | reproduced | matches |
|---|---|---|---|
| A1.1 fp16, gate on | 45 / 125 / 170 | 45 / 125 / 170 | **yes** |
| A1.1 fp16, gate off | 41 / 121 / 162 | 41 / 121 / 162 | **yes** |
| A1 bf16, gate on | 43 / 125 / 168 | 43 / 125 / 168 | **yes** |
| A1 bf16, gate off | *none published* | 40 / 121 / 161 | n/a |

Three of four cells reproduce their published counts exactly.

### Results (primary suite, n = 256)

| cell | A | B | C | D | **B − A answerable** | abstention A/B/C (answerable) |
|---|---:|---:|---:|---:|---:|---|
| A1.1 fp16, gate **on** | 66.4% | 57.4% | 37.9% | 69.4% | **−15.62** [−21.35, −9.90] | 1.0 / 10.9 / 26.0% |
| A1.1 fp16, gate **off** | 63.3% | 55.5% | 31.2% | 70.6% | **−13.02** [−18.23, −7.81] | 0.0 / 5.7 / 13.5% |
| A1 bf16, gate **on** | 65.6% | 57.4% | 37.5% | 69.4% | **−15.10** [−20.83, −9.38] | 1.0 / 10.4 / 25.5% |
| A1 bf16, gate **off** | 62.9% | 54.7% | 30.9% | 70.6% | **−13.02** [−18.23, −7.81] | 0.0 / 5.7 / 13.5% |

### The two axes separate cleanly

**Checkpoint / precision: no effect.** Gated, A1 and A1.1 differ by 0.5 points on
B − A and are *identical* on B (57.4%) and D (69.4%). Ungated they are identical
on every contrast reported here. The delta-rounding difference that justifies
A1.1 superseding A1 — 2.19% of the merged LoRA delta lost in fp16 against 17.25%
in bf16 — is real and its 43-vs-45 Showcase consequence reproduces, but it does
not touch this question.

**Gate: a real mechanism, not the cause.** The gate roughly **doubles the
abstention shift** on answerable rows — 26.0% vs 13.5% under random selection,
10.9% vs 5.7% under query selection. That confirms the +5.15625 `UNKNOWN` bias
is amplifying the mechanism Phase 0 identified. But removing it moves B − A only
from −15.62 to −13.02: **2.6 points of a 15.6-point effect, so roughly 83% of
the loss is not the gate.** D − A barely moves (−14.38 to −13.12). Every
interval still excludes zero.

Interaction (core x gate) on B − A answerable: **−0.52 points**. Negligible.

### What this settles

The primary-suite negative **survives removing the gate entirely and survives
swapping the checkpoint**. It was previously established on one checkpoint under
one gate setting; it now holds across both A-series cores and both gate states,
with all four cells agreeing and all intervals excluding zero.

A side result falls out: the gate's published +8 on all256 (170 vs 162)
decomposes almost entirely into unanswerable rows — 34.4% gated vs 20.3%
ungated — while answerable rows are essentially unchanged (77.1% vs 77.6%). The
gate does exactly what `CORE.json` says it does, and nothing else.

**This does not touch the LCR-dev finding.** That result is at a different length
on a different suite; neither the gate nor the checkpoint explains it, and it
stands unchanged.

**Not independent evidence.** All four cells share the same 256 rows and the same
cached contexts. Their agreement is robustness, not four confirmations.

## Caveats carried forward

* **Length.** "Long context" here means 1048–1481 tokens on the primary suite —
  measured with the model's own tokenizer, not estimated. The confirmatory suite
  below runs at 2112–2554 tokens.
* **Format.** Conditions B, C and D differ from A in two ways at once: text is
  missing, and retained runs are joined with a blank-line gap marker. All
  comparisons *among* B, C and D are format-matched; B − A and D − A are not.
  The conclusion does not rest on them alone — D loses to A while holding
  privileged information, and B beats C at a matched format and budget.
* **Power.** No significance is claimed. This corpus is underpowered by its own
  published analysis (~576 rows would be needed for 80% power at the effects
  seen; there are 256), and Stability192 has been read many times and is no
  longer a clean split.
* **Selector generality.** One frozen lexical configuration was tested, by
  design. A different selector family — or a much larger retained budget — is
  not ruled out by this result; what is ruled out is that *this* checkpoint has
  latent capability waiting behind a shorter context, since the answer-informed
  oracle could not find it either.
