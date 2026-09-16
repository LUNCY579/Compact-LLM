# The rehearsal pool is three times larger than I have been using, none of it is long, and the retention panel is a subset of it

**A full-file census of all seven untapped corpora — 148,000 rows, every row read, no
prefix sampling. Four findings, two of which correct earlier write-ups.**

Prefix sampling has burned this programme once already (`DATASET_REQUIREMENTS.md` §1.3:
a family-ordered `limit=4000` draw reported 0% fit when the truth was ~23%). Every
number below is a full pass over the file.

## 1. The inventory

| corpus | rows | p50 tok | p95 tok | **max chars** | fits 2048 | single-turn | classes |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3 reasoning | 50,000 | 125 | 198 | 894 | 100% | 22,948 (46%) | 20 |
| v3 integrated reasoning+dialogue | 30,000 | 132 | 202 | 940 | 100% | 13,912 (46%) | 8 |
| v3 natural communication | 30,000 | 122 | 180 | 801 | 100% | 14,104 (47%) | 12 |
| v3 preservation slice | 14,000 | 87 | 124 | 831 | 100% | 10,975 (78%) | 62 |
| short-response combined | 12,000 | 88 | 180 | 938 | 100% | ~88% | 44 |
| short-response new | 10,000 | 87 | 169 | 831 | 100% | ~87% | 25 |
| short-response preservation | 2,000 | 176 | 196 | 938 | 100% | ~95% | 31 |

The three large v3 corpora are **perfectly class-balanced** (every class exactly 2,500 /
3,750 / 2,500 rows) and **interleaved, not family-ordered** — the last previously-unseen
class appears at row 50, 31 and 29 respectively. They are safe to draw from by prefix,
unlike the transition corpora.

## 2. No corpus on this host can rehearse long context. At all.

**The longest row in all 148,000 is 940 characters — roughly 261 tokens. Zero rows
exceed 4,000 characters.**

This is not a marginal shortfall. The long-context suite that every rehearsal arm damaged
(161 to 144 / 137 / 136) scores retrieval over contexts two orders of magnitude longer
than anything available to rehearse with. The standing hypothesis — that long-context
damage could be bought back with long-context rehearsal — **cannot be tested with any
data in this repository**, and no combination, reweighting or filtering of these seven
corpora changes that.

`CONFIRMATION_INSTRUMENT_GAP.md` established there is nothing left to *confirm* on. This
establishes there is nothing available to *rehearse* with. The commissioned LCR corpus
(`docs/LCR_GENERATION_PROMPT_V2_ADDENDUM.md`) is not one lever among several. It is the
only one.

## 3. The 150-row retention panel is a literal subset of v3 reasoning 50k

`experiments/architecture_benchmark_001/data/monitor_probe.jsonl` carries ids of the form
`rineya_v3_sft__percentage_reasoning__reas_percentage_reasoning_s05__01577`. **All 150 of
its ids are present in the v3 reasoning corpus.**

The existing guard handles this exactly: all 150 carry one of the three panel
`reply_class` values, so the unconditional panel exclusion in `rehearsal_pool` removes
every one of them. Verified, not assumed. But it means any future use of v3 reasoning for
*any* purpose must keep that exclusion, and that the guard is load-bearing rather than
decorative.

**Template contact is the subtler problem.** The panel's rows share a generator with the
v3 corpora, so rows of *other* classes reuse its surface forms. Measured as shared 8-grams
over **user turns only** — system prompts are shared boilerplate and were excluded after
they produced a spurious 100% hit rate on the first pass:

| source | rows with >=30% of their user-turn 8-grams inside the panel |
|---|---:|
| v3 reasoning 50k (panel classes already dropped) | 572 / 42,500 (**1.3%**) |
| v3 integrated 30k | 504 / 30,000 (**1.7%**) |
| v3 natural comm 30k | 757 / 30,000 (**2.5%**) |
| **v3 preservation slice** | **0 / 13,250 (0.0%)** |

The preservation slice has **literally zero** 8-gram contact with the panel. That is why
it was the right rehearsal source, and it was a better choice than I knew at the time.

**This is fixable and cheap.** A strict *zero-8-gram-contact* exclusion — drop any
rehearsal row sharing even one 8-word span with any panel row — costs 5,000–7,400 rows
and still leaves ample pool:

| corpus | clean pool | rows the D1 budget consumes | headroom | compile rejections |
|---|---:|---:|---:|---:|
| v3 integrated 30k | 8,637 | 519 (54,269 tok, 8 classes) | 16.6x | **0** |
| v3 natural comm 30k | 9,132 | 546 (54,230 tok, 12 classes) | 16.7x | **0** |
| v3 reasoning 50k | 11,995 | 530 (54,214 tok, 17 classes) | 22.6x | **0** |
| v3 preservation slice *(baseline)* | 10,353 | 630 (54,159 tok, 53 classes) | 16.4x | **0** |

Every row compiles through the already-qualified `compile_row` path with **zero
rejections**, so no second serialization mode is introduced — the property RRP-009 relied
on holds for these sources too.

## 4. 29,764 clean, balanced rehearsal rows have never been touched

Every rehearsal arm in RRP-009 through RRP-012 drew from **one** source: the 14,000-row
preservation slice, 10,353 rows eligible. The three large v3 corpora are disjoint from it
— **zero id overlap**, verified row-by-row — and together supply **29,764 zero-contact
rows across 37 classes**, 2.9x the pool every prior arm used.

The trade is breadth of *class* against breadth of *source*: the preservation slice has
more classes (53) than the combined v3 pool (37), but the v3 pool has 2.9x the rows,
uniform class balance, and three distinct capability groups rather than one mixed slice.

**v3 integrated is the cleanest single source in the repository**: zero panel-class rows
and zero abstention-class rows, so the panel-exclusion rule is satisfied vacuously and it
cannot confound the abstention endpoint.

## 5. Correction: RRP-012's blend spanned 46.5% of the pool, not "everything"

`REHEARSAL_BLEND.md` reports M5 as testing the rule *"a rehearsal mixture must span every
capability you intend to keep."* M5 was a **two-component** blend —
`reasoning_minus_abstention` + `abstention` — covering **4,812 of 10,353 eligible rows
(46.5%)**.

There is a structural reason this was easy to miss. **4,000 rows of the preservation slice
(3,402 eligible) carry no `capability_group` field at all.** Every `capability_group`-based
filter in `REHEARSAL_FILTERS` is blind to them. They are not junk — they are 21 distinct
classes, disproportionately the short-response shapes (`factual_compression` 400,
`clarification` 280, `safety_boundary` 260, `context_recall` 260), and 860 are
abstention-class.

| selector | eligible rows | classes | share of pool |
|---|---:|---:|---:|
| `all` | 10,353 | 53 | 100% |
| `reasoning_minus_abstention` + `abstention` (**M5, as run**) | 4,812 | 23 | **46.5%** |
| `communication` + `integrated` + `reasoning_minus_abstention` | 6,745 | 33 | 65.2% |
| the above **+ ungrouped** | 9,287 | 48 | **89.7%** |

**What this changes and what it does not.** RRP-012's *finding* stands untouched: halving
the abstention component from 1,295 rows to 648 took abstention recovery from 13/64 to
**0/64** — a threshold, not a gradient. That is about dose within a component and does not
depend on the number of components.

What must be narrowed is the *scope of the falsification*. RRP-012 falsified **additivity
across two components**. It did not test a genuine span, and by its own threshold
mechanism a four-way even split — a quarter of the budget each — should protect *less*,
not more. The corrected reading: the rule "span every capability" is unsupported and the
mechanism argues against it, but it was refuted at n=2 components, not at span.
`REHEARSAL_BLEND.md` overstates its reach and is annotated accordingly.

## 6. There are 53,000 preference pairs and no code that can use them

Three preference corpora exist. This programme has never had DPO, preference or
process-supervision code — that absence is what made the earlier reasoning nulls
uninformative — so none of these has ever been touched.

| corpus | pairs | classes | chosen longer than rejected | median length delta |
|---|---:|---:|---:|---:|
| **v3 preferences** | 16,000 | 40 (balanced, 400 each) | 49% | **0** |
| short-response preferences | 5,000 | 1 | 23% | −14 |
| forge RL v2 `train_pairs` | 32,000 | 1 (`synthetic_preference`) | 8% | −6 |

**Only the v3 set is length-unbiased.** A preference corpus where the chosen answer is
reliably shorter or longer teaches length, not quality — a reward signal trained on the
forge pairs (chosen shorter in 92% of pairs, p90 delta = 0) would largely learn "be
terse." The v3 set's median delta of exactly 0 with a 49/51 split is the signature of
pairs that differ on content rather than length, and its 40 classes are balanced at 400
pairs each.

All three share a clean `{prompt, chosen, rejected, metadata}` schema, and no pair has
`chosen == rejected`.

**This is an option, not a plan.** Using it needs a preference-training implementation
that does not exist, and `CONFIRMATION_INSTRUMENT_GAP.md` means the result could be
measured but not confirmed. It is recorded here as a characterised, costed lever for the
operator to choose, not started.

## 7. What this licenses next

1. **Rehearsal source is the untested variable, and it is not length.** The v3 pool
   differs from the slice in scale and balance, not length. Any arm built on it tests
   breadth, and cannot test length.
2. **Long context is not addressable here.** Every rehearsal arm cost it significantly;
   nothing in this survey offers a way to buy it back. Arms must be registered expecting
   to pay it, not hoping to avoid it.
3. **Any v3-sourced rehearsal must use the zero-8-gram-contact exclusion**, not the
   reply_class exclusion alone. It is affordable (16–23x headroom) and it is what keeps a
   panel improvement from being partly practice on the panel's own surface forms.
4. **No promotion regardless.** `CONFIRMATION_INSTRUMENT_GAP.md` stands.

## Reproduce

Full-file censuses, no sampling:
`rineya-dataset-retention-records/dataset-survey-001/{corpus_survey,census,gap,elig,len,leak2,prov,clean_pool,yield,dpo}.py`
(outside the repository, per `docs/LANE_INTERLEAVE_HAZARD.md`).
