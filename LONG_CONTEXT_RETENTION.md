# Opallo did not lose long-context retrieval. It lost abstention and terse formatting.

**L1 verdict on the pre-registered test: significant, Opallo lower.** 161/256 → 135/256,
paired McNemar p = 1.06e-4. Taken alone that reads as a broken long-context core.

Taken alone it would also be wrong, and the decomposition matters more than the headline.

---

## The measurement

Both models run here, in one session, under one runtime, because the published 161/256 was
produced under Transformers 5.14.1 and this host pins 4.51.3. Greedy, 24 new tokens,
truncation forbidden, no row skipped.

Supa measured here: **161/256, showcase 41/64, stability 120/192.** Published: 161/256,
40/64, 121/192. The harness path reproduces the vault's frozen evidence exactly on the
total and within one row on each split, so the comparison below is on a validated
instrument rather than an improvised one.

## Decomposition

| scorer | Supa | Opallo | paired p |
|---|---:|---:|---:|
| strict first-line exact | 161 | **135** | **1.06e-4** |
| containment-tolerant | 162 | **152** | 0.087 — n.s. |
| tolerant, **answerable rows only** (192) | 149 | **150** | 1.00 |
| **unanswerable rows** (64) | **13** | **2** | both scorers |

**On answerable long-context questions Opallo has lost nothing.** Under a scorer that
accepts the right answer wrapped in a sentence, it scores 150 against Supa's 149, with 8
rows going each way. The entire strict-scorer loss on those 192 rows is presentation.

The tolerant scorer was audited before being believed: of the 17 rows it rescues for
Opallo, **zero** have a gold answer short enough for spurious substring matching, and every
one is the same pattern —

```
gold   : "a power relay that failed its second diagnostic"
Supa   : "power relay that failed its second diagnostic"     correct
Opallo : "The power relay that failed its second diagnostic." wrong
```

A leading article and a full stop. That single pattern is most of the
`causal_disambiguation` collapse from 31 to 7.

## What is genuinely lost

**Abstention.** On the 64 unanswerable rows, Supa says `UNKNOWN` 13 times and Opallo
**twice** — identical under both scorers, so this is not presentation. Opallo invents an
answer where the context does not contain one:

```
RLC-011-U  gold UNKNOWN   Supa "UNKNOWN"   Opallo "Saturday."
RLC-014-U  gold UNKNOWN   Supa "UNKNOWN"   Opallo "The current evidence does not identify the cause."
```

The second is the more interesting failure: Opallo *understands* that the answer is absent
and says so in prose, but no longer emits the token the protocol scores. Even the abstention
loss is partly a formatting loss — though 11 rows of it are the model answering anyway.

## Output length

245 of 256 outputs are longer, 2 shorter; mean generated tokens 2.95 → 4.30, sign test
p = 2.7e-70. Termination is unaffected: 256/256 for both.

**This partially revises the previous session's finding.** It concluded the verbosity shift
was "cosmetic" because on the Reasoning Forge the correct answer appeared in **zero** of 170
wrong outputs. That holds for forge and does not generalise: here the correct answer is
present in 17 of Opallo's wrong outputs. Cosmetic for *content* — the model still knows the
answer — but not cosmetic for *score*, and every instrument in this programme scores by
strict exact match. A shift that costs 26 rows on the family's flagship benchmark is not
cosmetic in any sense that matters.

## What this changes

The honest description of Opallo gains a real limitation and loses an imagined one.

* **Not true:** "Opallo broke Supa's long-context core." Retrieval over ~1,000-word
  contexts is intact — 150 vs 149 on answerable rows.
* **True:** Opallo is **26 rows worse on the Demo suite as that suite scores it**, which is
  how the family roster compares cores, and that is now disclosed in its README and
  `CORE.json`.
* **True and actionable:** the damage is two named behaviours, each with a corpus in this
  repository built for it.

| loss | rows | corpus that targets it |
|---|---:|---|
| terse answer formatting | 17 | `Rineya_Short_Response_Behavior_Corpus_v1.0` — 10,000 rows, median reply 33 chars |
| abstention / `UNKNOWN` | 11 | `Rineya_Abstention_Gate_Lab_v1`; Supa-A1 already ships a baked-abstention core |

That is a far better answer to "make up its weak abilities" than the reasoning work was
going to be. Reasoning is falsified on two instruments; **these two are measured, localized,
and have training data sitting in the repo.**

## Reproduce

```bash
PYG=C:/Users/clclc/BusinessProject/.venv-rab001-cu126/Scripts/python.exe
$PYG experiments/rab001_exploration/long_context_probe.py --model Supa --label supa
$PYG experiments/rab001_exploration/long_context_probe.py \
     --model Rineya_Published_Cores/Rineya-Opallo-1.0.0-devfreeze --label opallo
```

Artifacts `LONGCTX_supa.json`, `LONGCTX_opallo.json`, each carrying every prediction.
Pre-registration in `LONG_CONTEXT_RETENTION_PREREG.md`; the strata above were declared
reportable-but-not-tested there, and are reported without p-values for that reason. The two
scorers were **not** pre-registered — the tolerant one was added after seeing that 245 of
256 outputs grew — so the strict result is the pre-registered one and the decomposition is
explicitly post-hoc. It is reported as a mechanism, not as a second significance claim.
