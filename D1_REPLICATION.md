# D1 replicated: it holds long context in two draws of three, and fails in the third

**Three selection seeds, everything else fixed. Gate: 2/3 pass. The seed-17 result
was the best of three, not the typical one.**

Phase 1 found that D1 — the rung the *pre-registered* gate already recommended —
appeared to hold long context at 160/256 against Supa's 161, p = 1.0000. That was
one seed. This is three.

## The three draws

| seed | new skill /384 | panel | retained /47 | **gate** | long ctx /256 | vs Supa (paired) | `UNKNOWN` /64 |
|---|---:|---:|---:|:--:|---:|---|---:|
| 17 | 239 | 77 | 40 | **PASS** | **160** | −1, p = 1.0000 n.s. | 8 |
| 23 | 217 | 71 | 38 | **PASS** | **159** | −2, p = 0.8450 n.s. | 8 |
| 29 | 236 | 67 | **36** | **FAIL** | **148** | **−13, p = 0.0351 SIG** | 5 |
| *Supa* | *14* | *47* | *47* | — | *161* | — | *13* |

Every seed clears all three transition floors. Spread: new skill 217–239, panel
67–77, retained 36–40, long context 148–160, abstention 5–8.

## What replicates and what does not

**Replicates.** Acquisition — every seed clears every floor. Panel improvement —
67–77 against the source's 47, never below the ≥62 threshold. Abstention damage —
5–8 against Supa's 13, consistently degraded, never repaired.

**Does not replicate.** The two things that made seed 17 look like a free lunch:

- **Retention.** Seed 29 retains **36 of 47**, one below the gate. The
  pre-registered rule is explicit — *a failing seed is a failure and a majority does
  not rescue it* — so **D1 is 2/3, exactly as D3 was.**
- **Long context.** Seeds 17 and 23 are statistically indistinguishable from the
  untrained source (p = 1.0000, p = 0.8450). Seed 29 is a **significant loss**
  (148, p = 0.0351).

## The honest reading

**D1 is not a reliably strictly-better configuration.** It is a configuration whose
*typical* behaviour is much better than anything else measured, and whose worst draw
still fails.

That distinction matters and is worth stating precisely, because D1 remains the best
arm this programme has produced:

- No D3 seed was ever statistically indistinguishable from Supa on long context;
  **two of three D1 seeds are.**
- D1's worst long-context draw (148) is comparable to D3's single measured draw
  (150), and its best two (160, 159) are 10 rows better.
- D1 does this on **a quarter of D3's training data**.

So the dose finding survives: less data preserves more. What does not survive is the
claim that it preserves *enough, reliably*.

**This is the second time in this programme that a first seed flattered.** RRP-007
seed 17 passed retention by exactly zero margin and looked like a clean win until
seed 23 failed. D1 seed 17 looked like a free lunch until seed 29 failed. The
pattern is now well enough established to treat any single-seed result here as an
upper bound rather than an estimate.

## What this does not license

No promotion, no release. Beyond the 2/3 gate result, the instrument problem stands:
`CONFIRMATION_INSTRUMENT_GAP.md` establishes that every long-context and
literal-`UNKNOWN` abstention instrument in this repository is spent, reserved, or
locked. The long-context numbers above are **report-only** by registration, and D1
was selected by the pre-registered panel/reference gate *before* any long-context
measurement of it existed — that ordering is what keeps this from being selection on
a barred suite.

**Abstention is unrepaired in every D1 seed** (5–8 against 13). The only arm that
repaired it, M4, pays a *significant* long-context loss (137, p = 0.0003). Nothing
measured here gets both.

## What would change the picture

1. **More seeds.** Three is the registered minimum; 2/3 with one narrow miss (36 vs
   37) is not the same as 2/3 with a collapse. More draws would show whether seed 29
   is a tail or a coin flip.
2. **A dose between D1 and nothing.** The frontier was sampled at 384 / 768 / 1,536 /
   3,072 rows. Nothing below 384 has been measured, and D1's failure mode is
   retention, which improves as dose falls.
3. **Long-context rehearsal** — still the only intervention that would address the
   cause rather than the dose, and still blocked on a corpus that does not exist.

## Reproduce

```bash
bash rineya-dataset-retention-records/run-d1-replication.sh
```

Artifacts `PROCESS_d1r_s{23,29}.json`, `RETENTION_PROBE_d1r_s{23,29}.json`,
`LONGCTX_d1r_s{23,29}.json`; seed 17 is `PROCESS_rrp008_D1.json`,
`RETENTION_PROBE_rrp008_D1.json`, `LONGCTX_lc_D1.json`. Paired comparisons are exact
two-sided McNemar over the identical 256 rows against `LONGCTX_supa.json` at the same
implementation digest.
