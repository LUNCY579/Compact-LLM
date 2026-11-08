# The first non-featureless subset — and the feature is confounded by the bank's design

**REPORT-ONLY.** The 256-row Long-Context Demonstration Suite is **SPENT**, so nothing here may
promote a claim. Zero model calls: every input already existed. Script
`analysis/lc_loss_region.py`, receipt `LC_LOSS_REGION.json`.

## Why this was worth doing

**Every "which rows?" question in this programme has come back featureless** — per-item headroom
ceilinged at the noise floor; the alias rows the delta loses carried no distinguishing feature
(`ALIAS_RESOLUTION_INVESTIGATION.md` §3); the subgroup analyses found nothing. Three independent
attempts, three nulls.

Then the peer session found the two rotated arms lose the **same rows**. Reproduced here from
their per-row evidence: baseline λ=0 is correct on **162/256**; λ=0.25 loses **21** of those and
`ord23` loses **13**; the overlap is **10 against 1.69 expected** under independence — **5.93×**
— from an anchor loss and a row reshuffle, two mechanisms with nothing in common.

**A shared region is not noise, and regions can have properties.**

## What survives multiplicity correction

23 tests across six stratifications, Holm-corrected:

| test | k/n | raw p | **Holm p** |
|---|---:|---:|---:|
| `needle_profile = early` | 7/23 | <0.0001 | **<0.0001** |
| `task_type = count_with_exclusions` | 6/22 | 0.0005 | **0.0110** |
| `variant = paraphrase` | 7/51 | 0.0116 | 0.2436 |
| `difficulty = hard` | 9/91 | 0.0437 | 0.8740 |
| everything else | — | — | 1.0000 |

Two survive. **But they are not two features.**

## The confound, which is in the bank's construction and not in the result

| | |
|---|---|
| all **32** `count_with_exclusions` rows | have `needle_profile = early` |
| all **64** `early` rows | are `alias_resolution` (32) or `count_with_exclusions` (32) |

`count_with_exclusions` is **perfectly nested inside** `early`. And among the rows the baseline
gets right, `early` is *almost entirely* `count_with_exclusions`, because the baseline scores
**1/32** on the `alias_resolution` half. So of 23 baseline-correct early rows, ~22 are
`count_with_exclusions` and 6 are lost.

**The two surviving tests are the same test.** `paraphrase` and `hard` are confounded the same
way — six of the seven paraphrase losses *are* the count rows.

The ten rows, in full:

| id | needle | task_type | variant | difficulty |
|---|---|---|---|---|
| RLC-013-P, 021-P, 029-P, 037-P, 045-P, 053-P | early | **count_with_exclusions** | paraphrase | hard |
| RLC-057-B | early | alias_resolution | base | hard |
| RLC-006-U | middle | causal_disambiguation | unanswerable | hard |
| RLC-035-P | late | temporal_correction | paraphrase | medium |
| RLC-052-C | distributed | rule_exception | counterfactual | hard |

## What can and cannot be said

**Can:** the shared loss region is real and strongly concentrated — **six of ten rows are one
task type**, against a baseline-correct population where that type is 22 of 162.

**Cannot:** whether the operative feature is the **task** (`count_with_exclusions`) or the
**position** (`early` needle). **This bank cannot separate them**, because it never crosses them.
Any claim naming one is naming a design coincidence.

**The discriminating build is cheap and obvious:** `count_with_exclusions` items at
`middle`/`late`/`distributed` needle profiles, and some other task type at `early`. That is a
bank change, not a training run.

## One thing this bears on immediately

If both rotated arms lose *mostly the same one stratum*, then a loss/gain ratio computed over
all 256 rows is substantially **measuring how much of that one stratum each arm breaks**. The
peer session's conservation bands (`LC_CONSERVATION_PREREG.md`, witnessed in
`WITNESS_LC_CONSERVATION.md`) are pre-registered and stand as registered — but a conserved ratio
across arms that break the same stratum is a weaker fact than a conserved ratio across arms that
break different things, and the λ=0.5 reading should say which it is.

## The alias sub-stratum, reported and not leaned on

This suite carries its **own** `alias_resolution` stratum, n = 32 — an independent fourth bank
for the task ALIAS-001 measures. It is **below gate**: λ=0 scores **1/32**, so there is nothing
to preserve and nothing readable. Both rotated arms lose that single row. **n = 1; no direction
is claimed from it**, and it is recorded only so that nobody later mistakes it for a replication
attempt that failed.
