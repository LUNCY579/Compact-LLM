# WSD is worth about 8 panel rows, and it replicates

Every B28 arm in this programme inherited `cosine:0.03`, whose warmup ends after roughly 1.2
optimizer steps -- the model takes near-peak learning rate on its first step. Replacing it with
`wsd:0.2:0.2` (Warmup-Stable-Decay, 20% warmup, 20% 1-sqrt decay) and beta2 0.95 was worth +21
panel rows to H56. No B28 arm had ever run it.

Run on B28last20 at four seeds, 5,200-token budget, lr 1e-5. Same seed means the same data draw
and the same row order, so a per-seed pair differs **only** in the schedule.

    seed    cosine   wsd   delta    paired McNemar
      17        67    74      +7    3 vs 10, 13 discordant, p=0.0923
      23        58    62      +4    6 vs 10, 16 discordant, p=0.4545
      29        60    73     +13    6 vs 19, 25 discordant, p=0.0146
      mean    61.8  69.7    +7.9

**Pooled across the three seeds: 15 vs 39 discordant, p = 0.001496.**

Pooling is the right statistic here. The seeds are independent data draws -- seeds 17 and 23
select training rows with zero overlap -- so their discordant rows are disjoint and combine
cleanly. The per-seed sign test cannot beat p=0.25 at n=3 however consistent the direction is,
and two of the three individual pairs sit at or inside the 15-24 row seed-noise floor. Only the
pooled test has the power to decide, and 54 discordant rows is well clear of that floor.

## Why this one is trustworthy when today's other claims were not

Most single-seed results in this programme have not survived replication. Two dose claims were
refuted within two hours of each other, in opposite directions. What separates this one:

* the direction is consistent at 3 of 3 seeds;
* each comparison is **paired at fixed seed**, so seed-to-seed variance -- which spans 9 rows on
  this arm and dominates most effects measured here -- is removed by construction rather than
  averaged over;
* the pooled discordance is 54 rows against a noise floor of 15-24;
* the mechanism is understood and was predicted before measurement: a 1.2-step warmup takes
  near-peak lr on step one, and fixing it helps most where the model is most fragile early.

## What it does not establish

The effect size is uncertain. Per-seed deltas run +4, +7, +13, so "about 8 rows" is a mean over
three draws, not a reliable per-run expectation. Nothing here says WSD helps other arms, other
budgets, or other learning rates; H56 gained +21 under it and P56 only +4, so arm sensitivity is
already known to vary widely. And it is measured on one 150-row arithmetic panel -- the forge
probe shows WSD does not fix the verbosity regression (186/200 termination against cosine's
183), so this is a panel-capability improvement, not a general one.

## Best measured model

`B28last20 + wsd:0.2:0.2 + beta2 0.95`, 5,200 tokens, seed 17: **74/150**, kept 41 of 47,
gained 33, lost 6, at 314,618,880 trainable parameters -- 53% of full B28's, with embeddings,
final norm and the first 8 blocks frozen. Checkpoint saved.

At 3x dose the same recipe gives 64/150, and 74 vs 64 is 26 discordant rows at p=0.076 --
not significant. Whether 5,200 or 15,600 tokens is better is undetermined at one seed per
budget.
