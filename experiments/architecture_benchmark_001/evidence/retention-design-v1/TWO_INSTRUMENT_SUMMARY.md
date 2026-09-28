# Where the programme stands on two instruments

Two benchmarks, deliberately different. The **panel** is 150 arithmetic word problems with a
typed grammar scorer; the frozen source gets 47. **Forge** (`rineya_reasoning_forge_v1`) is 16
procedurally generated reasoning families -- deduction, scheduling, relational, truth puzzles --
with an independent solver verifying every answer; the source gets 32/200 and terminates on
199/200. Forge shares zero panel stems verbatim or digit-blinded, and its format confound is one
row against the panel's seventeen.

## Panel: WSD is worth about 7.5 rows and it replicates

B28last20, 5,200 tokens, lr 1e-5, four seeds. Each pair is at fixed seed, so only the schedule
differs.

    seed   cosine   wsd   delta   paired
      17       67    74      +7   3 vs 10, disc 13, p=0.0923
      23       58    62      +4   6 vs 10, disc 16, p=0.4545
      29       60    73     +13   6 vs 19, disc 25, p=0.0146
      41       62    68      +6   3 vs  9, disc 12, p=0.1460
    mean    61.75 69.25   +7.50

**Pooled: 18 vs 48 discordant, p = 0.0002871.** Only one of four individual pairs is
significant, because each sits at or inside the 15-24 row seed-noise floor. Pooling independent
draws is what gives the test power, and it is the reason to run four seeds rather than argue
about one.

Retention does not pay for it: kept goes [40,39,39,39] to [41,37,42,39].

## Forge: no capability was lost, by any model

    model                        exact    term    acc p     term p
    source                      32/200  199/200      -          -
    B28last20 cosine s17        29/200  183/200   0.508    0.00003
    B28 full cosine s17         27/200  184/200   0.302    0.00006
    B28last20 wsd s17           30/200  186/200   0.804    0.00024
    B28last20 wsd s29           32/200  187/200   1.000    0.00049
    B28last20 wsd s41           30/200  190/200   0.824    0.00391

**Accuracy is null for all five trained models**, p = 0.30 to 1.00. The seed-29 WSD model scores
*exactly* the source's 32/200 on forge while scoring +26 on the panel. Nothing here traded
reasoning for panel performance.

**Termination is lost by all five**, and WSD loses less: 15-16 rows under cosine against 13, 12
and 9 under WSD, with p weakening from 3e-5 to 3.9e-3. That is suggestive, not established --
only seed 17 has a matched cosine/WSD forge pair (183 vs 186), because cosine seeds 23/29/41 ran
under `SAVE='none'` in the noise-floor batch and left no checkpoints to probe.

`VERBOSITY_SHIFT.md` establishes what the termination loss is: a style shift, not a broken stop
mechanism. Roughly 160 of 200 outputs get longer against 30 shorter (p ~ 1e-22 in three models),
and on the rows that stop terminating the source was correct on **zero** of them -- which is
exactly why accuracy stays null while termination moves sharply.

## Best measured model

`B28last20 + wsd:0.2:0.2 + beta2 0.95`, 5,200 tokens, seed 17:

* panel **74/150** (source 47), kept 41 of 47, gained 33, lost 6;
* forge **30/200** against the source's 32 -- no measurable loss (p = 0.804);
* 314,618,880 trainable parameters, **53% of full B28's**, with embeddings, final norm and the
  first 8 blocks frozen;
* beats full-parameter finetuning's 73/150 at roughly half the trainable weights.

Expected panel score across seeds is about 69, not 74; 74 is the best of four draws.

## What is NOT established

* That 5,200 tokens is the right budget. At 15,600 the same recipe gives 64/150, and 74 vs 64 is
  26 discordant rows at p=0.076 -- not significant. Undetermined at one seed per budget.
* That WSD reduces the verbosity cost. One matched pair only.
* Forge's own seed-noise floor, which is unmeasured. Forge accuracy differences under roughly 15
  discordant rows should stay unread. The nulls above are consistent with noise rather than
  demonstrated equivalence -- though five independent nulls pointing the same way is meaningful.
* Anything about CED. No architecture arm beats ordinary finetuning of existing weights at
  matched budget; the cross-position read is significantly harmful (p=0.0107 under WSD).
