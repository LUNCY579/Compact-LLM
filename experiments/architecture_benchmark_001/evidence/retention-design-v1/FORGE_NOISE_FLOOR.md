# Forge's noise floor is half the panel's, and that settles what the nulls mean

Four WSD checkpoints of the **same configuration** -- B28last20, lr 1e-5, 5,200 tokens,
`wsd:0.2:0.2`, beta2 0.95 -- differing only in the seed, probed on the identical 200 forge rows.
Pure noise by construction.

    seed     exact       terminated
     17     30/200         186/200
     23     27/200         191/200
     29     32/200         187/200
     41     30/200         190/200
    spread       5               5
    sd         2.1             2.4

Same-config seed-pair discordance:

    pair        accuracy            termination
    s17~s23     5 vs 2, disc  7     3 vs 8, disc 11
    s17~s29     3 vs 5, disc  8     5 vs 6, disc 11
    s17~s41     6 vs 6, disc 12     1 vs 5, disc  6
    s23~s29     2 vs 7, disc  9     5 vs 1, disc  6
    s23~s41     4 vs 7, disc 11     4 vs 3, disc  7
    s29~s41     5 vs 3, disc  8     2 vs 5, disc  7

    ACCURACY floor    7-12 discordant, mean 9.2
    TERMINATION floor 6-11 discordant, mean 8.0

Against the panel's 15-24 (mean 18.8) on 150 rows, **forge is about twice as sensitive** while
measuring a harder task. The likely reason is scoring: forge's format confound is one row against
the panel's seventeen, so less of its variance is presentation artifact rather than capability.

## What this settles

**"No capability was lost" is established, not underpowered.** The accuracy comparisons against
the source run 15, 16 and 20 discordant rows -- *above* the 7-12 floor -- and stay null
(p = 0.302, 0.804, 0.824). Discordance clearing the noise band while the test remains null is
genuine equivalence. This is the same distinction that forced the withdrawal of the
`B28last20 ~ B28 full` efficiency claim on the panel, where 22 discordant rows against a floor of
18.8 meant "underpowered" rather than "equivalent". Here it lands the other way, and it lands
that way five times.

**Cosine's termination loss is real.** 16 and 15 discordant, both above the 6-11 floor, at
p = 3e-5 and 6e-5.

**WSD's termination loss is marginal.** 13, 12, and seed 41's **9 -- inside the floor**. That
model may carry no termination regression at all.

**"WSD reduces the verbosity cost" remains unestablished.** Cosine loses 15-16 rows and WSD 9-13;
the 3-7 row difference sits inside the 6-11 band. Only seed 17 has a matched cosine/WSD forge
pair anyway, because cosine seeds 23/29/41 ran under `SAVE='none'` and left no checkpoints.

## Consequence for how forge gets used

Forge accuracy differences of **12 or fewer discordant rows are unread**; 15 and above are
interpretable. That is a usable threshold, unlike the panel's, where the floor swallows most
effects this programme has chased -- `Hlast4` vs `B28last8` was 4 discordant rows, and a
same-config panel pair once reached p = 0.0352 on seed alone.

At 200 rows forge already beats the panel. It has 565 rows usable under a 2,048-token cap and 800
in total, so the floor can be tightened further if an effect ever needs it.
