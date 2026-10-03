# Superseded runs — retained as evidence of two measurement bugs

These are NOT results. They are kept because they are the evidence for two bugs found in my
own harness, and deleting them would erase how the bugs were detected.

## Bug 1 — the seed did nothing, so "3 seeds" would have been one run reported three times

`RIGOROUS_source_control_seed17_lr1e-05_s40.json` and `...seed29...` are **bit-identical**:

    step  0   dev CE 5.524462   task 38/100     (both seeds)
    step 20   dev CE 2.575193   task 40/100     (both seeds)
    step 40   dev CE 1.757339   task 43/100     (both seeds)

Training trajectories match to nine decimals. Nothing in the run consumed the seed: the data
order was `train[(step-1) % len(train)]`, weights load from fixed checkpoints, the model has
no dropout, and decoding is greedy. Reporting these as a replicated multi-seed result would
have been a fabricated claim of replication.

Fixed by making the seed shuffle TRAIN, so each seed trains on a different subset. Measured
overlap of the 40 rows actually used: 1/40, 3/40 and 1/40 across seeds 17/29/43.

## Bug 2 — leading-integer scoring inflated the task score

Task hits compared only the first integer. 31 of 50 `time_duration` answers are clock times
like `10:25 AM.`, so `10:55 AM` scored as correct on the hour alone; decimals collapsed the
same way (`62.5` vs `62.9` both read as `62`). That is ~21% of the task set scored too
leniently.

Fixed by comparing the normalised string or the FULL sequence of numbers, which keeps the
format tolerance that matters (`128 units.` still matches `128.`) while rejecting a different
value. Unit-checked against 10 cases including the clock-time and decimal traps.

## Also here

`...seed17_lr1e-05_s2.json` is the 2-step smoke test that caught a third problem — the frozen
source scores 0/20 on `arithmetic_chain`, making it useless as a retention probe.

Every number in these files is superseded by the re-run under
`outputs/rigorous/` with both fixes applied.
