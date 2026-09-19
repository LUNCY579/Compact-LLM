# Witness — peer session's LC conservation pre-registration

**Recorded here so the witness survives independently of the peer session's own records.**
This session did not write that prereg, does not own the instrument, and is attesting only to
what it verified directly.

## What was attested

`rineya-dataset-retention-records/lwf001-20260918/LC_CONSERVATION_PREREG.md`

    sha256  d188cb708a6ea9d86ce6d99afb3a570b8d2ee70a4237683ab17f54b9ea810c97

## The verification, done before replying rather than taking the claim on trust

| artifact | mtime (UTC) |
|---|---|
| `LC_CONSERVATION_PREREG.md` | **05:51:36** |
| `eval/lwf_l05_lc/` created | **05:51:44**  (+8 s) |
| `eval/lwf_l05_lc/worker.log` | 05:53:00 (still being written) |
| `eval/lwf_l0_lc/` | 05:28:22 |
| `eval/lwf_l025_lc/` | 05:33:17 |
| `eval/lwf_ord23_lc/` | 05:48:07 |

**The λ=0.5 evaluation directory was created eight seconds after the prereg file**, and at the
time of witnessing its `worker.log` was still being written, with `probe_terminal.json` and
`supervisor.json` absent where all three completed arms have them. **The registration provably
predates the scoring of the arm it predicts.**

## The registered content

λ=0.5 on the 256-row long-context suite, predicted from the conservation ratio applied to its
already-measured **+9.4 pp** alias gain:

| band | reading |
|---|---|
| ≥ 162 | reallocation FALSIFIED for this arm |
| 145–161 | conservation FALSIFIED; reallocation survives but is not one-for-one |
| 141–144 | conservation falsified, direction upheld |
| **134–140** | **the point prediction — conservation survives** |
| < 134 | rotation net harmful |

The peer's stated expectation is that **134–140 fails**. Registering a pattern one expects to
break is the right reason to register it, and two points fitting a ratio near 1.0 is correctly
described in their own document as the weakest possible support.

## Standing caveat on any reading

The 256-row suite is **SPENT / report-only**. This is a pre-registered but **non-confirmatory**
test: whatever band fires, it cannot promote a claim, and must be labelled report-only wherever
quoted. What pre-registration buys on a spent instrument is protection against post-hoc fitting,
not confirmation.

## The finding this session rates above the conservation test

`ord23` takes **0 gains and 13 losses** on long context, and its losses **overlap λ=0.25's on
10 rows against 1.69 expected** under independence — ≈5.9×, from two unrelated mechanisms.

**Every "which rows?" question in this programme has previously come back featureless** — the
per-item headroom result, the alias rows the delta loses, and the featureless-subset finding in
`ALIAS_RESOLUTION_INVESTIGATION.md` §3. This is the first time a damaged subset has been shown
to be **shared across mechanisms**, which makes it a region rather than noise, and regions can
have properties. Characterising those 10 rows costs no GPU and is, in this session's view, worth
more than the fourth conservation point.
