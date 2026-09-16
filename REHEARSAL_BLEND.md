# Spanning two capabilities protected neither. The rule I wrote was wrong.

**RRP-012. Three of four registered predictions falsified, including the one I said
I held most strongly. `DATASET_REQUIREMENTS.md` §2.1 is corrected as a result.**

> **Scope correction, added after `DATASET_SURVEY.md` §5.** The title overstates this
> document's reach. M5 was a **two-component** blend — `reasoning_minus_abstention` +
> `abstention` — covering **4,812 of 10,353 eligible rows (46.5%)**, not the pool. The
> cause is structural: 4,000 rows of the preservation slice (3,402 eligible) carry no
> `capability_group` field at all, so every group-based filter in `REHEARSAL_FILTERS` is
> blind to them.
>
> **The finding below is unaffected** — halving the abstention component from 1,295 rows
> to 648 took abstention from 13/64 to 0/64, which is about dose within a component and
> does not depend on how many components there were. What narrows is the *scope of the
> falsification*: this refutes **additivity across two components**, not "span". By this
> document's own threshold mechanism a wider split should protect *less*, so the rule
> remains unsupported — but it was not tested at span, and saying so is the difference
> between a result and an extrapolation.

RRP-011 found rehearsal protects what it resembles. From that I wrote a rule into
the requirements document — *"a rehearsal mixture must span every capability you
intend to keep"* — and then, per that document's own standard, tested it instead of
leaving it asserted.

It does not survive the test.

## The result

All arms: identical 768 transition rows at seed 17, identical ~108.4k rehearsal
token budget, identical recipe. Only the rehearsal source differs.

| arm | rehearsal | new /384 | panel | ret /47 | **long ctx** | **`UNKNOWN` /64** |
|---|---|---:|---:|---:|---:|---:|
| *Supa* | — | 14 | 47 | 47 | **161** | **13** |
| M3 | reasoning | 273 | 69 | 34 | **144** | 0 |
| M4 | abstention | 267 | **77** | 39 | 137 | **13** |
| **M5** | **half each, disjoint** | 275 | 68 | 36 | **136** | **0** |

The blend was built exactly as registered: **54,217 tokens from each component**,
pools disjoint after amendment 1 removed the `insufficient_information` overlap.

**M5 is worse than both of its own components on both axes it was built to
protect.** Long context 136, below M3's 144 and M4's 137. Abstention **0**, against
M4's 13.

## Predictions versus outcome

1. *"Abstention lands near M4's 13, expect ≥ 8. This is the prediction I hold most
   strongly."* — **Falsified completely. M5 scores 0.**
2. *"Long context ≥ 140, near M3's 144."* — **Falsified. 136**, below both components.
3. *"Panel retention ≥ 37."* — **Falsified, narrowly. 36.**
4. *"New skill stays in the 265–273 band."* — Held (275, marginally above).

I also pre-registered what falsification would look like: *"if M5 lands between M3
and M4 on both axes — roughly abstention ~6, long context ~140 — then blending
dilutes rather than spans."* **The actual result is worse than that prediction
too.** M5 did not land between its components. It landed below both.

## What this actually shows

**Protection is not additive, and it is not linear in dose.**

M4 gave abstention 1,295 rehearsal rows / 108,397 tokens and recovered abstention
completely — 13/64, exactly Supa's number. M5 gave abstention **648 rows / 54,217
tokens**, half as much, and recovered **nothing at all**. Halving the component did
not halve the effect; it erased it.

That is a threshold, not a gradient. Whatever abstention rehearsal is doing, it
needs a critical mass of the rehearsal budget to do it, and at half strength it is
indistinguishable from not being there.

**So a fixed rehearsal budget cannot be spread across capabilities.** The budget is
not a portfolio to diversify; it appears closer to a single choice. Adding a second
component did not buy partial protection of the second capability — it cost the
full protection of the first.

## The corrected rule

`DATASET_REQUIREMENTS.md` §2.1 said: *span every capability you intend to keep.*
**That is wrong at a fixed budget and has been rewritten.** What the evidence
supports:

> Rehearsal protects what it resembles (RRP-011), but protection requires a
> **critical share** of the rehearsal budget, not merely presence in it (RRP-012).
> At a fixed budget you are choosing **which one** capability to protect, not
> distributing protection across several. Decide which loss you least
> can afford, and spend the rehearsal on that.

Whether a **larger** rehearsal budget could protect two capabilities at once is
**untested**. M5 held the budget fixed at 50% of processed tokens precisely so
composition was the only variable. Raising the rehearsal share to, say, 67% and
splitting it would test it, and would need its own registration.

## Limits

Single seed per arm; four arms on one 256-row suite is four comparisons and no
per-arm significance claim is made. M5's panel retention of 36 and M3's 34 both sit
at or just below the D3 seed spread (35–37) and should not be over-read from one
seed. The abstention result does not have that problem: 13 → 0 is the full range of
the instrument, not a marginal move.

**Report-only, as registered.** `QUARANTINE.json` bars the Long-Context Demo for
`[TRAIN, SELECTION, CONFIRMATION]`. M4 remains the most interesting arm measured
and is **not** thereby a recommended configuration; a confirmation needs a fresh
instrument and its own registration.

Long context remains unrepaired in every arm ever measured here — 118 to 144
against Supa's 161 — and the reason was stated before any arm ran: **there is no
long-context training data in this worktree.**

## Reproduce

```bash
bash rineya-dataset-retention-records/run-blend.sh
```

Pre-registration with all four predictions, the explicit falsifying outcome, and
amendment 1 (the component overlap the build guard caught before anything trained)
is in `forge-v2-prereg-001/RRP012_BLEND_PREREG.md`. Artifacts
`PROCESS_rrp012_M5.json`, `RETENTION_PROBE_rrp012_M5.json`,
`LONGCTX_rrp012_M5.json`.
