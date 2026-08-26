# Cornerstone-1 x Cornerstone-2 pipeline plan

Goal, in the user's terms: **maintain response quality while compressing memory
as much as possible**, using significance and pooling together.

That is a *frontier* problem, not a single-point problem. The confirmed result
gives us exactly one point on it.

## 0. Where we actually stand

Confirmed on held-out stability-192 against a preregistered plan (`N-071`, `N-072`):

```text
B0 full cache      86/192 (44.8%)   155.1 MB
POOL13_SINK48      86/192 (44.8%)    45.4 MB    0.293x bytes, 0.770x wall
POOL1 = shipped    59/192 (30.7%)    45.4 MB
                   cluster p = 0.00088 vs shipped;  p = 1.0 vs full cache
```

One operating point: budget 256, pool 13, sinks 48, recent 64. Quality fully
preserved at under a third of the bytes.

**Open blocker.** `POOL13_SINK48` still ranks by `S/max(C,1)` beneath the
pooling, and the program has shown the raw score is inert (`N-046`, `N-054`).
[`halo_stability192_pooling_specificity_v1`](results/runs/halo_stability192_pooling_specificity_v1)
is running now to settle it, with a plan fixed before inference.

The outcome routes everything downstream:

| Control outcome | What the pipeline is |
|---|---|
| Real `S/C` beats its own eligible-pool shuffle (`p<0.05`) | a genuine C1 x C2 integration; both factors carry signal |
| It does not | **C2 alone**; the significance matrix is decorative here and the pipeline should optimise pooling/structure and stop paying for `S`/`C` state |

Building integration machinery before this returns would risk an elaborate
pipeline around a term that contributes nothing.

## 1. A prerequisite I cannot resolve alone

Every "Cornerstone-2" reference in this repository is either my own writing from
this session or the exclusion clause in
[CORNERSTONE1_ARCHITECTURE_HYPOTHESES.md](CORNERSTONE1_ARCHITECTURE_HYPOTHESES.md):

> New span pooling, contiguous-span preservation, or local-neighborhood scoring
> is out of confirmatory scope.

There is **no Cornerstone-2 specification, implementation, result set, or
provenance record here** — only the `max_pool1d` primitive at
`runtime/policy.py:121-133`. I have been treating "Cornerstone 2" as *the span
factor that clause names*. If it denotes a specific mechanism developed
elsewhere, the plan below is aimed at the wrong target and I need the spec or
its repository before writing integration code.

## 2. The frontier, which is the actual deliverable

We know quality is preserved at budget 256. Nobody has asked the compression
question: **how far down can the budget go before quality breaks?**

```text
budget   256 (known)  ->  0.293x bytes, ties full cache
budget   192, 128, 96, 64, 48  ->  unmeasured
```

Each budget needs its own best `(pool_width, sink_tokens, recent)`, because the
optimum almost certainly moves with budget: pooling width is a fraction of the
eligible pool, and `sinks=48` is a *larger share* of a small budget. Reusing
`pool=13, sinks=48` at budget 64 would be a category error — sinks plus recent
would consume the entire budget.

The deliverable is a curve: **smallest budget at which compressed accuracy is
still statistically indistinguishable from full cache**, plus the configuration
that achieves it.

## 3. Phased design

Every phase carries the same non-negotiables (section 4).

**Phase A — specificity (running).** Settles whether `S` is credited. ~75 min.

**Phase B — factorial at fixed budget 256.** The confirmed point was found by
ad-hoc sweeping across 14 arms with no multiplicity control; `pool=13` is a
winner's-curse survivor, not a demonstrated optimum. A clean factorial
`pool_width x sink_tokens` with an identity gate and a shuffled control per
cell, on showcase-64 for screening. Winner re-confirmed on stability-192.

**Phase C — budget descent.** For budgets 192/128/96/64, screen a small
budget-scaled configuration set on showcase-64, then confirm the best per budget
on stability-192. Produces the frontier curve. This is the phase that answers
"compress as much as possible".

**Phase D — family transfer.** Apply the chosen operating point to Halo
Uncensored, Lyra, Nimbus, Pulse. All five lose 10-13 answers with retention near
0.6 ([family report](PUBLISHED_FAMILY_REPORT.md)), so the target exists in each,
but transfer is a prediction until run. Each core keeps a separate identity; no
pooled claim.

**Phase E — registration.** One preregistered confirmation of the final
operating point, with the endpoint, gate, and analysis fixed before execution.
Only this phase may produce a deployment recommendation.

## 4. Rigor requirements — every phase, no exceptions

These are the practices that caught two of my own errors this program, and the
absence of one of them produced a third.

1. **Identity gate.** Every screen includes an arm that must reproduce a
   recorded baseline bit-for-bit. `POOL1 = 19/64` caught nothing this time
   because it passed — that is what it is for.
2. **Specificity control per claim.** A shuffled or random counterpart at
   matched structure. Five program findings died here; one survived.
3. **Held-out confirmation before any claim.** Screening on showcase-64,
   confirmation on stability-192 (48 packs, zero scenario overlap). Never claim
   from the screening split.
4. **Preregistered plan written to disk before inference.** One primary
   contrast. Everything else descriptive.
5. **Cluster-level analysis.** Scenario pack is the independence unit; 64 rows
   are 16 packs, 192 rows are 48. Case-level `p` is reported, never claimed on.
6. **Answer-blind inference.** Key opened only after the unscored artifact is
   durably written. Oracle arms are labelled bounds and never cited as
   performance.
7. **No `src/` drift.** Every file under `src/` is content-hashed by the
   Experiment F provenance record. Experimental variation goes in scripts, via
   `compression_callback` or in-process patching.
8. **Filter provenance.** The `RARE_MAX=3` retraction happened because a filter
   was carried from a diagnostic into an oracle without re-justification. Any
   filter must be justified *at each site where it is used*.

## 5. Known bug-risk surface

Concrete places this pipeline can go wrong, from what has already gone wrong:

- **Reused constants across contexts.** Root cause of the retracted oracle.
- **Silent arm inactivity.** `SPANLINK` fired on ~500 links per boundary and
  still changed nothing; without the instrumentation probe I would have reported
  a null for an arm that might not have been running. Instrument that an
  intervention actually alters selection before trusting its null.
- **Budget-scaling errors.** At small budgets `sinks + recent` can exceed the
  budget; `SignificancePolicy.allocation` raises when `require_heavy` is set,
  but a screen that catches the exception per-cell could silently skip cells.
  Assert cell coverage explicitly.
- **Process kills under memory pressure.** Two runs died silently with no
  traceback at ~5 GB free. Long runs go detached with incremental partials, and
  never concurrently with other heavy jobs.
- **Cross-process comparison.** `N-034` records that separate CPU/bf16 processes
  are not bitwise reproducible. Any contrast that matters runs in one process.

## 6. What this pipeline cannot deliver

- Nothing here makes the significance matrix work. If Phase A returns null,
  the honest framing of the whole program becomes: *a span/structure factor
  closes the compression gap; the significance score does not contribute.*
- No novelty. The [literature audit](LITERATURE_NOVELTY_AUDIT.md) verdict is
  `NO_NOVELTY_CLAIM`, and pooling over attention scores is squarely inside
  SnapKV's territory.
- No deployment claim without Phase E.
