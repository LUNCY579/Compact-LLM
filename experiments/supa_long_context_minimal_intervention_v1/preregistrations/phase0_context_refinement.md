# Preregistration — Phase 0, zero-training context refinement

Fixed before any model was run against conditions B, C or D.
Protocol version `1.0.0-pilot`.

## Question

Is Supa-A1.1's long-context weakness primarily a lack of reasoning capability,
or interference from irrelevant context competing for a small model's limited
representational capacity?

## Dataset

`Rineya_Long_Context_Demo_v1/data/rineya_long_context_demo_full.jsonl`,
256 rows / 64 scenario packs, SHA-256 recorded in each run manifest.

Chosen over the alternatives because it is (a) the suite the target checkpoint's
published result is defined on, so condition A has an exact expected value;
(b) genuine reasoning rather than needle retrieval — eight task types including
alias resolution, temporal correction, count-with-exclusions, causal
disambiguation, latest-verified-record and multi-condition identity; (c) crossed
with four variants per scenario (base, paraphrase, counterfactual,
unanswerable), giving distractors, paraphrase and evidence-sufficiency coverage;
and (d) carrying `supporting_facts`, without which condition D is impossible.

The suite is marked `demo/evaluation only; do not train on this exact suite`.
It is used here for evaluation only, in every phase.

## Conditions

All four share model, checkpoint, decoding, max output length, scorer, prompt
template, system message and seed policy. Only the context string differs.

* **A — full.** Original context.
* **B — query-selected.** BM25 over 192-token sentence-packed chunks; scored
  selection under a 384-token budget; ±1 neighbour expansion capped at 576
  tokens; document order restored.
* **C — random equal-token.** Same chunk count as B for that example, drawn
  uniformly at random from the same chunk set; document order restored.
* **D — oracle evidence.** Chunks localised from `supporting_facts`; ±1
  neighbour expansion under the same cap; document order restored.

**Structural anchor.** The document's first sentence (~30 tokens) is prepended
to B, C and D identically, and suppressed when chunk 0 is already selected.
Applying it to all three means it can never be what separates them.

**Gap marker.** Non-adjacent runs are separated by a blank line, identically in
B, C and D.

## Frozen selector parameters — no sweep

| parameter | value |
|---|---|
| chunk target | 192 tokens, packed on sentence boundaries, no split sentences |
| chunk overlap | 0 sentences |
| scorer | Okapi BM25, k1 = 1.5, b = 0.75, IDF over this document's chunks |
| stopwords | fixed 60-word list in `slcmi.context_selection` |
| selection budget | 384 tokens |
| neighbour radius | 1 |
| hard ceiling | 576 tokens |
| anchor | first 1 sentence |
| random seed | 20260905 + example index |
| oracle localisation threshold | 0.34 lexical overlap, **all** facts must clear it |

One revision was made to these numbers before any model ran, on the basis of a
CPU-only dry run: an earlier draft used a whole leading chunk as the anchor with
a 768-token ceiling, which left B at 56% of the original context and made D
collapse onto B on the first example inspected. The values above were then
frozen and no further tuning was done.

## Oracle validity

`supporting_facts` are abstracted statements, not spans: **0 of 424 appear
verbatim** in their context. Localisation is therefore lexical-overlap based and
its per-example confidence is recorded rather than assumed.

Oracle is declared **UNAVAILABLE**, not approximated, when:

* the row is unanswerable — its `supporting_facts` is a constant boilerplate
  string with no localisation signal; or
* any single fact fails the 0.34 threshold — a two-hop oracle missing one hop is
  not an oracle.

The gold `answer` field is never read when constructing D, and D's text is
always original context, never the annotation string.

## Analysis plan

Primary: strict exact match under the suite's own
`evaluate_predictions.py::normalize`.

Contrasts: B−A, B−C, D−A, D−B, each with a paired bootstrap CI (10,000
resamples, seed 20260905, resampling example indices so pairing is preserved)
and exact McNemar on discordant pairs.

Frames reported: all rows; answerable only; unanswerable only; oracle-eligible.
D is compared only within the oracle-eligible frame, never against an A computed
on a different row set.

Secondary, preregistered because the suite makes it possible: the **abstention
rate** per condition, split by answerable and unanswerable. If refinement raises
abstention, credit gained on unanswerable rows is a response-bias shift rather
than a reasoning gain, and the answerable-only frame is the honest read.

## Validity gate

Condition A over all 256 rows must reproduce the published Supa-A1.1 result —
45/64 Showcase, 125/192 Stability, 170/256 combined — exactly. Otherwise the
stage is reported INVALID and no contrast is interpreted.

## What would count as support

* **SUPPORTED** — B > A and B > C, both with intervals excluding zero.
* **NOT SUPPORTED** — B ≈ A or B < A, *and* D ≈ A or D < A, with B > C
  confirming the selector is not the failure.
* **INCONCLUSIVE / selector failure** — B ≈ A but D ≫ A: the model can reason
  over the right evidence and the cheap selector is failing to expose it. This
  must not be reported as model failure.

No claim of statistical significance will be made. This corpus is underpowered
by its own published analysis (~576 rows needed for 80% power; there are 256),
and Stability192 has been read many times and is no longer a clean split.

## Out of scope

No selector hyperparameter sweep. No embedding model. No second LLM. No
retrieval engineering beyond the frozen lexical selector above.
