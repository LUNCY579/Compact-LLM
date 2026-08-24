# Cornerstone-1 architecture hypotheses

These hypotheses are falsifiable research targets, not project claims. Every
new ChatBubble status is `NOT_RUN` at initialization.

## Mechanistic object

For surviving cache entry `i`, Cornerstone 1 carries cumulative received
significance `S_i` and realized scoring exposure `C_i`:

```text
R_i(e) = S_i / max(C_i, 1)^e
```

`C_i` belongs to the surviving semantic entry, not an absolute buffer slot.
Therefore every selection operation must apply the same gather indices to
`K`, `V`, `S`, `C`, and absolute token position.

The causal hypothesis is narrow: repeated compression gives early survivors
more opportunities to be scored. Raw cumulative attention can therefore mix
intrinsic usefulness with realized exposure. Explicit exposure state may
separate those quantities.

## Confirmatory hypotheses

### H1 — normalization under repeated compression

At a fixed final resident budget, fixed prompt, and fixed decode, `e=1` will be
more stable than `e=0` as the number of compression boundaries changes.

- Primary outcomes: compression-conditioned exact accuracy and
  all-required-evidence survival.
- Required interaction: method by boundary count.
- Failure condition: no paired improvement, or an age-only control matches it.

### H2 — realized exposure contains information beyond age

True entry-aligned `C` will outperform constant, age-derived, shuffled,
wrong-token, delayed, noisy, or head-averaged counts at matched storage.

- Primary diagnostic: rank and retention changes for required evidence.
- Failure condition: deterministic age or a perturbed count carries equivalent
  predictive and task value.

### H3 — retention mediates task quality

When normalization changes an answer from wrong to correct, the expected causal
path is:

```text
normalization changes evidence rank
  -> evidence survives a boundary
  -> later attention reaches that evidence
  -> answer becomes correct
```

An accuracy delta without trace evidence for this path remains mechanistically
unresolved.

## Exploratory architecture hypotheses

### H4 — significance-conditioned layer/head budgets

Concentration, entropy, stability, or churn of normalized significance may
allocate a fixed global KV budget better than uniform allocation. A policy must
not receive extra persistent bytes.

### H5 — bounded historical-attention prior

A small, bounded significance-derived logit prior may help future attention.
It may instead create rich-get-richer collapse. `lambda=0` must reproduce the
unmodified path, and normalized state must be compared with raw score, shuffled
state, and age-only bias.

### H6 — tiny significance memory

Normalized-significance aggregation into a small fixed number of memory slots
may retain useful information lost by hard eviction. Mean, recency, raw-score,
random, and zero-slot controls are required.

### H7 — tiny recovery controller

Low-dimensional significance diagnostics may predict when a larger budget,
memory path, or recomputation is worthwhile. A learned controller must be
trained on calibration data and assessed on held-out families; no reinforcement
learning is planned until simpler models establish predictiveness.

## Outcomes the program permits

The evidence may support any of these conclusions:

1. Cornerstone 1 is useful only for KV retention.
2. It transfers to a subset of architectural decisions.
3. It improves reasoning but costs too much.
4. It reduces compute but loses reasoning quality.
5. A positional or deterministic-age policy explains the result.
6. A learned component adds no value.
7. The effect is limited to retrieval.
8. The hypothesis fails outside its original setting.

Negative findings are retained in reports and are not tuned away.

## Cornerstone-2 exclusion

New span pooling, contiguous-span preservation, or local-neighborhood scoring is
out of confirmatory scope. If source-compatible pooling is required, it must be
configuration-controlled and reported as a separate factor. The confirmatory
Cornerstone-1 comparison disables pooling.

