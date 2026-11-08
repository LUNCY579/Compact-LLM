# Phase 1B preregistration draft — conditioning ladder

**Status: DRAFT, conditional, not executable.**

This registration activates **only** if Phase 1 v3 returns H1a **and** H1b **and**
H1c satisfied with the `direct_lookup` guardrail intact. At the time of writing
the v3 confirmatory matrix is still training and **no outcome has been
inspected**. If Phase 1 lands anywhere else, the correct follow-on is in the
Track C case that matches the actual result — not this document.

It lives on the retrofit branch because the ladder maps one-to-one onto the
retrofit's conditioning flags. It must be moved to the Phase-1 lab branch and
finalized there, with its own study ID and its own outcome-blind power audit,
before anything runs.

## Question

Given that the finite-difference signal is useful, how much additional accuracy
is obtained by letting it control more of the attention computation?

Accuracy is the primary endpoint. Latency, FLOPs, memory, and KV-cache cost are
recorded and are **not** gates.

## The ladder

Q, K, and V conditioning are separated because they do different things, and
assuming they are interchangeable — or additive — would hide the answer.

| arm | `condition_q` | `condition_k` | `condition_v` | what it tests |
|---|---|---|---|---|
| `cascade` | — | — | — | anchor: sequential depth, no trajectory signal |
| `history_control` | — | — | — | anchor: information-matched history pathway |
| `learned_delta` | ✓ | — | — | the Phase-1 mechanism (see mapping note) |
| `k_only` | — | ✓ | — | trajectory changes what is *retrievable* |
| `v_only` | — | — | ✓ | trajectory changes what is *carried* |
| `qk` | ✓ | ✓ | — | query/key interaction |
| `qv` | ✓ | — | ✓ | query/value interaction |
| `kv` | — | ✓ | ✓ | key/value interaction |
| `qkv` | ✓ | ✓ | ✓ | full conditioning |

Then, only after the best conditioning pathway is identified:
predictor/corrector, then trajectory-aware token significance, then the complete
operator. **Every intermediate arm is retained even if `qkv` is expected to
win** — the point is to learn which components contribute, not to crown the
biggest one.

### Mapping note, stated explicitly

Phase 1's `learned_delta` injects the control into the *head input*, which in
that lab is a slice of the hidden state feeding the head's own Q/K/V. A real
Qwen3 decoder projects Q/K/V from the **full** hidden state, so "head-input
injection" has no exact analogue here. The closest faithful analogue is
`condition_q`: it changes what the head looks for without altering the cached
keys or values.

This is a genuine architectural difference between the Phase-1 lab and the real
model. It must not be papered over: a Phase-1 result about `learned_delta`
transfers to `condition_q` **by analogy**, and Phase 1B re-establishes it
directly rather than assuming it.

### K and V conditioning interact with the KV cache

Conditioning K or V modifies tensors that would otherwise be cached. The
implementation applies the delta to the **current window only**, leaving cached
entries native, because a cached key written under one trajectory state must not
be silently reinterpreted under a later one. That makes `k_only`, `v_only`,
`qk`, `qv`, `kv`, and `qkv` behave differently in prefill than in incremental
decode, and the ladder must therefore report **cached-vs-uncached agreement per
arm** as a correctness precondition, not assume it.

`condition_q` has no such issue: queries are never cached.

## Inference

Identical to the registered Phase-1 v3 procedure and for the same reasons:

- estimator `crossed_cluster_t` (two-way cluster-robust, `t_{m-1}` reference);
- primary gate: lower 95% confidence bound above zero;
- the effect the study is *powered* for is reported descriptively and does not gate;
- `seed_cluster` and both pigeonhole variants reported as sensitivity analyses.

**Multiplicity.** The ladder has many arms, so the family-wise error rate must be
controlled. The registered procedure is a **fixed-sequence test** along the
containment order — `learned_delta` → best single target → best pair → `qkv` —
stopping at the first failure. A fixed-sequence procedure needs no alpha
adjustment while the ordering is preregistered, and here the ordering is forced
by strict containment. Comparisons *within* a rung (Q vs K vs V) are exploratory
and are reported as such.

**Interaction, not additivity.** Where compute permits, the pairwise arms are
used to test whether the pair effect equals the sum of its singles. Assuming
additivity would be an unregistered modelling choice.

**Power must be re-derived.** Phase 1B's increments are plausibly *smaller* than
the Phase-1 identification effect, and a smaller target needs more replication.
Run `phase1_power_simulation.py` against the Phase-1B effect ladder and report
the answer honestly, including "no candidate up through N seeds is adequate".
The Phase-1 seed count is **not** inherited.

## Excluded at this stage

Adaptive stopping, head skipping, token pruning, KV compression, sparse or
approximate trajectory updates, latency-chosen low-rank restrictions, and
distillation aimed at removing sequential computation. Those belong to a later
efficiency study. Efficiency metrics are recorded under `gating: false`.

## Claim ceiling

At most: *allowing an identified trajectory signal to additionally condition
Q/K/V increased held-out generated-task accuracy beyond conditioning one target
alone.* Not natural-language generalization, long-context behaviour, efficiency,
scaling, or broad architectural superiority.
