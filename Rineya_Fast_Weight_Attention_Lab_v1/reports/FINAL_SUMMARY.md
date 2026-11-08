# Final research summary — Supa-A1.1 Fast Weight Attention v1

Answers the 21 points in section 24 of the experiment brief. Every number here is
measured; nothing is estimated. Supporting detail in `RESULTS_P2B.md`,
`RESULTS_H7.md`, `MECHANISM_DIAGNOSTIC.md`, `LAYER_PROBE.md`, `RESOURCE_REPORT.md`.

| # | Item | Value |
|---|---|---|
| 1 | Worktree | `C:\Users\clclc\BusinessProject\RineyaBeta-supa-fwa-v1` |
| 2 | Branch | `research/supa-a1.1-fast-weight-attention-v1` |
| 3 | Base commit | `ee13448f11efaa7f14206dab95b277816b912ce6` (`research/positional-sequential-attention`) |
| 4 | Final commit | `0959c8368e13cb2832847e9be8e01bb65bf453cc` (37 commits on branch) |
| 5 | Pushed remote branch | `origin/research/supa-a1.1-fast-weight-attention-v1` — pushed, never force-pushed |
| 6 | Supa checkpoint | `Rineya_Published_Cores/Rineya-Supa-A1.1-1.1.0-devfreeze`, weights SHA-256 `a99f079d…dc4026`, resolved read-only from the sibling `Chat-Bubble` vault. Never written to. |
| 7 | FWA variant | Prefix-aligned **FALCON-2** (`x_t = phi(k_{t-1})`, read-after-write, per-column eta, ridge decay), auxiliary pathway merged as `h' = attn + alpha * P(m)` |
| 8 | Layers instrumented | **Layer 14 only** (0-based, of 28). A 28-layer zero-training probe ranks it **13th of 28** by leverage; no trained sweep was run |
| 9 | Added trainable parameters | **524,593** per instrumented layer (arms B, C, C-short); **525,618** for arm D (+1,025 significance gate) |
| 10 | Recurrent state size | **8,192 bytes** at rank 16 (8 KV heads x 16 x 16 x fp32), constant in sequence length |
| 11 | Training GPU-hours | **9.51** total over 18 training runs (P1 + P2 + P2b + C-short replication + memory probes) |
| 12 | Peak VRAM | Training **3.19 GiB** (6.91 GiB only in the rejected gradient-checkpointing probe). Eval: 1.32 / 1.80 / 3.64 GiB at 1K / 2K / 4K; 10.79 GiB at 8K (host-memory spill); **16K OOMs** |

## 13. A/B/C/D results (P2b, 168 items, two seeds, paired exact McNemar within seed)

| Arm | position | interference |
|---|---:|---:|
| A frozen backbone | 138/168 | 55/56 |
| B parameter-matched static control | 159 / 159 | 55 / 51 |
| C recurrent | 160 / 161 | 50 / 46 |
| D significance-gated | 161 / 160 | 55 / 51 |
| C-short (original 87.5-token horizon) | 163 | 55 |

| Contrast | seed 11 | seed 22 |
|---|---|---|
| C vs A | 25/3, p < 0.0001 | 26/3, p < 0.0001 |
| **C vs B** | 2/1, p = 1.00 | 3/1, p = 0.63 |
| **D vs C** | 3/2, p = 1.00 | 2/3, p = 1.00 |

**Arm B does all of the work.** It takes the frozen backbone from 138 to 159 with
no recurrent state at all. Arm C adds 1-2 further items, at neither seed
significantly.

## 14. Position-conditioned results

Arm A shows a **primacy deficit**, not lost-in-the-middle: accuracy rises
monotonically with position, 0.417 at 0.05 to 1.000 at 0.90. The preregistered
middle-vs-edge metric assumed the opposite deficit and was demoted (Amendment 5).

| position | 0.05 | 0.10 | 0.25 | 0.50 | 0.75 | 0.90 | 0.95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.417 | 0.708 | 0.792 | 0.917 | 0.958 | 1.000 | 0.958 |
| B | 0.917 | 0.792 | 0.958 | 1.000 | 1.000 | 1.000 | 0.958 |
| C | 0.875 | 0.792 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

The static control repairs the primacy deficit. H7 tested whether recurrence adds
anything there on a purpose-built 320-item suite: **rejected**, 18/4 (p = 0.0043)
at seed 11 and 8/11 (control ahead) at seed 22.

## 15. Context-length scaling

**No length degradation** for the frozen backbone: 0.857 / 0.786 / 0.821 at
1K / 2K / 4K. Difficulty in this task lives on the position axis, not the length
axis, so length-robustness claims are not supported either way.

## 16. Inference latency and memory effects

| Context | Arm A prefill | Arm C prefill | Peak VRAM (both) | Recurrent state | KV cache if used |
|---:|---:|---:|---:|---:|---:|
| 1024 | 5899 tok/s | 1401 tok/s | 1.32 GiB | 8192 B | 112 MiB |
| 2048 | 3630 tok/s | 1222 tok/s | 1.80 GiB | 8192 B | 224 MiB |
| 4096 | 1929 tok/s | 925 tok/s | 3.64 GiB | 8192 B | 448 MiB |
| 8192 | 164 tok/s | 154 tok/s | 10.79 GiB (spill) | 8192 B | 896 MiB |
| 16384 | OOM | OOM | — | — | — |

- **1.5x-4.2x slower prefill** at 1K-4K.
- **Zero VRAM saving**: peak is identical for A and C to within 0.01 GiB at every
  length, because the pathway is auxiliary and full attention still runs.
- The bounded state is real (8 KB, constant in T) and **buys nothing here**. Only
  Arm E (replacement) could convert it into a saving.
- Generation is worse than prefill implies: the wrapper refuses KV-cache mixing,
  so decoding is O(new_tokens x T).

## 17. Interference findings

**Section 14 of the brief is not answered by this lab, and that is a gap.** The
battery is ceilinged for gains (arm A scores 55/56) *and* too seed-noisy to
attribute a drop to an architecture (the same arm B scores 55 and 51). Within
seed, arm C loses 5 items to arm B at both seeds with zero gains, but arm B's own
cross-seed swing is 4 items, so no claim rests on it (Amendments 5, 7, 8, 9).
The blend-detection tool was built and deliberately **not run** against an
instrument that could not support its output.

## 18. Did significance gating help?

**No, not on its own terms.** Arm D never beats arm B. On position it is null
against C at both seeds (3/2, 2/3). On interference it restores *exactly* arm B's
score at both seeds (0/0 and 1/1 discordant) by repairing damage that arm C's
recurrence introduced. Its entire demonstrated value is cancelling a liability
that would not exist without the mechanism it is attached to.

## 19. Strongest evidence AGAINST the working hypothesis

1. **The parameter-matched static control reproduces the recurrent arm.** C vs B
   is 2/1 and 3/1 discordant out of 168. B alone delivers the whole gain,
   including the primacy repair that looked like the memory's best case.
2. **Recurrence adds variance, not signal.** On the 320-item suite arm B lands
   within 1 item across seeds (287, 288) while arm C-short moves 16 (301, 285)
   and disagrees with itself on 28 of 320 against B's 15. The one significant
   result in the whole experiment (p = 0.0043) was that variance landing well.
3. **The cost is unambiguous and measured**: slower prefill, no VRAM saving,
   quadratic generation, and a replicated within-seed interference cost.
4. **Two of my own mechanistic predictions failed.** The long-horizon init I
   argued for was harmful, and C-short — the horizon I argued against — was the
   best arm in P2b.

## 20. Decision

**NO-GO**, at layer 14, on a synthetic two-hop retrieval task.

GO required C > B at p < 0.05 in both seeds (got p = 1.00 and 0.63) or D > C at
p < 0.05 in both seeds (got p = 1.00 and 1.00). WEAK GO was considered and
rejected: C leads B by +1 and +2 items against a replicated -5 interference cost
and a 1.5-4.2x slowdown. REDESIGN was considered and rejected: it requires a
working memory that interference is spoiling, and there is no positional benefit
here to rescue.

## 21. Exact next experiment, if justified

**Arm E — replace attention rather than augment it**, at 1-2 layers, trained
long enough to recover, reporting a quality/memory tradeoff curve rather than a
win. Rationale: every negative result above concerns an *auxiliary* pathway,
which structurally cannot show the one thing FALCON-style architectures actually
claim — a bounded state substituting for a growing KV cache. That claim remains
untested, and this lab cannot test it without replacement.

**Do not** run a trained layer sweep first: its prerequisite is a suite
calibrated against a *trained* control, which has never been done, and the
zero-training probe shows the depth profile is a residual-norm artifact that a
learnable alpha can compensate.

## What would have gone wrong without preregistration

At seed 11, H7 read as +14 items, p = 0.0043, properly powered, in exactly the
regime theory predicted. It was a false positive and it did not get reported as a
result, because the both-seeds criterion and the exploratory label were both
fixed before the data existed. Twelve amendments record the design changes, four
corrections to my own errors, and two failed predictions of mine.
