# 4× KV reduction, no training, no measured capability loss — and why sharing could never do it

The displaced-layer law says every KV-sharing topology pays ~0.5 nats per removed bank, because
removing a bank displaces a layer **by construction**. Per-layer low-rank KV compression removes
no bank and displaces **nothing**, so the law predicts the price should not apply to it.

That was a falsifiable prediction. It is now measured, and it holds.

Exploratory, CPU FP32, frozen official Supa 1.7.0-devfreeze, **zero training**. Outside the
hashed root — no receipt, no gate. Open splits only; calibration drawn from TRAIN, scored on
DEV. No sealed or registered DEV access, no efficacy claim.

---

## Result

40 DEV rows (4,979 target tokens), 60 held-out TASK prompts. `kv` is cached values per token
per layer, relative to the source's 2,048.

| fit | rank | KV | dev CE | KL→source | top-1 | task | terminated |
|---|---:|---:|---:|---:|---:|---:|---:|
| source | — | 1.000× | 5.5782 | 0.0000 | 1.000 | 20/60 | 60 |
| whitened | 1024 | **0.500×** | **5.5782** | 0.0000 | 1.000 | **20/60** | 60 |
| whitened | 512 | **0.250×** | **5.5725** | 0.0431 | 0.901 | **19/60** | 60 |
| whitened | 293 | 0.143× | 5.6070 | 0.2355 | 0.771 | 6/60 | 60 |
| whitened | 146 | 0.071× | 5.9745 | 1.4996 | 0.474 | 0/60 | 58 |
| plain | 1024 | 0.500× | 5.5782 | 0.0000 | 1.000 | 20/60 | 60 |
| **plain** | **512** | **0.250×** | **8.2881** | 5.0151 | 0.133 | **0/60** | **0** |
| plain | 293 | 0.143× | 12.4488 | 9.2662 | 0.034 | 0/60 | 2 |
| plain | 146 | 0.071× | 13.8360 | 10.7057 | 0.000 | 0/60 | 0 |

**At 0.250× KV — a 4× cache reduction — the activation-whitened latent scores 19/60 against the
source's 20/60, with dev CE 0.0057 nats *below* the source's. With no training at all.**

At n=60 that is "indistinguishable", not "equal": Wilson intervals are [21.3%, 44.2%] for 19/60
and [22.7%, 45.9%] for 20/60. The honest claim is that no capability loss is detectable at this
sample size, not that none exists.

The cliff is between 0.250× and 0.143×: r=293 drops to 6/60 while its CE barely moves (5.607).
Once again CE fails to see a capability collapse — consistent with everything else in this
programme.

## Against the sharing frontier

The comparison the whole programme has been building toward. Sharing numbers are from
`ARCHITECTURE_COMPARISON.json` (24-row calibration split, source CE 5.5458, uniform 11.9312);
latent numbers are from this sweep (40-row dev split, source CE 5.5782). Different splits, so
compare each against **its own** source baseline — the gap sizes are not close enough for the
split difference to matter.

| KV budget | best sharing topology | latent (whitened) |
|---|---|---|
| 0.536× | C28: CE 12.5854 — **worse than uniform**, 0/180 even after training | — |
| 0.500× | local group 2: CE 12.9136 — worse than uniform | **exact, 20/60** |
| 0.250× | local group 4: CE 16.8612 — garbage | **CE −0.006 vs source, 19/60** |
| 0.143× | local group 7: CE 16.2356 — garbage | CE +0.029 vs source, 6/60 |
| 0.071× | local group 14: CE 15.1577 — garbage | CE +0.396 vs source, 0/60 |

Sharing is destroyed at 2× reduction. Latent compression is **exact** at 2× and still intact at
4×. Even at 14× (r=146) the latent's CE is +0.40 nats — better than *any* sharing point at
*any* budget, including the mildest one.

**The prediction was right, and the mechanism is the reason.** Sharing changes which K/V a layer
attends to, putting every displaced layer off-distribution. Low-rank compression changes only
how the same K/V is stored; each layer still attends to its own bank. There is nothing to
displace, so there is no 0.5-nats-per-bank toll to pay.

## The fit objective is the whole ballgame

At the same rank 512 and the same 0.250× budget:

    plain SVD    CE 8.2881   task 0/60   terminated 0/60   — degenerate, never emits EOS
    whitened     CE 5.5725   task 19/60  terminated 60/60  — indistinguishable from source

A 2.72-nat difference from the *objective alone*, at identical architecture, rank and cache
size. Plain SVD minimises ‖W − W_r‖_F, which is uniform over input directions. Whitened SVD
minimises E‖W n − W_r n‖² under the layer's real input covariance.

Why that matters so much here is visible in the covariance diagnostic:

    calibration 33,918 TRAIN tokens over 1024 dims (33.1 tokens/dim, not rank-deficient)
    effective rank (participation ratio)  3.53 – 5.27  of 1024
    worst condition number                1.08e8

The hidden-state covariance is dominated by a *handful* of directions — the well-documented
massive-activation/attention-sink structure of transformer residual streams. Plain SVD spends
its rank budget on directions the model essentially never activates. Whitening spends it on the
directions that carry signal.

**This nearly produced a false negative.** My first implementation had only the plain fit. Had
I stopped there, the recorded conclusion would have been "per-layer low-rank KV compression
fails at 4×" — the exact opposite of the truth.

## Controls

**Exactness control — PASSES.** `W_KV` is (2048, 1024) so rank ≤ 1024, and r=1024 must reproduce
the source. Both fits do, to FP32 roundoff through two matmuls:

    whitened r=1024   dev CE 5.578166   max|Δlogit| 1.841e-04   KL 0.0000   top-1 1.000
    plain    r=1024   dev CE 5.578166   max|Δlogit| 1.726e-04   KL 0.0000   top-1 1.000
    source            dev CE 5.578166

(An earlier 64-token smoke test reported 7.8e-05 for the same control. Both are FP32 roundoff;
the 1.8e-04 figures are the ones from this sweep and are the ones to cite.)

**This control already caught a real bug.** The whitened fit initially gave max|Δlogit| = **38.0**
at r=1024. Cause: `solve_triangular(chol.T, vh, upper=True)` computes `vh @ inv(chol).T`, not
`vh @ inv(chol)`. Without the control, the whitened arm would have been silently wrong at every
rank and the headline result would have been lost.

**No contamination.** Calibration for the whitened fit comes from TRAIN; CE, KL and the task
probe are scored on DEV and TASK, which are family-disjoint from TRAIN.

**Cache realism.** The incremental decoder caches up-projected K/V rather than the latent. That
is a speed choice, not a weakened claim — up-projection is a deterministic function of each
token's own latent, so expanding at write time and expanding the whole cache at read time give
bit-identical values. The reported cache cost is the latent size, which is what an
implementation would hold.

Verified at rank 256 on a 64-token smoke test: dense and cached decoding agreed to 1.1e-05
(plain) and 1.5e-05 (whitened). That check was **not** re-run inside the sweep above, so it is
evidence that the cache path is correct in general — not a per-variant measurement of the rows
in this table.

## What this does and does not establish

**Established.** On these weights, per-layer activation-whitened low-rank KV compression gives a
4× cache reduction with no capability loss detectable at n=60 and no training. It beats every
measured sharing topology at every measured budget, by margins far outside any plausible noise.
The displaced-layer law's prediction for a zero-displacement design is confirmed.

**Not established.**
- **One model.** Supa 1.7.0-devfreeze, 596M. The 2× exact point is shape arithmetic that holds
  for all ten Rineya cores; the 4× *measured* point is one model, one calibration set.
- **Small evaluation.** 40 dev rows, 60 task prompts, three numeric classes. 19-vs-20 is within
  noise; a larger probe could resolve a real gap. No seeds — the procedure is deterministic
  given a calibration set, so the untested variance is across *calibration sets*, not seeds.
- ~~**Ridge sensitivity untested.**~~ **SWEPT AND STABLE.** `RIDGE` was set to 1e-6 for Cholesky
  well-posedness and never tuned, against a covariance with condition number 1.08e8 — the largest
  open hole in this result. Across **seven ridges spanning 1e-8 to 1e-2**, rank 512 moves by
  **0.0059 nats** (CE 5.5725–5.5783) and scores **19/60 at every single one**. Rank 293 is
  likewise flat (5.6070–5.6229). The exactness control at r=1024 holds for every ridge
  (max |Δlogit| 1.03e-04–1.84e-04, top-1 1.000), as the algebra requires since
  `A = (W S) S⁻¹ = W` for any positive-definite whitening. The 4× result is not a tuning
  artifact. See `evidence/latent/LATENT_RIDGE_SWEEP_cpu.json`.
- **No latency claim.** RoPE is not decoupled, so DeepSeek's `W_UK`-into-`W_Q` absorption — the
  thing that makes MLA *fast* — is not implemented. This is a quality-versus-cache-size result
  only.
- **Not gated.** No receipt, no G1_QUALITY run, no sealed or registered DEV evaluation.

**Recommended next, in order:** ~~ridge sweep~~ (done — stable across six orders of magnitude);
a larger task probe across more capability classes, since 19-vs-20 at n=60 is the weakest link
that remains; then the 2× exact point is deployable today on any Rineya core with no experiment
at all.

## Reproduction

```bash
PY=C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe
cd C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification
$PY experiments/rab001_exploration/latent_sweep.py 40 60 256
```

~35 min CPU. Artifacts: `outputs/latent/LATENT_KV_SWEEP.json`,
`evidence/latent/LATENT_KV_SWEEP.json`.
