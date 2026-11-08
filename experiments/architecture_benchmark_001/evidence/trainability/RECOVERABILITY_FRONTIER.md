# The displaced-layer law is an *initialization* law. Training discharges ~80% of it — up to a cliff.

Three seeds, four arms, matched recipe, family-disjoint splits, a real retention probe. This is
the result that partially overturns "training does not rescue a displaced initialization": it
does, at small displacement. It also finds a sharp capability cliff that cross-entropy cannot
see.

Exploratory, CPU FP32, official Supa 1.7.0-devfreeze. Outside the hashed root — no receipt, no
gate. Open splits only; no sealed or registered DEV access; no efficacy claim.

## Design

| | |
|---|---|
| arms | `source_control` (0 displaced) · `ced_last4` (3) · `ced_last8` (7) · `C28` (13) |
| seeds | 17, 29, 43 — the protocol's registered training seeds |
| recipe | AdamW lr 1e-5, wd 0, clip 1.0, gradient checkpointing, **all** parameters trainable |
| budget | 40 steps, 40 **distinct** TRAIN rows per seed, ~5.2k supervised target tokens |
| TASK | 60 held-out prompts from classes the frozen source can do, **excluded** from TRAIN/DEV — so this measures **retention** |

Seeds shuffle TRAIN only, so between-seed spread is a clean measure of run-to-run variance.

## Built-in control — PASSES

Step-0 metrics must be identical across seeds, since the seed cannot touch an untrained model:

    source_control  dev CE  5.578166   task 20/60  identical across 3 seeds
    ced_last4       dev CE  6.533942   task  8/60  identical across 3 seeds
    ced_last8       dev CE  8.421801   task  1/60  identical across 3 seeds
    C28             dev CE 12.887974   task  0/60  identical across 3 seeds

## Result — a dose-response in displacement

Pooled over 3 seeds, n = 180 task prompts per arm, at step 40:

| arm | displaced | KV | dev CE | task | 95% CI | % of control | CIs disjoint? |
|---|---:|---:|---:|---:|---|---:|---|
| `source_control` | 0 | 1.000× | 1.7857 | 73/180 (40.6%) | [33.7%, 47.9%] | — | — |
| **`ced_last4`** | **3** | **0.893×** | **2.0657** | **69/180 (38.3%)** | **[31.5%, 45.6%]** | **95%** | **No** |
| `ced_last8` | 7 | 0.750× | 2.1978 | 36/180 (20.0%) | [14.8%, 26.4%] | 49% | Yes |
| `C28` | 13 | 0.536× | 3.1104 | 0/180 (0.0%) | [0.0%, 2.1%] | 0% | Yes |

**`ced_last4` is statistically indistinguishable from the ordinary control after 40 steps** —
38.3% vs 40.6%, overlapping intervals. `ced_last8` recovers to half the control and is
significantly worse. `C28` never answers a single prompt correctly across 180.

## The law, re-stated

| arm | init penalty | per displaced | 40-step penalty | per displaced | **discharged** |
|---|---:|---:|---:|---:|---:|
| `ced_last4` | +0.9558 | 0.3186 | +0.2800 | 0.0933 | **71%** |
| `ced_last8` | +2.8436 | 0.4062 | +0.4121 | 0.0589 | **86%** |
| `C28` | +7.3098 | 0.5623 | +1.3247 | 0.1019 | **82%** |

Two findings, and the second is the important one:

**1. ~0.5 nats/bank is an initialization price, not a permanent one.** Roughly 80% of it is
discharged by 5.2k supervised target tokens — a trivial budget. The residual is a nearly flat
0.06–0.10 nats per displaced layer regardless of how many are displaced. Every prior statement
in this programme that priced sharing at ~0.5 nats/bank was describing the *untrained conversion*
and should say so.

**2. There is a capability cliff between 7 and 13 displaced layers that CE does not see.** `C28`
discharges 82% of its CE penalty — the same fraction as `ced_last8` — and still scores 0/180.
Likelihood recovery and capability recovery are different processes, and CE tracks the first.
No metric in the registered gate ladder would have caught this.

## Correcting the earlier claim

`MULTISEED_RESULT.md` and `ARCHITECTURE_PATTERNS.md` concluded "training does not rescue a
displaced initialization" from the C28-vs-control pair alone. That conclusion is **correct for
C28 and wrong as a general statement about CED**. The corrected claim:

> Training rescues a displaced initialization when the displacement is small. At 3 displaced
> layers recovery is complete to within measurement error; at 7 it is partial; at 13 it does not
> occur at all within this budget.

The C28 numbers themselves are unchanged and stand.

## What this does and does not establish

**Established.** At a matched recipe on family-disjoint data with three seeds and a 180-prompt
retention probe, displacement damage is largely *trainable away* at 3 layers and not at all at
13. The step-0 identity control passes for all four arms.

**Not established.**
- **Budget.** Everything rests on ~5.2k supervised target tokens against a registered ceiling of
  1,000,000. Whether C28's 0/180 is structural or merely under-trained is **not settled here**;
  a 500k-token matched run is underway to test exactly that.
- **Where the cliff sits.** It is somewhere in 8–12 displaced layers; no arm was run between 7
  and 13.
- **Probe breadth.** Three numeric classes. Retention of other capabilities is unmeasured.
- **Worth it?** `ced_last4`'s full recovery buys only a **10.7%** KV saving — and per-layer
  low-rank compression gets **4×** with no training at all (`evidence/latent/`). This result is
  scientifically informative about the law; it is not a competitive KV-reduction path.

## Reproduction

```bash
PY=C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe
cd C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification
for s in 17 29 43; do
  for a in source_control ced_last4 ced_last8 C28; do
    $PY experiments/rab001_exploration/rigorous_pilot.py $a $s 40 1e-5 40 60 20
  done
done
$PY experiments/rab001_exploration/aggregate_rigorous.py
```

~10 min per cell, 12 cells. Artifacts: `outputs/rigorous/RIGOROUS_*.json`,
`outputs/rigorous/AGGREGATE_MULTISEED.json`.
