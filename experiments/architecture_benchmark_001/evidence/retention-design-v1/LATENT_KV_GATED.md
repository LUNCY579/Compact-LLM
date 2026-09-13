# Latent KV through the retention gate: 2× is exact, 4× is not free, and whitening is everything

The latent low-rank KV result was this programme's only live deployment candidate, and it had
never faced a capability gate. It was measured with the legacy number-sequence scorer on 60 rows
and reported as **"4× KV reduction at no measurable capability cost"** — 19/60 against the
source's 20/60.

Rerun on the 150-row typed panel against the **paired** 47-item reference from the
`G1_RETENTION_B28` receipt at digest `a329eb7b`, that conclusion changes in an instructive way.

Exploratory: same package scorer, outside the receipt system. No receipt, no gate opened.
Calibration covariance from TRAIN (33,918 tokens), never DEV and never the probe.

## Result

| arm | KV | exact? | source | arm | reference | **lost** | new | terminated | verdict |
|---|---:|:--:|---:|---:|---:|---:|---:|---:|---|
| `whitened r1024` | **0.500×** | **yes** | 47 | 47 | 47 | **0** | 0 | 148/150 | **PASS** |
| `whitened r512` | **0.250×** | no | 47 | **47** | 55 | **8** | 8 | 148/150 | FAIL |
| `whitened r293` | 0.143× | no | 47 | 24 | 55 | 31 | 8 | 150/150 | FAIL |
| `plain r512` | 0.250× | no | 47 | **0** | 47 | **47** | 0 | **0/150** | FAIL |

Both bare policies agree on every arm.

## 1. The 2× is exact, and the exactness control holds

At rank 1024 the reparameterisation is exact by construction, and it behaves that way: **0 losses
in all three classes**, reference equals the source set exactly (47, not 55), spectral energy
retained 1.000000. This is the programme's second bitwise-identity control alongside `ced_last1`,
and like that one it passes cleanly.

**A 2× KV reduction with zero measured capability cost and zero training is the strongest
positive result in this programme.**

## 2. The 4× is *not* free — and only a paired reference can see why

`whitened r512` scores **47 of 150 — identical to the source's 47.** A rate comparison reports no
change whatsoever. That is exactly what the legacy 19/60-vs-20/60 reading said, and it is
misleading.

The reference is **55**, not 47. The arm answers **8 questions the source gets wrong** and
**loses 8 the source gets right**. Net rate change zero; actual churn 8 in each direction —
**17% of the source's correct answers are forgotten.** By class the damage is uneven:
`ratio_proportion` loses 5 of 16, `percentage_reasoning` 2 of 21, `time_duration` 1 of 10.

This is the single clearest demonstration in this programme of why retention has to be measured
against a **paired, immutable reference** rather than as an aggregate rate. Two models with the
same score are not the same model.

Whether an 8-for-8 swap is acceptable is a product decision, not a measurement one. What is no
longer defensible is "no measurable capability cost".

## 3. Whitening is doing all the work

`plain r512` and `whitened r512` are the **same rank and the same 0.250× KV**. The only
difference is whether the low-rank fit minimises activation-space error under the layer's real
input covariance or plain parameter-space error.

    whitened r512   47/150 correct, 148/150 terminated
    plain    r512    0/150 correct,   0/150 terminated

Plain SVD at 4× does not merely degrade — it **terminates on nothing at all**. Same compression
ratio, same parameter count, total collapse. The activation-covariance fit is not a refinement;
it is the entire mechanism.

## 4. A caveat on the covariance that deserves attention before anyone ships 4×

`covariance_health` reports **effective rank 4–5 of 1024** over 33,918 calibration tokens. The
token count is comfortable (33 tokens per dimension); the *anisotropy* is extreme. The activation
covariance is dominated by a handful of directions, which is precisely why whitening buys so
much — and also why a rank-512 whitened fit is determined by a very low-dimensional subspace of
the calibration distribution.

Nothing here tests how that fit behaves on inputs unlike TRAIN. The 4× number is a
same-distribution result, and the 8 lost answers may be the visible edge of that.

## What this changes

- **2× (rank 1024) is the shippable variant**, on an exactness proof rather than a hope.
- **4× (rank 512) is downgraded** from "free" to "costs 17% of the source's correct answers and
  returns a different 17%". It remains far the best compression candidate in this programme, but
  it is not a no-op.
- **7× (rank 293) is out**: 31 of 55 lost.
- The registered zero-loss rule fails 4×, as it failed the preserving training dose. That is the
  same uncalibrated-threshold finding, not a new one — and here the gate's *magnitude* separation
  is what carries the information: 0, 8, 31, 47 losses across a clean ordering.

## What this does not establish

- **No memory, latency or throughput claim.** Every KV figure in this programme is analytic. The
  2× and 4× only exist if a serving stack actually stores the rank-r latent. Nobody has measured
  a byte or a millisecond.
- **No receipt.** Latent KV has no gate lineage — no `G1_LATENT` stage, no parity checks. CED has
  15 mandatory checks; this has none. That is the gap to close before "shippable" means anything.
- **One panel, one seed, greedy, 24 new tokens**, three numeric classes, a grammar whose label
  review is pending and which is known to reject the source's own `It ends at …` phrasing.
- **Same-distribution calibration only**, per §4.

## Reproduction

```bash
PY=C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe
cd C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification
for T in latent_whitened_r1024 latent_whitened_r512 latent_whitened_r293 latent_plain_r512; do
  $PY experiments/rab001_exploration/retention_probe.py "$T" "$T" cpu
done
```

~5–9 min per arm; the TRAIN covariance is computed once and cached. Artifacts:
`evidence/retention-design-v1/RETENTION_PROBE_latent_*.json`.
