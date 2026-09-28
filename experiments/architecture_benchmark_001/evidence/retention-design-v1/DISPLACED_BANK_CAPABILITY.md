# The "usable budget: 2–3 banks" finding does not survive a capability measure

This programme's KV-sharing law is one of its headline results: every removed bank displaces
exactly one layer by construction, each displaced layer costs **~0.5 nats** of cross-entropy, CE
tracks the displaced *count* rather than the topology, and therefore the **usable budget is 2–3
banks (~10% KV)**.

The exchange rate is real. The budget conclusion drawn from it is not. Measured on answers
instead of likelihood, **three displaced banks — the top of the claimed usable budget — leaves a
model that answers 2 of 150 held-out questions the source answers 47 of, and terminates on 53.**

## Method

`retention_probe.py` builds each arm in memory with exactly the construction `rigorous_pilot.py`
uses — `to_ced(encoder_layers = 28 − N, decoder_layers = N, donor_layer = 28 − N)` — and scores
it on the 150-row held-out probe with the package's own typed grammar. The frozen source side is
re-scored from the raw answers stored in the `G1_RETENTION_B28` receipt at digest `a329eb7b`; the
run aborts unless that reproduces both frozen references exactly (47 permissive / 41 strict), so
the reference is provably the one the registered gate uses.

`ced_lastN` displaces **N − 1** banks: `ced_last1` is the identity case.

Exploratory. Outside the receipt system: no receipt, no gate, no stage opened, no efficacy claim.

## Result

| arm | displaced banks | source | arm | reference | **lost** | terminated | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `ced_last1` | **0** | 47 | **47** | 47 | **0** | 148/150 | **PASS** |
| `ced_last2` | 1 | 47 | 43 | 52 | **9** | 149/150 | FAIL |
| `ced_last3` | 2 | 47 | 23 | 50 | **27** | 131/150 | FAIL |
| `ced_last4` | 3 | 47 | **2** | 48 | **46** | 53/150 | FAIL |
| `ced_last8` | 7 | 47 | **0** | 47 | **47** | **0/150** | FAIL |
| `D40` (width 1024→896, 0 banks) | — | 47 | 0 | 47 | 47 | 113/150 | FAIL |

Both bare policies agree on every verdict.

**`ced_last1` is the control and it holds exactly.** Zero displaced banks is the identity
transform, and the arm reproduces the source: 47/47, zero losses in all three classes, and all 24
recorded samples byte-identical to the source's. The programme's standing rule — "`ced_last1` and
`local group 1` must be bitwise identity" — is satisfied by the retention probe as well, which is
what makes the rest of the column trustworthy.

## Why this contradicts the CE-derived budget

Cross-entropy degrades smoothly and roughly linearly in the displaced count — that is the
measured law, and nothing here disputes it. Capability does not:

    displaced banks   0      1      2      3      7
    answers kept     47     43     23      2      0
    terminated      148    149    131     53      0

One bank already costs 9 of 52 reference answers. Two costs half of them. **Three — the top of
the claimed usable budget — costs 46 of 48, and termination collapses to about a third.** By
seven the model emits nothing that terminates at all.

A smooth per-bank price in nats, extrapolated linearly, says 2–3 banks is a modest 1–1.5 nat
expense. The answers say the model is gone. This is the same likelihood-versus-capability
decoupling the retention gate was built for, now visible in the *construction* budget rather than
in a training run — and it means the "~10% KV is usable" figure was never a capability claim.

## A second, smaller finding

The arms are not strictly worse than the source on every row: `ced_last2` answers **5** questions
the source gets wrong, `ced_last3` **3**, `ced_last4` **1**. Reference sizes exceed 47
accordingly. That churn is small and it never compensates — losses run 9, 27, 46 against gains of
5, 3, 1 — but it is real, and it is why the reference rule is a union rather than the source's
set alone.

Note that the `gained` field in the per-class counts is **identically zero** wherever the
reference is that union, because every arm-correct id is in the union by construction. The
meaningful quantity is `reference − source`, now reported as `arm_correct_outside_source`. An
adversarial pass flagged this as a vacuous metric; these runs confirm it empirically.

## What this does not establish

- **Initialization only.** No arm here was trained. The programme has measured that training
  moves CE enormously without moving capability, so a trained `ced_last3` is a separate question
  and is not answered here. What is answered is that the *construction* budget cannot be read off
  the CE curve.
- **One panel, one seed, greedy, 24 new tokens, 150 prompts of three numeric classes**, scored by
  a grammar whose label review is pending and which is known to reject the source's own
  `It ends at …` phrasing on 7 rows. A finite-panel measurement, not a population claim.
- **Not a receipt.** These arms have no `G1_QUALITY` receipt, so they cannot hold a
  `G1_RETENTION` one. This is an exploratory probe that uses the same scorer.
- The `to_ced` construction is one topology. The KV-sharing law says CE tracks the displaced
  count and not the topology; whether *capability* is equally topology-blind is untested.

## Reproduction

```bash
PY=C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe
cd C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification
for N in 1 2 3 4 8; do
  $PY experiments/rab001_exploration/retention_probe.py "ced_last$N" "ced_last$N" cpu
done
```

~3–8 minutes per arm on CPU, one at a time. Artifacts:
`evidence/retention-design-v1/RETENTION_PROBE_ced_last{1,2,3,4,8}.json`.
