# The first CED-family arm to pass a capability gate — and what that does *not* mean

RAB-001 falsified CED **conversion by displacement**: splitting 28 layers into 14 encoder + 14
decoder destroys the model, and the displaced-bank sweep measured the cliff (0 banks keeps 47/47;
3 banks leaves 2 of 48; 7 banks terminates on nothing).

D28-001 proposes a different transition. H56 and P56 keep **all 28 decoder blocks with their
self-attention intact** and *add* a memory read after each, with the read output projection
initialised to **zero**. Nothing is displaced. The prediction is that they reproduce the source
exactly at initialisation.

Run on the 150-row typed panel against the paired 47-item reference at digest `a329eb7b`, real
official Supa 1.7.0-devfreeze weights, built on GPU and scored on CPU (the reference's device):

| arm | parameters | KV banks | transition | source | arm | **lost** | terminated | verdict |
|---|---:|---:|---|---:|---:|---:|---:|---|
| **H56** | 1,156,087,424 | 57 | additive, zero-init read | 47 | **47** | **0** | 148/150 | **PASS** |
| **P56** | 1,156,087,424 | 57 | additive, same-token sigmoid | 47 | **47** | **0** | 148/150 | **PASS** |
| **D56** | 1,036,516,352 | 56 | identity-expanded depth | 47 | **47** | **0** | 148/150 | **PASS** |
| C56 | 979,890,816 | 29 | memory-only *replacement* | 47 | **0** | 47 | **0/150** | FAIL |
| C28 | 568,786,304 | 15 | 14+14 split (the old arm) | 47 | **0** | 47 | 1/150 | FAIL |

All three additive arms are **byte-identical to the source on every recorded sample (24/24)**.
Both bare policies agree on every arm.

## What this establishes

**The transition is the whole story, not the architecture.** C56 is the *same two-stack
shared-memory idea* as H56 — one encoder, one decoder, a shared K/V bank — built as a
*replacement*: the decoder loses its self-attention and reads shared encoder K/V instead. It
produces token salad and terminates on **nothing** (`'60 units.'` → `searchDataevityevityevity…`).
H56 keeps the self-attention and *adds* the read. Same idea, opposite outcome.

That is the first positive structural result for the CED family in this programme, and it
converts the earlier negative into something precise: **shared causal memory is not what killed
the converted arms — removing the decoder's own attention to make room for it is.**

**First execution on real weights.** Upstream, D28-001 had only random tiny fixtures and
meta-device shape accounting: `official_weights_loaded: false`,
`evidence_profile: local_scoped_random_weight_mechanics`, and an
`official_preflight` of `BLOCKED_OR_INTERRUPTED — missing or mismatched official source asset`.
The 1,156,087,424-parameter count is now confirmed against real tensors rather than meta ones,
and the construction is verified function-preserving at full scale.

## What this does **not** establish, and it is the larger half

**H56 passing is expected by construction, so it is necessary and nowhere near sufficient.** With
the read output projection at zero the memory branch contributes exactly 0 to the residual
stream. H56 at initialisation *is* B28 plus 560M parameters of dead weight. Scoring 47/47 tests
that the construction is wired correctly — it tests nothing about whether shared memory helps.

D28-001 says so itself: *"A zero-output identity test alone is insufficient."* Its `active_path`
fixture exists precisely because a zero branch makes every downstream check vacuous.

Three specific gaps:

1. **H56 and P56 are indistinguishable here — necessarily.** P56 is the control that isolates
   *context* from *parameter count*: its read sees only the same token's memory. At
   initialisation both are B28, so both score 47/47. The control only becomes informative after
   the read branch is trained. This run cannot separate them and was never able to.
2. **D56 also passes**, and it is an ordinary depth control with no memory at all. So a 47/47 at
   init distinguishes *nothing* among the additive arms — it only separates additive from
   replacement.
3. **The cost is real and unpaid.** 596M → 1.156B parameters, 28 → 57 KV banks. This is
   **more** memory, not less. FWT-001 orders it correctly: viability first, compression second.
   Nothing here has shown viability beyond "does not break at init".

## The next test, and what would falsify it

The live question is whether the added read pathway becomes *useful* rather than merely
harmless. That is D28-001's P2 engineering smoke — 25 steps, added-parameters-only AdamW at
lr 1e-5, H and P matched on data order and optimizer policy — followed by a retention probe on
each.

The informative outcomes:

- **H improves on the probe and P does not** → the *context* in the shared memory is doing work.
  That is the result the architecture is for.
- **H and P improve equally** → the gain is parameter count, not memory. The architecture is not
  earning its keep.
- **Both lose reference answers** → the added branch is a liability under training, and the
  retention gate catches it the way it caught lr 1e-4.

Note the last one is live: the dose test measured that at lr 1e-4 this source loses 26 of 47
reference answers while CE improves 3.93 nats. P2 runs at lr 1e-5, where the same test lost 6 of
47. **A 25-step smoke is also a null dose** — the registered engineering lane changed 0 of 150
answers on B28 — so it may well show nothing at all, and that would not be evidence of safety.

## Reproduction

```bash
PY=C:/Users/clclc/BusinessProject/.venv-rab001-cu126/Scripts/python.exe
cd C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification
for A in H P D56 C C28; do
  $PY experiments/rab001_exploration/retention_probe.py "dual28_$A" "dual28_$A" cpu
done
```

H56 is 1.156B parameters: construction peaks at source + result, so it is built on the GPU and
moved to the CPU for scoring. It needs roughly 6.5 GiB free host at the move — a first attempt
was stood down by `hostbudget.guard` at 1.39 GiB free and is preserved as such. Artifacts:
`evidence/retention-design-v1/RETENTION_PROBE_dual28_*.json`.
