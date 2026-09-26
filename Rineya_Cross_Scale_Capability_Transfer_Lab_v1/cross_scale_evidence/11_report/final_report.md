# Cross-Scale Capability Transfer into Rineya Supa

## Abstract

This study tested whether parameter information from official Qwen3-4B and
Qwen3-8B could improve the 596,049,920-parameter Rineya Supa checkpoint without
requiring a larger model at inference. All model construction used FP32
arithmetic and emitted exact Supa/Qwen3-0.6B-shaped checkpoints. Development,
calibration, and a fresh private sealed bank were hash-separated; Invisible-
Limit TEST2 was denylisted and never accessed.

Direct cross-scale truncation, norm-ranked subsets, embedding-ridge alignment,
and embedding-Procrustes alignment produced structurally valid but incoherent
models. A target-basis spectral operator was viable only at strength 0.10. Its
4B and 8B deltas were strongly aligned (cosine 0.932; sign agreement 0.926), and
a mean-consensus injection at alpha 0.15 improved MERGEDEV from 47/59 to 48/59
with no breaks. The gain did not replicate on DEV-FULL: both models scored
144/180, with one fix and one regression and a slightly negative scenario-
macro delta. No finalist was frozen and the sealed holdout was not accessed.

The bounded conclusion is negative: scale-consistent spectral information is
measurable and can alter behavior, but the tested transfer procedures did not
produce a robust overall Supa improvement.

## Research design

The preregistration fixed hypotheses, coefficient grids, preservation gates,
scenario-cluster statistics, successive halving, and sealed-access rules before
model inspection or scoring. The study used:

- calibration: 59 rows / 49 scenarios;
- MERGEDEV-fast: 16 rows / 13 scenarios;
- MERGEDEV: 59 rows / 49 scenarios;
- DEV-FULL: 180 rows / 148 scenarios;
- sealed: 239 rows / 197 scenarios, generated and hashed but never read for
  model development or confirmation.

Evaluation was greedy, thinking-disabled, batch-size one, CPU float32, seed 17,
with semantic capability, format adherence, completion, invalid output, and
scenario identity recorded separately.

## Architecture audit

| Model | Layers | Hidden | MLP | Q heads | KV heads | Head dim | Effective parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 28 | 1024 | 3072 | 16 | 8 | 128 | 596,049,920 |
| Supa-A1.1 | 28 | 1024 | 3072 | 16 | 8 | 128 | 596,049,920 |
| Qwen3-4B | 36 | 2560 | 9728 | 32 | 8 | 128 | 4,022,468,096 |
| Qwen3-8B | 36 | 4096 | 12288 | 32 | 8 | 128 | 8,190,735,360 |

All tokenizer bundle hashes are identical. Official 0.6B redundantly stores a
bit-identical tied output head; Supa stores the tie implicitly. Qwen3-8B is
untied, requiring an explicit embedding/output collapse policy for direct
width projection.

## Projection ladder

### Deterministic coordinate projection

Eight P1 controls covered prefix, uniform full-depth, and edge-preserving depth
maps with head-aware prefix or norm-ranked width selection. Every arm had exact
keys/shapes, finite logits, and 596,049,920 parameters. Every arm failed the
functional gate with repetitive or malformed output and missing EOS.

### Linear basis alignment

- 4B embedding ridge: sample RMSE 0.019, held-out RMSE 0.965, held-out cosine
  0.335, target-cycle error 0.715; incoherent output.
- 4B scaled Procrustes: held-out RMSE 0.859, cosine 0.536, cycle error
  3.2e-6; incoherent output.
- 8B scaled Procrustes: held-out RMSE 1.028, cosine 0.275, cycle error
  3.2e-6; incoherent output.

The 8B donor was harder to align than 4B despite greater source capacity.

### Target-basis spectral transfer

P4 retained official 0.6B singular vectors, embeddings, norms, and tensor
Frobenius norms while transplanting normalized top-eight donor singular-value
profiles. Strength 1.0 collapsed. Strength 0.10 passed every functional check
for both donors. This is partial spectral transfer, not intact donor compression.

## Baselines and controls

### MERGEDEV

| Model | Semantic | Scenario macro | Format | Completion | Invalid |
|---|---:|---:|---:|---:|---:|
| Official Qwen3-0.6B | 42/59 | 0.789 | 45.8% | 66.1% | 0 |
| Supa ungated | 47/59 | 0.755 | 88.1% | 100% | 0 |
| Supa published gate | 47/59 | 0.755 | 88.1% | 100% | 0 |
| Projected 4B spectral control | 42/59 | 0.776 | 39.0% | 59.3% | 0 |
| Projected 8B spectral control | 43/59 | 0.810 | 47.5% | 67.8% | 0 |

The published Supa logit offset changed zero responses. The 8B projected
control was one case above 4B/Qwen but remained four cases below Supa.

## Scale-trajectory diagnostics

For the viable strength-0.10 P4 deltas:

- global cosine: 0.9320;
- angular deviation: 21.25 degrees;
- sign agreement: 0.9264;
- magnitude correlation: 0.9145;
- delta4/base L2 ratio: 0.00260;
- delta8/base L2 ratio: 0.00301.

Module cosines were 0.91-0.95 across attention and MLP projections. Embeddings
and norms were unchanged by construction. This is strong evidence for a common
spectral scale statistic, but not by itself evidence for transferred capability.

## Candidate funnel

Representative fast-screen controls:

- 8B absolute interpolation, alpha 0.01: tied Supa at 12/16;
- 8B task vector, alpha 0.20: tied Supa at 12/16;
- mean consensus, alpha 0.10: tied Supa at 12/16;
- mean consensus, alpha 0.15 and 0.20: 13/16 vs Supa 12/16.

Alpha 0.15 won the frozen smaller-coefficient tie-break. FP16 retained the
intended perturbation across 304,319,933 coordinates; its exact weight SHA-256
is `5758b7af3334af517b6b38c63605e6d74b62617c75bb9e69544e815f8f713a51`.

### MERGEDEV promotion

Consensus alpha 0.15 scored 48/59 versus Supa 47/59, fixing one
counterfactual case and breaking none. Exact McNemar p=1.0; scenario-bootstrap
95% CI was [0.000, 0.061]. Format rose from 88.1% to 89.8%; completion remained
100%.

### DEV-FULL confirmation

| Model | Semantic | Worst family | Scenario macro | Format | Completion |
|---|---:|---:|---:|---:|---:|
| Supa | 144/180 | 0.083 | 0.7703 | 91.1% | 100% |
| Consensus alpha 0.15 | 144/180 | 0.083 | 0.7658 | 91.1% | 100% |

The candidate fixed one paraphrase case and broke one mathematics case. Exact
McNemar p=1.0. Scenario-bootstrap 95% CI was [-0.0203, 0.0068], with 0.233
bootstrap probability of a positive macro delta. The promotion tool returned
zero eligible finalists.

## Advanced methods

TIES, DARE, DARE-TIES, Fisher estimation/merging, and layer/module injection
are implemented and covered by deterministic pure and end-to-end serialization
tests. Real-checkpoint outcome experiments were not run because the
preregistered localization/advanced gate required a positive larger-development
signal. Running them after DEV-FULL tied would increase adaptive-search bias.

## Answers to the twenty research questions

1. **Can 4B improve Supa?** Not robustly under tested methods. It contributes to
   a consensus with a MERGEDEV-only gain.
2. **Can 8B improve Supa?** Not robustly. Its projected control was slightly
   better than 4B, but still below Supa.
3. **Is 8B a better donor?** Inconclusive. It scored one projected-control case
   higher, while its basis-alignment distortion was substantially worse.
4. **Best projection method?** Target-basis rank-8 spectral transfer at strength
   0.10; it was the only coherent donor-derived projection family.
5. **Best interpolation coefficient?** Development selected mean-consensus
   alpha 0.15, but it did not confirm on DEV-FULL.
6. **Does small alpha beat 50/50?** Small transfer was necessary for coherence.
   A 50/50 outcome comparison was not authorized after full-strength controls
   collapsed, so no benchmark superiority claim is made.
7. **Task vector versus absolute interpolation?** Representative versions tied
   Supa on the fast screen; neither showed a positive signal.
8. **Are delta4 and delta8 aligned?** Yes, strongly: cosine 0.932 and 92.6% sign
   agreement.
9. **Evidence for a common scale trajectory?** Yes for target-basis spectral
   statistics; no robust evidence that it improves behavior.
10. **Which layers are most transferable?** Not estimated; localization was
    outcome-gated after DEV-FULL.
11. **Which modules are most transferable?** Behavioral transfer was not
    localized. Descriptively, all attention/MLP module deltas aligned strongly.
12. **Attention or MLP deltas?** No outcome comparison. Their cross-donor
    cosines were similarly high (approximately 0.91-0.95).
13. **Middle/late concentration?** Not tested after the localization no-go.
14. **Do TIES/DARE reduce interference?** Tooling exists; no authorized outcome
    result.
15. **Does Fisher weighting help?** Tooling exists; no authorized outcome
    result.
16. **Can 4B+8B consensus beat either donor?** It beat both projected controls
    and Supa by one MERGEDEV case, but tied Supa on DEV-FULL.
17. **Can the final model beat prior strong Rineya baselines?** No robust final
    model was established.
18. **Are improvements preserved outside development selection?** No. The
    MERGEDEV gain disappeared on DEV-FULL.
19. **Does inference remain approximately 0.6B?** Yes: 596,049,920 effective
    parameters and 1,192,134,784 weight bytes; no donor required.
20. **Compute required?** Frozen-valid evaluation covered 869 rows and 5,012
    generated tokens in 1,598.3 seconds of model-execution wall time (0.444
    CPU-hours), with 0 GPU-hours. Early projection/merge wall times were not
    instrumented precisely and are not retroactively invented.

## Final conclusion

Useful scale-associated parameter statistics can be projected into a 0.6B
target basis, and 4B/8B spectral deltas show a coherent shared direction. Under
the frozen funnel, however, injecting that direction did not produce a robust
overall Supa improvement. The correct scientific result is a bounded negative
finding with a reproducible failure boundary—not a claim of compressed 8B
capacity.

No sealed confirmation, recovery LoRA, or release candidate was run because no
training-free candidate passed DEV-FULL promotion.
