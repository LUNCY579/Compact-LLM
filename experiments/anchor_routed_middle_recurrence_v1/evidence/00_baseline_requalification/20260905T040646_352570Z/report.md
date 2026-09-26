# Baseline requalification

Status: PASS.

Question: does the original anchor-layer hierarchy reproduce under the new
cache-free harness? Setup: same eight smoke prompts, seed 17, CPU FP32/eager,
four threads, canonical Supa custom class and abstention offset. Exact model,
source, runtime, dataset hashes and timings are in run_manifest.json.

Mechanical qualification: qualification.json records inactive/restored bit
identity, historical baseline greedy-ID identity, cache prefill/decode audit,
skip effects and tolerance checks. Final manifest records before/after state and
checkpoint hashes, zero trainable parameters and gradients, and peak RSS.

| Layer | Historical skip cosine | Reproduced | Absolute difference |
|---:|---:|---:|---:|
| 0 | 0.383896 | 0.383896 | 0.000000 |
| 1 | 0.303229 | 0.303229 | 0.000000 |
| 2 | 0.062311 | 0.062311 | 0.000000 |
| 8 | 0.977926 | 0.977926 | 0.000000 |
| 19 | 0.713742 | 0.713742 | 0.000000 |

Interpretation: early anchor necessity and L8 removal tolerance reproduce only
if all checks pass. Expected negative-control corruptions end the affected case
early; unlike the historical decoder, the new decoder does not finish a known
loop. First-token skip comparisons precede those generation stopping decisions.

Non-claims: skip tolerance is not repeatability or efficacy; intact cache
equivalence does not establish cache semantics for recurrent passes.

Next gate: middle-layer frozen reapplication screen.
