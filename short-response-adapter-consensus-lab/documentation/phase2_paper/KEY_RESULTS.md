# Key results

## v2.2 TEST cornerstone (N=4,050)

| Method | Exact accuracy |
|---|---:|
| Backbone | 0.025% |
| Abstention specialist | 13.012% |
| Raw probability K=7 | 14.099% |
| Track A | 14.593% |
| Hard plurality K=7 | **17.259%** |

- Hard minus Track A: +2.667 pp; 328 rescues / 220 harms; Holm
  `p=1.363e-5`.
- Hard minus raw probability: +3.160 pp; 226 rescues / 98 harms; Holm
  `p=4.358e-12`.
- Versus Track A, the net +108 successes consist of +230 abstention successes
  and −122 answerable successes.

## Same-split DEV teacher-forced and free-running (N=4,050)

| Method | Gold-prefix all-token example | Free-running exact |
|---|---:|---:|
| Hard plurality | 25.259% | 25.309% |
| Raw probability | 26.617% | 26.593% |
| Calibrated probability | **26.765%** | **26.790%** |
| Reciprocal rank top-5 | 18.469% | 18.321% |

The ordering is preserved. Calibrated probability beats hard free-running by
+1.481 pp (Holm `p=2.773e-4`); reciprocal rank trails hard by −6.988 pp (Holm
`p=2.189e-49`).

## Committee and adaptive systems

- Selected free-running K=1…7 accuracy: 26.099%, 25.012%, 27.185%, **27.333%**,
  25.877%, 26.395%, and 25.309%. K=4 is best.
- Removing Grounding improves K=7 by 1.086 pp; Procedure is neutral; Track A,
  Track B, General, Abstention, and Social contribute positively in the deployed
  committee context.
- Adaptive accuracy is 25.111% versus hard K=7 at 25.309% (`p=0.546`), with
  56.579% fewer mean expert forwards and mean latency of 0.928 s versus 1.919 s.
- Adaptive behavior improves answerable accuracy by 1.168 pp and reduces
  abstention accuracy by 9.074 pp.

## Confirmation and transfer

- Calibrated probability on v2.2 TEST: 13.827%; 3.432 pp below hard, Holm
  `p=1.073e-15`; statistically indistinguishable from raw probability.
- On v2.1.2 (N=1,320), hard scores 0.076% versus raw probability at 3.030%;
  hard minus raw is −2.955 pp, Holm `p=1.528e-10`.

Canonical expanded tables are in [paper/TABLES.md](../../paper/TABLES.md), and
the final machine fact packet is
[final_synthesis_facts.json](../../results/phase2_mechanism/final_synthesis_facts.json).
