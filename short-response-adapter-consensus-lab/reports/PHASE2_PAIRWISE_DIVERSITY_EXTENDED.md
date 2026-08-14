# Extended pairwise expert diversity

Status: **PAIRWISE_DIVERSITY_EXTENDED_COMPLETE**

| Expert pair | Mean JS (TEST hard path) | DEV disagreement | DEV double fault | DEV correctness MI (nats) | Success Jaccard | Failure Jaccard |
|---|---:|---:|---:|---:|---:|---:|
| procedure / abstention | 0.4048 | 0.6719 | 0.3133 | 0.0337 | 0.4532 | 0.4548 |
| grounding / abstention | 0.3788 | 0.6498 | 0.2832 | 0.0348 | 0.4691 | 0.4266 |
| procedure / social | 0.3705 | 0.6661 | 0.3594 | 0.0731 | 0.4973 | 0.5274 |
| grounding / social | 0.3691 | 0.6658 | 0.2994 | 0.0364 | 0.4478 | 0.4363 |
| track_a / social | 0.3157 | 0.5885 | 0.1845 | 0.0711 | 0.4943 | 0.3092 |
| general / social | 0.3046 | 0.5908 | 0.2137 | 0.0749 | 0.5023 | 0.3532 |
| general / abstention | 0.2818 | 0.5676 | 0.2012 | 0.0639 | 0.5272 | 0.3475 |
| track_a / abstention | 0.2774 | 0.5622 | 0.1745 | 0.0612 | 0.5230 | 0.3070 |
| track_b / social | 0.2703 | 0.5780 | 0.1948 | 0.0730 | 0.4976 | 0.3251 |
| track_a / procedure | 0.2655 | 0.4139 | 0.1882 | 0.1329 | 0.6517 | 0.3995 |
| track_b / procedure | 0.2593 | 0.4256 | 0.1970 | 0.1328 | 0.6539 | 0.4147 |
| general / procedure | 0.2549 | 0.4185 | 0.2181 | 0.1475 | 0.6672 | 0.4560 |
| track_b / abstention | 0.2466 | 0.5515 | 0.1840 | 0.0627 | 0.5251 | 0.3219 |
| abstention / social | 0.2439 | 0.4494 | 0.5257 | 0.4343 | 0.8464 | 0.8783 |
| general / grounding | 0.2007 | 0.3413 | 0.2087 | 0.1536 | 0.7171 | 0.4826 |
| track_a / grounding | 0.1960 | 0.3330 | 0.1841 | 0.1487 | 0.7109 | 0.4383 |
| track_b / grounding | 0.1940 | 0.3337 | 0.1929 | 0.1509 | 0.7138 | 0.4550 |
| grounding / procedure | 0.1883 | 0.3455 | 0.3695 | 0.3190 | 0.8027 | 0.7481 |
| track_b / general | 0.0976 | 0.1705 | 0.1978 | 0.3374 | 0.9323 | 0.7845 |
| track_a / track_b | 0.0810 | 0.1373 | 0.1800 | 0.3163 | 0.9360 | 0.7742 |
| track_a / general | 0.0604 | 0.1381 | 0.1950 | 0.3595 | 0.9415 | 0.8054 |

Distributional and correctness diversity are descriptive. Large values are not automatically useful: low-competence experts can be diverse for the wrong reason.

Raw TEST SHA-256: `d3733391b53dc53e7d9ca899d7015ba6ff0b2faa7c28e52a2778a0827c45c32d`.
