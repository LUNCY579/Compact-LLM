# Published-family long-context search report

Date: 2026-08-26. Primary metric: normalized strict exact match on `Rineya_Long_Context_Demo_v1`.

Rename note: Supa was formerly named Taria, and Baya was formerly named Lumina. The rename changed identity metadata only; frozen weights, hashes, parity evidence, and scores are unchanged. Historical artifact and publisher-script paths retain their original names for auditability.

## Result

The best measured answer-blind family cascade reaches:

| Split | Exact | BP packs | CF packs | UNKNOWN | All-four packs |
|---|---:|---:|---:|---:|---:|
| Showcase 64 (selection) | **48/64 (75.0%)** | 13/16 | 11/16 | 9/16 | 7/16 |
| Stability 192 (frozen validation) | **128/192 (66.7%)** | 36/48 | 27/48 | 19/48 | 14/48 |
| Complete 256 | **176/256 (68.8%)** | 49/64 | 38/64 | 28/64 | 21/64 |

**Baya 2.0 System46** is the recommended efficiency/performance tradeoff.
**Supa 2.0 System48** is the nine-arm search ceiling. Both are versioned
routed-system bundles, not single dense checkpoints.

All routing rules consume predictions only. They never inspect answer keys at inference time.

## Published cores

Two immutable BF16 dense cores were added transactionally to `Rineya_Published_Cores`:

| Core | Role | Dense Showcase | Dense Stability | Dense all-256 | Weight SHA-256 |
|---|---|---:|---:|---:|---|
| Supa 1.7.0 | low-scale paired-answer arm | 40/64 | 121/192 | 161/256 | `962abc5b…dd24b` |
| Baya 1.7.0 | high-scale abstention arm | 40/64 | 112/192 | 152/256 | `1737cd1e…8bba` |

Supa is Nimbus plus the disjoint Stage-A checkpoint-200 long-context LoRA at scale 0.8125. Baya uses scale 1.25. Dense merging changed 5/256 and 7/256 raw predictions respectively relative to PEFT execution; their actual dense scores and parity differences are recorded in each `CORE.json`.

The dense Baya-high/Supa-low UNKNOWN gate scores 42/64, 124/192, and 166/256. The higher family frontier uses PEFT/task-vector arms around the published anchors rather than pretending dense parity was exact.

## Fresh same-protocol baselines

Every active published core was run from its actual vault directory with BF16 loading, greedy decoding, thinking disabled, 24 generated tokens, seed 17, context 4096, and no truncation.

| Core | Geometry | Showcase | Stability | All 256 |
|---|---:|---:|---:|---:|
| Aegis | 19L | 2/64 | 20/192 | 22/256 |
| Lyra | 28L | 28/64 | 86/192 | 114/256 |
| Nimbus | 28L | 30/64 | 83/192 | 113/256 |
| Halo | 28L | 29/64 | 86/192 | 115/256 |
| Pulse | 28L | 31/64 | 89/192 | 120/256 |
| Halo Uncensored | 28L | 31/64 | 96/192 | 127/256 |
| Baya | 28L | 40/64 | 112/192 | 152/256 |
| Supa | 28L | 40/64 | 121/192 | 161/256 |

## Per-anchor frontier

| Anchor | Best tested recipe | Showcase | Stability | Baseline gain on Showcase |
|---|---|---:|---:|---:|
| Pulse | negative-vector two-arm UNKNOWN gate | 43/64 | 118/192 | +12 |
| Halo | checkpoint-200 scale 0.875/1.25 gate | 41/64 | 112/192 | +12 |
| Lyra | checkpoint-200 scale 0.8125/1.25 gate | 41/64 | 118/192 | +13 |
| Nimbus | checkpoint-200 scale 0.8125/1.25 gate | 45/64 | 124/192 | +15 |
| Halo Uncensored | baseline retained; transferred delta harmed it | 31/64 | 96/192 | 0 |
| Aegis | native Pulse-1.0/Aegis dense LERP at 0.5 | 6/64 | not promoted | +4 |

Pulse’s best single adapter is the exact negative vector `1.25*delta200 - 0.25*delta250`: 41/64 Showcase and 117/192 Stability. A stronger negative arm, `1.5*delta200 - 0.5*delta250`, supplies complementary abstentions used by both final cascades.

## Search coverage

The combined evidence contains 147 complete 64-row reports with unique run names. Tested families include:

- all eight active published-core baselines;
- checkpoint-200 global scales, including negative and extrapolated values;
- checkpoint-225/250 raw and norm-matched controls;
- exact checkpoint soups and SWA;
- four predeclared negative task vectors;
- per-layer, depth, attention/MLP, and output-projection scale profiles;
- structured LoRA rank pruning;
- family-local adapter transfer;
- native 19-layer Aegis/Pulse dense interpolation;
- exhaustive ordered UNKNOWN gates over all available prediction vectors;
- millions of modal-vote combinations;
- ordered non-UNKNOWN consensus cascades.

Checkpoint continuation, convex soup, ordinary profile shaping, and rank pruning all stopped after failing their predefined improvement gates. Halo Uncensored stopped after both transfer sentinels lost accuracy and abstention remained zero.

Dense all-family TIES, DARE-TIES, SLERP, Fisher merging, activation-guided grafting, and evolutionary search were not claimed as executed: the repository does not yet contain verified implementations for those operations, and they are not equivalent to the adapter methods above. The implemented fast matrix was exhausted under explicit stop rules.

## Final cascades

### Baya 2.0 System46 — three-arm efficiency frontier

1. Nimbus + checkpoint-200 at scales 0.8125 and 1.25, using high only when it says `UNKNOWN`.
2. Pulse + `1.5*delta200 - 0.5*delta250`; use it only when it says `UNKNOWN`.

This improves the prior Nimbus gate by one Showcase and one Stability hit without a label-dependent selector.

### Supa 2.0 System48 — nine-arm search ceiling

Start from the three-arm state, then apply three ordered consensus pairs. Preserve an existing `UNKNOWN`; otherwise replace only when both pair arms agree on the same non-`UNKNOWN` normalized answer:

1. Halo Uncensored baseline + Nimbus scale 1.0625.
2. Pulse baseline + Pulse checkpoint-250 scale 0.75.
3. Halo scale 0.25 + Halo scale 0.75.

The Showcase-selected chain gained three cases with no Showcase losses. Frozen Stability improved from 124/192 for the original Nimbus gate to 128/192.

## Reproducibility

- `configs/final/pulse_strong_over_nimbus_gate46.json`
- `configs/final/nimbus_pulse_consensus_cascade_showcase48.json`
- `scripts/build_abstention_scale_ensemble.py`
- `scripts/build_consensus_cascade.py`
- `scripts/run_adapter_scale_sweep.py`
- `Rineya_Published_Cores/_publish_taria_lumina.py`

Generated predictions, manifests, reports, and per-candidate evidence are under `artifacts/rineya_published_core_search_v1/` and `artifacts/rineya_long_context_weight_geometry_v1/`.

## Caveats

Showcase was used for selection. Stability was evaluated after each recipe was frozen and must not be used for further tuning. The nine-arm result has substantially higher latency than a single core. These synthetic controlled-reading results do not establish broad long-document capability.
