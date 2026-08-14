# Phase 2 Rescue, Harm, and Existing-Trace Analysis

**Status: complete for analyses derivable from the frozen full-TEST artifact; new-inference mechanisms remain pending.**

The cornerstone verification passed before this analysis ran. Global paired comparisons retain their confirmatory interpretation under the frozen comparison family. Category, domain, difficulty, length, agreement, coalition, and trace-profile cuts are descriptive/exploratory and are not presented as independently confirmatory tests.

## Central rescue/harm result

| Candidate vs baseline | Both correct | Rescue | Harm | Both wrong | Rescue rate | Harm rate | Net gain | 95% paired bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hard K=7 vs Track A | 371 | 328 | 220 | 3131 | 8.10% | 5.43% | +2.67 pp | [+1.53 pp, +3.80 pp] |
| Hard K=7 vs Probability K=7 | 473 | 226 | 98 | 3253 | 5.58% | 2.42% | +3.16 pp | [+2.30 pp, +4.02 pp] |

Hard voting is not winning by preserving every correct baseline answer. Against Track A it rescues **328** cases and harms **220**; against probability mean it rescues **226** and harms **98**. The mechanism must therefore explain asymmetric error correction, not simple dominance.

## Answerable versus abstention composition

| Comparison | Class | N | Candidate acc. | Baseline acc. | Rescue | Harm | Net gain |
|---|---|---:|---:|---:|---:|---:|---:|
| Hard K=7 vs Track A | answerable | 3510 | 12.59% | 16.07% | 98 | 220 | -3.48 pp |
| Hard K=7 vs Track A | abstention | 540 | 47.59% | 5.00% | 230 | 0 | +42.59 pp |
| Hard K=7 vs Probability K=7 | answerable | 3510 | 12.59% | 7.95% | 178 | 15 | +4.64 pp |
| Hard K=7 vs Probability K=7 | abstention | 540 | 47.59% | 54.07% | 48 | 83 | -6.48 pp |

Behavioral policy metrics make the answer/abstain tradeoff explicit:

| Method | Overall | Answerable | Abstention | False abstention | Failure to abstain | Mean tokens |
|---|---:|---:|---:|---:|---:|---:|
| Track A | 14.59% | 16.07% | 5.00% | 3.87% | 95.00% | 4.03 |
| Abstention expert | 13.01% | 0.00% | 97.59% | 91.37% | 0.19% | 10.92 |
| Probability K=7 | 14.10% | 7.95% | 54.07% | 54.90% | 45.93% | 7.50 |
| Hard K=7 | 17.26% | 12.59% | 47.59% | 25.36% | 52.41% | 5.73 |

## Task-category rescue/harm

| Task category | N | Hard−Track A | Rescue / harm | Hard−Probability | Rescue / harm |
|---|---:|---:|---:|---:|---:|
| aggregate_delta | 270 | +0.00 pp | 0 / 0 | +0.00 pp | 0 / 0 |
| capacity_fit | 270 | -2.22 pp | 34 / 40 | +10.74 pp | 33 / 4 |
| compare_metric | 270 | -3.33 pp | 14 / 23 | +9.63 pp | 31 / 5 |
| conflicting_evidence | 270 | +70.37 pp | 190 / 0 | +10.74 pp | 47 / 18 |
| constraint_preserving_transform | 270 | -9.63 pp | 4 / 30 | +12.22 pp | 36 / 3 |
| constraint_satisfaction | 270 | +2.22 pp | 17 / 11 | +9.63 pp | 26 / 0 |
| dependency_order | 270 | -9.26 pp | 10 / 35 | +8.52 pp | 24 / 1 |
| diagnostic_priority | 270 | +0.00 pp | 0 / 0 | +0.00 pp | 0 / 0 |
| evidence_binding | 270 | -0.74 pp | 0 / 2 | +0.37 pp | 2 / 1 |
| evidence_synthesis | 270 | -11.85 pp | 0 / 32 | +0.00 pp | 0 / 0 |
| filter_then_rank | 270 | +0.37 pp | 12 / 11 | +3.33 pp | 9 / 0 |
| insufficient_information | 270 | +14.81 pp | 40 / 0 | -23.70 pp | 1 / 65 |
| rule_exception | 270 | -2.96 pp | 5 / 13 | +5.19 pp | 15 / 1 |
| state_transition | 270 | -8.52 pp | 0 / 23 | +0.74 pp | 2 / 0 |
| two_hop_relation | 270 | +0.74 pp | 2 / 0 | +0.00 pp | 0 / 0 |

## Free-running hard-versus-soft divergence

Hard and probability mean produce exactly the same complete token sequence on **2246/4050 (55.46%)** examples. They diverge on **1804 (44.54%)**.

| Paired outcome | All examples | Divergent examples | Median first divergence t | Soft winner has zero top-1 supporters | Higher-confidence hard dissenter |
|---|---:|---:|---:|---:|---:|
| both_correct | 473 | 0 | None | 0 | 0 |
| hard_only | 226 | 226 | 0.0 | 0 | 178 |
| probability_only | 98 | 98 | 0.0 | 0 | 96 |
| both_wrong | 3253 | 1480 | 0.0 | 1 | 1371 |

At each first divergence the two methods still share an identical accepted prefix. Their expert top-1 proposals match across separately generated traces on every audited divergence; any selected-token difference therefore comes from aggregation, not different local model states. This is a free-running trajectory description, not yet the required teacher-forced causal comparison.

## Coalition and directional-confidence trace evidence

The Hard-K7 trace contains **23,204** token decisions and **107** distinct winner-support masks.

| Mask | Members | Tokens | Token share | Sequence-correct association | EOS share |
|---|---|---:|---:|---:|---:|
| `1111111` | track_a, track_b, general, grounding, procedure, abstention, social | 8133 | 35.05% | 20.30% | 40.05% |
| `1111100` | track_a, track_b, general, grounding, procedure | 2754 | 11.87% | 16.49% | 8.50% |
| `1110011` | track_a, track_b, general, abstention, social | 2470 | 10.64% | 24.86% | 0.00% |
| `1111000` | track_a, track_b, general, grounding | 971 | 4.18% | 11.95% | 1.13% |
| `1110000` | track_a, track_b, general | 896 | 3.86% | 41.41% | 28.46% |
| `1111010` | track_a, track_b, general, grounding, abstention | 850 | 3.66% | 12.82% | 0.00% |
| `1111011` | track_a, track_b, general, grounding, abstention, social | 627 | 2.70% | 25.52% | 0.64% |
| `0000011` | abstention, social | 591 | 2.55% | 7.11% | 0.00% |
| `1110111` | track_a, track_b, general, procedure, abstention, social | 390 | 1.68% | 37.44% | 0.00% |
| `1101000` | track_a, track_b, grounding | 345 | 1.49% | 2.90% | 0.29% |
| `0100011` | track_b, abstention, social | 325 | 1.40% | 56.00% | 0.00% |
| `1110010` | track_a, track_b, general, abstention | 325 | 1.40% | 45.54% | 0.00% |
| `1111110` | track_a, track_b, general, grounding, procedure, abstention | 321 | 1.38% | 3.74% | 0.31% |
| `1110100` | track_a, track_b, general, procedure | 314 | 1.35% | 7.01% | 24.84% |
| `1011000` | track_a, general, grounding | 294 | 1.27% | 1.02% | 0.00% |

Coalition sequence-correct association is descriptive: a token coalition inherits the final sequence label and is not itself a gold-token accuracy measure.

| Expert | Winner-support rate | Mean top-1 p | Mean margin | Mean entropy | p when supporting | p when dissenting |
|---|---:|---:|---:|---:|---:|---:|
| track_a | 88.74% | 0.8112 | 0.7673 | 0.9670 | 0.8776 | 0.2878 |
| track_b | 89.01% | 0.7578 | 0.7157 | 1.4158 | 0.8141 | 0.3017 |
| general | 85.13% | 0.7562 | 0.7080 | 1.4073 | 0.8529 | 0.2031 |
| grounding | 68.78% | 0.7376 | 0.6590 | 1.3653 | 0.8266 | 0.5414 |
| procedure | 59.00% | 0.6978 | 0.6427 | 1.7862 | 0.8485 | 0.4811 |
| abstention | 63.81% | 0.8059 | 0.7771 | 1.1325 | 0.9650 | 0.5254 |
| social | 58.40% | 0.6141 | 0.5595 | 2.3294 | 0.8575 | 0.2723 |

These profile differences establish that experts have different directional-confidence distributions along the hard trajectory. They do **not** establish calibration error because gold next-token labels are unavailable on free-running prefixes; DEV teacher forcing and temperature fitting remain necessary.

## What this artifact supports now

- **SUPPORTED:** Hard K=7 has positive net paired rescue over Track A and probability mean on the complete evaluated TEST corpus.
- **SUPPORTED:** The gain includes both rescues and harms; hard voting is an asymmetric error-correction policy, not an everywhere-dominant predictor.
- **SUPPORTED (descriptive):** Hard and soft aggregation often choose different early tokens from identical expert proposals, after which autoregressive trajectories diverge.
- **SUPPORTED (descriptive):** Expert top-1 confidence, entropy, margin, agreement, and coalition participation differ materially in the frozen trace.
- **UNRESOLVED:** Calibration mismatch is sufficient to explain the hard-vote advantage.
- **UNRESOLVED:** Specialization rather than generic diversity causes the gain.
- **UNRESOLVED:** Teacher-forced local gains are smaller than free-running gains, as required for a trajectory-correction claim.
- **UNRESOLVED:** Removing the abstention expert preserves or destroys the answer/abstain frontier.

## Provenance and reproduction

- Raw result: `results/sealed/v22_frozen_finalists.jsonl` (`d3733391b53dc53e7d9ca899d7015ba6ff0b2faa7c28e52a2778a0827c45c32d`)
- Cornerstone audit: `reports/phase2_cornerstone_verification.json` (`dfb75e4f901a6da946ed04aee03223e90bb033185757436e9df3123b97be8543`)
- Analysis script: `experiments/phase2_mechanism/analyze_existing_results.py` (`389c31aa2a3662a7df0aad81995836d83c94033b5bf7cbf851abe0240a32954a`)
- Machine result: `results/phase2_mechanism/existing_results_analysis.json`

```powershell
$Python = "..\.venv\Scripts\python.exe"
& $Python experiments\phase2_mechanism\verify_cornerstone.py
& $Python experiments\phase2_mechanism\analyze_existing_results.py
```
