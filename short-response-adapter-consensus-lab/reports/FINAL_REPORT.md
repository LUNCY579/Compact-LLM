# Final Report: Multi-Adapter Token Consensus for a Short-Response Model

## Executive conclusion

Exact hard next-token voting is a real, correctly implemented inference method, and it substantially improves this otherwise incapable frozen backbone. It is not the best method or best single behavior policy in the tested setting.

On the preregistered 32-example sealed packet, K=7 hard plurality scored **0.250** versus **0.000** for the backbone: paired difference +0.250, 95% bootstrap CI [0.094, 0.406], Holm-adjusted exact McNemar p=0.0469. It did not convincingly beat official Track A (0.156; difference +0.094, CI [−0.063, 0.250]) and lost to the abstention specialist (0.500) on aggregate balanced accuracy. Raw probability mean scored **0.375**, numerically above hard voting by 0.125 (CI [0.031, 0.250]); the adjusted paired test was not significant on only 32 cases.

The main scientific finding is therefore:

> Adapter consultation can buy capability over the base model, but hard plurality has no demonstrated advantage over ordinary soft fusion and does not beat the strongest single expert. Most aggregate gain comes from handling unsupported cases; answerable reasoning remains weak.

This is a bounded result, not a claim about the complete 4,050-case sealed split.

## Evidence map

| Evidence | Scope | Authoritative artifact |
|---|---|---|
| Candidate inventory | 36 dense models, 186 adapters | `reports/candidate_inventory.json` |
| Anchor reproduction | 8 cases per anchor, exact output/token/correctness | `results/summaries/baseline_reproduction.json` |
| Hard K sweep | K=1/2/3/5/7, 16 paired DEV cases | `results/runs/v22_hard_vote_primary_smoke.jsonl` |
| Fusion ladder | eight methods, 8 paired DEV cases | `results/runs/v22_fusion_comparators_smoke.jsonl` |
| Negative controls | 18 conditions, 144 rows | `results/runs/v22_negative_controls_smoke.jsonl` |
| EOS ablations | four policies | `results/runs/v22_*eos*_smoke.jsonl` and ordinary K=5 rows |
| Adaptive consultation | 16 paired DEV cases | `results/runs/v22_adaptive_consultation_smoke.jsonl` |
| Oracle RAG | 16 paired official prompts | `results/runs/v22_oracle_rag_consensus_smoke.jsonl` |
| Frozen sealed | 256 rows, 32 examples | `results/sealed/v22_frozen_finalists.jsonl` |
| Paired inference | bootstrap/McNemar/sign-flip/Holm | `results/summaries/{pilot,sealed}_statistics.json` |
| Source integrity | zero changed protected sources | `artifacts/manifests/official_sources_after.json` |
| Completion audit | all required hashes, rows, configs, and protected paths pass | `reports/completion_audit.json` |

## Sealed results

| Frozen condition | Correct | Answerable | Abstention | False abstention | Mean tokens | Mean latency | Mean forwards |
|---|---:|---:|---:|---:|---:|---:|---:|
| Backbone K=1 | 0.000 | 0.000 | 0.000 | 0.000 | 6.72 | 0.57 s | 6.72 |
| Track A K=1 | 0.156 | 0.188 | 0.125 | 0.000 | 4.34 | 0.25 s | 4.34 |
| Abstention K=1 | 0.500 | 0.000 | 1.000 | 1.000 | 11.00 | 0.71 s | 11.00 |
| Full K=7 hard plurality | 0.250 | 0.063 | 0.438 | 0.188 | 6.13 | 3.83 s | 42.88 |
| Full K=7 probability mean | 0.375 | 0.063 | 0.688 | 0.563 | 8.19 | 5.37 s | 57.31 |

All conditions ended with EOS, were nonempty, stayed below 32 tokens, and had zero repetition failures. K=7 uses 201,341,896 adapter-storage bytes, 30,277,632 resident adapter parameters, 348,680,192 active parameters per expert forward, about 858 MB peak allocated VRAM, and about 1.87 GB peak process RSS. Hard voting was roughly 6.7× slower than the measured backbone condition and 15× slower than Track A; probability mean was slower again because it generated longer outputs.

## Paired sealed inference

| Comparison (A − B) | Difference | 95% bootstrap CI | Exact McNemar p | Holm p |
|---|---:|---:|---:|---:|
| Hard K=7 − backbone | +0.250 | [0.094, 0.406] | 0.0078 | 0.0469 |
| Hard K=7 − Track A | +0.094 | [−0.063, 0.250] | 0.4531 | 0.4531 |
| Hard K=7 − abstention | −0.250 | [−0.438, −0.063] | 0.0215 | 0.1074 |
| Hard K=7 − probability mean | −0.125 | [−0.250, −0.031] | 0.1250 | 0.3750 |
| Probability mean − backbone | +0.375 | [0.219, 0.531] | 0.00049 | 0.00342 |
| Probability mean − Track A | +0.219 | [0.031, 0.406] | 0.0654 | 0.2617 |

Only improvements over the zero-scoring backbone survive this comparison-family correction. The packet is underpowered for smaller method differences.

## Committee scaling and controls

The balanced 16-case DEV hard-vote curve was nonmonotonic: K=1 best individual 0.500, K=2 0.500, K=3 0.125, K=5 0.188, and K=7 0.438. K=7 was the only committee that recovered both answerable and abstention behavior. More experts are therefore not sufficient; membership and coalition structure matter.

Duplicate copies matched the corresponding single adapter on 100% of token sequences for hard, probability, and logit aggregation. Weak/social injection damaged narrow committees. Artificially scaling the weak expert's logits did not alter hard votes by magnitude, while it changed soft-fusion behavior—exactly the calibration sensitivity that control was designed to expose. A fixed random-pool proxy performed surprisingly well on the tiny packet, so semantic specialization has not been isolated as the cause of benefit.

## EOS and short-response behavior

Ordinary EOS remained the primary rule. On the 8-case K=5 EOS packet, ordinary, strict-majority, confidence-threshold, and minimum-three-token conditions all scored 0.125. Mean output lengths were 6.69–7.50 tokens, EOS rate was 1.0, and repetition rate was 0. No EOS ablation improved capability. Consensus preserved shortness; probability fusion shifted more cases toward the 11-token canonical abstention.

## Disagreement and adaptive compute

On sealed hard K=7 rows, mean disagreement ranked errors extremely well: AUROC 0.992 and AUPRC 0.995; vote entropy AUROC was 0.979 and pairwise-JS AUROC 0.969. It was not a calibrated error probability (Brier 0.337, ECE 0.451), so it should be treated as a ranking signal until calibrated on disjoint data. All estimates are optimistic/unstable at n=32.

The separate adaptive extension started with Track A, Track B, and general, then consulted four more experts only when core agreement was below 0.67. On the 16-case DEV packet it produced exactly the same token sequences as full K=7 on 16/16 examples, matched correctness 0.438, reduced mean expert-token forwards from 42 to 22, and reduced mean latency from 3.27 s to 1.68 s. It triggered at least once on 13/16 examples. This is a useful compute-allocation result, but it was deliberately not promoted after DEV and was not run on sealed data.

## Oracle RAG

Using official v3-large oracle evidence prompts and official functional/grounding definitions, Track B and full K=7 both scored 16/16 across eight supported, five unsupported, and three conflict cases. Every response was grounded. Consensus added no quality at this oracle ceiling and retained a 9.56-token mean. Real retrieval was not evaluated in this lab.

## Answers to the required questions

### Q1 — Does hard voting outperform the base model?

Yes on the frozen 32-case sealed packet: 0.250 versus 0.000, with a corrected paired p=0.0469. The scope is this resource-aware packet, not the full TEST set.

### Q2 — Does it outperform the best individual adapter?

No. It trails the abstention specialist in aggregate (0.250 vs 0.500), although that specialist is behaviorally degenerate: 0% answerable and 100% false abstention. Hard K=7 also does not significantly outperform Track A.

### Q3 — Does it outperform the average adapter?

On the 16-case DEV characterization, K=7 hard scored 0.438 versus mean K=1 adapter correctness 0.170 (excluding the backbone). All seven adapters were not rerun on sealed, so there is no sealed average-adapter claim.

### Q4 — Does hard voting outperform soft fusion?

No. Raw probability mean scored 0.375 sealed versus 0.250 hard. The 32-case adjusted paired test is inconclusive, but there is no evidence for a hard-vote advantage.

### Q5 — Does expert specialization matter?

Membership matters, but specialization as a causal factor is unproven. K=7 was better balanced than narrow specialist committees, while a random-pool proxy was competitive on a tiny control packet.

### Q6 — Does random-seed diversity help independently?

Unanswered. No compatible, matched same-task multi-seed adapter family exists in the official pool; training one was outside the resource-aware scope once existing behavioral diversity proved sufficient for the primary experiment.

### Q7 — Does mixed adapter-type diversity help?

Unanswered. PEFT capability probes passed for LoRA, DoRA, PiSSA, AdaLoRA, and a minimal mixed model, but fair matched adapters were not trained. Software support is not quality evidence.

### Q8 — Does committee size show a useful scaling curve?

No monotonic curve. Quality fell at K=3/K=5 and recovered at K=7, while cost grew approximately with expert-token forwards.

### Q9 — Does disagreement predict mistakes?

Yes as a ranking signal on the sealed pilot (hard K=7 AUROC 0.992/AUPRC 0.995), with the small-sample caveat. No as a calibrated probability without further calibration (Brier 0.337, ECE 0.451).

### Q10 — Can disagreement allocate additional thinking compute?

Promising on DEV: adaptive K=3→K=7 exactly matched full K=7 outputs with about 48% fewer expert-token forwards and 49% lower latency. Not sealed-confirmed.

### Q11 — What happens to brevity and EOS?

Shortness is preserved: sealed means are 6.13 hard and 8.19 probability tokens, P95 is 11, EOS/completion rates are 1.0, and repetition is 0. Soft fusion increases canonical abstention and false abstention.

### Q12 — What is the latency/VRAM/storage cost?

Sealed hard K=7 averages 3.83 s, 0.63 s TTFT, 42.88 forwards, 858 MB allocated VRAM, 1.87 GB RSS, and 201 MB adapter storage. Probability mean averages 5.37 s and 57.31 forwards. The shared backbone exists once.

### Q13 — Is it useful after accounting for compute?

Against the zero-capability backbone, yes if latency is secondary. Against a specialized single policy or probability mean, hard voting is not on the best observed quality frontier. Adaptive consultation is the most promising cost reduction but needs sealed confirmation.

### Q14 — Which official source backbone benefits most?

Unanswered comparatively. Only the frozen v2.2 anchor has a clean seven-expert same-base pool. D_greedy was exactly reproduced but its adapters have a different lineage, so a fair second-backbone committee was not fabricated.

### Q15 — Are improvements statistically convincing?

Only the committee-versus-backbone differences survive Holm correction. Differences against Track A, abstention, and between hard/soft aggregation are not statistically decisive at n=32.

### Q16 — What failure modes were discovered?

- Exact token-score ties exposed unstable `topk`; stable `argmax` fixed baseline equivalence.
- KV-cache and batched/hotswap paths changed bf16 decisions/logits.
- A first sampler overrepresented the lexically first stratum; it was archived and replaced with primary-class interleaving.
- Windows tool timeouts left child workers alive; duplicate rows were audited/repaired and exclusive locks added.
- Masked EOS created `0 × log(0)` NaNs; numerically safe entropy/JS fixed it.
- Committees often trade answerability for canonical abstention rather than learn reasoning.

### Q17 — What should run next?

Run a larger frozen DEV replication followed by a newly frozen sealed confirmation of only three conditions: Track A, K=7 probability mean, and adaptive K=3→K=7. Stratify power around answerable cases, fit/validate an uncertainty threshold on disjoint DEV folds, and add a matched multi-seed LoRA family. Only after that should matched DoRA/PiSSA/AdaLoRA training or a second D_greedy-base committee be attempted.

## Reproducibility and integrity

The environment and PEFT probes are recorded in `reports/peft_capability_probe.json`. The official Git commit was `1487782e7ad5d431323518d5a8b33a394299c3e9`. Before/after protected-source manifests have identical internal SHA256 `899141435d813de64b29a732f6b393202077be2ba1ad900cfa6e9134d38b9b00`; `changed_sources` is empty. The official repository and both excluded mini-labs remain unmodified.

Exact Windows commands are in the lab `README.md`.
