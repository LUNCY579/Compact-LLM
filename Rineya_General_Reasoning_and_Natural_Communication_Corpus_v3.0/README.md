# Rineya General Reasoning and Natural Communication Corpus v3.0

Rineya v3.0 is a train-only curriculum expansion for the approximately 454M-parameter short-response model. It preserves the product contract—read the full thread, reason internally, then return the shortest natural response that fully performs the requested function—while expanding reasoning, multi-turn state updates, relationship continuity, trust boundaries, role fidelity, and evidence-grounded dialogue.

## Resolved planning total

The generation prompt contains a numerical conflict: its group table says 40,000 reasoning rows, while its per-class quota says 2,500 rows for each of 20 reasoning classes, which equals 50,000. This package follows the more specific per-class quotas:

| Group | Classes | Rows per class | Total |
|---|---:|---:|---:|
| Reasoning | 20 | 2,500 | 50,000 |
| Communication | 12 | 2,500 | 30,000 |
| Integrated | 8 | 3,750 | 30,000 |
| **New SFT** | **40** |  | **110,000** |

## Recommended files

- `data/sft/rineya_v3_new_all_train_110000.jsonl`: all new v3 SFT.
- `data/sft/rineya_v1_v2_preservation_slice_14000.jsonl`: deterministic preservation only; no regenerated preservation.
- `data/sft/rineya_v3_recommended_combined_124000.jsonl`: recommended starting mix.
- `data/dpo/rineya_v3_preferences_train_16000.jsonl`: 400 class-specific pairs per class; use only after accepting an SFT checkpoint.

Do not append the complete v1 or v2 corpus to the recommended combined file again. Do not append official DEV, SEALED, RAG DEV/TEST, or preference-DEV data.

## Quality architecture

- 16,504 template families; no family exceeds 8 rows.
- True four-row counterfactual families change one decisive fact.
- At least 30% multi-turn and 20% counterfactual coverage per class.
- Latest corrections override earlier state in compositional conversations.
- Targets contain no chain-of-thought and have a hard 24-proxy-token limit.
- Every row includes a speech act, quality rationale, likely failure, uniqueness claim, shard identity, and pending human-review status.
- DPO rejections are short, fluent, class-specific errors rather than generic verbose caricatures.

## Evaluation isolation

The build scanned the available v1/v2 corpora plus evaluation-like files in the supplied v2.1.1 repository snapshot. Exact overlap is zero. A sampled near-duplicate audit is included. Re-run the packaged validator against the current local repository before training because local DEV/SEALED files may have evolved after the supplied snapshot.

## Human review

Automated checks establish structural validity, deterministic reproducibility, exact isolation, length bounds, family quotas, and contract checks. They do not establish human publication approval. Review `reports/human_audit_packet_520.md` and use the 40,000-ID audit pool for larger sampling.

## First controlled training experiment

1. Keep the accepted v2-trained teacher checkpoint immutable.
2. Train one QLoRA epoch on the 124,000-row recommended file, optionally adding the approved RAG training corpus once.
3. Re-run the same frozen chat and RAG DEVs.
4. Compare category tails, real-RAG abstention, target length, and anti-collapse statistics—not only aggregate quality.
5. Freeze one SFT candidate before DPO or sealed testing.
