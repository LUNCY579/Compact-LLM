# Rineya General Reasoning and Natural Communication Corpus v2.0

Train-only data for the Rineya1.1 research iteration. The corpus expands general reasoning, multi-turn context use, and natural human-like communication while preserving the short-response contract.

## Recommended files

| Use | File | Rows |
|---|---|---:|
| First teacher SFT run | `data/sft/rineya_v2_recommended_combined_40000.jsonl` | 40,000 |
| New-data-only ablation | `data/sft/rineya_v2_new_all_train.jsonl` | 32,000 |
| Reasoning-only ablation | `data/sft/rineya_v2_reasoning_train.jsonl` | 16,000 |
| Communication-only ablation | `data/sft/rineya_v2_natural_communication_train.jsonl` | 9,600 |
| Integrated dialogue ablation | `data/sft/rineya_v2_integrated_reasoning_dialogue_train.jsonl` | 6,400 |
| v1 preservation control | `data/sft/rineya_v1_preservation_slice_8000.jsonl` | 8,000 |
| Post-SFT DPO | `data/dpo/rineya_v2_preferences_train_12000.jsonl` | 12,000 |

The recommended 40,000-row file already contains the 8,000-row v1 preservation slice. **Do not append the full v1 corpus to that file again.** To build your own mix, use `rineya_v2_new_all_train.jsonl` instead.

## Package maps and reports

- `reports/curriculum_map.md` explains all 40 capability classes and their failure-specific DPO contrasts.
- `reports/semantic_audit_report.json` records deterministic checks over 15,200 recomputable or contract-grounded rows.
- `reports/full_quality_audit_report.json` checks file unions, role contracts, text integrity, exact DPO linkage, and available evaluation snapshots.
- `reports/schema_validation_report.json` validates all seven JSONL artifacts against the included schemas.
- `reports/reproducibility_report.json` confirms that all seven core data files regenerate byte-for-byte from seed `20260816`.
- `reports/sample_audit_packet.md` contains 400 stratified SFT examples and 120 stratified DPO examples for human review.

## Design priorities

1. Structural diversity over repetition.
2. Short final answers without chain-of-thought targets.
3. Explicit counterfactual and minimally changed cases.
4. Multi-turn corrections, plan revisions, and context updates.
5. Natural informal language, empathy, disagreement, compromise, and repair.
6. Grounded answers, conflict detection, and abstention when facts are absent.
7. Balanced reasoning and communication instead of math-only specialization.
8. Preservation replay from v1.0 to reduce behavioral forgetting.

## Hard trainer contract

Every SFT row is JSONL, starts with a system message, alternates user/assistant roles, and ends with the assistant target. No row contains a top-level `question` field, so Phase 9 does not rebuild the prompt. Every DPO prompt is pre-rendered and ends at the Qwen assistant generation prefix.

## Before training

1. Run `scripts/semantic_audit.py --root .` to recheck the 15,200 deterministic/template-grounded rows and DPO source linkage.
2. Run `scripts/full_quality_audit.py --root . --eval-root C:\PATH\TO\repository` for union, role, text, DPO-linkage, and exact-snapshot checks.
3. Run `scripts/validate_dataset.py` against the exact current repository DEV/SEALED files.
4. Human-review `reports/sample_audit_packet.md`, especially safety, empathy, roleplay, and DPO negatives.

Automated semantic checks are intentionally narrower than human review: they verify recomputable answers and explicit context contracts, not whether every open-ended social reply is the uniquely best wording.
## Deterministic regeneration

The packaged generator defaults to a safe sibling folder ending in `_regenerated`, so it does not overwrite this release. Supply the extracted v1.0 corpus used for preservation:

```powershell
python .\scripts\generate_dataset.py `
  --v1-root C:\PATH\TO\Rineya_Short_Response_Behavior_Corpus_v1.0 `
  --eval-root C:\PATH\TO\short-response-model-phase-0-1-2
```

Use `scripts/tokenizer_length_report.py` with the exact local Qwen tokenizer before training if exact target-token statistics are required.

