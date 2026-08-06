# Phase 9 grounded-recovery plan — v1.7.2

## Governing decision

The v1.7.1 integrity changes were necessary but not sufficient. They fixed prompt budgeting, decode provenance, task-contract cost, bounded zero-reference comparisons, and teacher preflight. The remaining problem was experimental order: a broad warm/narrow SFT/KD sweep could still consume substantial runtime before proving that any inherited student could learn Oracle evidence use.

v1.7.2 makes the grounded capability ladder the primary path:

1. **Teacher Oracle preflight.** Exact evidence bypasses retrieval. Prompt, grounding, abstention, conflict, injection, social, and decode behavior are inspected before student training.
2. **Warm initialization baseline.** The untrained 344M depth-copy student is evaluated on the same Oracle packet.
3. **Deliberate overfit wiring test.** A deterministic 12–24-row set covers facts, abstention, conflict, injection, procedure, and social behavior. The exact rows are repeated for training and reevaluated after saving/reloading. This tests labels, masks, optimizer steps, adapter attachment, publication, reload, and prompt parity.
4. **Compact warm SFT sweep.** The official first three settings are `5e-5/600`, `5e-5/1200`, and `1e-4/1200`, LoRA rank 16. No broad Cartesian grid is created.
5. **Oracle selection.** A candidate must improve over the warm initialization and demonstrate non-degenerate supported answers, abstentions, conflict handling, procedure use, social replies, and passthrough injection resistance. Missing denominators or all-zero behavior produce `NO_VALID_QUALITY_FRONTIER`.
6. **Real retrieval.** FTS5, selection, calibration, distractors, conflict, and injection behavior are tested only after Oracle use works.
7. **Progressive KD.** KD starts from the selected SFT artifact, never as the main direct-from-initialization path.
8. **Architecture comparison.** Narrow random controls remain locked until both the overfit and full Oracle stages pass.

## Training mixture

The training-only balanced view targets:

| Behavior | Share |
|---|---:|
| Supported evidence extraction | 45% |
| Missing-evidence abstention | 25% |
| Conflicting evidence | 10% |
| Injection-resistant evidence use | 10% |
| Procedures | 5% |
| Social conversation | 5% |

The source reviewed/curriculum files are preserved. Rebalancing writes a separate deterministic sampled view and provenance manifest.

## Loss emphasis

The ordinary completion CE/KL objective is retained. Four observable token groups receive explicit weights:

- first answer token: `2.0`;
- required evidence-anchor tokens: `1.75`;
- approved abstention sequence: `2.0`;
- EOS/end-of-turn boundary: `1.5`.

No hidden-state or attention matching was added. This keeps the experiment debuggable.

## References and comparisons

Evaluation reports explicit numerator/denominator evidence for supported answers, abstentions, conflicts, procedures, social replies, and adversarial resistance. Every candidate is kept distinct against:

- the accepted teacher;
- the exact untrained student initialization;
- the SFT predecessor when evaluating KD.

The gate therefore reports candidate→teacher, initialization→SFT, SFT→KD, and initialization→KD without near-zero division.

## Artifact contract

A clean run writes under one new root:

```text
artifacts/phase_9_grounded_recovery_v1_7_2/
├── warm_344m_initialization/
├── dataset/
│   ├── train.jsonl
│   ├── dev.jsonl
│   ├── test.jsonl
│   ├── train_mixed_curriculum.jsonl
│   └── train_grounded_balanced.jsonl
├── evaluation_teacher_oracle/
├── evaluation_warm_initialization_oracle/
├── overfit_probe/
├── warm_sft_sweep/
│   ├── lr5e5_s600/
│   ├── lr5e5_s1200/
│   ├── lr1e4_s1200/
│   ├── selection.json
│   └── SELECTION_REPORT.md
├── grounded_recovery_ladder.json
├── GROUNDED_RECOVERY_LADDER.md
├── evaluation_teacher_full/
├── evaluation_selected_sft_full/
└── progressive_kd_from_selected_sft/   # only when requested
```

Phase 9B remains blocked until the ordinary Phase 9 human gate names and approves a specific student.
