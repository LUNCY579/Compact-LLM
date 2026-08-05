# Phase 2C — Importance-Guided Targeted Recovery

Phase 2C starts from a **human-approved Phase 2B reduced checkpoint** and asks:

> Which surviving layers can receive a small LoRA/QLoRA adapter and recover nearly all of the quality of all-surviving-layer tuning with materially less trainable state?

It preserves the physical weight-memory reduction already achieved by Phase 2B. It does not reinterpret the negative Phase 2A result: Phase 2A trained tiny subsets on the original architecture, while Phase 2C trains selected surviving layers after concise behavior was learned and redundant depth was removed.

## Default command

```powershell
.\scripts\phase_2c_targeted_recovery\run_phase_2c.ps1 `
  -Phase2BProfile balanced `
  -QLoRA `
  -GradientCheckpointing
```

Resume an interrupted run with the **identical decision-bearing settings and unchanged model inputs**:

```powershell
.\scripts\phase_2c_targeted_recovery\run_phase_2c.ps1 `
  -Phase2BProfile balanced `
  -QLoRA `
  -GradientCheckpointing `
  -Resume
```

`run_config.json` records SHA-256 fingerprints of the reduced checkpoint, Phase 2B catalog/registry, datasets, benchmark, seeds, LoRA settings, and ranking weights. Resume prints exact nested differences instead of a generic mismatch.

## v1.4.7 correctness and resume hardening

The v1.4.6 Phase 2C reasoning rows contained mechanically correct assistant targets but no explicit correctness contracts. Because Evaluator v4.5 intentionally treats categories and reference wording as non-normative, fluent wrong answers such as `I need a full tray first.` could receive an undeserved high score on arithmetic cases.

v1.4.7 adds strict, prompt-grounded contracts to every deterministic implication, ordering, comparison, arithmetic, contradiction, set-membership, temporal, and clarification row. Correct alternate wording remains accepted, while missing or directly wrong answer content receives proportional quality cost.

The data migration changes only evaluation metadata. Stable IDs, user messages, and assistant training targets are byte-semantically unchanged. A verified migration path therefore:

1. reuses completed adapters;
2. reuses raw generated replies;
3. rescales/rescores every durable reply under the new contracts;
4. rebuilds generation summaries and output-deviation evidence;
5. reuses teacher-forced NLL rows because tokenized messages and targets are unchanged;
6. continues partial generation automatically.

Do not manually run a child evaluator after installing v1.4.7. Rerun the original Phase 2C wrapper with `-Resume`.

The shared atomic writer retries transient Windows access/sharing violations. Progress sidecars are best-effort and cannot discard a completed example. Generation and teacher-forcing evaluators both persist one row per completed stable ID and receive `--resume` from the Phase 2C runner.

Intermediate Trainer checkpoints are disabled by default for Phase 2C, while the final adapter is staged, validated, and published safely. Use `-KeepTrainingCheckpoints` only when those checkpoints are intentionally needed. `-MinimumFreeDiskGB` controls the pre-training free-space check.

## Data

The Phase 2C mixture retains the approved Phase 1 train/dev records and adds deterministic, mechanically verified concise-reasoning cases for implication, ordering, comparison, arithmetic, contradiction detection, set membership, temporal reasoning, and clarification. Added train/dev entities are disjoint, exact prompt-target leakage is rejected, and sealed data is never read.

The development file deterministically interleaves original and Phase 2C cases. Therefore, the default 64-case importance calibration and 180-case candidate evaluation contain both conversational controls and concise-reasoning examples instead of accidentally evaluating only the first original-data records.

## Importance signals

1. First-order Taylor sensitivity of assistant-target loss to each surviving layer output.
2. Target-token NLL increase when one surviving layer is temporarily bypassed.
3. Final-stage Phase 2B raw-deletion damage when the matching registry is available.

The combined ranking chooses **where limited updates may be useful**. It is not evidence that a layer can be deleted.

## Required controls

- `all_remaining`
- `top1`, `top2`, `top4`, `top8`
- `complementary4`
- `random4`, `low4`
- `top4_attention`, `top4_mlp`

Candidates are evaluated against both the independently trained `all_remaining` control and the frozen Phase 2B model. The decision engine uses Evaluator v4.5 quality dimensions, tail/category behavior, issue cost, target NLL, exact stable-ID output comparison, trainable parameters, training time, peak training VRAM, and adapter size. The old `legacy_heuristic_score` remains report-only and never selects a profile.

Every recommended profile receives repeated-seed evidence. Selective repeats are compared with a separately trained `all_remaining` control using the same seed. If a failed finalist exposes a new recommendation, that new finalist is repeated before publication.

## Outputs

- `artifacts/phase_2c_targeted_recovery/run_config.json`
- `artifacts/phase_2c_targeted_recovery/importance/combined_layer_ranking.json`
- `artifacts/phase_2c_targeted_recovery/candidate_registry.json`
- `artifacts/phase_2c_targeted_recovery/model_catalog.json`
- `reports/PHASE_2C_TARGETED_RECOVERY_REPORT.md`
- `reports/PHASE_2C_STOP_GATE.md`

No profile enters Phase 3 until the stop gate says `Human decision: PASS` and the exact profile is listed in `Approved profiles`.

## Quality evaluator v4.5

This phase uses Quality Evaluator `4.5` and the shared relative-opportunity engine. Informal or mildly ungrammatical language is permitted when understandable; references are examples rather than unique answers; and behavioral restrictions are enforced only when explicitly present in the response contract. Deterministic Phase 2C reasoning tasks now carry explicit prompt-grounded correctness contracts because their answers are objectively verifiable.

Promotion never depends on an absolute score, pass-rate, completion, category, NLL, or issue-count floor. The declared proportional resource benefit must be strictly greater than combined proportional quality opportunity loss. See `RELATIVE_OPPORTUNITY_EVALUATION_POLICY.md`, `EVALUATION_V4_5_MIGRATION.md`, and `PHASE_2C_RELIABILITY_AND_CORRECTNESS_HOTFIX_v1.4.7.md`.
