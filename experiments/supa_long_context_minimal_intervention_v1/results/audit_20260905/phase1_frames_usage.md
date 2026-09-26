# Descriptive target-frame helper

Run from repository root after bounded outputs are complete:

```powershell
& '../Chat-Bubble/.venv/Scripts/python.exe' 'experiments/supa_long_context_minimal_intervention_v1/results/audit_20260905/analyse_phase1_frames.py'
```

Default stages are `phase1` and `phase1_a1`. Outputs, in this audit directory:

* `phase1_frames.json`: compact machine-readable counts, accuracy, abstention,
  paired differences, and raw-source hashes.
* `phase1_frames.md`: accuracy and abstention tables by all rows,
  answerability, variant, task type, and split for both cores.

Optional `--stages phase1_smoke` selects the smoke; `--output-stem NAME` keeps a
separate snapshot. Output files are restricted to this audit directory. The
helper imports only the standard library and the official scorer; it extracts
the existing Phase 0 abstention function and `UNKNOWN_FORMS` through AST without
importing model/provenance modules. Thus it uses the exact existing abstention
definition, applied to independently normalized raw predictions.

Only COMPLETE arm receipts with the exact expected target IDs and verified
metadata/scoring contribute values. Missing or still-running arms contribute no
scores. Stage COMPLETE additionally requires the production root COMPLETE
receipt, a stable snapshot, and exact raw-target-prediction agreement between the
root file and the analyzed per-arm files. Other stages remain PARTIAL or INVALID.
No arm selection or advancement decision is produced. Use the separate
mechanical, preservation, and gate audits as well.

Paired differences include net correct/abstention changes and discordant counts.
Composite score changes are also decomposed into answerable and unanswerable
net-correct contributions. There are no hypothesis tests or inferential claims.
The aggregate decomposition describes observed responses, not their full cause.

## Validation performed

```powershell
& '../Chat-Bubble/.venv/Scripts/python.exe' 'experiments/supa_long_context_minimal_intervention_v1/results/audit_20260905/analyse_phase1_frames.py' --stages phase1_smoke phase1 phase1_a1 --output-stem phase1_frames_validation
```

This produced `phase1_frames_validation.json` and `.md`. Every partition's
denominators, correct counts, abstention counts, and matched deltas add back to
the corresponding all-row values. The bounded Arm A manifest's recorded source
and configuration hashes still matched after helper creation and execution.
Nothing under frozen `src/`, `scripts/`, or `configs/` was changed.

At this validation snapshot, smoke was COMPLETE; bounded A1.1 had only completed
Arm A and remained PARTIAL; bounded A1 had no completed arm and remained PARTIAL.

The smoke illustrates why these frames matter: B's net gain from 20/32 to 22/32
comes entirely from unanswerable correct responses increasing from 2/8 to 4/8.
Answerable A and B both score 18/24 with no correct/incorrect flips and no
abstentions. C scores 4/24 on answerable rows and abstains on 16/24. These are
smoke observations, not bounded training conclusions.
