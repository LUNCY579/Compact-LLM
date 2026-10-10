# Rineya intelligence quality foundry

An isolated, quality-first dataset program for the centralized Titan experiment.
All new work lives here; sibling corpora and the active branch remain read-only.

Read `docs/END_TO_END_PLAN.md` for the full pipeline and retention design,
`docs/LONG_CONTEXT_CURRICULUM.md` for the task curriculum, and
`docs/REVIEW_WORKFLOW.md` for every-row supervision. `config/quality_policy.json`
defines hard preparation boundaries. `docs/RESEARCH_SOURCES.json` records primary
research; those studies do not establish gains for this Titan student.

Current reviewed release: `releases/pilot-v3/release_manifest.json`. See
`DATA_CARD.md` and `docs/FIRST_CYCLE_REPORT.md` for measured evidence and limits.
Cycle 002 added a durable source/parent registry and independently reviewed the
next distributed-reasoning design. See `cycles/002-source-registry/CYCLE_REPORT.md`
and `cycles/002-source-registry/REGISTRY_GUIDE.md`. The current catalog uses the
source-lineage correction v2; the original seed adapter is historical evidence.
No legacy rows have been admitted. Cycle 003 froze the executable dossier
catalogue and independent parser/solver, pinned exclusions for exposed fixtures,
and completed one fixed 54-attempt birth of unsolved original worlds. Those worlds
are prospectively registered. Two public views per world agree with a separately
implemented author solver; one intended ambiguous parent is quarantined. Exact
Titan lengths are 1,622–3,168 tokens. Dependence and row-review gates remain open;
none has been admitted. See `cycles/003-dossier-implementation/PROGRESS.md` and the current
`state/cycle_status.json`. Do not repeat that birth or replenish rejected slots.

## First research pilot

The intended pilot is72 independent generated worlds across six structured task
families: authoritative state, relational aggregation, precedence planning,
contextual rule induction, instruction hierarchy and answerability. Roles are48
TRAIN,12 CAL and12 exposed DEV. Short/medium/long are generation profiles;
`profiles/` records actual native tokenizer lengths. They are not claims of native
model qualification. No CHECK/FINAL is materialized or opened.

The generator and verifier are separate implementations. Every canonical row is
joined to the public prompt, independently solved, negatively tested, tokenized
and sent to two assigned reviewer agents with gold/specification hidden. Reviews
bind exact bytes and carry concrete evidence and limitations. Same-model agents
are procedural independence, not calibrated human review or independent providers.

All candidate/audit/profile/review/release outputs are write-once. Never patch a
released row: create a new revision and rerun the affected gates. Existing errors
and failed attempts remain evidence. Only TRAIN messages and final targets enter
the SFT candidate export; private world specifications, oracle reasoning and CAL/DEV
do not enter that file.

## Run commands

Use the locally available tokenizer runtime at
`C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe` or another pinned
CPU environment containing transformers/tokenizers. Token profiling disables
Torch/TensorFlow/Flax and network access before importing the tokenizer. No model
weights are read and no training code exists here.

From this directory, using `python` as that executable:

```powershell
python -m unittest discover -s tests -v
python scripts/pipeline.py build --out candidates/new-pilot
python scripts/pipeline.py audit --batch candidates/new-pilot --out audits/new-pilot
python scripts/token_profile.py --batch candidates/new-pilot --out profiles/new-pilot
python scripts/pipeline.py packets --batch candidates/new-pilot --out packets/new-pilot
# Assigned agents independently review the packets before the release step.
python scripts/pipeline.py release --batch candidates/new-pilot --audit-dir audits/new-pilot --profile-dir profiles/new-pilot --packet-dir packets/new-pilot --reviews reviews/new-pilot/semantic.jsonl reviews/new-pilot/adversarial.jsonl --out releases/new-pilot
```

Use new version directories for reruns. Candidate manifests reject changed source
bytes. Qualify a revised producer in a new batch, preserving older manifests.
The scheduled supervisor must acquire the local dataset-cycle lock and follow
`docs/REVIEW_WORKFLOW.md`; it prioritizes fixing defects over generating volume.
The demonstration seed above recreates the existing pilot worlds. Such a rerun
is a reproducibility check, not fresh data or new independent evidence. A new
mechanism requires explicit parent/structure/source reservation and revised design.

## Limits

A reviewed release has status `REVIEWED_RESEARCH_PILOT`, with training_ready=false.
It is a reviewed candidate for later integration, not a replacement for the current
frozen5,120-row E1 dose. Historical semantic overlap, broader structural diversity,
calibrated prose review, native training/capture/length integration, baseline
preservation enrollment and model-gain evidence are separate outstanding gates.
Only qualified before/after behavior can establish acquisition or retention.
