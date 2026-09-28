# Amendment 005: E1 plan and preflight contracts

Frozen before any E1 model work. Native E0 is still blocked by host RAM. This
amendment permits only input materialization, execution-plan construction,
receipt validation and response-import mechanics; it does not launch a model.

## Data and sequence order

Bind all three actual TRAIN token streams to their measured file/row hashes,
tokenizer/template identity and source lineage. The language component previously
stored per-row token hashes/counts only. A separate tokenizer-only export must
recompute its arrays/masks and match every prior measurement, preserving all prior
conversation as masked context and the final assistant reply/EOS as the sole target.
No preparation module that sets USE_TORCH=0 may be imported by a native worker.

The fixed candidate dose remains4096 acquisition +768 deterministic replay +256
language replay =5120 sequences,680532 nonpadding and63546 supervised tokens.
Only TRAIN rows are admitted. No source-correct filtering, row replacement, extra
epoch or CHECK/FINAL access. Preserve the original row order before shuffling.

For seed101, initialize Python random.Random(101), shuffle the4096 acquisition
rows, then shuffle the combined1024 replay rows with the same RNG. Each of256
optimizer steps consumes16 acquisition and4 replay rows: four acquisition then
one replay, repeated four times. No additional within-family quota is introduced.
Each input row occurs exactly once. All attempts/tokens count, including a failed
step; a plan is a schedule, not proof that any tokens were trained.

## Exact schedule and cursor

Warmup rounding is ceil(0.05*256)=13 updates. For one-based update u,
LR=1e-4*u/13 for u<=13, then LR=1e-4*(256-u)/(256-13). The final optimizer call
therefore has LR0: it still consumes its batch and updates optimizer moments but
has no direct parameter displacement. This convention is prospective and explicit,
not an observed optimum or a standard-library scheduler default. No LR convention
will be changed after seeing outcomes without a new registered comparison.

Checkpoint/DEV cadence remains0/64/128/192/256; select the final checkpoint only.
Plan cursors exist only at completed optimizer boundaries. They bind the full
schedule, exact data/token prefix, config and identities. A valid accounting cursor
does not prove that optimizer parameters, RNG or a native checkpoint were restored;
those require independently verified checkpoint/RNG artifacts and E0 qualification.

External identity fields are exact Titan parent weight SHA256; primary model-receipt
SHA256; normalized data/file-binding aggregate SHA256; critical source inventory
SHA256; and frozen planning-config SHA256. Record the40-character Git commit
separately. Never compare a receipt digest as if it were the checkpoint weight hash.

## Fail-closed evidence boundary

Seed101 is the only initial plan seed. Seeds211/307 require the later registered DEV
decision and their own frozen plans. E0 pass status alone is insufficient: validate
the actual bound config/source/runtime/model/trainable/resource/check/cost bundle.
E0's e1_authorized flag is always false and is not a native training grant.

Enrollment imports must preserve every attempted response and exact expected row
coverage, including source errors, twins and eight-row worlds. They must not turn
derivatives into independent samples. Language source review cannot supply calibrated
language grading. Data/control tests do not certify model execution or preservation.

Current preflight remains no-native-executor/no-launch even if preparatory evidence
is structurally valid. E0, current-source enrollment, language evaluation/calibration,
long-context runtime qualification and the actual native executor remain distinct
missing requirements. No resource floor, numerical tolerance or claim scope is reduced.
