# Balanced-quality implementation

The roadmap is implemented as offline quality-card validation, data-admission
readiness assessment, public aggregate custody and exact existing-owner handoffs.
It does not execute reviewers or native models, admit rows, train, open confirmation,
change serving policy, or mutate an active owner's experiment.

## Quality cards

`quality_card.py::build_quality_card` and `validate_quality_card` provide the shared
`gd001_quality_card_v1` interface. Cards cover instruction following, grounded
retrieval, arithmetic/state tracking, composition, answerability/abstention and
language/long-context preservation. Missing domains are `UNMEASURED` with null
endpoints; no zero score or overall winner is invented.

Each card binds an exact 500M-700M source/tokenizer, full baseline and candidate
checkpoint identities, runtime, prompt/decoding/readout policy, data/split, original
endpoint thresholds, public receipts and study-specific resource accounting.
Paired source-incorrect repairs remain separate from source-correct damages. Fixed
source-success membership and denominators must match the candidate's evaluated
membership. Formatting, semantic correctness, abstention and termination stay
distinct endpoints. A net gain cannot hide losses or inadequate support.

Evidence levels are `mechanical`, `native_qualification`, `exploratory`, and
`confirmed`. Confirmation metadata needs the source/scoring qualification,
DEV selection lock, independent authorization and single-use completion chain.
No bank is opened by card validation. Receipt digests are not signatures:
external file custody, human review and model execution remain supplied provenance.
Every result explicitly denies execution, confirmation and promotion authority.

Intervention type describes the mechanism, not demonstrated generalization:
adapter on/off with fixed policy is a learned-weight comparison; a bias or prompt
change on identical checkpoint bytes is a serving comparison. A previously trained
adapter can be evaluated with zero new optimizer updates. Historical training
custody remains separate, and calibration improvement is not reasoning acquisition.

The complete executable schema example is `tests/test_quality_card.py::build_fixture_card`.
It contains invented TOY identities and is never a native source or efficacy result.

## Data admission

`data_admission.py::assess_data_admission` checks a retained physical packet against
its exact dose/source/tokenizer/task/disposition/consumption bindings. It verifies:

- Exact target correspondence to independently reviewed solutions, actual renderer
  and response masks, supervised EOS including legitimate empty outputs, original
  filtering/truncation decisions, and repeated occurrence accounting.
- Rights/source-use evidence, parent/family split closure, raw/rendered overlap,
  independent solver/scorer qualification, actual native rendering/EOS/loss evidence,
  and separately qualified production grading.
- Complete fixed224 reviewer assignment/outcome/history metadata, unique fresh
  contexts, commitment before candidate release, no forbidden exposure, executed
  custodian interface evidence and independent history review. Missing, uncertain,
  malformed or invalid-exposure slots remain in the original denominator.

The fixed224 study is an EXCLUDED finite diagnostic, not production-grader or learner
qualification. Positive control judgments cannot substitute for production gates.
The validator does not open reviewer packets, candidates or private truth. Nested
fixture evidence, including rights and calibration protocols, cannot establish
production readiness.

`make_pending_packet` prepares an explicit `PENDING_OWNER_GATES` request from public
metadata while physical source/dose and production gates are absent. Its assessment
always returns `BLOCKED` with exact missing prerequisites. This supports the current
dataset state without reading training or excluded-control payloads.

Even `READY_FOR_OWNER_ADMISSION_REVIEW` admits zero rows and grants no training
permission. Existing owner authenticity/admission review remains necessary.

## Public ledger and owner packets

`roadmap.py::prepare_roadmap` consumes a signed `gd001_balanced_quality_roadmap_v1`
specification with explicit owners, public file references, observations and readiness
dependencies. Every reference binds an absolute path, full-file SHA256 and
`access: public_aggregate`. Only bounded JSON/Markdown/text documents are read;
protected bank paths are rejected before content access, and nested references
inside documents are not followed. Original metadata bytes are retained externally.

Imported historical summaries without full source/runtime/policy/endpoint custody
remain incomplete observations, never fabricated qualified cards. Genuine complete
cards can be supplied through `quality_card_evidence_ids`; validation preserves their
actual evidence level. Cross-source ranking and a scalar overall quality score are
disabled. Explicit fixtures cannot acquire exploratory labels.

The packet writes `quality-scorecard.json`, `owner-handoffs.json`, exact public-byte
snapshots, their receipt inventory, `ROADMAP_PACKET.json`, and `OWNER_HANDOFFS.md`.
The packet binds scorecard, handoff and snapshot-receipt digests. Verification detects
mutation without overwriting past artifacts. Mutable upstream owners need a fresh
source review before any later execution.

Additional owner adapters prepare:

- **M1:** verify the existing direct-interpreter proposal, executable/DLL custody and
  completed direct-parent identity fixture. Preserve failed/consumed attempts,
  original source/sidecar identity, environment, guards, pins and command. This
  preparation grants zero retries; fresh review and separate owner authorization
  are still required before the existing watchdog can run.
- **Supervised acquisition:** bind the owner's exact existing draft, including its
  actual seed, 540 whole sequences, 27 updates, 20 sequences/update, q/v rank8 scope,
  completion-plus-EOS objective, full LR list and zero-LR final update. Capture the
  proposed runtime and qualification resources without borrowing them for GD.
  Propose preserving chunk-boundary snapshots at 9/18/27 and initial GD DEV steps
  0/9/27 only under prospective owner approval, original standard exports and parity.
  Absolute paths are unresolved; no hook is wired and no campaign is launched.

The supervised draft is made decision-ready by binding decisions that exist and
listing actual remaining gates. Missing sustained native fit, M0 scoring, fresh
prior-correct language/context guards, evaluation margins, study budgets and complete
execution authorization remain blockers; they are not assigned convenient defaults.

## Priority and owner boundaries

1. Complete the existing owner's auditable fixed224 diagnostic and production/data
   admission gates. Canonical dataset mechanics alone admit no rows.
2. Original M1 owner reviews the direct-interpreter retry; after genuine M1 PASS,
   M2 separately qualifies its observer and fresh 96CAL/192CHECK comparison at live
   3.75 and selected bias, with full generation/replay and independent CHECK authority.
3. SYNTH-CRS owner completes the frozen-source baseline and sustained qualification,
   then reviews the exact source/surface-varied/condition-diverse acquisition study.
   Preserve difficulty/order confounds. Oracle R acquisition needs a separate plan
   if signal remains insufficient; discarded qualification updates cannot supply it.
4. DSD remains the first conditional distillation route: same backbone, seeds17/29,
   three methods, 12 updates/cell, 72-update ceiling, teacher7/8 with2 paired wins and
   zero harms. Its retention replay is evaluation-only; adding rehearsal requires
   an amendment. RDD's separate-source residency and RSD's real SFT parent stay gates.
5. SBHM remains on hold. TAPS's newly accepted restricted negative result is retained;
   the existing owner completes final mission/reproducibility reconciliation.
   Pruning/merging/recurrence require discriminating ordinary/norm/compute controls.

Use the existing owner chats and worktrees. This implementation sends no messages,
alters no other owner source or registry, and starts no competing worker.

## Commands and verification

```powershell
python -m experiments.rineya_generalization_dynamics_v1.roadmap_cli check-card --input C:/Users/clclc/BusinessProject/gd001-records/OWNER/quality-card.json
python -m experiments.rineya_generalization_dynamics_v1.roadmap_cli check-data --packet C:/Users/clclc/BusinessProject/gd001-records/OWNER/data-admission-packet.json
python -m experiments.rineya_generalization_dynamics_v1.roadmap_cli prepare --spec C:/Users/clclc/BusinessProject/gd001-records/OWNER/roadmap-spec.json --output C:/Users/clclc/BusinessProject/gd001-records/OWNER/new-roadmap-packet
python -m experiments.rineya_generalization_dynamics_v1.roadmap_cli verify --output C:/Users/clclc/BusinessProject/gd001-records/OWNER/new-roadmap-packet
```

Prepare output under a fresh external BusinessProject/gd001-records directory.
Keep TEMP/environments/caches under BusinessProject outside source and `.codex`.
The CLI has no launch, dispatch, train, native-evaluate, confirmation or deploy command.
GD's 1,800-second/16,000-forward DEV-plus-parity envelope applies only to its reviewed
future run, not source training, reviewer turns or another experiment.

Tests cover count/identity/type mutation, fixed-support retention, incomplete domains,
fixture provenance, reference custody, protected access, EOS/masks/empty outputs,
all224 assignment/history joins, original-owner launcher custody and packet integrity.
Fixtures, tiny tensor stubs and public aggregate validation do not establish any new
Rineya quality gain. Previous GD code, historical receipts and local changes remain
preserved; new native proposals must bind the current expanded code digest.
