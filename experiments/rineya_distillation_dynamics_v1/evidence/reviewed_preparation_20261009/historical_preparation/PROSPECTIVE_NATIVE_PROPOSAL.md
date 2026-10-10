# RDD-001 selected preparation — 2026-10-05

Status: engineering design reviewed; **native invocation blocked and not authorized**.
Preparation only: no model loads, native evaluations, optimizer updates or
confirmation-bank reads. CPU tensor/metadata fixtures are mechanical evidence.
The worktree is `C:/Users/clclc/BusinessProject/ChatBubble-rineya-distillation-dynamics-v1`,
branch `research/rineya-distillation-dynamics-v1`, starting at
`2d7be4b5c2ba8a610b09d17455d519eecaa421e6`. Existing evidence and GD owner edits
are retained. No conversion, stock-Qwen substitution or imported permit is allowed.

## Frozen selected comparison

Use `experiments/rineya_distillation_dynamics_v1/configs/selected-pilot.json`:

| Cell | Rollout / objective | LR | Seed | Treatment steps |
|---|---|---:|---:|---:|
| A-lr0-s17 | Fixed teacher / full-vocabulary FKL | 1e-5 | 17 | 2 |
| B-lr0-s17 | Current student / full-vocabulary FKL | 1e-5 | 17 | 2 |
| D-lr0-s17 | Current student / full-vocabulary RKL | 1e-5 | 17 | 2 |
| SFT-lr0-s17 | Verified solution / response CE | 1e-5 | 17 | 2 |

A−B isolates rollout source at FKL; B−D isolates direction on current-student
prefixes. The same zero-B initialization, task/update schedule, one independent
replay example per update, replay weight 0.1, sampling temperature 1, rank 8,
alpha 16, response cap 32 and context cap 1024 apply throughout. Each of the two
steps saves a retained checkpoint and original standard-PEFT child export.
Evaluation has greedy plus two draws per registered DEV row. Realized response
token counts are reported; matching update/task exposure does not match tokens
or compute. No capped/wrong rollout is filtered. Teacher-bank creation cost is
retained for each consumer. No source×direction interaction, replicated method
ranking, broad retention claim or long-context preservation claim is supported.

The one LR is frozen for this proposal because four cells with the same rate
separate the two prespecified contrasts without introducing a search. 1e-5 is
the lower historical engineering placeholder, selected conservatively for this
two-step pilot. This is a design review, not empirical LR calibration or an
execution grant. The native LR/source reviewer attestation remains unset.
The acquisition gain +0.03 and each-source retention drop ≤0.02 gates are retained.
All four cells and both steps are required; a no-learning pilot can select nothing.

## Exact proposed assets

Student: `C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Experimental-1.0.0-devfreeze`.
Weight SHA-256: `537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7`.
RDD snapshot: `4e6e8f7f0ef89662c2ec910280a72501e05f2e89cf88498a62edfa029af0609f`.

Fixed candidate teacher: `C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze`.
Weight SHA-256: `962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b`.
RDD snapshot: `a1da3fdd639d368c45db9e3ba8b8a0237e2c59abe81ce88493f514c59768c339`.

Both store 596,049,920 BF16 tensor elements in standard Qwen3 layouts and have
byte-identical tokenizer/template assets. Header counts are not runtime-qualified
parameter counts. Supa is frozen as the prospective candidate because it satisfies
the static tokenizer/layout prerequisite and fits the same serialized size class;
its name supplies no competence evidence. Baya is inventoried as complementary
evidence, not an adaptive teacher fallback. Halo fails strict tokenizer backend
identity; no asset modification is proposed.

RDD will use EOS 151645 from model config and tokenizer pad 151643 with no padded
batch or prompt truncation. This differs from the historical central E0 EOS list
[151645,151643]/pad151645. Fresh scope-matching qualification must resolve that
contract; neither old CE512 evidence nor generation_config is silently imported.

## One prospective native action

After the missing prerequisites are genuinely supplied, reviewed and registered,
the next proposed action is **one supervised qualification-only invocation**.
It is a separate future decision. It must stop before treatment, selection and GD
diagnostics. No automatic retry or chained campaign is proposed.

```powershell
$py = 'C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe'
& $py -B -m experiments.rineya_distillation_dynamics_v1 qualify `
  --registration 'C:/Users/clclc/BusinessProject/rdd001-records/selected-native-v1/registration.json' `
  --output 'C:/Users/clclc/BusinessProject/rdd001-records/selected-native-v1/qualification-once' `
  --allow-model-load --allow-training --ack-host-available
```

The registration/output paths above do not exist and were not created. The flags
describe the required future grant; this document is not that grant. A clean,
reviewed source checkout and newly frozen implementation hashes are required.
The qualification-only worker cannot enter `run_arm` or read confirmation.

Its catalog must have exactly 16 distinct public TEACHER_DEV parents with empty
solutions and fresh owned train/replay/DEV roles. Proposed small DEV enrollment:
8 acquisition, 8 harder transfer and 8 retention parents from a reviewed short
skill source, plus 2 train and 2 replay parents, all disjoint. The actual catalog
and approved historical parent/raw-prompt exclusions remain missing. SFT/replay
references must pass the declared final-integer/exact grader and fit cap32 including
one EOS. Hard transfer changes operation/dependency structure, not just constants.
No retention source is certified by a label; content and baseline reliability need
review. No confirmation catalog is opened or generated by this preparation.

The screen greedily evaluates all 16 student/teacher cases. Teacher accuracy must
be at least 0.6 and advantage at least 0.1 (thus at least 10 teacher successes and
2 net extra successes out of 16). Wrong and capped attempts stay in denominators.
If the screen passes, qualification performs two disposable updates, one FKL and
one RKL, each from the same original adapter with a new optimizer. It compares
dense versus chunked full-vocabulary values/hidden gradients, verifies finite
scoped backward and frozen teacher/base hashes, then exact adapter and native
logit save/reload. Finally it restores the source bitwise and records restoration.
No treatment optimizer is reused. Failures, partial prompt inventories, bootstrap
probe attempts and restoration failures remain immutable.

At cap32 this invocation has at most 35 generation attempts / 1,120 generated
tokens and 1,130 backbone forwards (including manual probes), excluding parameter
loading and output projections. These are prespecified accounting bounds implied
by this plan, not measured cost or FLOPs. The unchanged hard supervisor limits are
1,800 seconds, 4 GiB process-tree RSS; admission is twice 7.4 GiB host available,
4 GiB GPU and disk, with running headroom 6 GiB host / 0.5 GiB GPU. Every actual
forward checks guards; qualification can fail before completing its horizon.
Model serialized size alone does not prove the two-model process fits 4 GiB RSS.

## Prerequisites still blocking the action

1. Matching genuine RDD-admissible E0 evidence. The historical
   `PASS_E0_UNPADDED_CE512_SCOPED` receipt is preserved and is not accepted by
   the existing `PASS_E0_SCOPED_NO_EFFICACY_CLAIM` contract. Its narrower context,
   masking/stopping and 1.25 GiB host reserve do not qualify this proposal.
2. Independent exact-native versus standard-HF output/lineage parity for both
   exact assets, plus reviewed feedback at public student error prefixes.
3. The fresh reviewed catalog, exclusion metadata, content/context review and
   unopened confirmation digest commitment required by registration. The sealed
   bank content is not an input to preparation or qualification.
4. Actual named LR/source review and future explicit model-load/disposable-update
   authorization. Current approval is offline preparation only.
5. An available owned host and normal shared GPU lock admission. Other owner
   Python processes were observed; no worker was stopped or lock stolen.

## GD handoff and exact parity boundary

Future standard PEFT saves preserve both q/v factors, tensor roundtrip hashes,
exact base/tokenizer identity and original/derived/disposable artifact roles.
The complete RDD consumption ledger includes all qualification, teacher-bank,
source baseline, DEV, acquisition feedback, solution CE and independent replay
exposures with raw/rendered public text, IDs, purpose and policy version. Its GD
gradient projection retains source/filter/consume/unused occurrences and repeats.
Public DTO projection is explicitly identified; it does not claim reconstruction
of an upstream raw dataset. Keep the full approved source catalog independently.

The GD owner at `C:/Users/clclc/BusinessProject/ChatBubble-rineya-generalization-dynamics-v1`
has the active chat `Prepare GD-001 attachment contract`. That chat owns GD edits.
Its `gd001_attachment_v1` also needs a genuine prospective producer approval,
original post-save receipts for steps1/2, a real completion receipt and final
checkpoint binding. This preparation creates none of those execution declarations.

Actual custom-LoRA versus loaded HF/PEFT full-logit and greedy token parity is
still required at both original snapshots under a prespecified tolerance/prompt
policy. PEFT is absent from the pinned native environment; an independently
reviewed runtime is a prerequisite, not an install performed here. Metadata,
roundtrip serialization or an attachment audit is insufficient native parity.

GD must exclude every RDD public raw prompt/parent, including qualification and
DEV, not just the rendered gradient prompts. The rendered-versus-raw overlap gap
has been reported to the GD owner. Handoff stays prospective until all these
requirements are met. No GD attachment or capability result is claimed.

Next concrete decision: review this fixed Titan/Supa pair and 1e-5 design, then
obtain the missing exact E0/export/data/host evidence. Only after that can the one
qualification-only action be considered. Treatment remains a later explicit
decision; neither the 24+6 campaign nor the 12+6 screen is proposed.
