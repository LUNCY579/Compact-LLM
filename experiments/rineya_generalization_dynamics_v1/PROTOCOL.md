# GD-001 protocol v1

## Scope and hypotheses

H1: On one native 0.6B post-training trajectory, clean-task performance and misleading-
context performance need not rank checkpoints identically.

H2: Selecting an eligible checkpoint by development misleading-context macro accuracy
can improve the corresponding **fresh confirmation** score over the final checkpoint.
The primary contrast is `generalization_minus_final`; `ordinary_minus_final` is a
secondary comparison. Both may be zero or negative. No winner is assumed.

H3 (descriptive): R+S can fail even on cases whose R and S clean controls are solved.
Item-level repair/damage and three-checkpoint reversals are descriptive behavioral
measurements, not proof of circuit competition or the paper's causal explanation.

The paper's section 3.6 also examines fine-tuning phenomena. Its scope is therefore not
literally restricted to pre-training, but this Rineya 0.6B recipe and selection policy
are new hypotheses, not established replications.

## Registered comparison

Use one trajectory: one base, tokenizer, training manifest, architecture, adapter
configuration, prompt format, decoding policy and runtime. Register an initial base
(step zero) and 2-19 strictly ordered training snapshots; 6-10 saved snapshots are a
reasonable first real pilot. Never compare unrelated model-family names as if they
were adjacent checkpoints. The last registered training snapshot is the final control.

Full-model and LoRA snapshots cannot be mixed within a trajectory. The base and every
snapshot have content-addressed file inventories. Metadata presence alone does not
qualify real weights. The actual loaded base count must match the declared count.
Lineage is caller-declared: hashes establish identity, not independent reconstruction
of the upstream training run. Preserve upstream training receipts alongside this run.

## Data and linkage

Six paired families each have a clean and a misleading variant with an unchanged
answer and declared wrong shortcut. There are no handpicked failures from Rineya's
historical banks. Arithmetic demonstrations remain mathematically correct. Retrieval
traps use explicitly obsolete records, not contradictory authoritative evidence.

R/S/RS triplets share a `skill_case_id` and latent quantities. Report composition both
unconditionally and conditional on solving the matching R and S clean controls.
Retain zero-support denominators as null, never as 0% or 100% success.

DEV uses additive transformations and ACTIVE/ARCHIVED record templates. Confirmation
uses subtraction and CURRENT/RETIRED record templates, new entity namespaces, changed
position ordering and distinct seeds. These are explicit distribution shifts within
synthetic skill families, not proof of broad out-of-distribution reasoning.

All pairs and linked skill cases remain together in their split. The training manifest
must cover the entire training trajectory. Audit normalized exact prompt overlap and
parent-ID overlap. This is **not semantic decontamination** and cannot detect a training
manifest that omits examples. Do not train on these generators or tune on confirmation.
Public generator code is not a cryptographic seal or an independent external benchmark.

## Generation and likelihood are separate endpoints

Greedy, single-beam generation, no sampling or repetition penalty, explicit EOS,
registered non-thinking plain/chat prompt format, default 512 input tokens and 16 new
tokens. Prompts are never truncated. Length-limited completions cannot earn generated
correctness even if their visible prefix equals the answer. The integer parser accepts
only a complete integer, optional sign and surrounding whitespace; explanations,
reasoning tags and extra answer text fail formatting.

Likelihood independently scores every token of `tokenizer.encode(" " + candidate,
add_special_tokens=False)` appended to the **same already-tokenized generation prompt**.
This explicit token-boundary convention is not implicit joint-string retokenization.
Prompt loss is excluded; logits are shifted once. Record total log probability, answer
token counts, total-margin preference and a length-normalized diagnostic. Unequal
candidate lengths can favor different scores. Neither pairwise preference nor a
normalized score substitutes for generated correctness.

Record own-clean conditional robustness and a second conditional score on the fixed
reference-clean subset. The former diagnoses the current checkpoint; its denominator
changes across checkpoints and cannot alone establish improvement. Absolute all-item
macro misleading accuracy is the selection endpoint, not a moving conditional subset.

## Eligibility and DEV-only selection

The following are **new exploratory GD-001 settings**, fixed before model calls. They
are not imported from the paper, not replacements for existing Rineya gates, and not
statistical non-inferiority tests:

- Each of five acquisition/transfer families has clean accuracy at least 0.50.
- At least eight fresh retention-control prompts were solved by the reference.
- At least 95% of those reference-solved retention prompts remain correct.
- Clean five-family macro accuracy does not fall below the reference (default minimum
  acquisition delta 0.0). This default does not require or prove positive acquisition.

A higher acquisition margin can be preregistered in a new policy before `init`.
Thresholds, budgets, code and suite hashes are frozen in the manifest. Selection ranks
eligible trained snapshots by robust macro accuracy or ordinary clean macro accuracy;
ties choose the earlier training step, then checkpoint ID. The reference is never a
trained candidate. When no snapshot qualifies, confirmation stays blocked; do not
relax thresholds after inspecting the scores. The final snapshot remains the control
even when it fails eligibility.

The fresh retention probe is narrow copying/authority selection. Passing it cannot
establish broad language, long-context, abstention, safety or existing-skill retention.
Those require separately registered, selection-admissible instruments and a new
protocol version before any wider claim. Historical barred banks remain barred.

## Confirmation access and uncertainty

The selection lock binds all development receipts, policy and manifest. Confirm only
the reference, final, ordinary-selected and generalization-selected checkpoints,
deduplicating repeated choices. No selection function consumes confirmation data.

`confirmation.STARTED.json` is created exclusively **before** confirmation generation
or item access. A failed/interrupted attempt consumes the bank. The CLI does not resume
or rerun confirmation, replace its seed, or delete its receipt. The single-writer lock
is not a substitute for the one-use receipt. Preserve failures and register a genuinely
new follow-up study rather than clearing files to recover a favorable outcome.

Report paired wins/losses and a deterministic 2,000-resample 95% bootstrap interval.
Resample independent parent units within arithmetic strata and whole linked R/S/RS
cases within a skill stratum; the latter has weight three so the endpoint remains a
five-family macro average. A cluster sign test is descriptive. Neither the interval
nor the test includes variation across training seeds or corrects multiple exploratory
contrasts. A positive pilot calls for an independently registered, multiple-seed
replication; it does not authorize promotion.

## Safety and evidence levels

`mechanical_fixture` and `native_pilot` are incompatible provenance labels. Fixtures
can never be relabelled by a CLI flag. There are no fixture-dependent native fallback
paths. Content hashes detect accidental changes, not malicious owners who rewrite all
receipts; these are digests, not digital signatures or OS-level access controls.

Native inference requires an explicit flag, frozen local assets, available dependencies
and registered resource guards. Defaults: CPU FP32, two threads; 7.4 GiB free host RAM,
1 GiB free disk; CUDA additionally needs 7.4 GiB free VRAM and no foreign compute
process reported by nvidia-smi. Guards cannot be lowered. One checkpoint is loaded at
a time, in inference mode. No optimizer, gradient update, training loop, download,
source-weight modification, checkpoint averaging or production routing change occurs.
Use the host's existing scheduler too: GD-001 locks coordinate only GD-001 processes;
preflight observations cannot prevent an unrelated process starting later.

Native evaluation also freezes an aggregate budget before registration: by default
1,800 seconds and 16,000 actual model forward calls per split. Loading, scoring and
unloading share the wall budget; all checkpoints and resumed DEV attempts share the
forward budget. Generation and both candidate-likelihood forwards count. Immutable
attempt receipts retain spent resources on failures; an interrupted attempt without a
completion receipt blocks resumption because its consumption is unknown. Completed
checkpoint receipts are reused without spending additional model forwards.

A completed sweep replays its validated receipts without opening another budget
attempt. Registry verification failures after scoring permanently invalidate that
split; restoring the original assets cannot rehabilitate its predictions. Keep the
invalidation receipt and use a new registration after resolving the asset change.

Time checks are cooperative before loads and model forwards. One in-flight load or
forward may finish after the deadline; a hung native operation requires a supervisor
that terminates only its own worker. The forward limit prevents the next counted call
from starting. Budget exhaustion is a blocked outcome, not permission to enlarge a
registered budget or discard its receipts. Use a new registration for a changed budget.

Every outcome retains `permits_model_promotion: false`.
