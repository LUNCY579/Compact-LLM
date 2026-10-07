# Native-host handoff

## Current status

All implemented offline checks can run from the repository root. The saved
`VALIDATION.json` distinguishes standard-library tests, tiny CPU optimizer tests,
and NOT_RUN native Rineya qualification. No laptop work or Codex task was started.
No 27B teacher, paid API, model download, GPU allocation or optimizer on Rineya was
requested implicitly by this code. A prior grant to a different experiment does
not carry over.

## Host prerequisites

Obtain a fresh resource-approved run scope: immutable local 0.6B checkpoint and
tokenizer hashes, qualified loader, explicit CPU/GPU choice, memory headroom,
per-call cancellation/timeouts, total token/wall budgets, checkpoint destination,
and allowlisted trainable parameter names. Preserve old weights. Use an isolated
worker with only approved tools; the in-memory RHD environment is not a secure
terminal sandbox. No API key or external provider adapter is included.

Qualify the plain RHD JSON envelope before adoption. Native loaders and packed
Rineya formats remain owned by the inherited loader track. Do not assume all family
artifacts are Hugging Face-compatible. The adapter below wraps already-loaded
resources and imports PyTorch only after explicit authorization.

## Collection and training API

```python
from experiments.rineya_recursive_self_rewrite_v1.core import Limits, enroll
from experiments.rineya_recursive_self_rewrite_v1.pipeline import discovery, reconstruct
from experiments.rineya_recursive_self_rewrite_v1.training import hf_backend, train_sft
from experiments.rineya_recursive_self_rewrite_v1.data import compile_dataset, match_controls
from experiments.rineya_recursive_self_rewrite_v1.evaluation import evaluate_panel, compare_panel

# model/tokenizer: loaded by the approved Rineya host; never created by this example.
# tasks/ancestry: audited RHD Task objects and complete parent-origin declarations.
# Native token limits below are EXAMPLES, not a hardware qualification or a grant.
catalog = enroll(tasks, ancestry)
backend = hf_backend(model, tokenizer, checkpoint_id=checkpoint_sha256,
                     authorized=model_calls_authorized, max_context=4096)
limits = Limits(tokens=100_000, calls=160, wall_seconds=120, new_tokens=256)
training_tasks = [t for t in tasks if t.split == "train"]
found = discovery(training_tasks, backend, limits, attempts=2, seed=37)

# Freeze eligible source_ids, source selection, budgets and recipe before training.
# Use reconstruct(..., arm="one_pass") and reconstruct(..., arm="recursive")
# with the SAME student as executor; planner/critic backends are explicit arguments.
# Then compile_dataset(...), write_run(...), verify_run(...) before native training.

receipt = train_sft(
    model, tokenizer, compiled_dataset, authorized=training_authorized,
    initial_checkpoint=checkpoint_sha256, tokenizer_id=tokenizer_sha256,
    trainable_names=approved_parameter_names, optimizer_steps=4,
    tokens_per_step=128, learning_rate=1e-5, max_context=4096, seed=37,
)
# Save to a NEW local destination using the qualified host's safe checkpoint path.
# Verify reload fidelity before evaluating. Never treat decreased loss as efficacy.
```

Reload the same untouched parent before training EACH arm. Do not apply sequential
arms to one mutated model. `match_controls` must compare complete actual receipts
and dataset manifests. A hand-authored metadata string is not evidence that a loader
used the claimed weights; the host must bind identities to real bytes.

The reference trainer can select existing adapter parameters but does not attach
LoRA modules, optimize quantized base weights, automatically select layers, or add
retention replay. Those are separate host treatments. Keep the no-replay screen
small and explicitly unqualified for retention until evaluation. Interrupted runs
must discard partially updated models; there is no native resume/checkpoint manager.

## Evaluation and ownership

Use `evaluate_panel` on calibration and independent retention_eval tasks only. It
constructs H0 executions without runbooks/reviewers. Compare complete panels using
`compare_panel`; archive native token/cost ledgers alongside results. Register an
external sealed evaluator before a confirmation claim; this runner refuses sealed
access rather than adapting against the confirmation set.

Next additions, only after native mechanism evidence: independent retention replay,
new structured-output/file-repair environments with their own security qualification,
external task-aware semantic-leakage review, native durable logging/resume,
licensed teacher adapters, and integration with CSS/independent coefficient studies.
Do not merge all these treatments into the first experiment or change production
inference merely because its CPU protocol tests passed.
