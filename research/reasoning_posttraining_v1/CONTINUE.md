# Continue RRP-001

## Mandatory scope

Read [GOAL.md](GOAL.md), [STATUS.json](STATUS.json), [RESULTS.md](RESULTS.md),
[CLAIM_LEDGER.json](CLAIM_LEDGER.json), [SOURCE_AUDIT.md](SOURCE_AUDIT.md),
[EXPERIMENT_REGISTRY.json](EXPERIMENT_REGISTRY.json) and the package
[AGENTS.md](../../experiments/reasoning_posttraining_v1/AGENTS.md).

**The goal is still active and is not close to complete.** A qualified instrument, a met
microfit criterion, a verified resume and a published branch are not the model-quality
deliverable. Nothing measured so far supports any reasoning, preservation or
generalization claim, and nothing may be reported as one.

Worktree: `C:/Users/clclc/BusinessProject/ChatBubble-rineya-reasoning-posttraining-v1`.
Branch: `research/rineya-reasoning-posttraining-v1`.

Refresh actual state before trusting anything below:

```powershell
git status --short; git rev-parse HEAD
git ls-remote --heads origin research/rineya-reasoning-posttraining-v1
```

## What is now done, and exactly what it proves

Three supervised launches ran against real Supa 1.7 weights, each with a terminal
supervisor receipt and a settled budget entry. Receipts are committed under
`evidence/inspection-20260915/rrp001-runs/`; checkpoint tensors (~11.3 GiB) stay on disk
only, under `rineya-reasoning-posttraining-records/rrp001-journal/`.

| run | outcome |
| --- | --- |
| `qualify-001` | PASS. 596,049,920 unique / 314,618,880 trainable in 220 tensors, recounted natively. Source state byte-identical before and after. 0 optimizer steps. 32/32 approved targets reproduced by the native tokenizer. 2048-token shape qualified. |
| `train-001` | 105/105 updates. Fixed 32-world TRAIN probe 8/32 -> 29/32 -> **31/32**, response CE 1.496 -> 0.0855 -> 0.0122. Pre-registered criterion was >= 31/32: **met**. Frozen weights unchanged. |
| `resume-001` | PASS, 15/15 comparator checks. 103 updates from `checkpoint-00002` in a fresh process reproduced the candidate's final parameters **bit-identically** and every per-row prediction. |

**This is engineering evidence only.** The probe rows are the same 32 rows that were
trained on, so 31/32 measures fitting, not generalization. What it does establish is that
the optimizer, label masking, EOS handling, generation path, scorer, checkpoint/resume and
budget accounting all work end to end on real weights.

Do not treat `checkpoint-00105` as a quality candidate. Do not export, merge, quantize or
register it.

## Authorization state

[MODEL_EXECUTION_AUTHORIZATION.json](MODEL_EXECUTION_AUTHORIZATION.json) records the
operator's verbatim session grant and the numeric limits this agent instantiated from it
(labelled agent-derived, not operator-quoted). [GRANT.json](GRANT.json) is the code-enforced
envelope, bound to the exact prepared bytes and plan digest.

The RRP-001-MICROFIT envelope is now largely spent and **its single
`adaptive_model_selection_rounds` is consumed**. Remaining measured headroom is in
`STATUS.json`. Any further training requires a **registered successor experiment with its
own envelope** — do not stretch this grant. Cloud, teacher, RL and release lanes remain
pinned at zero and were never opened.

`human_data_review` is still **NOT_PERFORMED**. This agent did not and will not
self-certify it. Until the operator signs `REVIEW_PACKET.md` (85 worlds prepared), no
DEV/retention comparison may be presented as a quality claim.

## Exact reproduction

```powershell
$env:PYTHONPATH = 'C:/Users/clclc/BusinessProject/ChatBubble-rineya-reasoning-posttraining-v1'
$py = 'C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe'
& $py -m pytest experiments/reasoning_posttraining_v1/tests -q   # 76 passed, 0 skipped
```

Native commands take `--source --prepared --plan --grant --journal --run-id --timeout
--confirm-model-execution`; see the three `launch.json` files in the evidence tree for the
exact argv actually used. Bindings are `prepared-004` + `preflight-002`
(`data_contract_sha256 66aa9a9d...`, `plan_sha256 9208fc92...`); the source is
`ChatBubble-ced-native-loader-qualification/Supa`, read-only.

**A native run requires a clean checkout of this branch.** `controller.checkout()` refuses
a dirty tree, and `binding` includes the package code hash and HEAD, which
`validate_checkpoint` enforces. Consequence, learned the hard way (CL-013): you cannot
edit or commit **any** repo file between a candidate attempt and its resume without
invalidating the resume.

## The next work, in order

None of step 1 needs a model, a GPU or a budget. Do it first.

1. **Build the measurement instrument.** This is the real blocker and it does not exist:
   - a repeatedly usable DEV bank, a preservation/abstention DEV, and a genuinely fresh
     confirmation bank kept out of the optimizing loop (`confirmation` is still
     `NOT_ESTABLISHED` in the data contract);
   - UNKNOWN and IMPOSSIBLE strata — present quotas cover only `EXACT_PRIMITIVE`,
     8 families x 4 worlds;
   - shortcut baselines (constant-label, cue-copy, last-mentioned) and a positive-control
     exact solver, so a broken instrument cannot be misread as a model result;
   - same-world short/long and early/middle/late evidence variants, with operator depth
     varied separately from distractor length.
2. **Predeclare** primary contrasts, family weights, cluster units, practical margins,
   uncertainty method and the allowed number of model-selection reads — before any arm runs.
3. **Register a successor** experiment and request one concrete bounded envelope for it.
4. **Then** run the A/B/C ladder. Report it as review-pending unless the packet is signed.

The baseline probe already decomposes the source model's failures into three distinct
mechanisms, which should not be chased with one undifferentiated mixture:

- **abstention miscalibration** — the source emitted 12 literal `UNKNOWN`s against 6 true
  `UNKNOWN` references, over-abstaining on computable items and answering `amber` where the
  truth was `UNKNOWN`;
- **response-contract violation** — `dialogue_memory` 0/4, emitting prose
  ("Use Workshop for my ve...") instead of `venue=Workshop`;
- **execution and binding errors** — `program_trace` arithmetic (`12` vs `1`) and
  `relational_join` wrong-entity (`OrenHill` vs `MiraGlass`).

The one world that 8 exposures never fitted is a `signed_ledger` arithmetic case
(`211` vs `145`) — a useful, specific falsification target.

## Known instrument defects to fix in the successor

- **CL-014**: `engine.final_evaluation_sha256` hashes `final.json`, which
  `native_runtime.evaluate` fills with `generation_wall_seconds`. It is nondeterministic by
  construction and useless as a resume criterion. Move timing out of the hashed artifact.
  The comparator works around it by hashing with timing excluded.
- **CL-008**: `qualify()` charges 2 evaluation calls while performing ~16-17 forwards.
  Under-charging, inside the plan's 32-call allotment, but not faithful accounting.
- **CL-009**: evaluation-call headroom is 30. A retried resume needs 32 and would be
  refused. Fails closed, correctly; needs an amendment, not a quiet top-up.
- **CL-011**: checkpoints exist only at update 2 and the final update, so a late abort
  forfeits ~102 updates. `Design.checkpoint_every` is ignored by this engine.
- **CL-004**: budget-journal tail truncation is undetectable by construction. Run evidence
  without a reservation *is* refused, which covers every worker attempt.
- Promote `evidence/inspection-20260915/rrp001-runs/verify_resume.py` into the package.

Preserve negative results. Never extend a test until significance appears, never pool seed
measurements into artificial sample size, and keep `CLAIM_LEDGER.json` growing rather than
resolving contradictions silently.
