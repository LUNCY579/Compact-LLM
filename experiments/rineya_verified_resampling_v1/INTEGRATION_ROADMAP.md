# Integration roadmap and implementation handoff

**Specification only. Names below describe proposed new modules, not implemented APIs or runnable commands.**

## 1. Existing integration surfaces

All inherited references are bound to parent commit `ba683ac0fcadbf04e4e179200d035b6a980e388f`; source keys are in [SOURCES.md](SOURCES.md).

| Existing surface | Proposed reuse | Boundary to preserve |
| --- | --- | --- |
| Centralized `CENTRALIZED_EXPERIMENT_REPORT.md` | Exact Titan identity, template/geometry, prepared data, known resource failures | Recorded/static evidence is not current native qualification |
| Centralized `EXPERIMENT_REGISTRY.json` | Align with E3 objectives, E4 sampling, E5 composition, and SIDE inference studies | Do not amend E0/E1 or call this draft a fully registered native experiment |
| PRLC README and contracts | Inspect public prompt, task, episode, and local-loader boundaries before adapting | The native loader does not automatically support every custom family artifact |
| RHD README and contracts | Independent attempts, bounded local environments, replay audits, TRAIN-only compilation | Keep private reviewer information out of student/selection views; no native trainer implied |

Inspect the actual implementation and applicable repository instructions before changing code. Reusing a schema requires semantic equivalence, not just matching field names. No centralized registry or inherited source file is changed by the planning delivery.

## 2. Stage A: CPU-only implementation after approval

Proposed files under this directory:

| Proposed module | Responsibility |
| --- | --- |
| `contracts.py` | Separate immutable public task/candidate records from private oracle/audit records |
| `sample_bank.py` | Deterministic attempt identities, append-only status records, sample-bank validation |
| `selectors.py` | Seeded random and registered vote selection using allowlisted public fields only |
| `oracle_evaluation.py` | Privileged grading, prefix coverage, combinatorial coverage estimates, paired reports |
| `budget_accounting.py` | Attempted/returned/failed/pending work, tokens, wall time, tools and storage |
| `report.py` | Distinguish oracle/selected/greedy/sampled outcomes, incomplete banks, costs and status |
| `tests/` | Mechanical, adversarial, leakage, splitting, and estimator tests |

Do not add `train_export.py`, a native loader, a paid API client, or a production route in the first CPU-only step. Do not mutate PRLC/RHD merely to make imports convenient. Use toy synthetic records that cannot be mistaken for real Titan outputs.

### Proposed data boundaries

A private task catalog contains source/revision/parent/split, public prompt, and separately privileged reference fields. A public task view contains only the user-visible prompt and opaque identity required for bookkeeping. Hidden task-family labels or expected abstention must not reach selectors or routing policies.

A public candidate record contains attempt identity, permitted model/policy identity, output text/token IDs as appropriate, termination state, and explicitly allowed observed features. It contains no oracle correctness. Candidate ordering is frozen independently of grading.

A private grading record maps an attempt to an independently bound verdict and grader revision. A selection record contains candidate choice or an explicitly registered fallback, selector identity, and publicly justified decision features. A cost record is separate from correctness and records failed work too.

A run manifest binds source code, source assets, runtime, task/split population, sampler/selector/parser identities, resource ceilings, and permissions. Host paths are deployment configuration, not hard-coded assumptions. Do not copy private records into public DTOs by serializing entire objects.

### Minimum CPU acceptance suite

1. Coverage formula endpoints, invalid k/n/c, all-wrong/all-correct banks, and exact small-bank enumeration agree; matched prefix metrics are not confused with random-subset estimates.
2. Correct-minority/wrong-majority examples separate oracle success from vote failure. Seeded random and tie handling are deterministic and do not inspect gold.
3. Gold, expected abstention, hidden family/source labels, and oracle scores cannot enter a selector view; reject forged fields, identity collisions, and post-serialization leakage.
4. Legitimate empty text/set outputs, malformed outputs, repeated answers, truncation, EOS, no-valid-candidate pools, and fallback abstention follow the explicit contract.
5. Parent/split collisions and protected roles fail closed. Future export-role checks reject DEV/selection/CHECK/FINAL and retention-evaluation records.
6. Attempt seeds are independent of oracle labels and run scheduling; repeated/reordered/resumed writes cannot silently create new samples or overwrite earlier attempts.
7. Pending/failed/timeout records remain in accounting; incomplete banks cannot be reported as complete. Budgets stop work before a prohibited operation and preserve partial evidence.
8. Parent-cluster reporting preserves paired rows and exposes undefined denominators. CPU fixtures explicitly report native calls and training updates as zero.

These are planned tests. No test count or passing result is asserted for this delivery.

## 3. Stage B: exact-host inference qualification

Separate approval is required. Build an inference adapter only after inspecting the exact current qualified loader and source assets. Verify recorded hashes, actual loaded geometry/ties, prompt token IDs, full stochastic-token path, independent RNG streams, EOS, dtype, inference state, maximum intended shapes, and original single-answer parity.

Native collection is not the inherited E0 training command. A training-capable command can perform updates even when the research intention is inference-only; do not invoke it as a shortcut. Bind an actual inference-only operation with optimizer access disabled.

Preserve the centralized resource floors and process containment. Start sequentially, record cold-start and steady-state memory/time, and qualify failure/resume accounting. New source or runtime revisions need requalification. Do not retry unchanged blocked host state or treat this sandbox's CPU characteristics as the user's Windows RTX 4070 host.

## 4. Stage C: frozen-model pilot and reporting

Before generation, complete the null launch fields in `plan.json` through an explicit immutable amendment. Freeze the 128 parent IDs and decoding/selection policies on disjoint calibration inputs. Read [PROTOCOL.md](PROTOCOL.md), not a copied chat summary, for denominators and entry gates.

Generate the entire fixed bank without oracle-informed early stopping. Run allowed selectors on public views and evaluate using the isolated oracle. Store raw attempted outputs locally under explicit data policy; do not commit sensitive candidate/gold payloads by default. Produce a report that distinguishes complete, incomplete, blocked, and scientifically unfavorable outcomes.

Do not merge a selector into production or declare a stronger model from this screen. Qualified preservation and new protected confirmation are separate requirements.

## 5. Conditional successor experiments

### Selection and verification

Only when coverage supports it, train or calibrate a verifier on permitted disjoint training/calibration parents. Compare against random/vote and direct solver controls. Record verifier compute and test access. Keep a larger-model judge an explicit larger-system arm, not an uncounted part of a sub-1B claim.

### Long context

After 2048/4096 runtime qualification, reuse the prepared paired-world input contract. Separate answerability, evidence placement, and distractor effects. Preserve world clustering. Qualification at a short prompt does not certify long-context fitting or retention.

### Family portfolios and composition

Require compatible, qualified donor identities and frozen allocation. Compare sixteen total attempts from one model against sixteen total across variants. Count model-loading/switching and verifier cost. Compare output-level complementarity to a separately qualified merge, not to weight cosine alone.

### Adaptive inference and architecture controls

Freeze a stopping/routing rule on disjoint calibration parents. Compare 1/2/4/8/16 adaptive allocation with fixed budgets at matched measured costs. Treat recurrence, longer reasoning, retrieval, cache changes, and stronger-model baselines as separate interventions with qualified runtime/quality controls.

### Verified training data and RHD

A later `train_export.py` may accept only successful, complete, correctly bound TRAIN records. Preserve unsuccessful attempt costs; enforce parent caps and explicit selection weights. A correct final answer alone does not qualify all intermediate rationale or tool steps.

For RHD, resample entire independently reset trajectories, then use actual replay verification. Keep student-only, privately reviewed, and teacher-generated arms distinct. Export current-action targets without private reviewer notes. No unreviewed code execution or privileged references in deployed trajectories.

Any training experiment separately freezes checkpoint, objective, optimizer/update budget, target-token budget, replay, and evaluation. RSFT/GRPO/MaxRL comparisons cannot inherit a biased success-only rollout distribution as though it were unchanged on-policy RL. The target is held-out pass@1 improvement with preservation, not merely a larger harvested pool.

## 6. Bounded handoff for the next implementation agent

> Work in `LUNCY579/ChatBubble`, branch `research/rineya-verified-resampling-v1`. First fetch and inspect the live branch; record its starting commit and check repository instructions. Read this package and the pinned centralized/RHD/PRLC references. This handoff specifies CPU-only implementation, not model execution; obtain the user's explicit implementation approval before making additional changes. Do not switch to main, amend E0/E1, modify weights, relax guards, call paid teachers, download models, train, or open CHECK/FINAL. Implement only Stage A contracts, sample-bank bookkeeping, public-view random/vote selectors, isolated oracle evaluation, accounting, and tests. Use synthetic fixtures labeled as such. Keep unresolved native-plan values blocked. Test leakage, split identity, estimator semantics, valid empty outputs, failure accounting, and replayable seeds. Report exact files and actual test outcomes; do not imply native efficacy. Commit and push only to this dedicated branch after approval, then verify the remote head and changed paths. Stop and report conflicts or qualification gaps instead of silently changing the experiment.

This text is a saved handoff, not a background task or an authorization to execute it now.
