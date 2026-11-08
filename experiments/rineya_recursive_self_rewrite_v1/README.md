# Rineya Recursive Self-Rewrite — RSR-001

A student-first reconstruction extension to RHD-001. **Implemented and CPU-tested;
no native Rineya result, trained checkpoint, terminal-benchmark reproduction, or
retention improvement is claimed.**

Dedicated branch: `research/rineya-recursive-self-rewrite-v1`.
Parent: `research/post-training-mimo-centralized-v1` at
`19551ddebc87c93161c303865ce6d0f130de0093`.
The RHD contracts/environment are reused in place, pinned by Git blob hashes.
No upstream experiment, production inference, checkpoint, existing branch,
laptop schedule, or unrelated game is modified.

## Implemented path

```
fixed student -> H0 resampling vs H0/H1/H2 discovery -> verified source
             -> planner -> restricted runbook -> critic -> bounded revision
             -> SAME student, NEW environment, H0 plus private training guidance
             -> replay + public-evidence sufficiency -> per-action SFT
             -> H0-only calibration and retention evaluation
```

`core.py` implements strict bounded runbooks, identity-bearing callbacks, source
binding, ancestry enrollment, and a multi-role resource ledger. `runtime.py`
uses real deterministic RHD tool execution, never a reported success bit alone.
`pipeline.py` implements discovery, reconstruction, critic binding, fresh-execution
provenance, and a separately labeled RHD PASS/REPLACE correction treatment.
`data.py` re-audits sources and descendants, enforces per-parent caps, separates
self/teacher-assisted data, and removes private guidance from rewritten **inputs**.
`evaluation.py` implements full paired coverage, pass@k, parent-cluster bootstrap,
and calibration/retention plus per-source/skill breakdowns.
`training.py` provides an opt-in adapter for an already-loaded causal LM and a
real PyTorch optimizer with exact supervised-token quotas and parameter allowlists.
`artifacts.py` provides exclusive private bundles, integrity checks, deterministic
replay, and reconstruction of the exported dataset for verification.

## Run locally

From the repository root, Python 3.11 or newer:

```bash
python -m unittest discover -s experiments/rineya_recursive_self_rewrite_v1/tests -v
python -m experiments.rineya_recursive_self_rewrite_v1 plan
python -m experiments.rineya_recursive_self_rewrite_v1 demo --tasks 12 --out ./runs/rsr-demo
python -m experiments.rineya_recursive_self_rewrite_v1 verify --out ./runs/rsr-demo
```

The default pipeline uses only the standard library. Seven additional optimizer /
backend tests run when PyTorch is installed; otherwise unittest reports them as
skipped. Nothing downloads or installs dependencies automatically. The demo's
executor, planner and critic are **hand-written fixtures**. Their successes verify
plumbing, not learned skills. Demo one-pass and recursive outcomes are not evidence
that recursion improves a model.

Every output directory must be new. The run contains the private catalog, discovery
records/call logs, full runbooks/critique journals, rewritten executions, SFT data,
and a manifest written last. The `verify` command checks file hashes, replay,
source lineage, fresh sandboxes, approvals and recompiled dataset identity.
Do not publish private run outputs. Existing bundles cannot be overwritten.

## Research treatments

| Treatment | Implemented behavior | Deployment evaluation |
| --- | --- | --- |
| Base | Frozen initial checkpoint | H0 only |
| Direct | Source-visible controller-context SFT | H0 only |
| One-pass | One planner/critic round per candidate; fresh execution | H0 only |
| Recursive | Bounded rejection, revision and fresh execution | H0 only |
| Corrected | Separate recursive arm with RHD student-first PASS/REPLACE | H0, no reviewer |

Direct intentionally retains the source-visible harness hint. It is a
**source-context control**, not a claim to reproduce every original terminal
harness convention. Reconstruction excludes runbooks, critic conversations,
reviewer notes and rejected proposals from both prompts and targets.
Self and teacher-assisted reconstruction must be exported separately.

Discovery gives both strategies the same total per-task attempt slots and resource
caps: six H0 attempts versus two each from H0/H1/H2 by default. Actual calls, tool
use, characters and, for qualified native backends, tokens are recorded separately.
Equal caps do not imply identical consumed compute. Planner and critic calls count
toward reconstruction costs. Failures and budget-exhausted attempt slots remain in
the denominator.

H1 exposes state derived from public observations. H2 provides bounded public-error
repair guidance. **Unlike the paper's verifier-triggered source harness, H2 does not
query the hidden task grader to decide whether to continue.** The terminal grader
is consulted only after the model has finished. That deliberate difference protects
the H0 evaluation boundary.

## Native integration

See `HOST_HANDOFF.md`. The API supports actual model generation and optimizer
updates, but deliberately does not load weights or select a trainable scope for the
operator. Use the project's qualified checkpoint loader and obtain fresh host
approval. The ordinary CLI is offline-only. `hf_backend` uses exactly the same
plain serialized RHD envelope and prompt-token boundary as the SFT encoder; it does
not silently add a chat template. Native compatibility must be demonstrated first.

The reference trainer loss-masks complete action sequences into exact token quotas
without truncating their input context. A target segment that crosses an update
boundary is resumed in the next update. This matches supervised tokens and update
counts, **not forward FLOPs**. `match_controls` rejects mismatched parent/source
coverage, checkpoint/tokenizer identities, trainable scope, steps, target tokens,
learning rate, seed, context limit or retention policy.

Independent retention replay is not implemented in this reference trainer; its
receipt is explicitly `none-mechanism-screen`. Retention evaluation is implemented.
Do not relabel this as a retention-trained or retention-positive checkpoint.

## Deliberate limits

The implemented environment covers bounded arithmetic, record retrieval,
retrieve-then-check composition and evidence-missing abstention. It is not a shell,
filesystem sandbox, general JSON-transformation engine, or code-repair benchmark.
Runbooks use a finite procedural vocabulary, not arbitrary free-form instructions.
The evidence predicate is conservative for these tasks, not a general semantic
leakage proof or calibrated-abstention validator. Ancestry and exact normalized
prompt collisions are checked; semantic near-duplicate/structural holdout audits
remain the data owner's responsibility.

In-process callbacks are trusted research code, not an OS security boundary. The
ledger checks wall time between calls; it cannot preempt a hung Python callback.
Live adapters need separate enforced timeout/cancellation, process isolation,
provider-cost limits, durable per-call logging and native resume. The native API
has no checkpoint save/reload manager; the host must save to a new directory and
verify reload fidelity. A failed training run may already have changed weights:
discard that in-memory model and reload the pinned parent before retrying.

No sealed confirmation task can execute through this adaptive CLI/API. A separately
frozen, independently administered confirmation evaluator is a future gate. Neither
training loss nor fixture success authorizes promotion. See `PROTOCOL.md`,
`protocol.json`, `SOURCES.md` and `VALIDATION.json` for evidence and next steps.
