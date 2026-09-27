# Exact-host handoff: one rollback-only native qualification

## What is ready, and what is not

Run on the user's existing Windows GPU host with the local immutable **Supa**
export pinned in the Titan soup parent, the existing delivered MAXRL-COMP-002
corpus, and this branch checked out. No downloading or new environment install is
automatic. Missing dependencies/paths/pins/lease/memory cause refusal.

This command tests direct-coordinate loading, one discovery step, exact reset,
one masked step and rollback. It uses next-token loss on one existing atomic
TRAIN prompt for autograd plumbing. It does NOT learn a task from verified
responses, evaluate Rineya accuracy, establish efficacy or authorize full training.
A full native objective/collector integration still needs implementation and tests.

## 1. Checkout and verify

PowerShell, from the existing ChatBubble repository:

```powershell
git fetch origin
git switch research/titan-clare-sparse-composable-skills-v1
git status --short
git rev-parse HEAD
git merge-base --is-ancestor 816a1e8f4d1ab1c9d1ab847697d962a141521bac HEAD
python -m pytest -q experiments/clare_sparse_composition_v1/tests
```

Git can create a local tracking branch from the uniquely named remote branch.
Do not switch or overwrite an unrelated dirty worktree. Use a separate worktree
when needed. The native runner demands this exact branch, bound HEAD and no
tracked modifications. Parent/main must remain untouched.

## 2. Bind facts, never invent them

Copy `configs/host.example.json` to a NEW untracked `host.local.json` and supply:
`model_dir` (existing immutable pinned export), `data_dir` (existing full corpus),
`eligible_names` (reviewed real transformer parameter names, identical across
controls and <=8,000,000 total coordinates), `smoke_row_id` (actual atomic R TRAIN
ID), `git_commit` (current HEAD), and `runtime` (exact output below).

```powershell
python -c "import json; from experiments.clare_sparse_composition_v1.native import runtime; print(json.dumps(runtime(), indent=2))"
```

The initial scientific eligible pool is intentionally not guessed from a model
name. This smoke's small pool is not a license to claim all-model support
allocation. Bind trainable inventory and any study-specific pool restriction
before native efficacy experiments. No test answers may inform that choice.

## 3. Check, then execute only the tiny smoke

Use NEW filenames for every attempt:

```powershell
python -m experiments.clare_sparse_composition_v1.cli preflight --binding host.local.json --out clare-preflight-001.json
python -m experiments.clare_sparse_composition_v1.cli native-smoke --binding host.local.json --config experiments/clare_sparse_composition_v1/configs/learned_disjoint.json --authorize-smoke --out clare-native-smoke-001.json
```

The second command includes explicit authorization for this tiny smoke only.
It takes the inherited exclusive lease with zero waiting, checks two consecutive
>=7.4 GiB free-host-RAM readings, checks conservative VRAM needs, and guards each
step at >=6.0 GiB free RAM. Thresholds are not CLI-tunable. Local-only loaders
refuse remote code. Token length must fit 2..512 without truncation. Existing
checkpoint/tokenizer hashes and architecture/inventory are verified before load.
The command never writes new model weights and restores the changed pool.

Passing requires `SMOKE_ONLY_PASS`, `reset_exact=true`, `rollback_exact=true`,
matching identities and exactly two plumbing optimizer steps. Review resource
peaks and the mask before registering a native study. Errors after entering native
code may report counters as null because the wrapper cannot reconstruct partial
progress; do not interpret null as zero. A refused preflight reports known zeros.

## Single next experiment

Complete the bound one-row rollback smoke above. Only afterward implement and
qualify the native skill-loss adapter (including response/EOS/log-prob masks and
verifier separation), enroll existing evaluation/guard banks, and freeze actual
token budgets. Then the first efficacy comparison is A/D/E at rho=.80 and one
registered order, repeated across the three predeclared seeds—not a rho sweep.
