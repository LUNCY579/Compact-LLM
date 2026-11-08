# Rineya MaxRL Composable Skills v2

**MAXRL-COMP-002 | 41,500 generated primary examples | DATA MECHANICS PASS, NATIVE UNQUALIFIED**

Repository: `LUNCY579/ChatBubble`.
New branch: `research/maxrl-composable-skills-data-v2`.
Code commit: `847cd27e9427c8c5e3acaf81191b2136a628b130`.
Registration: `c96c660c0a3be4599e76afb7a00dd5ac57e818a0`.
Unchanged v1 parent: `4b36c5e7dfbe4233be6ec1d7cf5614f7fc26189a`.

This corpus targets three different questions: acquiring individual skills with
MaxRL, combining separately learned skills, and measuring/mitigating forgetting.
It cannot guarantee no forgetting or make learned deltas orthogonal by construction.
No pretrained model, tokenizer, training run, or native evaluation was executed.

## What changes from v1

Four skills use one typed planning interface: **R** resolves references; **U** replays
state updates and filters eligible items; **S** selects an exact-size subset with a
target sum; **P** orders those items under precedence constraints. Combined tasks
have a shared answer: lookup determines candidates, updates affect eligibility
and values, selection determines the subset, and precedence determines its order.
They are not separate questions concatenated into one prompt.

All 15 nonempty skill signatures are evaluated. TRAIN/CALIBRATION include atomic
R/U/S/P and only three pair bridges RU/US/SP. **RS/RP/UP, all four triples, and RUSP
are absent from TRAIN, CALIBRATION, and WARMSTART.** Warmstart is atomic only.
For a clean independent-delta experiment, train each donor ONLY on its atomic
manifest; bridge and answerability training are explicitly separate interventions.

Every active skill of every core row has a local counterfactual test: change that
stage alone, reject a transferred formerly valid certificate, and independently
recover a different valid certificate. There are 56,100 such auxiliary checks.
They show local stage dependence, not absence of alternative shortcuts.

## Delivered counts

| Data role | Primary rows | Intended use |
|---|---:|---|
| train: atomic skills | 12,000 | Four independent 3,000-row donor banks |
| train: pair bridges | 9,000 | Three separate 3,000-row adaptation banks |
| train: answerability | 1,000 | 500 feasible/impossible pairs; optional separate bank |
| calibration | 1,250 | Exposed feasibility/mixed-success diagnostics, not training |
| development | 4,150 | Coefficient/recipe development, all signatures |
| test | 4,150 | Exposed held-out instances, all signatures |
| challenge | 4,150 | Longer/harder worlds and reserved intros, all signatures |
| warmstart | 600 | Disjoint, optional, atomic supervised initialization |
| replay_train | 1,600 | Fresh legacy-domain rehearsal, separate from guards |
| guard_dev | 1,200 | Exposed preservation development bank |
| guard_test | 1,600 | Exposed preservation test bank |
| guard_challenge | 800 | Harder exposed preservation bank |
| **Total** | **41,500** | **10,450 parent clusters, not 41,500 independent worlds** |

There are 1,200 answerability twin pairs across all roles. Twin specifications differ
only in target by +1, inside the reachable numeric range; one is feasible and the
other is proved impossible by independent finite search. The protocol's abstention
is `{"answer":"UNKNOWN"}`, not the bare historical literal UNKNOWN endpoint.

All provided evaluations are exposed, NOT sealed. The dataset is a synthetic
short-answer method adaptation, not a natural-language, long-context, or clinical
preservation benchmark. See `DATA_CARD.md` for construction biases and limitations.

## Run from the extracted ZIP root

The ZIP contains the full, already-generated corpus under `data/`, this package,
and the unchanged v1 source dependency. No generation step is required for use.

```bash
python -m pytest -q experiments/maxrl_verifiable_data_v1/tests experiments/maxrl_composable_skills_v2/tests
python -m experiments.maxrl_composable_skills_v2.corpus audit --data ./data --receipt ./fresh_audit.json
```

To regenerate the full corpus into a NEW directory, without needing the v1 data ZIP:

```bash
python -m experiments.maxrl_composable_skills_v2.corpus build --out ./regenerated_data --v1-zip ./data/diagnostics/v1_exclusion_hashes.json
```

The `--v1-zip` argument accepts the original v1 ZIP OR the included exact exclusion
inventory. The actual delivered build used the original verified v1 ZIP. Default
seed `260922002`; `--scale pilot` is for debugging and is not an independent study.
The full build/audit may take several minutes. Existing output paths are refused.

Generation and verification are standard-library only. Tests used Python 3.13.5
and pytest 9.0.2. Exact replay across other Python versions is not established.
Git contains source, documentation and evidence; the approximately 465 MB of
rendered data is delivered in the compressed conversation attachment, not Git blobs.

## Data and trainer boundaries

`canonical/` binds every row to its public prompt, verifier specification, skills,
parent, role and hashes. `policy/` contains ID plus system/user messages; only
messages enter the policy. `verl_jsonl/` contains the usual prompt/reward-model
columns; ground_truth is a verifier specification, not a preferred answer trace.
The callback is `verifier.compute_score(data_source, solution_str, ground_truth,
extra_info=None)`. Package importability and native trainer integration remain to
be qualified on the actual training host.

`oracles/` contains independent reference search results and checked negatives.
They are audit data, NOT on-policy samples. `diagnostics/` contains same-parent
stage counterfactuals, matched answerability pairs, world lineage and v1 exclusions.
Do not feed diagnostic world records or oracle fields to the actor. `manifests/`
provides explicit row-ID banks and proposed curriculum usage contracts.

`sft/warmstart.messages.jsonl` has only `messages`, ordered system/user/assistant,
with final-answer-only targets; lineage is a separate sidecar. Warmstart both
objective arms identically or neither. SFT alone is not MaxRL. Online MaxRL needs
actual independently sampled, fixed-size prompt groups from each current policy.

## Preservation enrollment

Replay rows and evaluation guards are disjoint. A guard becomes a source-correct
preservation instrument only AFTER actual baseline responses have been recorded.
The included evaluator rejects missing/duplicate rows, mismatched prompt hashes,
bank changes, and unmatched tokenizer/decoding metadata. It reports gains AND lost
source-correct rows, keeps a zero denominator undefined, and resamples parents
rather than pretending sibling views are independent. It does not auto-promote.
See `STUDY_GUIDE.md` for baseline enrollment and before/after comparison commands.

## Optional conversion, not claimed as completed

```bash
python -m experiments.maxrl_composable_skills_v2.export parquet --data ./data --out ./parquet_data
python -m experiments.maxrl_composable_skills_v2.export tokenizer --data ./data --tokenizer /absolute/local/tokenizer --out ./tokenizer_check.json --context 4096 --response-budget 128
```

The actual Parquet attempt exited 2 because PyArrow is absent; installation was
network-blocked. **No Parquet files were produced.** The converter has a roundtrip
check but its successful native-engine path remains unexecuted. All JSONL is present.
The tokenizer command is local-only and never truncates; even a successful check
would establish prompt lengths only, not response masking, EOS or training readiness.

## Executed evidence

Combined source suite: **462 tests passed** (177 unchanged v1 + 285 v2).
Persisted full-corpus audit: **41,500 re-solved rows; 116,528 accepted reference
certificates; 581,000 rejected negatives; 56,100 live stage controls; 1,200 matched
answerability pairs.** No exact cross-split or provided-v1 collisions were found.
All ten published v2 Python source blobs match the locally tested bytes.
No remote CI execution or native model success is claimed.

Full data manifest SHA-256:
`86ac8f0d851f2178752357f62249eab5a23fd3de8fa032e7373614891df4cd44`.

## Scientific premise and sources

Today's inspected Rineya composition synthesis supports investigating independent
per-delta scales; it does not establish orthogonality as the causal explanation
or certify retention. The clean recipe is `theta0 + sum(alpha_i * delta_i)` with
separate controls, positive acquisition checks and explicit norm/cross-term
accounting. Orthogonal projection, rehearsal and bridge training are separate
experimental factors. No previous registered experiment is altered here.

- MaxRL paper: https://arxiv.org/html/2602.02710v3
- Rineya synthesis: `LUNCY579/rineya-dataset-retention-records@12174cb371d457a19ad2584f44aa624e47064163`, `COMPOSITION_SYNTHESIS_20260921.md`.
- GEN2 registration: same repository, `2e8af4d84d02cea43c2003f02701ae060d5b562a`.
- Official data/reward interfaces: https://verl.readthedocs.io/en/latest/preparation/prepare_data.html and https://verl.readthedocs.io/en/latest/preparation/reward_function.html .

Native source competence, token budgets, full historical overlap, human review,
statistical margins and actual acquisition/retention/composition effects remain open.
