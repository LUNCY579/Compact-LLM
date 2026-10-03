---
license: cc0-1.0
language:
- en
task_categories:
- text-generation
tags:
- synthetic
- reinforcement-learning
- verifiable-rewards
- planning
- long-context
- chain-of-thought
pretty_name: Rineya Reasoning Forge RL v2
---
# Rineya Reasoning Forge RL v2.0.0

**20,000 original, verifier-backed canonical task worlds, with separately generated RL, reasoning-SFT, and concise-answer SFT views.** Created September 14, 2026 for Cheng Li's Rineya research.

This is a **machine-validated post-training candidate**, not a trained model or a demonstrated improvement. It is a companion to the uploaded Rineya Reasoning Forge v1, not a replacement or a relabeling of its 10,000 rows. No external dataset rows, model-generated rollouts, weights, or private benchmark examples were imported. No repository was modified.

## Start here

| File or directory | Contents |
|---|---|
| `data/rl/*.jsonl.gz` | Final-only RL prompts; no gold answers or traces in the prompt |
| `data/rl_reason/*.jsonl.gz` | The same worlds with an explicit reasoning-then-final output contract |
| `data/sft_final/{train,validation}.jsonl.gz` | Concise answer demonstrations |
| `data/sft_reason/{train,validation}.jsonl.gz` | Solver-generated structured solution traces, then concise answers |
| `metadata/*.jsonl.gz` | Verifier-only worlds, references, objective certificates, traces, provenance, and context spans |
| `process/train_transitions.jsonl.gz` | **36,752** verified valid/invalid local state-transition examples |
| `preferences/train_pairs.jsonl.gz` | **32,000** ID-linked synthetic preference pairs |
| `episodes/*.jsonl.gz` | **10,000** causal, teacher-forced dispatch decision views, four per continuous-replanning parent |
| `probes/` | **384** diagnostic views from **64** existing IID-test parent worlds |
| `pilot/train_256.jsonl`, `pilot/validation_128.jsonl` | Balanced short-input subsets; native token lengths remain unqualified |
| `DATA_CARD.md` | Scope, exact counts, limitations, and provenance |
| `docs/TRAINING_PLAYBOOK.md` | Native-model gates and matched SFT/RL/conciseness experiments |
| `docs/FORMAT_AND_REWARDS.md` | Output contracts, reward behavior, and safe framework boundaries |
| `docs/CONTINUOUS_AND_LONG_CONTEXT.md` | What sustained state, planning, and context length mean here |
| `docs/REVIEW_CHECKLIST.md` | Stratified review protocol before optimization |
| `docs/SOURCES.md` | Primary research/documentation consulted; no source text used as corpus rows |
| `PREVIEW.md` | Abbreviated real TRAIN examples |
| `schemas/` | JSON Schemas for public prompt, SFT, and prediction records |
| `configs/` | Review, reward-policy, and controlled-experiment contracts |
| `reports/` | Executed checks and their limits |

The view counts are **not additive independent-example counts**. A reasoning target, final-only target, preference pair, process row, pilot row, and layout probe inherit the same parent group. Never resplit derived views independently.

## Composition

| Split | Canonical worlds | Per family | Purpose |
|---|---:|---:|---|
| TRAIN | 16,000 | 2,000 | Optimization candidates |
| Validation | 1,600 | 200 | Development and checkpoint selection |
| IID test | 1,600 | 200 | Same-family developer-visible holdout |
| Transfer test | 640 | 80 | Larger planning spaces, longer relation chains, or longer replay horizons |
| Length stress | 160 | 20 | Evaluation-only very-long inputs |
| **Total** | **20,000** | **2,500** | Unique canonical formal cases |

Eight families: ledger replay, register-machine execution, versioned routes, prerequisite action planning, deadline scheduling, constrained budget selection, evidence-chain resolution, and continuous replanning.

TRAIN inputs have **512–4,112 whitespace-delimited words**. Length-stress inputs have **8,192–16,400 words**. These are **not tokenizer-certified token windows**. Neither native Rineya tokenization, model execution, optimizer updates, nor GPU-fit measurement was performed.

## Executed release checks

The final source passed **222 unit/integration tests**. Separate audit solvers checked all **20,000** canonical references and complete structured traces; the release validator replayed **198,770** stateful/action transitions. JSON Schema checked **75,200** public RL/SFT view rows. All process labels, preference directions, causal snapshots, and paired probes were checked. The original/new exact normalized user-prompt overlap was **zero**; semantic overlap against private benchmarks was not tested.

The concise public references are at most **80 characters** across the full corpus. Some reasoning references are much longer (the maximum stress response is **79,502 characters**), so concise public output must not be confused with low total inference cost. Native token lengths are still unmeasured. See `reports/QUALIFICATION_SUMMARY.json` and `reports/VERIFICATION.md`.

## Reproduce and verify

Core generation, rewards, environments, and evaluation need **Python 3.10+ and the standard library only**. The executed release runtime is Python 3.13.5. No API keys, network, GPU, or model download are required.

From the extracted bundle directory:

```bash
python -m unittest discover -s tests -v
python scripts/verify_release.py --root . --fast
python scripts/verify.py --root .
python scripts/audit_release.py --root .
```

Generate a fresh full corpus into a **new** directory, never over an existing release:

```bash
python scripts/forge.py --output rebuilt --seed 20260914
python scripts/verify.py --root rebuilt
python -m forge_rl.probes --root rebuilt
```

The core build produces all canonical data, SFT views, metadata, preferences, process labels, and episode views. The separate probe command adds the matched test diagnostics. To make a small smoke corpus, add `--scale 0.002`; its counts are intentionally different from the full release. Same-seed byte reproducibility was tested on a complete 64-world smoke build, not claimed from an unexecuted second full release build. Deterministic data bytes are intended within the same supported runtime; the report embeds Python version.

## Evaluate your model outputs

Save one JSON object per line:

```json
{"id":"COPY_AN_ACTUAL_ROW_ID","prediction":"{\"value\":17}"}
```

Use IDs from the matching prompt file. The example value above is illustrative, not a gold answer.

```bash
python scripts/score_predictions.py --root . --split validation \
  --predictions my_validation_predictions.jsonl --output my_validation_scores.json
```

For reasoning-mode outputs, use `--mode reason_then_final` and the matching `data/rl_reason` prompts. For sampled RL rollouts, include integer `sample_id` values from zero and pass `--samples-per-prompt 4`. Optional `finish_reason`, `completion_tokens`, and `reasoning_tokens` are recorded inputs, not inferred token counts.

The evaluator reports coverage, missing outputs, strict format, task correctness, plan feasibility, per-family/length/layout/status results, pass@k for complete groups, and nonzero group-reward variance. Duplicate or unknown prediction IDs fail. Missing predictions count as wrong in the full-set accuracy denominator. Explicitly truncated completions do not count as completed correct answers. Gold-reference self-scoring tests are software checks, never model results.

## Use the RL reward callback

```python
from forge_rl.adapters import VerifiableReward

reward_fn = VerifiableReward(root=".", split="train", mode="final_only")
# In a separately qualified TRL environment, pass reward_fn as reward_funcs.
# Pass only dataset['prompt'] to the model. Preserve dataset['id'] for the callback.
```

The callback checks exact prompt/reference hash alignment. Its signature follows the documented custom-reward boundary, but **no native TRL trainer integration was executed**. This is not a complete PPO/GRPO training script or an implementation of any named paper's full method.

## Internal reasoning versus public answers

The reasoning view contains **original, solver-generated task solution traces**. They are not transcripts of a model's private reasoning. Each trace exposes the task decomposition and structured checkpoints needed to replay a witness; it does not list every rejected search branch.

Textual `<think>` and `<final>` delimiters do **not** make text private. A qualified inference runtime must separate the reasoning field from the public response. Final-only training is also included, but removing reasoning tokens at inference may change accuracy. Test that comparison rather than assuming latent reasoning has been preserved. Public response length and total generated-token cost are different measurements.

## Rights and scope

Generated data is offered with **CC0-1.0 intent** and original code with **MIT terms**, matching the uploaded v1 bundle's stated approach. This is not a legal warranty or a claim that AI-generated text necessarily has copyright protection. See `LICENSE_DATA.md` and `LICENSE_CODE`.

Do not use this controlled synthetic corpus as your sole broad-language recovery corpus, do not train on the test files, and do not infer CED/Supa architectural superiority or a general reasoning improvement from these software checks.

## Additional validation and packaging

`python scripts/validate_schemas.py --root .` optionally checks every public RL/SFT row using `jsonschema`; it is not a core dependency. `scripts/audit_release.py` accepts `--original PATH_TO_V1.zip` to repeat the source-overlap check.

For a newly generated and reviewed release tree, `scripts/freeze_manifest.py --root NEW_ROOT` creates its immutable file manifest; `scripts/package_release.py --root NEW_ROOT --output NEW.zip` packages exactly those files. Existing manifests and archives are never overwritten. Archive CRCs and SHA-256 hashes check file integrity, not model quality.
