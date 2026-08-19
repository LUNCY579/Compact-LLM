# Experimental Protocol

## 1. Primary hypothesis

A compact state extracted from the complete solution history may contain enough information to supervise the next correct reasoning decision without replaying the full prior reasoning sequence.

The practical comparison is:

```text
FULL:
original problem → complete uninterrupted reasoning + final answer

SPLIT:
original problem + compact state → next reasoning transition
```

The SPLIT training rows are globally shuffled and do not preserve trajectory adjacency.

## 2. Primary adapters

### Adapter A — FULL

- Source: `data/full_context/train.jsonl`
- Message field: `messages`
- Unit of supervision: one complete problem trajectory
- Native TRAIN rows: 43,200

### Adapter B — SPLIT

- Source: `data/split_transitions/train.jsonl`
- Primary message field: `messages`
- Unit of supervision: one compact-state transition
- Native TRAIN rows: 170,941

Train both adapters from the exact same unmodified base checkpoint. Do not continue Adapter B from Adapter A or vice versa.

## 3. Fixed controls

Hold these constant unless the change is itself a declared ablation:

- base model checkpoint and revision;
- tokenizer and chat template;
- LoRA target modules;
- LoRA rank, alpha, dropout, and bias setting;
- optimizer and scheduler;
- learning rate and warmup policy;
- precision and quantization mode;
- gradient accumulation;
- maximum input and completion lengths;
- gradient clipping;
- checkpoint interval;
- checkpoint-selection metric;
- decoding parameters;
- evaluation prompts and graders;
- random-seed list.

Run multiple seeds when compute permits. A minimum of three seeds is preferable for the primary result.

## 4. Budget interpretations

A single comparison cannot simultaneously match all of rows, source problems, target tokens, input tokens, optimizer steps, and epochs because the two representations differ structurally. Report more than one budget regime.

### Regime N — Native recipe

Train one native epoch over each dataset.

This answers:

> What happens when each representation is used in its natural complete form?

It does not isolate representation alone because SPLIT receives more rows and repeated original prompts.

### Regime U — Matched optimizer updates

Use the same number of optimizer updates for both adapters. Shuffle independently and stop at the common update budget.

This controls optimization duration but not necessarily tokens or source-problem coverage.

### Regime T — Matched tokenizer tokens

Count actual training tokens using the base model tokenizer and stop both runs at the same cumulative token budget. Report input, target, and padded tokens separately.

This is the strongest primary budget control when practical.

### Regime P — Matched source-problem exposure

Use `scripts/build_epoch_view.py` to choose one SPLIT transition per problem per epoch. Each SPLIT epoch view then contains 43,200 TRAIN records, matching FULL by problem count.

This isolates one-step supervision at matched source-problem frequency, but it samples only part of each trajectory per epoch.

## 5. Evaluation layers

### 5.1 Final-answer evaluation on the shared held-out problems

Use the 2,400 TEST problem IDs for both adapters.

For FULL:

```text
problem → complete response → extracted final answer
```

For SPLIT stateful rollout:

```text
problem + initial compact state
→ next step + updated state
→ next step + updated state
→ ...
→ done + final answer
```

The optional `stateful_rollout_messages` view trains parseable JSON state updates for this purpose. Limit rollouts to the source problem's known maximum plus a small safety margin during controlled testing; for unseen external problems use a fixed global limit.

Report:

- exact final-answer accuracy;
- completion rate;
- invalid-format rate;
- mean rollout steps;
- overrun/early-stop rate;
- total generated tokens;
- correction or contradiction rate.

### 5.2 Oracle-state next-step evaluation

Evaluate SPLIT on TEST transitions with the gold compact state provided. Report:

- exact or normalized next-step match;
- answer-equivalence score;
- updated-state validity;
- `done` classification accuracy;
- category-level accuracy;
- performance by step index.

This isolates learned transition quality from rollout error propagation.

### 5.3 Free-rollout gap

Define:

```text
free-rollout gap = oracle-state transition success − free-rollout final success
```

A large gap indicates that local transitions were learned but errors accumulate when the model consumes its own generated states.

### 5.4 Generalization evaluation

Create a separately generated or human-authored test set whose problem templates and numerical ranges were not used for training. Do not tune on it. Suggested strata:

- same rules with unseen values;
- longer trajectories than TRAIN;
- new wording for known operations;
- mixed-category compositions;
- state corruption and recovery;
- irrelevant-history insertion;
- removal of progressively more history.

## 6. Primary metrics

The most informative primary metrics are:

1. final-answer exact accuracy on shared held-out problems;
2. category-macro accuracy;
3. oracle-state next-step accuracy;
4. free-rollout final accuracy;
5. free-rollout gap;
6. accuracy as a function of trajectory length;
7. accuracy as a function of step index;
8. robustness to omitted obsolete history;
9. robustness to corrupted compact states;
10. matched-token learning curves.

Secondary metrics:

- target NLL;
- response length;
- invalid output rate;
- wall-clock time;
- peak VRAM;
- examples/second and tokens/second.

These efficiency metrics are useful diagnostics but are not the research hypothesis.

## 7. Required ablations after the two-adapter pilot

The first experiment can remain strictly FULL versus SPLIT. If the result is promising, the highest-value follow-up ablations are:

### State sufficiency curve

Retain progressively more historical information:

```text
minimal state
25% recent history
50% recent history
75% recent history
complete history
```

### Order ablation

Use the exact same transition records under two loaders:

- globally shuffled;
- trajectory-ordered.

This directly isolates presentation order from state decomposition.

### State corruption

Inject one incorrect derived fact and test whether the model:

- follows it blindly;
- detects the inconsistency;
- repairs the state;
- abstains.

### Longer-chain extrapolation

Train on trajectories of 2–6 transitions and test on 7–12 transitions.

## 8. Leakage controls

- Never split after transition expansion.
- Never use TEST results for learning-rate, checkpoint, prompt, or decoding selection.
- Keep DEV and TEST problem IDs fixed across all adapters.
- Do not train on `samples/paired_examples.jsonl`; it contains examples from multiple splits for inspection.
- If creating an external test set, deduplicate by normalized prompt and generator parameters.

## 9. Reporting template

For each run, record:

```text
base checkpoint:
adapter condition:
dataset view:
seed:
LoRA configuration:
optimizer / scheduler:
learning rate:
maximum sequence length:
training rows seen:
unique problem IDs seen:
input tokens:
target tokens:
padded tokens:
optimizer updates:
wall-clock time:
peak VRAM:
selected checkpoint:
DEV metrics:
TEST metrics:
```

The paper-level comparison should include both absolute scores and the difference from the untouched base model.
