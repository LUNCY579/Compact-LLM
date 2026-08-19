# Rineya Long-Context Demonstration Suite v1.0

A controlled, original long-context question-answering suite for demonstrating whether a Rineya-family model can read a dense context and return a short exact answer.

## Why this design

A familiar *Harry Potter* or television-*Sherlock* summary can look impressive, but it is not a clean test of context use: a pretrained model may already know the story. This suite instead uses invented fantasy, detective, science-fiction, expedition, museum, research, civic, and family-enterprise settings. The decisive facts exist only inside each supplied context.

The suite also avoids cherry-pickable one-off questions. Every scenario is a four-row pack:

1. **Base** — the ordinary answerable question.
2. **Paraphrase** — a differently worded question with the same answer.
3. **Counterfactual** — one decisive fact changes, so the correct answer must change.
4. **Unanswerable** — the decisive fact is missing or unresolved, so the answer must be `UNKNOWN`.

A model that succeeds on all four variants is much less likely to be guessing, relying on a familiar narrative, or repeating the same answer regardless of the context.

## Contents

- **64 original scenario packs**
- **256 unique prompts**
- **192 unique contexts** because each base context is intentionally reused once for its paraphrased question
- Context length: **853–1183 words**, mean **1012.43**
- **16 showcase packs / 64 rows** for public demonstrations
- **48 stability-test packs / 192 rows** for internal evaluation
- **8 domains**
- **8 task families**
- Early, middle, late, and distributed evidence placement
- Medium and hard cases
- 64 explicitly unanswerable cases

## Directory layout

```text
data/
  rineya_long_context_demo_full.jsonl
  all_prompts.jsonl
  all_answer_key.jsonl
  showcase_prompts.jsonl
  showcase_answer_key.jsonl
  showcase_base_full.jsonl
  stability_prompts.jsonl
  stability_answer_key.jsonl

scripts/
  run_local_hf.py
  inspect_token_lengths.py
  evaluate_predictions.py
  make_html_report.py
  validate_suite.py

examples/
  prediction_format.jsonl

DATA_CARD.md
PROMPT_TEMPLATE.txt
dataset_stats.json
schema.json
requirements.txt
MANIFEST.sha256
```

## Recommended workflow

### 1. Create an environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Check exact token lengths before inference

Word count is not token count. Use the tokenizer belonging to the exact Rineya checkpoint:

```powershell
python scripts/inspect_token_lengths.py `
  --tokenizer "C:\path\to\Rineya" `
  --input data/stability_prompts.jsonl `
  --max-context 4096 `
  --json-output outputs/token_lengths.json
```

Do not silently truncate these prompts. A truncated context changes the benchmark.

### 3. Run deterministic inference

```powershell
python scripts/run_local_hf.py `
  --model "C:\path\to\Rineya" `
  --input data/stability_prompts.jsonl `
  --output outputs/rineya_run1.jsonl `
  --max-new-tokens 24 `
  --max-context 4096
```

The runner uses greedy decoding and does not remove reasoning text. If a model emits a long explanation, that remains part of the prediction and can reduce strict exact-match performance.

### 4. Evaluate

```powershell
python scripts/evaluate_predictions.py `
  --key data/stability_answer_key.jsonl `
  --predictions outputs/rineya_run1.jsonl `
  --json-report outputs/rineya_report.json `
  --markdown-report outputs/rineya_report.md
```

For repeated-run stability, pass multiple prediction files:

```powershell
python scripts/evaluate_predictions.py `
  --key data/stability_answer_key.jsonl `
  --predictions outputs/run1.jsonl outputs/run2.jsonl outputs/run3.jsonl `
  --json-report outputs/stability_report.json
```

### 5. Build a public HTML showcase

Run the model on `data/showcase_prompts.jsonl`, then:

```powershell
python scripts/make_html_report.py `
  --full data/rineya_long_context_demo_full.jsonl `
  --predictions outputs/showcase_predictions.jsonl `
  --output outputs/rineya_showcase.html `
  --model-name "Rineya"
```

The default HTML report displays the 16 base showcase cases. Full contexts are collapsible.

## Metrics that matter

Do not advertise only a handful of correct examples. Report at least:

- **Strict exact accuracy**
- **Base/paraphrase both-correct rate**
- **Counterfactual pair correct-and-changed rate**
- **Unanswerable accuracy**
- **All-four-variants-correct rate**
- Accuracy by context-length bucket
- Accuracy by evidence position
- Short-response rate
- Repeated-run agreement when nondeterministic decoding is tested

The strongest single number in this suite is **all-four-variants-correct**. It requires the model to preserve the answer under paraphrasing, change it when the context changes, and abstain when the answer is absent.

## Public-demo guidance

Use the 16 base showcase cases for readable screenshots or a live presentation, but keep the complete stability split out of promotional materials until evaluation is finished. State the decoding settings, model checkpoint, prompt count, context-length range, and whether the result is strict exact match.

For a live demo, randomly select the scenario before inference instead of preselecting a successful item. Showing the random seed or drawing the item on screen makes the presentation more credible.

## Training-use warning

This version is an evaluation/demo asset. **Do not fine-tune on these exact rows and then report performance on them.** If the idea becomes a training dataset, create a disjoint training release with new characters, event graphs, phrasings, question templates, and counterfactuals. Keep this v1 suite frozen as an untouched test.

## Scope and limitations

The suite tests dense reading, multi-step fact resolution, state tracking, corrections, rule exceptions, counting, causal disambiguation, and calibrated abstention. It does not establish broad real-world long-context competence, literary interpretation, retrieval over hundreds of thousands of tokens, or performance on natural documents. The prose is synthetic and structurally controlled so that correctness is auditable.

## Text provenance

Every scenario was written specifically for this package. No text was copied from *Harry Potter*, modern *Sherlock* adaptations, or another franchise. The final publisher should choose and attach the desired dataset license before public release.
