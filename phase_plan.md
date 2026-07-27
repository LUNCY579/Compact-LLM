Based on the files available, the project is currently at **Phase 0 transitioning into Phase 1: Dense SFT Baseline**.

You have built most of the infrastructure needed to validate data, run the untouched model, establish a benchmark, train an all-layer LoRA/QLoRA adapter, and later scan individual transformer layers. I do not have execution logs proving that these stages have passed, so the project should not yet be treated as being in the layer-selection phase.

The blueprint requires this order:

```text
Prove the data and baseline
→ prove the short-response behavior
→ study training-layer efficiency
→ align preferences
→ test model soups
→ optimize context
→ experiment with inference sparsity
```

It explicitly says behavior must be proven before training efficiency and sparsity are pursued. 

# Complete workflow

## Phase 0 — Environment and data contract

```text
Install dependencies
→ confirm CUDA works
→ prepare train/dev/sealed data
→ validate schema
→ check leakage
→ perform human audit
```

At this stage, you are not changing the model yet. You are proving that the experimental foundation is trustworthy.

### `install_all.ps1`

Creates or reuses the Windows Python 3.11 virtual environment, installs the CUDA-enabled PyTorch build first, installs the remaining requirements, checks dependency consistency, and verifies CUDA detection. 

Its job is essentially:

```text
Set up a reproducible GPU-capable development environment.
```

### `requirements.txt`

Defines the broad project environment:

* PyTorch;
* Transformers;
* PEFT and TRL;
* datasets and validation;
* evaluation metrics;
* LoRA/QLoRA;
* experiment tracking;
* retrieval;
* quantization;
* profiling;
* testing.

It intentionally covers later stages as well as the current stage, so installing it does not mean all those systems have already been implemented. 

### `requirements-wsl-optional.txt`

Contains optional Linux/WSL-only acceleration tools such as Triton, vLLM, DeepSpeed, and xFormers. These are for later optimized-runtime work and should not destabilize the main native-Windows environment now. 

### `verify_environment.py`

Imports the essential libraries and reports:

* Python version;
* package versions;
* CUDA availability;
* PyTorch CUDA runtime;
* detected GPU;
* available and total VRAM.

It fails clearly when an essential dependency or CUDA setup is broken. 

---

## Phase 0A — Dataset preparation

You generate or curate records and divide them into:

```text
data/train/train.jsonl
data/dev/dev.jsonl
data/sealed/sealed.jsonl
```

The purposes are different:

* **train:** model learns from these;
* **dev:** repeatedly used to compare configurations;
* **sealed:** reserved for final unbiased evaluation.

The blueprint requires separation between training, development, calibration, and sealed evaluation, with no exact or near duplication across them. 

### `validate_dataset.py`

Checks that every record:

* follows the JSON schema;
* has a unique ID;
* identifies the correct split;
* begins with a system message;
* ends with an assistant training target.

It also prints the distribution of `reply_class` values in each split. 

Its job is:

```text
Is the dataset structurally valid and internally consistent?
```

### `check_leakage.py`

Compares train, dev, and sealed splits for:

* shared template families;
* strongly similar normalized prompts;
* near-duplicate examples.

It removes common superficial prefixes before comparison so a renamed wrapper such as “Quick question” does not hide duplicated content. 

Its job is:

```text
Are the evaluation results likely to be genuine,
or did the model already see essentially the same scenario?
```

### `build_human_audit_packet.py`

Samples records by split and response category and creates:

```text
reports/HUMAN_DATA_AUDIT_PACKET.md
```

A human reviewer marks each sampled example as:

```text
ACCEPT
REPAIR
REJECT
```

The script uses a fixed seed, so the audit packet is reproducible. 

Its job is:

```text
Catch naturalness, personality, ambiguity, and quality problems
that automated validators cannot reliably detect.
```

This is important because a JSON record can be structurally perfect while containing bad dialogue.

---

# Phase 1 — Establish the untouched base model

Once the environment and pilot data are ready, run the original model before training.

## `smoke_inference.py`

Downloads and loads the base model—currently `Qwen/Qwen3-0.6B`—and runs one short-response inference request.

It records:

* prompt;
* generated reply;
* input tokens;
* output tokens;
* generation time;
* peak allocated VRAM;
* GPU device.

It disables Qwen’s thinking mode and uses deterministic generation. 

Its question is simply:

```text
Can this model load and generate correctly on this computer?
```

A smoke test is not a quality evaluation. It proves the basic path works.

## `evaluate_base_fixed.py`

Runs the model over the pilot benchmark and measures behavior more systematically.

It checks properties such as:

* nonempty output;
* complete utterance;
* expected token range;
* required phrases;
* forbidden phrases;
* whether clarification is asked;
* whether the model unnecessarily echoes the user.

It can evaluate either:

```text
base model only
```

or:

```text
base model + PEFT adapter
```

It also measures timing and VRAM. 

Its job is:

```text
Create the baseline score that every trained model must beat.
```

Without this baseline, you would know that training changed the model but not whether it improved it.

---

# Phase 1A — Train the dense behavioral reference

The word **dense** here means inference still executes the complete underlying transformer. Training uses LoRA or QLoRA adapters rather than full-parameter updating.

## `train_dense_sft.py`

This is the main supervised fine-tuning script.

It:

1. loads train and development JSONL records;
2. loads the base tokenizer and model;
3. optionally loads the base in 4-bit for QLoRA;
4. identifies attention and MLP projection modules;
5. attaches LoRA adapters;
6. masks prompt tokens so the loss focuses on assistant responses;
7. trains using Hugging Face `Trainer`;
8. saves the adapter and tokenizer;
9. writes a training manifest containing configuration, metrics, seed, target modules, dataset counts, and peak VRAM.

It can train:

```text
--selected-layers all
```

or a specific set such as:

```text
--selected-layers 12
--selected-layers 10,11
```

That allows the same controlled implementation to support both the initial all-layer reference and later selective-layer experiments.  

For the current stage, use:

```text
--selected-layers all
```

The all-layer LoRA result is the behavioral reference against which one-, two-, and four-layer models will later be compared.

The blueprint requires this dense SFT baseline before layer or sparsity research. 

---

# Stop Gate 1

After training, evaluate the adapter using the same benchmark and generation settings used for the base model.

Compare:

```text
Base model
vs.
All-layer SFT adapter
```

The trained model should demonstrate:

* shorter responses;
* complete utterances;
* natural texting style;
* personality adherence;
* response variety;
* no collapse into generic phrases;
* acceptable performance across your benchmark categories.

The blueprint says not to begin layer selection until the dense SFT model produces concise, complete, correct, and varied replies on the development suite. 

For your role-playing specialization, “correct” should primarily mean:

```text
correctly follows the supplied personality,
mood, relationship, and conversation context
```

rather than encyclopedic factual accuracy.

---

# Phase 2 — Layer-selection research

## `scan_layers.py`

This orchestrates controlled single-layer experiments.

It:

1. reads the model configuration to determine the number of transformer layers;
2. selects a coarse set spanning early, middle, and late layers;
3. runs `train_dense_sft.py` separately for each selected layer;
4. evaluates each resulting adapter;
5. collects metrics such as:

   * heuristic score;
   * median reply tokens;
   * P90 reply tokens;
   * complete-utterance rate;
   * evaluation VRAM;
   * training VRAM;
   * number of target modules.

This directly implements the blueprint’s first selective-layer study. 

The comparison should eventually contain:

```text
all-layer LoRA
best single layer
best adjacent pair
best selected four layers
random four layers
low-ranked control layers
```

The blueprint requires controls so reduced parameter count alone is not mistaken for intelligent layer selection. 

### Important filename check

Your current `scan_layers.py` calls:

```python
evaluate_base.py
```

But the file available here is named:

```text
evaluate_base_fixed.py
```

If your repository does not also contain `evaluate_base.py`, the layer scan will fail when it tries to start evaluation. Either:

```text
rename evaluate_base_fixed.py → evaluate_base.py
```

or update this line in `scan_layers.py`:

```python
eval_script = Path(__file__).with_name("evaluate_base_fixed.py")
```

This should be corrected before running the scan.  

Do not run the expensive scan yet unless Stop Gate 1 passes.

---

# Later blueprint stages

After the selective-layer study:

## Phase 3 — Preference alignment

Build chosen/rejected pairs and train with DPO:

```text
short + natural + in-character
beats
verbose, generic, repetitive, or personality-breaking
```

The blueprint recommends DPO before online RL because your desired behavior is naturally expressible through preference pairs. 

## Phase 4 — Model soups

Train several compatible selective adapters with different seeds, then test uniform and greedy averaging.

Only accept a soup when it beats the best individual model without critical regression. 

## Phase 5 — Context system

Add external memory, retrieval, context budgeting, and stable-prefix caching.

This is separate from the core fine-tuning work.

## Phase 6 — Sparse simulation

Freeze the accepted dense reference and experimentally mask MLP blocks at ratios such as:

```text
100%
80%
65%
50%
35%
```

At first, this measures quality tolerance and does not claim speed improvement.

## Phase 7 — Sparse runtime

Only after simulation passes:

* train block routers;
* add confidence-based dense fallback;
* prototype Triton or custom kernels;
* measure actual wall-clock speed and memory.

## Phase 8 — Release candidate

Run the full ablation matrix and preserve:

* model hashes;
* tokenizer version;
* dataset manifests;
* seeds;
* hardware;
* raw metrics;
* accepted adapter or soup recipe;
* quantization settings.

The blueprint permits shipping a dense quantized model when custom sparsity does not provide measurable value. 

# Where you are now

Based on the available files:

```text
Environment tooling              Prepared
Dataset validation tooling       Prepared
Leakage tooling                  Prepared
Human audit tooling              Prepared
Base smoke test                  Prepared
Base evaluation                  Prepared
All-layer SFT training           Prepared
Single-layer scan                Prepared
DPO                              Not shown yet
Model soups                      Not shown yet
Activation profiling             Not shown yet
Sparse router/runtime            Not shown yet
```

Therefore, the project is best described as:

> **Phase 0 implementation is mostly prepared; Phase 1 has not been proven until the commands have been executed and reviewed.**

# Immediate next step

Run the project in this order:

```powershell
.\install_all.ps1
.\.venv\Scripts\Activate.ps1
python verify_environment.py

python scripts\validate_dataset.py
python scripts\check_leakage.py
python scripts\build_human_audit_packet.py
```

Review and approve the audit packet. Then:

```powershell
python scripts\smoke_inference.py
python scripts\evaluate_base.py
```

After the baseline report exists, train a small all-layer SFT experiment:

```powershell
python scripts\train_dense_sft.py `
  --selected-layers all `
  --max-steps 100 `
  --gradient-checkpointing
```

Evaluate that adapter against exactly the same benchmark. The next formal decision is **Stop Gate 1**:

```text
Did the all-layer SFT adapter clearly learn short,
natural, varied responses without becoming incomplete or generic?
```

Only after that answer is supported by evidence should `scan_layers.py` become the next active experiment.
