# Supa Long-Context Minimal Intervention Pilot

Lab id: `supa_long_context_minimal_intervention_v1`
Protocol version: `1.0.0-pilot`
Target checkpoint: **Rineya-Supa-A1.1-1.1.0-devfreeze** (read-only)

**2026-09-05 audit correction:** read `RESULT.md` and
`preregistrations/audit_20260905_phase1_repairs.md` before using the historical
motivation below. Phase 0 does not establish a causal interference verdict or a
perfect-retrieval ceiling. The bounded curriculum requires a 3840-token limit;
the code's preservation classes are counting_sets/conditional_rules. The merged
Stage A adapter used CC0 long-context data plus short replay, not this LCR SFT
corpus. Phase 1 uses isolated arm directories and complete compatible results.

## Research question

Supa-A1.1 scores 170/256 on the Rineya Long-Context Demonstration Suite. This
pilot asks, in the cheapest order possible, *why the other 86 rows fail*:

1. **Phase 0 — is it interference?** Does removing irrelevant context expose
   latent capability the full-context score hides? Inference only, no training.
2. **Phase 1 — is it capability?** Can a tiny adaptation of the *late* attention
   pathway improve long-context reasoning while the catastrophic early stem stays
   frozen?
3. **Phase 2 — conditional.** Only if Phase 1 clearly wins: does widening the
   adapter to two more soft-significant layers add anything?

The pilot is deliberately small. A clean negative is a successful outcome.

## Motivation from existing ablation evidence

`Rineya_Cross_Scale_Capability_Transfer_Lab_v1/cross_scale_evidence/15_supa_layer_ablation_screen/`
identity-skips each of the 28 decoder layers in turn and measures collapse:

| band | layers | behaviour under identity skip |
|---|---|---|
| critical | 1, 2, 0 | catastrophic generation collapse |
| soft-significant | 19 > 27 > 25 > 17 > 18 | degraded but coherent |
| mostly null | much of the middle stack (layer 8 least sensitive) | near-null |

**This is an inference-sensitivity map and nothing else.** It is *not* a
training-importance ranking, *not* a pruning prescription, *not* evidence that
redundant-at-inference layers should be deleted, and *not* evidence that the
most sensitive layers carry the best training gradients. It is used here only to
motivate one cheap candidate intervention.

The working design rule is: **protect the stem; adapt the late reasoning/readout
path.**

## Hypotheses

* **H0 (interference).** A tiny model's limited representational capacity is
  contested by irrelevant context. Removing it should raise accuracy at equal
  model and equal decoding.
* **H1 (intrinsic).** If refinement has little ceiling, the checkpoint needs
  intrinsic adaptation, and a small late attention LoRA on layers 17-19 can
  deliver it without touching layers 0-2.
* **H2 (NoRA-init).** Column-normalising the LoRA down-projection at init is
  either better at matched budget or equally good materially sooner.

## Phase definitions

### Phase 0 — zero-training context refinement

Four input conditions over the **same** evaluation examples, same model, same
checkpoint, same decoding, same max output length, same scorer, same prompt,
same seed policy. Only the context string differs.

| condition | context |
|---|---|
| **A** full | original context, unchanged (canonical baseline) |
| **B** query-selected | BM25 chunks under a fixed token budget, restored to document order |
| **C** random equal-token | same chunk count as B, chosen at random, restored to document order |
| **D** oracle evidence | chunks localised from the dataset's own `supporting_facts` metadata |

Frozen selector (one configuration, no sweep — see `preregistrations/`):

* chunk target **192 tokens**, packed on sentence boundaries, no overlap;
* scored by **Okapi BM25** (k1=1.5, b=0.75), IDF over the document's own chunks;
* scored-selection budget **384 tokens**, then ±1 neighbour expansion capped at
  **576 tokens**;
* structural anchor = the document's **first sentence only** (~30 tokens),
  prepended identically to B, C and D so it can never be what separates them;
* non-adjacent runs separated by a blank line, identically in B, C and D.

Condition **C is mandatory**: without it, B - A cannot distinguish a *relevance*
benefit from a plain *brevity* benefit.

Condition **D** reads `supporting_facts` only; the gold `answer` field is never
consulted and the emitted text is always original context, never the annotation
string. D is marked **UNAVAILABLE** rather than approximated when the annotation
cannot be localised (see `RESULT.md` for the exact eligibility count).

### Phase 1 — tiny intrinsic attention adapter

Three arms on the same frozen evaluation suite, all at **full context**:

* **Arm A** — frozen canonical Supa-A1.1, no training.
* **Arm B** — attention-only LoRA on layers **17, 18, 19**.
* **Arm C** — attention-only **NoRA-init** on the same layers, identical budget.

`theta_train = {q_proj, k_proj, v_proj, o_proj}_{L17,L18,L19}`.

Layers **0, 1, 2 stay completely frozen** — not trained, not looped, not
altered, not removed. MLP/gate/up/down projections, embeddings, the LM head,
norms and every other layer are untouched in Phase 1.

Arms B and C share dataset, data order, seed, rank, optimizer, LR, scheduler,
context length, accumulation, step count and precision. NoRA-init differs from
ordinary LoRA **only at initialization**.

### Phase 2 — conditional expansion

Runs **only** if Phase 1 produces a clearly positive adapter. Compares
L17-19 against L17,18,19,25,27 using the winning method, everything else
identical. 25 and 27 are the remaining soft-significant layers; this is a
narrowly motivated extension, not a layer search.

## Decision gates

* **Phase 0 → Phase 1:** Phase 1 runs regardless of Phase 0's sign; Phase 0
  determines *interpretation*, not permission. Phase 0 must be mechanically
  correct (condition A reproduces the published baseline exactly) first.
* **Phase 1 positive:** ≥ 2 absolute points on the primary long-context
  composite **or** ≥ 5% relative error reduction, with average preservation
  regression ≤ 1 absolute point.
* **Phase 1 → Phase 2:** licensed only by a clearly positive Phase 1 adapter.
  A null or negative Phase 1 stops the programme; it does not open a layer sweep.
* **Phase 2 preference:** keep L17-19 unless L25/L27 give a clear additional
  improvement, judged per additional trainable parameter and per additional
  training cost.
* **2x2 composition check:** licensed only if Phase 0 query selection wins
  independently **and** the adapter wins independently. Inference only, no
  retraining.

## Non-go rules

No pruning or deletion of layers. No retraining layer 8 because it looks
redundant. No recurrence, RNN/GRU/LSTM, Mixture-of-Recursions or layer looping.
No RoPE modification. No KV-cache research. No temperature experiments. No
attention-architecture change. No MoE/router. No full finetune. No training all
28 layers. No MLP training in the first adapter experiment. No massive new
dataset. No 16K/32K long-context training. No hyperparameter sweep. No large
embedding model. No second LLM for retrieval. No combining mechanisms in one
training run. **No modification of the canonical Supa-A1.1 checkpoint.**

Unrelated opportunities found during repository inspection are out of scope.

## Resource constraints

Side-car friendly by design: it must coexist with other work on the same laptop
(RTX 4070 Laptop, 8 GB, shared).

* CPU-only preprocessing, cached to disk so the GPU stage never redoes it.
* One model load per stage; batch size 1; explicit teardown between stages.
* Short 2K-4K training examples; tiny adapters; small evaluation first.
* Bounded smoke run before every larger confirmation.

Measured for every stage into `results/<stage>/resource.json`: peak RSS, peak
GPU allocated/reserved, wall-clock, model loads, plus trainable parameter count
and step count for training stages.

## Checkpoint protection

The canonical core is read-only. `slcmi.provenance` hashes all ten checkpoint
files before and after every stage and raises if any differs; the frozen-model
stages additionally hash the full parameter state before and after inference.
Large weights are gitignored and live only in the primary checkout, so the
canonical path is `../Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-A1.1-1.1.0-devfreeze`.

## Layout

```
configs/            frozen stage configs
preregistrations/   decisions fixed before each stage ran
scripts/            entry points (one per stage)
src/slcmi/          selector, evaluation, statistics, provenance
results/<stage>/    per_example.jsonl, summary.json, run_manifest.json, resource.json
logs/               raw stdout per stage
tests/              mechanical checks
RESULT.md           the report
```

## Evaluation suites

| key | source | role | contexts | oracle |
|---|---|---|---|---|
| `demo` | `Rineya_Long_Context_Demo_v1`, 256 rows | **primary** | 1048–1481 tok | lexical localisation; 0/424 facts verbatim |
| `lcr_dev` | dev split of `Rineya_LCR_SFT_v1`, 200 rows sampled | confirmatory | 2112–2554 tok | **exact** localisation; 2688/2688 facts verbatim |

`demo` is primary because Supa-A1.1's published 45/125/170 is defined on it, so
condition A carries an exact validity gate. `lcr_dev` is the confirmatory
replication: longer contexts and an exactly localisable oracle, which the primary
suite cannot provide. It is disjoint from the Phase 1 training split and from the
demo suite (0 shared contexts, questions or titles), and carries no validity gate
because no published baseline exists for it.

## Commands

Run every command from the repository root.

```bash
PY=../Chat-Bubble/.venv/Scripts/python.exe   # torch 2.12.1+cu126, transformers 5.14.1, peft 0.19.1
S=experiments/supa_long_context_minimal_intervention_v1/scripts
```

Mechanical tests (no model, no GPU):

```bash
../Chat-Bubble/.venv/Scripts/python.exe -m pytest experiments/supa_long_context_minimal_intervention_v1/tests -q
```

Phase 0 — primary suite, smoke then full:

```bash
../Chat-Bubble/.venv/Scripts/python.exe experiments/supa_long_context_minimal_intervention_v1/scripts/phase0_build_conditions.py --smoke
```

```bash
../Chat-Bubble/.venv/Scripts/python.exe experiments/supa_long_context_minimal_intervention_v1/scripts/phase0_run_eval.py --smoke
```

```bash
../Chat-Bubble/.venv/Scripts/python.exe experiments/supa_long_context_minimal_intervention_v1/scripts/phase0_build_conditions.py
```

```bash
../Chat-Bubble/.venv/Scripts/python.exe experiments/supa_long_context_minimal_intervention_v1/scripts/phase0_run_eval.py
```

```bash
../Chat-Bubble/.venv/Scripts/python.exe experiments/supa_long_context_minimal_intervention_v1/scripts/phase0_analyse.py
```

Phase 0 — confirmatory suite:

```bash
../Chat-Bubble/.venv/Scripts/python.exe experiments/supa_long_context_minimal_intervention_v1/scripts/phase0_build_conditions.py --suite lcr_dev
```

```bash
../Chat-Bubble/.venv/Scripts/python.exe experiments/supa_long_context_minimal_intervention_v1/scripts/phase0_run_eval.py --suite lcr_dev
```

Phase 1 — adapter mechanics receipt (CPU), smoke, then the bounded run:

```bash
../Chat-Bubble/.venv/Scripts/python.exe experiments/supa_long_context_minimal_intervention_v1/scripts/phase1_verify_adapters.py
```

Use the resumable fixed-sequence controller from the repository root:

```powershell
& 'experiments/supa_long_context_minimal_intervention_v1/logs/run_phase1_resumable.ps1'
```

It sets the fixed CUDA launch and allocator environment, checks free VRAM,
runs one active model process at a time, validates/skips complete arms and uses
explicit `--resume` for interrupted arms. Re-run the same controller after a
Codex restart. The budget remains300 total attempts per adapter. A partial arm
never licenses a winner or Phase2. The controller ends with documents pending;
the final gate and any licensed Phase2 remain separately reviewed.

Direct recovery requires the same runtime environment as the original arm:

```powershell
$env:CUDA_VISIBLE_DEVICES = '0'
$env:CUDA_LAUNCH_BLOCKING = '1'
$env:PYTORCH_ALLOC_CONF = 'backend:native,garbage_collection_threshold:0.7,per_process_memory_fraction:0.8'
$env:PYTORCH_CUDA_ALLOC_CONF = $env:PYTORCH_ALLOC_CONF
& '../Chat-Bubble/.venv/Scripts/python.exe' 'experiments/supa_long_context_minimal_intervention_v1/scripts/phase1_run.py' --core a11 --arms B --max-steps 300 --resume
```

Canonical files and frozen scientific identities must still match. Never use a
remaining-step budget on recovery. Legacy attempts without checkpoints cannot
be resumed. See the audit amendment and RESULT.md for the source changes and
limitations. Read live append-only segment progress while an arm runs; avoid
opening `latest.pt` concurrently with its atomic replacement. After actual
process termination, `--combine`/`--resume` safely validates committed state under
an OS lock.

Audit snapshot of every frozen constant:

```bash
../Chat-Bubble/.venv/Scripts/python.exe experiments/supa_long_context_minimal_intervention_v1/scripts/emit_frozen_config.py
```
