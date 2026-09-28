# OPALLO-BEH-001: format and literal-abstention screen

**Design and offline mechanics implemented. Native qualification, calibration and model
screen remain unexecuted.** This is a prospective engineering screen, not an efficacy test.

## Evidence boundary

The cited scientific subject is `08910b691ae66175f5202fb9f0724ce0ac6bebc5`.
The reviewed parent is `a36de96b30eb03265db3a18e464a2e2a23ee67bf`, two commits later.
That parent already has a registered three-seed repair/control training experiment,
reported as not started because of host memory. This additive companion preserves its
registration, data, thresholds, source files and pending outcomes.

Historical RAB scores are Supa 161/256 and Opallo 135/256. The explicitly post-hoc
containment analysis gives 149/192 versus 150/192 on answerable rows. That motivates
behavioral repair; it does not establish retrieval equivalence or a new confirmatory
scorer. Seventeen rescued Opallo outputs and eleven fewer UNKNOWN reports are not
disjoint additive components of the 26-row net loss.

Source inspection found two important measurement distinctions:

1. The RAB probe compares its first physical line after lowercasing, replacing periods
   with spaces and collapsing whitespace. The suite normalizer additionally removes
   articles, answer labels, quotes and other punctuation. These are different scorers.
   Historical results retain the actual RAB rule; the new tool names that rule explicitly.
2. The probe's `says_unknown_on_unanswerable` counts a substring anywhere in the response.
   Historical 13/64 versus 2/64 is retained as a source-reported counter, not independently
   verified full-response literal UNKNOWN. Prose abstention is distinct from unsupported
   answering. No raw historical prediction reclassification was performed in this task.

Retrieval training, CED changes, architecture search, WSD or soup selection, general
reasoning and release remain outside this experiment. The old Demo and arithmetic panel
are exposed retention instruments, not fresh sealed tests.

## Fixed model and four controls

All conditions use the published Opallo dense artifact, not its FP32 training checkpoint:
`Rineya_Published_Cores/Rineya-Opallo-1.0.0-devfreeze/model.safetensors`, SHA-256
`e726d6bea1139eddfbcdadeb4c8d7b979e7a55e43490d9787ab99a3ab12ee76b`.

| Arm | Prompt policy | Output decision |
|---|---|---|
| B | Original task contract | Original greedy response |
| F | B plus the fixed `FORMAT` sentence in `screen.py` | Original greedy response |
| G | Exactly B | Frozen UNKNOWN probability gate |
| FG | Exactly F | The same gate and threshold as G |

B already requests UNKNOWN when the requested record field is absent. F changes only
response-format wording, leaving the record, question, evidence availability and UNKNOWN
instruction identical. The inference tests instruction sensitivity, not a proven pure
internal formatting mechanism. No condition changes weights, token limits or stopping.

The gate reuses the gate lab's exact-UNKNOWN first-token signal:
`p = exp(first-step log P(exact UNKNOWN token))`. When `p >= tau`, its output is exactly
`UNKNOWN`; otherwise it preserves the original response. The signal is calculated from
full-vocabulary logits, not top-k mass, answer text, or answer-key labels. The tokenizer
must verify single-token encoding and exact decoding of UNKNOWN. Token IDs and another
core's thresholds are not transferable assumptions. Normalized-probability gating and
constant-logit-bias baking are different mechanisms.

G and FG replay the B and F captures offline, so four conditions require only two model
prompt policies. Report actual capture token and latency costs; offline output substitution
is not evidence of native early-termination savings.

## Fresh fixtures and calibration lock

The deterministic builder produces 16 calibration scenarios / 64 rows and 32 check
scenarios / 128 rows. Each scenario has base, paraphrase, changed-value counterfactual,
and missing-field versions: calibration 48 answerable / 16 unanswerable; check 96 / 32.
Complete noun phrases include legitimate articles, which remain part of their target.
Gold-dependent extraction, clipping and article removal are not interventions here.

The splits have disjoint scenario IDs, entity codes and answer values. They share a
synthetic generator: they are not template-held-out, secret, cryptographically sealed, or
representative of general long-context performance. Prompt exports contain only IDs and
messages. Answer values appearing in an answerable record are legitimate task evidence.

Before native execution, a reviewer must inspect all 48 packs / 192 targets and record a
new overlap audit against pinned training data, historical Demo prompts, the exposed
panel, and prior repair-selection material. The audit records normalized full-prompt and
latest-user matches, near-template review, source hashes, counts and exclusions. Existing
corpus leakage reports do not establish this union audit. Necessary exclusions require a
new registration before model outcomes. Current fixture status is not leakage-cleared.

Freeze fixture hashes, reviewed capture recipe and implementation digest before calibration.
Only calibration B and F are captured first. The single threshold is selected on B from:
`off, .001, .003, .01, .03, .1, .3, .5, 1.0`.
Selection maximizes literal UNKNOWN on unanswerable calibration rows subject to zero new
literal UNKNOWN on answerable rows. Ties prefer off, then the largest numeric threshold.
F calibration is diagnostic, not a second selection source.

The spec and both calibration capture hashes must have a git receipt before check outcome
access or capture. Check is scored once, with no prompt, threshold-family or split changes.
The program recomputes calibration to detect altered specs; chronological freezing still
requires the external receipt.

## Registered engineering decision

FG versus B is the primary contrast. F and G are attribution controls, not alternative
opportunities for promotion. On the same 128 check IDs, FG must simultaneously achieve:

* At least +4/96 answerable hits under the historical RAB first-line rule and +4/96
  full-response literal answer matches, ignoring only outer whitespace for the latter.
* At least +4/32 full-response, case-sensitive literal UNKNOWN on unanswerable rows.
* Zero losses of baseline-correct answerable rows and zero new literal UNKNOWN on any
  answerable row. Aggregate gains cannot offset individual preservation losses.

Coverage, finite signals, successful status, actual EOS termination and untruncated inputs
are prerequisites. A baseline ceiling means insufficient room on this fixture, not a
reason to lower thresholds. Joint failure parks this recipe. A one-behavior benefit may
motivate a separately registered follow-up, not alternate-winner selection on check.

The +4 thresholds are minimum-useful engineering criteria, not power calculations,
significance tests, equivalence margins or restoration to Supa. Report paired scenario
outputs, raw counts, both loss directions, emitted characters, actual generation tokens
and termination. Three answerable siblings do not create 96 independent scenarios;
seed-by-row repetition likewise does not create independent trials. No p-value gates
this screen. Containment is not an implemented success endpoint.

Blind review of unanswerable outputs must distinguish literal UNKNOWN, semantic abstention
in other words, unsupported answers and ambiguous responses. Annotation disagreements
are resolved without changing exact-match labels. Improved literal compliance alone is
not a claim of reduced hallucination.

## Native qualification and resource limits

The shipped offline tool does not implement a model loader, native capture adapter,
watchdog or baking path. These remain prerequisites, not completed integration.
The inherited `run_local_hf.py --emit-logprobs` is only a candidate collector: the parent
records an accelerate/runtime incompatibility, and its confidence helper checks the
tokenizer EOS ID rather than the RAB probe's complete official EOS list.

Qualification must hash the weights, tokenizer/config/template assets, complete generation
config, EOS list, collector source and exact package/runtime manifest. It records UNKNOWN
encode/decode, device, compute dtype, attention backend and TF32 settings. Artifact storage
dtype is not automatically execution dtype. All arms use one qualified isolated runtime,
greedy decoding, thinking disabled, 24 new tokens and 4096 context capacity, without
truncation, extra adapters or logit modifications before confidence capture.

Confidence-on/off parity and one repeat use the first eight calibration B rows, preserving
text, input IDs, generated IDs and termination. The main capture already includes these
eight once, so this adds 16 generations: **400 maximum before retention**
(128 calibration + 256 check + 16 parity/repeat). Actual termination is checked against the
full official EOS list, not inferred from the helper's tokenizer-only flag.

Execution is serial, one model job at a time, within the existing memory safeguards and a
separately qualified environment. The native supervisor must enforce 900 seconds per
capture and 3600 seconds for the screen. Asset, parity or resource failures are BLOCKED,
not scientific negative results. Inference is lower-overhead than training but is not
assumed to fit a busy host. The offline program does not enforce these native limits.

## Retention and future training

A qualified successful screen only advances to retention review. Freeze FG, then compare
B against FG on the exposed 256 Demo rows, preserving the actual RAB scorer, literal
metrics, full predictions and paired answerable-loss ledger. This costs at most 512
additional generations and needs no Supa reload. This is an exposed diagnostic, not new
confirmatory evidence. Advancing the narrow policy requires no new false UNKNOWN and no
lost baseline-correct answerable row.

The policy is opt-in for closed-context, exact-value responses; other behavior remains
unchanged. Global use, weight modification or baking requires independent data, native
gate parity, full arithmetic-panel retention against a same-runtime baseline, and separate
release qualification. The parent training experiment's mean floors of 62/37 do not by
themselves qualify a changed serving policy.

The short-response corpus is a future targeted SFT source rather than an immediate
12,000-row training job. Its `output_only` and `factual_compression` classes are relevant;
`knowledge_boundary` and `uncertainty` need case-by-case contract review rather than
blanket UNKNOWN relabeling. Human review is still pending in its data card. The gate lab
provides a mechanism, not transferred evidence of an Opallo benefit.

The parent's repair targets use natural-language abstention, not literal UNKNOWN. Its
recorded 5,302 versus 5,233 supervised-token budgets differ by 69 tokens and are approximate
rather than exact matches. Fixed row selection is distinct from training-order seeds;
three retrained controls are not all the shipped BF16 artifact. These observations are
recorded without editing that registration. Any later new training design requires a
separate row/loss-mask audit, matched actual token/update budgets, fixed checkpoint and
update operator, and paired seeds. Neither prompt success nor failure of a 14-row mixture
settles general abstention teachability. Retrieval and efficacy claims remain closed.
