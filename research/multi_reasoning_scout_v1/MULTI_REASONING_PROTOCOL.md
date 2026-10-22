# MRS-1: frozen open development mechanism scout

This is an inference-only experiment on exact retained dense Rineya models. It is OPEN / DEVELOPMENT / SELECTION-EXPOSED FOREVER. It has no confirmation, training, deployment, or release authority. Passing fixture tests does not establish model efficacy.

The primary existing candidate is `d1-seed17-bf16-export`, with frozen Supa 1.7 as a secondary unadapted reference. Both are 596,049,920-parameter plain dense Qwen3 models, stored in BF16 and evaluated in FP32, eager attention, deterministic greedy decoding, TF32 off, batch one, one loaded model, no custom logit processor or abstention bias. Exact assets and lineage are in MODEL_MANIFEST.json and MODEL_LINEAGE.json. D1's retained export is a distinct identity from the original FP32 training state. Historical DenseReference results do not qualify this runner.

The immutable bank has 120 unique worlds: 30 each for register transitions, counterfactual event deletion, prerequisite scheduling, and evidence chains with superseding facts. Eight format-smoke worlds and 16 CAL worlds are excluded from the 96-world CHECK endpoint. All derivatives of a world stay together. Gold and world specifications are separate from public input. Only the problem text, fixed role instructions, and explicitly projected candidate fields enter a model prompt; IDs, split, difficulty, answerability, gold and oracle results do not.

Each independent role returns a small JSON object with string `answer` in `0,1,2,3,UNKNOWN` and optional short `state`. R2 additionally requires `sufficient_evidence` and `issues`; R3 requires `candidate_valid` and `issues`. Confidence calibration is not introduced. State validity means task-native validity of the final state/time; free-text intermediate reasoning is not certified as a trajectory.

| Arm | Procedure | Logical calls | Allowed new tokens |
|---|---|---:|---:|
| A0 ordinary direct | R0 normal solver | 1 | 64 |
| A1 token-budget direct | Same R0, single longer allowance | 1 | 320 |
| A2 repeated direct | R0 and three benign fixed wordings of the same ordinary request, then S1 | 5 | 320 |
| A3 primary M1 | R0, backward R1, premise-audit R2, mechanical-reconstruction R3, then S2 | 5 | 320 |
| A4 analysis only | Any correct solver/checker candidate, scored after generation | 0 additional | 0 |

R0 is generated once and reused by A0/A2/A3. No temperature sampling or hidden retries create diversity. R1/R2/R3 each see the original problem and no earlier outputs. Their procedures are specified computationally, not as personalities. An independent R3 recomputes its own represented state: its validity flag is not a judgment on another role's answer.

Selector controls are S0 (majority over valid terminated answers; ties favor first valid role), S1 (one model call seeing all four candidate answers plus structural format/termination status), and S2 (one model call also seeing short state and checker fields). S2 is A3's final answer. S1 is an extra diagnostic call outside A3's five-call budget. A2 uses S1 on its four ordinary attempts. Malformed outputs remain in raw records; selectors receive an empty parsed object and explicit format failure instead of invalid raw text. Both selectors may synthesize a novel answer; novel correct synthesis is counted separately from candidate-selection efficiency. None sees labels.

The paired S1-minus-A2 diagnostic holds the answer-only selector fixed across candidate strategies; A3-minus-S1 isolates the effect of additional checker fields for the differentiated candidate set. A3-minus-A2 remains primary but changes both the role computations and the information exposed to the selector.

The total unique per-item execution is 11 calls and at most 960 generated tokens: shared R0, A1, three repeated-direct variants, A2 selector, three independent roles, A3 selector, and S1 diagnostic. Prompt lengths and actual outputs differ. This is token-allowance matching and call-count matching, not FLOP or latency matching. A single longer generation cannot reproduce multiple full prefills. Actual prompt/output/total tokens, generation/forward calls, elapsed time, sampled peak RSS and GPU peak allocation are recorded. Logical per-arm costs and unique experiment costs are kept separately.

An optional five-call M2 function implements solver -> premise checker -> revised solver -> constraint checker -> selector. It is tested but disabled in this cheap experiment. Enabling it requires a new prospective budget/experiment; it is not an unreported extra arm.

The parser accepts one whole JSON object, allows surrounding whitespace, rejects duplicate keys, invalid types, extra fields, markdown fences, prose, and nonfinite JSON. There is no parser repair or label-dependent extraction. Unfinished outputs count as nontermination and are not valid answers. No early stopping or prompt rewriting follows correctness. SMOKE requires at least 6/8 valid terminated outputs separately for R0, R1, R2, R3, A2 selector and A3 selector; failure blocks CAL/CHECK. CAL is recorded but not used to tune anything in MRS-1.

Preload guards require 6 GiB free host RAM, 6 GiB free GPU memory and 8 GiB free disk. Running guards require 1.5 GiB free host and 0.5 GiB free GPU, with 6 GiB allocation/RSS caps and 128 MiB new evidence per model. These do not lower the existing MQ/RRP guards. There are at most two current models, one load each, no retry after a loaded attempt, 1,322 generation requests/model including deterministic qualification probes, 115,232 reserved output tokens/model, and a hard foreground watchdog of 1,800 seconds/model. Four additional numerical qualification forwards are explicitly recorded. No foreign process, old lock, checkpoint, or historical evidence is removed. A refused preload uses no model launch. No CPU or cloud fallback is implicit.

Qualification is fresh on the exact model, source, runtime, protocol and bank: strict load, finite FP32 tensors, tied storage, cached/uncached logits, causality, repeated greedy identity, public input length checks, and unchanged in-memory weights. The longest direct prompt supplies the numerical cache/causality probe; every actual role prompt is independently bounded at generation time. Qualification is mechanical, not a model quality pass.

Evidence is create-only, raw-output preserving and bound to source files, repository SHA, model asset hashes, bank hash, prompt hash and protocol hash. `FREEZE.json` is created before any model generation. Failures preserve completed call records and never become a complete paired result. No partial intersection is used in the future comparison.
