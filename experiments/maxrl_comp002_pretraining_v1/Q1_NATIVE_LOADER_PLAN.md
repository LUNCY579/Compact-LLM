# COMP-RS-001 Q1: bounded native loader identity probe

Registered before a model load. This is a local read-only mechanical probe,
following Q0 source/data and Q1 asset/tokenizer receipts. It does **not**
authorize optimizer construction, a backward pass, a rollout group, training,
TEST/CHALLENGE reads, or a claim of native training readiness.

- Candidate anchor: exact local Supa 1.7 file set bound in
  `evidence/Q1_ASSET_CANDIDATE.json` (BF16 Qwen3, 28 layers, 310 stored tensors).
- Candidate runtime: `C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe`,
  Python 3.11.15, PyTorch 2.10.0+cu126, Transformers 4.57.1. No network or
  remote-code loading. Standard `AutoModelForCausalLM.from_pretrained` loads
  BF16 on CPU, then transfers to CUDA. This runtime has no Accelerate package;
  there is no device-map shortcut or silent fallback.
- Resource launch gate: two host-available RAM readings at least 5.5 GiB,
  5 seconds apart; at least 6 GiB free disk; CUDA available; no foreign heavy
  Python GPU process; exclusive `rineya-gpu-0.lock` lease labeled
  `comp-rs-q1-native-loader`. Stop if the gate fails. No process killing or
  stale-lease theft.
- Bound work: one model load, exact comparison of every stored tensor against
  the loaded state before and after **two** inference-only forwards on one
  synthetic short prompt. No generated completion, sampling, verifier call,
  gradient, optimizer, checkpoint write or model update. Record actual class,
  aliases, parameter inventory, runtime, device, memory peaks and raw-logit
  repeatability in a create-only receipt. End and release the lease on failure.
- The synthetic prompt is exactly: system `You are a concise assistant.`;
  user `Reply with the JSON object {"answer": [1]}.` Render through the
  bound tokenizer with `enable_thinking=False`, `add_generation_prompt=True`
  and `add_special_tokens=False` when re-tokenizing. It is not a source
  competence score.

The probe checks a 20-minute wall ceiling at stage boundaries; the launching
session monitors the intervals. Its code,
asset receipt and lease-module hashes are recorded. Later native completion,
EOS, selected-logprob, gradient, source-competence and enrollment checks need
their own bounded receipts. The Q0 template's `training_authorized=false`
remains true.
