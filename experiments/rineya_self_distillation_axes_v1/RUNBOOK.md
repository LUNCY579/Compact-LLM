# DSD-001 smallest native proposal, 2026-10-05

The current deliverable is an offline, byte-bound review packet. No native model
has been loaded, qualified, evaluated or trained in this preparation. The original
README/PROTOCOL/VERIFICATION and controlled evidence remain historical records.
This runbook registers one tiny successor proposal; it does not authorize execution.

## Source and ownership

Worktree: `C:/Users/clclc/BusinessProject/ChatBubble-rineya-self-distillation-axes-v1`.
Branch: `research/rineya-self-distillation-axes-v1`; starting/current HEAD at preparation:
`e1737660cd0feacd6efd4177ea1d1579bf0cd865`. Initial tracked/untracked status was clean.
This chat, `01a10d65-b003-7630-990f-eea2fe498f8c`, was the only visible DSD owner.
No matching non-shell DSD process was observed. Recheck both ownership and processes
at a later launch; never create a competing worker or remove another owner's lease.

Exact eligible source: **ungated Rineya-Supa-1.7.0-devfreeze**, core ID
`rineya.core.supa.v1.7.0-devfreeze.28L.qwen3-nimbus-lcr-stage-a-step200-scale-0p8125`,
at `C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze`.
The worktree copy lacks the bulk weights/tokenizer assets. Static byte/header reads
verified 310 BF16 tensors, 596,049,920 serialized parameters, Qwen3/28 layers,
nonquantized, no sliding window/scaled RoPE, and explicit tokenizer template.

Weight SHA256: `962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b`.
Complete 14-file DSD manifest: `495dee5eef1eb5365188ee589961da75909feb4a7446ac4f2d0760e48602d6b4`.
Selected parameter: `model.layers.12.self_attn.q_proj.weight`, shape `[2048,1024]`,
2,097,152 FP32 master parameters. This is a bounded subspace adaptation, not full
model fine-tuning. The shared source is writable; immutable custody means no writes
plus matching full pre/post hashes, including failure paths. Do not change attributes
or permissions on the shared source. Other experiments' parity/permits do not qualify DSD.

[Static source receipt](C:/Users/clclc/BusinessProject/dsd001-preparation-20261005/source-inspection.json)
and [full custody receipt](C:/Users/clclc/BusinessProject/dsd001-preparation-20261005/source-custody.json)
record the exact evidence. The source inspection's MODEL_QUALITY_STATUS line125
citation is imprecise; use line118 for registered weight identity and line122 for
the frozen Supa baseline. Its preserved receipt bytes were not edited.

## Fixed scientific contrast

Compare `sd-student-forward-ema0` against `sd-teacher-forward-ema0`, with `sft`
from the same verified solutions. This isolates rollout source; it holds forward
`KL(q||p)` and a frozen teacher constant. No controlled outcome chose this setting.
Forward KL first tests matching the privileged teacher's distribution without
adding a KL-direction or teacher-drift confound. It is a hypothesis, not a winner.
The primary paper separately identifies rollout/coupling/KL axes:
[arXiv:2609.39494v1](https://arxiv.org/abs/2609.39494v1).

Two paired order/sampling seeds17 and29; six individually selected cells; twelve
updates each, **72 total**. The same per-seed 12-row stream from a16-row TRAIN bank
is frozen before outcomes. Seeds reorder one bank; they are not independent
training draws. AdamW(.9,.999), eps1e-8, decay0, clip1, cosine LR1e-5, batch1,
BF16 operands/FP32 masters, math SDPA, no cache/thinking. SFT includes reference+EOS.
No20-arm grid,126-cell pilot, controlled campaign, EMA search or SFT LR search is
approved. One SFT LR cannot establish cross-method superiority.

Candidate bank:16 TRAIN,12 IID DEV,12 harder DEV,8 arithmetic replay DEV,
8 ordinary spatial replay DEV, and8 separate teacher-gate TRAIN-only rows.
The explicit modular-state rule is `s <- (a*s+b) mod17`, with signed coefficients.
Two independent solvers (sequential updates and composed affine map) verify solutions.
IID/TRAIN/gate use2 operations; harder transfer uses5–7 operations under the same
rule. Canonical inputs and equivalent initial-state/composed-map pairs are disjoint.
Shared algorithm families are intentional; this is compositional DEV transfer,
not broad independent-family generalization. No model output filters data.

The row schema's `admitted_for` declares a proposed split. The sidecar remains
`CANDIDATE_NOT_ADMITTED`; semantic/rights/admission review and native tokenizer
budgets must pass separately. Gate rows never enter updates or student DEV.
Retention is independent evaluation replay only; no rehearsal is added to training.
Both IID and hard rows use the loader's acquisition task name; `comparison.py`
separates their IDs. Legacy pooled acquisition endpoints are not primary here.

## Prospective teacher gate and zero-update checks

The initial privileged teacher must get **>=7/8 correct, >=2 paired wins versus
the clean source student, zero paired harms** on the isolated gate rows. Greedy
generation,32-token limit, same scorer, full reference solution. A clean-source
ceiling that prevents two wins makes this contrast uninformative: stop and preserve
the result. Do not substitute easier rows or reduce the threshold. This necessary
gate does not prove reasoning competence or deployment transfer; reference copying
alone cannot be reported as student acquisition.

Only the teacher receives references in gate/TRAIN contexts. Student inference,
IID/harder/replay generation and subsequent behavioral scoring always use original
prompts. Answer/reference sentinel mutations must leave student model inputs and
predictions unchanged. The two models score the same sampled response with separate
causal offsets (`prompt_length-1`), detached full-vocabulary KL, and detached rollouts.
Rollout source, KL direction and EMA remain separate fields. Exception restoration,
no-live-graph swaps, exact frozen/synchronized EMA endpoints and frozen hashes persist.

Zero-update qualification will verify exact local loading/no missing keys; raw-BF16
versus selected-FP32 full-logit parity (exact equality, no relaxed tolerance); representative
five objectives for each seed; full512-token no-update SFT backward; finite selected
gradients; teacher/source/frozen parameter+buffer identity; actual math-SDPA profiler
events; complete token budgets without truncation; custody; competence and resources.
Maximum-shape SFT backward is a memory/dispatch probe with synthetic EOS padding.
It is not maximum-shape KL, worst-case training/optimizer fit or capability evidence.
Comparison cannot launch merely because this qualification passes.

Proposed native runtime: Python3.11.15, torch2.10.0+cu126, Transformers4.57.1,
safetensors0.7.0, tokenizers0.22.2, currently present in the read-only BusinessProject
MQ environment. No environment installation/upgrade is authorized. The Hermes
torch2.6/Transformers5.14 environment is a different runtime. Offline CPU fixtures
use the MQ interpreter and append Hermes site-packages only to locate pytest9.1.1;
Torch/native dependencies still resolve from MQ first. This does not prove native compatibility.

## Approval packet and commands

The [exact proposal](C:/Users/clclc/BusinessProject/dsd001-preparation-20261005/packet/proposal.json),
[candidate custody](C:/Users/clclc/BusinessProject/dsd001-preparation-20261005/packet/data-custody.json),
and [packet receipt](C:/Users/clclc/BusinessProject/dsd001-preparation-20261005/packet/packet-receipt.json)
bind source, code, template, data, per-seed configs, row streams and limits.
Every new attempt/environment/cache belongs under BusinessProject outside tracked source;
never use `.codex`. Native qualification is a separate explicit decision. A review
must bind the **proposal file SHA256**, phase `zero_update_qualification`, experiment
`DSD-001`, reviewer, and exact separately approved screen/gate data hashes.
No approval file is created by preparation. A changed byte requires a fresh packet/review.

Preparation only (safe offline, create-only output):

```powershell
& C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe -B -m experiments.rineya_self_distillation_axes_v1.proposal --output C:/Users/clclc/BusinessProject/dsd001-preparation-20261005/packet --custody C:/Users/clclc/BusinessProject/dsd001-preparation-20261005/source-custody.json --inspection C:/Users/clclc/BusinessProject/dsd001-preparation-20261005/source-inspection.json --workspace C:/Users/clclc/BusinessProject/ChatBubble-rineya-self-distillation-axes-v1 --business-root C:/Users/clclc/BusinessProject
```

Future command, **not executed and not authorized by this runbook**:

```powershell
& C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe -B -m experiments.rineya_self_distillation_axes_v1.reviewed_qualification --packet C:/Users/clclc/BusinessProject/dsd001-preparation-20261005/packet/proposal.json --review C:/Users/clclc/BusinessProject/dsd001-preparation-20261005/operator-qualification-review.json --output C:/Users/clclc/BusinessProject/dsd001-native-qualification-attempt1 --execute-qualification
```

Do not use legacy `qualify`/`train` to bypass this packet. Comparison needs a later
reviewed bounded runner and separate explicit training approval after qualification;
there is no auto-launch path from qualification or plan creation.

## Limits, failure and interpretation

Qualification:0 updates, <=2048 forwards,1,048,576 processed tokens,2048 generated
tokens,30 minutes. Cell:32 response/512 total tokens, <=12 updates, <=5528 SD or5132
SFT forwards, <=3M processed tokens, <=5504 generated tokens,30 minutes. Comparison:
<=16,576,512 processed input tokens,32,256 sampled tokens plus768 SFT reference
tokens. These analytical upper bounds include all baseline/final-three evaluations;
equal updates are not equal tokens/FLOPs. Record actual costs, entropy, stop reasons,
preclip gradients, per-step and cumulative update norms.

Before qualification require6GiB host/4GiB CUDA free and5GiB disk; during require
1.5GiB host/.5GiB CUDA free. Record load and stage allocated/reserved peaks and
host/process peaks without resetting after load. The old GPU snapshot is availability
only, not a fit certificate. Resource/time guards sample before/after forwards and
at stage boundaries; they cannot preempt a stalled native kernel or certify continuous
free-memory floors between samples. Any nonfinite/missing gradient, privilege/parity/identity/
custody/budget/resource violation stops with a new failure receipt and partial evidence.
No automatic retry, fallback, shorter row, dropped case, source edit or foreign permit.
Preserve failure and source rehash results. No resume, cloud launch or promotion.

Primary acquisition: IID gain averaged at steps10,11,12, plus newly correct/damaged
rows. Harder transfer is separate. Final retention reports macro-task delta and each
task's source-correct retained/total, damages and repairs; zero denominator is undefined.
Require at least4 source successes per8-row retention task to interpret preservation.
The exploratory tradeoff flag requires positive IID gain, >=2 new IID successes
averaged over final-three checkpoints, and <=1 source-success damage per retention
task. This coarse DEV allowance is not a production no-regression margin.

A weaker update alone is not retention success. If less damage accompanies smaller
updates without matched acquisition, label the result attenuation-confounded or
inconclusive. Do not retune after retention outcomes. A later matched-dose/SFT-LR
control, independent training draws and fresh authorized confirmation need new
prospective decisions. Six cells/one bank cannot certify broad efficacy or serving.

Next decision: review the exact candidate/design/runtime packet and approve only the
bounded zero-update qualification, or reject it with a recorded reason. Training
remains a later decision. Historical evidence and serving policy remain untouched.
