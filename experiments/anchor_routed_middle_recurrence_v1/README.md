# Anchor-Routed Middle Recurrence (ARMR), experiment v1

ARMR is an experiment label. Does Supa-A1.1 tolerate iterative reuse of a
pretrained middle operator while its indispensable early stem remains intact?
Layer necessity, repeatability, useful refinement, router value, and training
recovery are separate causal questions.

The starting observation is the identity-skip screen at `fde553c` in
`Rineya_Cross_Scale_Capability_Transfer_Lab_v1/cross_scale_evidence/15_supa_layer_ablation_screen/`.
L0/L1/L2 skip catastrophically changes logits; L8 skip has mean cosine 0.9779.
Removal tolerance does not establish repeatability. The old ranking actually
places most middle layers above 0.90 cosine; do not generalize 0.71–0.89 to all.

## Frozen sequence and gates

1. Audit repository, checkpoint, runtime and cache. Requalify intact, inactive,
   skip L0/L1/L2/L8/L19, restore. Require each historical mean cosine within
   0.02, anchor cosines below 0.5, L8 above 0.95, exact inactive/restored logits
   and greedy IDs, unchanged state hash and checkpoint files. Stop on failure.
2. Independently test the 12 prespecified layers in the JSON configuration at
   two and three TOTAL applications (one and two extra passes). Candidates span
   highly removable, moderately sensitive, and late layers. No early repeats;
   catastrophic skip controls already qualify the stem hypothesis. Every pass
   remains at the original site; the ordinary upper stack is unchanged.
3. A two-application arm licenses fixed recurrence development qualification
   only if ALL eight smoke prompts complete without safety failure, all observed
   site RMS ratios lie in [0.25,4], mean logit cosine >=0.90, baseline-to-arm KL
   <=0.5 nats, no increase in invalid/loop outputs, and no smoke task-score loss.
   These pragmatic mechanical thresholds are frozen before the screen, not
   efficacy thresholds. Three applications are independently judged.
4. If any candidate passes, select highest mean cosine among passing middle
   arms, breaking ties by lower layer index. Evaluate A0/A1/A2 on the entire
   existing MERGEDEV-fast set. A2 requires its own smoke pass. Test A3 only for
   an adjacent pair both passing repeat-2; choose pair by mean constituent cosine.
   A3 must first pass the same smoke gate. Stop an arm on destructive behavior.
5. A router scaffold is licensed only if fixed recurrence passes development
   preservation (no overall score loss, no family loss over 0.10, no increased
   invalid rate) AND supplies at least one paired corrected development answer
   with no loss of total correctness. If no arm qualifies, stop at this research
   checkpoint. This conservative additional gate avoids an unmotivated router.

Each (arm,prompt) is a separately bounded case-run. A safety failure immediately
ends that case and is recorded; it is never retried. Other prespecified cases
remain independent experiments. NaN/Inf, cache access, model mutation, memory
limit, or unexpected runtime exceptions abort the stage. Repetition (three
identical suffix periods, at least six tokens), substantial non-Latin script
corruption on these English prompts, and prompt time budget stop generation.
Negative skip controls may fail these output checks as expected; their failure
is evidence, never a passing recurrence result. A 24-token cap is truncation,
not proof of EOS failure. Unicode anomalies are contextual flags, not a universal
definition of invalid language. No thresholds change after seeing outcomes.

## Architecture and cache semantics

`embedding -> L0 -> L1 -> L2 -> ordinary lower/middle stack -> repeated site ->
ordinary upper stack -> norm -> Supa abstention-aware LM head`.

A scoped wrapper calls the original bound decoder forward, preserves its
argument objects, and restores the original attribute on exit, including errors.
Total applications 1 and disabled mode are exact passthrough. No duplicated
weights, new token positions, optimizer, or training. The cache strategy is
**full-prefix recomputation without KV**: `use_cache=False` AND all cache
arguments absent/None. Reject cached calls before the model creates or updates
any cache. Qwen3 attention updates a non-None cache regardless of use_cache.
Repeated calls receive identical position IDs, mask, and RoPE tensors. Cached
recurrence, token-level routing, compile, and gradient checkpointing are excluded.
An intact cached/full-prefix probe is only a baseline audit, not cached ARMR.

## Future independently gated arms

| Arm | Mechanism | Training |
|---|---|---|
| A0 | intact Supa | none |
| A1 | best layer, 2 total applications | none |
| A2 | best layer, 3 total applications | none |
| A3 | adjacent safe middle block, 2 total applications | none |
| A4 | sequence router + distinct STOP, max 2 extra passes | router/gates only |
| A5 | A4 + loop-specific ordinary LoRA | router + LoRA |
| A6 | A4 + independent anchor injection | router/gates + anchor |
| A7 | strongest positive arm + tiny stem recovery LoRA | minimal PEFT |

For A4, use 3–4 measured candidate operators, RMS-normalized state query and
small learned keys plus STOP. Initialize residual sigmoid gate bias at -8
(g~0.000335); record max logit delta, cosine, KL and output identity before
training. Exact gate zero must return the original tensor. Capture anchor after
L2 once per forward; optional identity projection and lambda=0 make injection
inert. STOP ends internal depth; EOS alone ends generation. Compare fixed best,
fixed block, round-robin, seeded random, uniform mixture if feasible, and offline
oracle before claiming routing value. Include identity compute, parameter-matched
MLP and non-recurrent FLOP controls for causal gains. Analyze route/STOP frequency,
task and step dependence, gate magnitudes, and collapse. A collapsed router can
mean fixed depth suffices. Implement these only after the gates above pass.

T0 characterizes frozen operators. T1 freezes all backbone parameters and trains
only router/gates (plus independently registered anchor projection). T2 is
loop-specific LoRA only after T1 value; NoRA is optional only with validated local
implementation. T3 tiny zero-initialized early adapters requires a positive
capability signal with drift. No dense early tuning. A5/A6/A7 remain separate.

## Measurement and interpretation

Record text/IDs/length/EOS/cap/safety events, output and format scores, Unicode
and script flags, loops, first-token cosine/KL/top-1/top-k/margin/gold rank,
teacher-forced canonical-answer NLL when a single answer exists, and each pass's
RMS/L2/variance/cosine/delta norm. Reference normal single-pass state; also record
successive-state cosine, delta cosine and ratio rho. Rho is a trajectory
diagnostic, not a contraction proof. Suffix oscillation, persistent drift, and
small deltas are descriptive, not evidence of reasoning. First-token logits do
not characterize the entire generation distribution. Smoke task gains are
reported descriptively and cannot establish capability improvement.

Tier 1 is the historical eight-prompt mechanical screen. Tier 2/3 use the existing
unsealed MERGEDEV-fast preservation/reasoning families. Tier 4 long-context work
requires earlier qualification and separates retrieval from reasoning. Sealed
data stays closed throughout this task; the existing firewall is unchanged and
the new runner additionally allowlists exact development paths before reading.
No long-context, training, or sealed efficacy claims are licensed by smoke.

Runs use exclusive directories; manifests record source/config/dataset/checkpoint
hashes, branch/commit, runtime, seed, hardware, dtype, times, RSS and status.
No checkpoints are written. A dense-matmul FLOP estimate and real prefill/decode
latency are recorded separately; CPU timing is descriptive and serial-order
confounded. KV memory is zero; GPU memory is inapplicable. Hidden tensors/logits
stay in memory; small scalar results and text are versioned.

## Reproduction

Use the original ablation environment: Python with PyTorch 2.12.1+cu126,
Transformers 5.14.1, pytest, psutil and numpy; inference remains CPU FP32/eager.
Set `RINEYA_SUPA_PATH` to the immutable local Supa-A1.1 directory.

```powershell
python -m pytest -q experiments/anchor_routed_middle_recurrence_v1/tests
python -m compileall -q experiments/anchor_routed_middle_recurrence_v1/src experiments/anchor_routed_middle_recurrence_v1/scripts
python experiments/anchor_routed_middle_recurrence_v1/scripts/run_experiment.py --stage baseline
python experiments/anchor_routed_middle_recurrence_v1/scripts/run_experiment.py --stage screen --baseline <passing-baseline-run>
```

The screen requires a matching passing baseline receipt. Fixed recurrence and
router commands will be documented only when implementation is licensed.
