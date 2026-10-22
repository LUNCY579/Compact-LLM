# Prospective bulk native successor — not an admission

Owner: implementation engineer. No model/tokenizer execution or protected data
access by this agent. Coordinator owns source freeze, independent review, resource
admission, parent entry/supervisor, native lock and dispatch. Existing OPEN24,
GPU4c1 and original v1 sources remain untouched.

Population is 288 new source-visible OPEN cases in 144 parent worlds, six
categories, two variants per world; 192 DEVELOPMENT cases/96 worlds and 96
OPEN_VALIDATION cases/48 worlds. Parent worlds must be disjoint across splits.
Shared generator code/dependence is reported, not renamed away. Category6 evidence
swap variants may have different gold/answerability; no same-gold assumption.
The source-visible validation surface is exploratory transfer, not confirmation.

Each deterministic policy runs once per case at seed17: intact A0, unit A1, six
fixed A2 constants (.8,.9,1,1.1,1.2,prior prefix mean1.1473425965417514), unchanged
A3. A5 runs two actual seed17/29 prefix RNG streams. Thus11 physical evaluations
per case =3168. Deterministic duplicates are not additional observations. The
DEV fixed winner uses equal-category means of the two-variant parent means, with
lower-alpha ties. A3 pool admits exactly one DEV generation per case with full
prefix/iteration provenance; longer responses contribute more pool observations.
Freeze bank/token/runner/config/case/parent/pool identities before every validation
forward. Validation primary comparator remains DEV-selected; validation grid and
per-case constant oracle are retrospective diagnostics, not reselection.

Timing counterbalance is prospectively deterministic: sorted parent index and
variant index select nine-policy rotation/reversal. Every case sees all nine
policies exactly once; no order depends on labels/outcomes. A5 remains a later
post-freeze phase and its order limitations are explicit. Original full-model
rollouts are never reused to create counterfactual trajectories. Paired first
logit diagnostics are computed afterward from each case's actually observed first
logits and stored in a separate nonphysical readout_pairs journal, because A0 may
occur after another policy. This adds readout computation but zero model forwards.
All timing is instrumented/descriptive; no production speed or timing-repeat claim.

Fresh inherited SMOKE uses its original scorer/cap24: 48 legacy/new/noextra/unit/
original-A3 identity cases, 72 regular nine-policy control cases including the
extra static constant, and8 DEV-pool A5 cases =128. Fresh max357 checks noextra,
unit, original A3, all scales, forced exception restoration and actual A5 pool;
18 forward attempts include one expected partial exception, and two cache
invocations reject before model computation. Source/FP32 state/custom head,
positions/masks, L0-L2 one-shot, buffer devices, cache/gradient/parameter/source
preservation and numerical guards remain required. No training, KV, deeper
recurrence, precision change or original-reader monkeypatch is admitted.

Complete count forecast:3296 physical case evaluations; generation cap28416
(3168*8 +128*24), plus18 maximum-shape attempts and2 cache-rejection attempts =
28436 API invocation cap.288 constant-oracle rows and96 frozen-validation aliases
are nonphysical reuse with copied costs excluded. Missing partial compute is null.

Read-only timing source: runs/native-gpu-reference-002/result/forwards.jsonl,
only completed calls with1<=prefix_tokens<=357, no outcomes/quality filtering.
There are2065 calls, observed shapes24–88, mean.0735847897s, p99.1200511000s,
maximum.5056918000s. That maximum is the first cold call. Charging it separately
leaves2064 observations, maximum.1467086000s. No prior observation qualifies
shapes89–357. A provisional complete-cap estimate using the post-cold maximum
times1.5 is6257.709 seconds, plus the measured cold call,200 seconds cold setup/hash allowance and1800
seconds finalization reserve,8258.214 seconds. The pessimistic all-call cold
maximum scenario exceeds the available allowance and is not admitted as a fit.
These are feasibility estimates only; fresh fully executed max357/warm qualification
cost must forecast the complete remaining matrix before any DEV model outcome.
Cold actual time and setup/hash work remain charged once in elapsed worker cost.

Prospective parent deadline18000 plus1800 reserve is19800, below the stated
20867.829 seconds remaining. No retry is available. Host parent7GiB/worker6GiB,
running host reserve1GiB/RSS ceiling6GiB; initial GPU free6GiB, allocated peak5GiB,
global free1GiB; disk/evidence reserves remain unchanged. Guards are not lowered.
Budget/resource/numerical failure retains all partial journals and suppresses
primary comparisons; no survivor-selected worlds or favorable truncation.

New schema: READY_FOR_BULK_FP32_DISPATCH; independent review
PASS_BULK_FP32_DISPATCH with admission_sha256. Explicit bulk_bank_path/sha256,
bulk_manifest/audit/positive_controls/source_sha256 and
bulk_token_binding_path/sha256, plus bulk_normalizer_source_sha256 (the imported
frozen open_bank.py normalizer), qualified gpu_helpers_sha2564c1, canonical
checkpoint/runtime/dispatch hashes and resource contract. Token packet binds all
288 IDs/parents/variants/splits and exact rendered IDs, input<=350/output8. Root
owns new bulk_binding.py, bulk_entry.py, bulk_supervisor.py and admission. Worker
entrypoint is `python -B -m experiments.rineya_taps_adaptive_recurrence_campaign_v2.bulk_native --source <canonical> --out <exclusive> --admission <reviewed>`.
Do not call the current OPEN24 runner as a fallback or mutate its reader guards.
