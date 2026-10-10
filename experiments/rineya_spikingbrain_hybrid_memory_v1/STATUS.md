# SBHM-001 qualification and native handoff

## Current publication status, 2026-10-09

**State:** `AUDITED_RECURRENCE_HOLD_QUALITY_PILOT_INCOMPLETE`.
The [portable research archive](evidence/research_closeout_20261009/README.md)
publishes the October 5 audited closeout, isolated `quality_followup` implementation,
reviewed synthetic corpora, partial native receipts and subsequent offline findings.
Historical files below retain their dated state; they are not current run instructions.

SBHM's negative result is unchanged: C 172/224 distant versus D 171/224,
zero D-only correct cases, and both damage 4/22 source-correct synthetic retention
cases while repairing 11 source-incorrect cases. Additional recurrence spending
remains on justified hold. Aggregate 29/64 retention is not broad preservation.

The separately authorized balanced-order pilot completed three native
qualifications and one frozen-anchor measurement of all 672 DEV rows. It stopped
at the unchanged 6 GiB runtime free-RAM guard during the control-training stage.
No completed training receipt/delta, trained DEV measurement or comparison exists.
Partial optimizer-step count, failure phase and exact failing RAM sample are unknown.
The ordering hypothesis therefore remains untested. The fresh anchor's 21/64
source-correct retention cases across five skills are a different denominator from
SBHM's 22/64; three fresh skills have zero source-correct coverage.

Offline analysis localizes first-record copying to revision selection, records
answer-shape and record-count shortcut opportunities, and preserves raw numeric
lexemes to avoid parser-created errors. A hypothetical output-wrapper diagnostic
and relational engineering fixtures establish no intrinsic Rineya improvement.
One seed, synthetic DEV and dependent world variants limit these discoveries.

The reviewed repeat proposal remains plan-only and unexecuted. Any later native
attempt requires separate exact authorization, fresh ownership/headroom checks,
a new exclusive output root and unchanged guards. Publication performs no model
run, promotion, policy change or automatic follow-up. Full receipts, source/code/
corpus bindings, limitations and the next concrete decision are indexed in the
archive. Weights, binary deltas, teacher targets, environments and grants stay local.

## Historical completed matrix, 2026-10-04

**Date:** 2026-10-04. **State:** `NATIVE_FOUR_ARM_DEV_COMPLETE_NO_RECURRENCE_BENEFIT`.

The fresh FP32 matrix completed all 22 commands with exit 0. **The recurrent
addition did not recover extra distant evidence above local-only in this pilot.**
On 224 distant DEV rows (112 counterfactual worlds), C scored 172/224 (76.79%)
and D 171/224 (76.34%): D-minus-C = **-0.45 percentage points**, with zero D-only
correct rows and one C-only correct row. The matched-near contrast was identical,
so the descriptive distance-specific increment is zero. This is a negative
result for this source/site/recipe, not a universal recurrent-memory conclusion.

| Arm | Distant /224 | Matched near /224 | Synthetic retention /64 | Retained A-correct /22 |
|---|---:|---:|---:|---:|
| A frozen source | 51 (22.77%) | 104 (46.43%) | 22 (34.38%) | 22 |
| B original + recovery | 172 (76.79%) | 179 (79.91%) | 29 (45.31%) | 17 (77.27%) |
| C local + recovery | 172 (76.79%) | 172 (76.79%) | 29 (45.31%) | 18 (81.82%) |
| D local + recurrent + recovery | 171 (76.34%) | 171 (76.34%) | 29 (45.31%) | 18 (81.82%) |

Retention is deliberately separate: C and D each damaged 4 of the 22 source-
correct retention cases and repaired 11 source-incorrect cases. B damaged 5 and
repaired 12. Broad original-model capability retention is not established by
these 64 synthetic tasks. Strict compact-JSON and normalized exact-match scores
were equal; neither measures semantic JSON equivalence.

Stratification found no D-only distant repair in any task or distance band.
Both scored 54/70 at the largest 1040-1058-token evidence gap; the one lost
distant case was an incorrect lookup value in the medium gap band. Near's
one lost case was a different update example, also a wrong value rather than
formatting. Missing-evidence behavior collapsed: C/D returned null on all 32
missing-category cases at each position, so they failed all 16 answerable
variants and passed only the 16 absent variants. The 50% category score is not
calibrated evidence handling. Worlds are the dependent-unit grouping, with
counterfactual/position variants kept together; no confidence interval or
promotion threshold is applied to this one-seed DEV result.

Instrumented distant-row runtime (guards included; loading/tokenization
excluded; CUDA peaks include the loaded model) was:

| Arm | Median TTFT (s) | Median decode tokens/s | Maximum allocated VRAM (GiB) |
|---|---:|---:|---:|
| A | 0.279 | 30.38 | 2.496 |
| B | 0.294 | 28.89 | 2.496 |
| C | 0.442 | 26.71 | 2.489 |
| D | 0.624 | 25.11 | 2.493 |

All 2048 prediction cache snapshots match their source-geometry formulas.
Among 511 C/D rows with equal cached prefix length, D uses exactly 0.5 MiB
more than C. The other row has unequal completion length and is excluded from
that paired storage comparison. The slow sequential reference has no optimized
speed or energy claim, and recurrent state did not improve this quality/cost
tradeoff. Raw cost distributions and counterfactual results are in
`completed_matrix_supplement.json`.

All B/C/D controls completed 512 attention-distillation and 512 answer/EOS
recovery updates, with exactly the same TRAIN row/token/label exposure. Every
post-update delta was reloaded and independently qualified before evaluation.
Each arm evaluated all 512 DEV rows. All 10 native inference qualification
stages passed at 2048-token prefixes. Native cache snapshots matched the source
geometry: A/B 448 MiB, C 433 MiB and D 433.5 MiB at 2048 tokens. D saves only
14.5 MiB (3.24%) of total KV storage and adds 0.5 MiB compared with C; other 27
sites still retain full attention. Added memory parameters and unequal reference
compute prevent an exclusive recurrent-route attribution.

The successful run empirically fits its actual 42-259-token TRAIN inputs in
FP32 on this host/GPU with the unchanged admission/runtime/RSS/VRAM guards.
It does not qualify the full 512-token training cap or larger/multi-site models.
No cache cleanup was needed in the successful stage receipts. Teacher targets
were exactly 479.607 MiB within the 512 MiB cap. The full CPU suite is 136 passed,
zero skipped, including the originally skipped pinned integration and recovery
loss/gradient/update equivalence checks.

Authoritative raw artifacts are in
`runs/titan_transition_v2_20261004_memory_efficient/` (four evaluation receipts,
training records/deltas, qualification receipts, teacher inventory and
`comparison.json`). Exact copies of 50 JSON receipts/manifests (6.38 MB) are
preserved in `evidence/native_20261004/completed_receipts/`; hashes verify each
copy against the source run. Binary deltas/targets remain in the source run.
The complete log and independent audit/supplement are in
`evidence/native_20261004/`. The interrupted full-logit attempt remains separate
and contributes no control to the completed comparison. No model promotion,
optimized-kernel, energy, broad quality or multi-seed result is claimed.

## Execution history

The first native matrix stopped at D recovery on the unchanged 1 GiB VRAM
reserve guard. Its completed A/B/C and D-distillation artifacts are preserved
under `runs/titan_transition_v2_20261004`; no recovered D or complete comparison
exists for that attempt. A guarded repeat stopped after recovery step 23 with
2.25 GiB allocated, 3.46 GiB peak allocation and 5.17 GiB reserved, demonstrating
substantial inactive allocator cache pressure. See `D-recovery-memory-probe.json`.

The tested rerun materializes vocabulary logits only at supervised answer/EOS
predictor positions, keeping the entire prefix/backbone and the same causal mean
loss. CPU pinned-library tests compare selected logits, loss, every trainable
gradient and an AdamW update for B/C/D. Runtime checks may release only inactive
PyTorch cache under reserve pressure, then apply the identical 1 GiB guard;
cleanup before/after memory is recorded. No data, precision, architecture, steps,
optimizer or guard threshold changed. The complete suite now has **136 passes,
zero skips**. All controls are rerun under new exact provenance; old results
will not be pooled with new D. Current config is
`configs/titan_transition.v2.memory_efficient.json`, current root is
`runs/titan_transition_v2_20261004_memory_efficient`, and current log is
`evidence/native_20261004/matrix-memory-efficient-execution.txt`.

The prior native attempt below is historical and does not establish full fit.

Fresh admission now passes with approximately 19.6 GiB free host RAM. The
registered matrix is running under `runs/titan_transition_v2_20261004`, with
its complete console log at `evidence/native_20261004/matrix-execution.txt`.
Frozen A passed actual-checkpoint native inference qualification at 2048 tokens;
full-versus-64-token-chunk maximum logit error was 0.00002265 with equal greedy
tokens. Teacher capture completed for all 256 unique TRAIN rows, exactly
502,904,576 serialized target bytes (479.607 MiB). A and B completed DEV
measurement. B and C completed 512 distillation plus 512 recovery updates each,
with finite losses/gradients and fresh recovered-delta qualification. C DEV
evaluation is running; D remains pending. Training lengths actually exercised
are at most 259 tokens; the 512-token cap is not a worst-shape fit certificate.
No completed four-arm behavioral contrast exists yet.
The October 3 RAM-blocked receipts below remain historical evidence.

The formerly skipped `transformers==4.51.3` integration passes on tiny random
CPU Qwen3 models. It checks A/B identity, wrapper equality, full/chunk/token
and greedy parity, request isolation, cache bounds and D zero-mix equivalence
to C. Separate ignored `.venv` (Torch 2.10.0+cpu) and `.venv-cuda` (Torch
2.6.0+cu124) preserve the shared runtime. CUDA availability and device identity
were queried; GPU model computation has not occurred. Current dated test and
environment receipts are in `evidence/qualification_20261003/`.
The final suite is **109 passed, 0 skipped**, including an interior converted
site followed by standard attention. See `cuda_stack/middle-site-final.json`;
syntax and package checks also pass.

The selected newest dense Titan source is Titan-Transition 1.0.0-devfreeze from
the owner's main Chat-Bubble repository. All 310 tensor names/dtypes/shapes and
the pinned tokenizer/config contract match standard Qwen3: 28 layers, width 1024,
16 query heads, 8 KV heads, explicit head dimension 128, QK norms, default RoPE
theta 1e6 and tied embeddings. Its 596,049,920 FP32 parameters occupy 2.22046 GiB
before runtime memory. Observed weight SHA256:
`a4d73353db9a91cbba7b4cd7c8fe4fe6822670d10fe523b26a6d651bb9509567`.
The complete 17-file source digest is
`c169fba225bfa58804c966caba97c147d9239b7dbcd2f3b7be87cdd05e44125a`.
A separately copied, hash-verified read-only snapshot is under ignored
`.native-source/titan-transition`; published files are unchanged.

The admitted stronger corpus has 256 TRAIN/512 DEV rows: seven evidence tasks and eight
synthetic retention tasks, independent answer/name/filler/record-order RNG,
factual counterfactuals and matched near/distant position controls. All 768 raw
prompt targets were independently replayed without importing the generator's
oracle. Independent review checks 336 counterfactual pairs, 336 position pairs,
exact token spans, required-fact distances, split/template/world separation,
group coverage and full teacher-budget admission. The owner expressly requested
dataset generation/review with agents; receipts do not claim human inspection.

The registered 512 updates per stage cover every TRAIN row twice. Full unique
teacher targets are estimated at 479.607 MiB, including headers, with 32.393 MiB
below the unchanged 512 MiB cap. Native training fit remains unknown. The first
v1 proposal remains rejected/unreviewed because its recent distractors encoded
distant answers; its files and disqualification receipts are preserved.

Comparison now reports paired D-minus-C strict/normalized discordances and
percentage-point differences, separate A-correct retention/repairs/damages,
per-family and registered distant/near/retention groups. Invalid raw metrics,
pairing, group coverage and incomplete/unbound training records fail closed.
Design/review references are hash-checked; cache admission counts actual
serialized target-file bytes before writing.

At the October 3 handoff no pretrained inference, distillation, recovery or DEV
comparison had run. Repeated fresh host free-RAM checks were below 7.4 GiB admission despite 31.22 GiB
installed RAM. Runtime 6 GiB RAM/4 GiB worker RSS/1 GiB VRAM reserve guards are
unchanged. Do not infer fit from the RAM upgrade, CUDA availability or weight
size. Actual-source and arm-specific stage receipts supersede this historical
block only as each new native stage completes.
The final source/corpus/config/code-bound, dated grant is recorded from the
owner's existing instructions. Final preflight and an actual A-qualification
attempt both exited 2 on the RAM guard before creating a qualification directory
or loading a model. Preflight reports no other blocker. The complete 22-command
matrix is saved plan-only; see `native-v2-admission.json`, `registered_v2_pilot.json`
and `matrix-v2-plan.json`. Use `configs/titan_transition.v2.json` for this pilot;
the earlier proposed config points to rejected v1 and must not be executed.

At 2048 tokens, theoretical persistent FP32 cache is 448 MiB for A/B, 433 MiB for
C and 433.5 MiB for D: D saves 3.24%, before activations/training graphs/transients.
D's converted site is 1 MiB local KV plus 0.5 MiB recurrent state. These are
geometry estimates, not native peak-memory/speed/energy measurements.

The initial forget projection has a 16-token half-life; other 27 sites keep full
attention. D/C evidence concerns one trained intervention, not an exclusive
recurrent route. Short synthetic retention and exact JSON string scoring remain
narrow DEV measures. Correlated counterfactual/position rows share a world;
future uncertainty must respect that unit. No broad quality or promotion claim.

## Archived 2026-10-02 delivery

**Date:** 2026-10-02. **State:** `IMPLEMENTED_OFFLINE_TESTED_NATIVE_BLOCKED`.

## Delivered

Independent PyTorch local/recurrent reference, standard-Qwen3 native bridge,
selective LoRA conversion, streaming teacher targets, attention distillation,
answer/EOS recovery, safe adaptation deltas, cache/state-reset diagnostics,
per-row DEV measurements, exposure-checked four-arm comparison, and a plan-only
or explicitly authorized 22-command matrix. No existing research file changes.

## Executed here

- `python -m compileall -q sbhm tests`: PASS.
- `PYTHONPATH=. pytest -q tests --disable-warnings`: **56 passed, 1 skipped**.
- Tiny random CPU fixtures execute teacher capture, B/C/D distillation, recovery,
  safetensors save/reload and output equivalence. These are not pretrained-model
  results. Other tests cover causal/chunk parity, persistent storage bounds,
  independent request state, recurrent reset, finite gradients, explicit head
  dimension, controls, split/manifest/receipt guards and comparison pairing.
- Optional actual-Transformers tiny-random-Qwen3 integration: **SKIPPED**, because
  `transformers==4.51.3` is not installed. This missing integration remains a
  prerequisite, not a passed native test.
- Example native preflight: **BLOCKED**, exit 2, no model loaded.
- Matrix plan-only path: prints 22 commands without creating a run directory.

Raw scoped output is under `evidence/`. The runtime is CPU-only torch 2.10.0,
Python 3.13.5; no CUDA execution occurred. No GitHub Actions were added or used.

## Not executed / not established

No exact Rineya checkpoint was supplied to the runner. No pretrained inference,
training, model download/upload, protected-bank access, speed comparison, energy
measurement, native memory-fit qualification, or capability evaluation occurred.
The runtime lacks the pinned Transformers/tokenizers/accelerate stack, CUDA and
7.4 GiB free host RAM. The example also has no source lock, reviewed native
corpus, or dated operator grant. It intentionally cannot authorize execution.

The native path is implemented but only standard ungated HF-format Qwen3 is
supported. Custom Supa/gated/pruned-export loaders must be inspected and qualified
individually; a public Qwen model cannot silently replace the selected Rineya
source. Full-model training, optimized kernels, multi-site conversion, beam and
padded batching, checkpoint-resume, substantive semantic scoring and full spiking
hardware execution remain outside this first pilot.

## Next authorized handoff

Run the optional tiny official-library integration test in the pinned isolated
environment. Select and lock the exact local source; freeze the single-site
pilot and reviewed disjoint TRAIN/DEV data. Review the matrix and resource
budget. Obtain explicit stage-specific native permission; then qualify A/B/C/D
on the actual source/host before training. Any new trained delta requires fresh
qualification before evaluation. Preserve every negative/blocked attempt and
report raw paired retention/cost evidence before considering broader conversion.

A one-site mechanics result is not an overall Rineya improvement. SBHM remains
complementary to the existing research; this delivery does not merge, retire,
replace, or mark any other branch obsolete.
