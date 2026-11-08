# Titan joint quality instrument repair

An additive instrument successor for the existing Titan owner. The v1 source and receipts remain unchanged. Query-blind policies exposed perfect structural shortcuts in the original draft, so its corpus is not admissible for a reasoning-gain claim. It builds verified closed-world TRAIN/PILOT data, binds the exact Titan sources and tokenizer, prepares matched exposure, tests shared-denominator learning and full resume on tiny CPU fixtures, and checks paired quality gates. It supplies no native model worker or execution grant.

Source lives here, outside both active Git worktrees. All generated data, receipts, caches and pytest temporary files belong in the sibling `C:/Users/clclc/BusinessProject/Titan-Joint-Quality-Pilot-v2-runs` directory. Canonical models, environments and historical evidence are read-only.

## Implemented behavior

- Six domains and twelve genuinely different TRAIN/PILOT task mechanisms; two algorithmically separate teachers solve the rendered prompts without metadata or gold.
- Exactly 2,000 TRAIN rows and 384 PILOT rows, with two variants per world. PILOT prompts and references are separate. Confirmation is a 768-row future contract; generation, reads and scoring remain closed.
- Same-world opposite-query pairs across all twelve mechanisms, matched wrappers and distinct truths; irrelevant facts are identical within each pair. Content deduplication, split-family and nuisance audits remain separate from independent teacher, ancestry and instrument review.
- Full local Qwen serialization: thinking off, response plus one EOS, masked trailing newline; no truncation or padding. No Transformers/model import is used for offline tokenization.
- Deterministic whole-parent selection, processed-token mixture within five points of 50/25/15/10, at most 54,172 tokens per arm, whole rows at most 1,024 tokens and groups at most 2,048. Every group contains reasoning targets.
- Same sequences, order, groups and reasoning labels in both arms. Control forwards every input but masks nonreasoning CE; joint adds preservation, grounding and format CE. Both divide by the same planned target count.
- The reasoning term has the same normalization at the same parameter state. Added joint gradients can change clipping and Adam trajectories; identical realized parameter updates are not claimed.
- Exact inherited CPU-moment Adam and token-progress WSD primitives, checksum guarded. Tiny CPU loss-gradient, frozen-parameter and full checkpoint/resume proofs; real tokenizer vocabularies and native calls are refused by this fixture engine.
- Strict semantic/format scoring, duplicate-key JSON rejection, source-correct baseline freeze, individual harms, recovery and all-or-incomplete paired gate checks. The analyser accepts fixtures only and always leaves promotion/quality flags false.

## Source contract

The parent is plain Qwen3 Titan-Sparse FP32, SHA `a28629c9d3454fbf2e884fd7ca442c1b406e7caf4b36d34b2c0614d0add31284`. Reference sources are ungated Supa 1.7 `962abc5b...dd24b` and Titan-Transition `a4d73353...09567`. Geometry is 596,049,920 parameters, 310 tensors, 28 layers, width 1,024 and tied vocabulary 151,936. Tail layers 8-27 contain 220 tensors/314,618,880 parameters. The inherited sparse delta is not a runtime-pruned model; joint adaptation can densify its new update.

The historical RRP runtime was Torch 2.10.0+cu126/Transformers 4.57.1. The existing interpreter used for these offline checks has a newer runtime. Metadata/tokenizer/fixture checks do not transfer native qualification. Do not transplant MQ's Supa loader lock, per-objective normalization or six-cell permit into this experiment.

## Offline commands

Use the existing interpreter without installing or upgrading packages. From this source directory:

```powershell
$py = 'C:\Users\clclc\BusinessProject\Chat-Bubble\.venv\Scripts\python.exe'
$runs = 'C:\Users\clclc\BusinessProject\Titan-Joint-Quality-Pilot-v2-runs'
& $py -B -m titan_joint_quality_v1 build-data --out "$runs\data-NEW"
& $py -B -m titan_joint_quality_v1 source-audit --out "$runs\source-NEW.json"
& $py -B -m titan_joint_quality_v1 plan-exposure --data "$runs\data-NEW" --source "$runs\source-NEW.json" --out "$runs\exposure-NEW"
$env:OMP_NUM_THREADS = '1'
& $py -B -m titan_joint_quality_v1 fixture-pair --out "$runs\fixture-NEW"
& $py -B -m pytest -q -p no:cacheprovider --basetemp "$runs\pytest-NEW" tests
& $py -B -m titan_joint_quality_v1 handoff --data "$runs\data-NEW" --source "$runs\source-NEW.json" --plan "$runs\exposure-NEW" --out "$runs\handoff-NEW"
```

Every output must be new. `native-run` and `confirmation` exit 2 before creating output or importing a model. Blocked source/token/exposure checks also exit 2. Old prototype receipts remain intact; when generator source changes, create a new dataset epoch rather than editing its manifest.

## Quality acceptance

Joint needs at least five percentage points more semantic reasoning than Sparse and performance above the prospectively reviewed shortcut floor; it may trail control by at most two points. It must lose none of Sparse's frozen correct cases and preserve/restore at least one additional source-correct parent over control without offsetting source harms. Unsupported answers, false abstention and format failures cannot increase. Ambiguous free prose requires blind semantic review.

Freeze Sparse-correct preservation and Supa-correct/Sparse-wrong recovery sets before candidate results. Instrument review must establish usable per-domain denominators; zero losses over an empty set is not retention evidence. Full recovery of the prospective Supa-correct recovery panel is required before release qualification. No population noninferiority follows from zero observed losses.

The fixed recipe is answer+EOS, tail 8-27, FP32, LR1e-5, token-based WSD20/60/20 with 1-sqrt decay to0.1 peak, betas(.9,.95), epsilon1e-8, decay0 and clip1. One final checkpoint per arm; no selection from intermediate or partial runs.

## Existing owner handoff

Continuation belongs in [Read Codex goal objective](codex://threads/01a0dfa3-9cdf-7703-a362-32f559ec15a7). This implementation neither messages that chat nor launches a competing worker. The current Titan/TAPS campaigns, published weights and serving policy stay intact.

The owner must review the new corpus/teacher/overlap/floors, integrate the exposure/masks into its contained native lane, and qualify this exact source/objective/tokenizer/runtime. Baselines, native source-correct freezing and raw scientific result review are not implemented by the fixture analyser. The native count forecast still requires exact qualification counts and measured complete costs. The proposed six hours includes qualification, failures, both arms, all five evaluated configurations and an 1,800-second finalization reserve. No previous permit may be reused.

After a useful pilot: separately register one failure-directed curriculum/preservation/format comparison, replicate on independent training draws 17/23/29, then obtain fresh-family confirmation and export/serving qualification. Composition and learned sparsity stay downstream of a demonstrated useful donor. None of these future native stages is dispatched here.

## Opposite-direction research

The repaired instrument generates the world before choosing opposed questions. Every two-row parent must have identical non-query public inputs and requested wrapper, with distinct verified truths. A deterministic, stateless query-blind policy can therefore be correct on at most one member of each pair. This is an information-bound instrument check, not a neural quality gain. All original quality margins, token/cost limits, no-retry rules and native/confirmation closures remain unchanged.

Visible format instructions must agree with metadata and targets. Format crossover tests preserve the world and truth while changing only the requested wrapper; they diagnose convention transfer separately from content acquisition.

Historical pruning repairs retention in three matched draws (+1/+1/+4 retained answers) but does not establish zero forgetting or uniform quality improvement. A future opposite-update comparison must include dense, stable top25-percent and per-tensor norm-matched uniform scaling of the SAME new child delta; never add the inherited Titan update twice. No such model comparison has run here.
