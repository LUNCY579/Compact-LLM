# CLARE-RINEYA-001 — capacity-aware sparse skill composition

Can learned, capacity-aware sparse parameter allocation (CLARE, arXiv:2609.17026) make additive
R/U/S/P skill deltas in the 596M Rineya model interfere less, retain better and compose better on
held-out skill combinations than dense sequential training, independent-delta addition, magnitude
pruning, or random disjoint allocation at identical capacity?

- **Protocol:** [CLARE_RINEYA_001_PROTOCOL.md](CLARE_RINEYA_001_PROTOCOL.md) (registered before any native arm)
- **Phase 0 audit:** [CLARE_RINEYA_AUDIT.md](CLARE_RINEYA_AUDIT.md)
- **Configuration:** [configs/clare_rineya_001.json](configs/clare_rineya_001.json)
- **Smoke procedure and host handoff:** [NATIVE_SMOKE_AND_HANDOFF.md](NATIVE_SMOKE_AND_HANDOFF.md)
- **Result schema:** [RESULT_SCHEMA.json](RESULT_SCHEMA.json)
- **Status and measured results:** [RESULTS.md](RESULTS.md)
- **Future work (not in 001):** [GRADIENT_AWARE_REUSE_DESIGN_NOTE.md](GRADIENT_AWARE_REUSE_DESIGN_NOTE.md)

## Canonical branch

**`research/titan-clare-sparse-composable-skills-v1-claude` is the canonical CLARE-RINEYA-001
branch** (operator decision, 2026-09-27). A parallel agent implemented the same brief on
`research/titan-clare-sparse-composable-skills-v1` (author LuncyNever, `76230adc`, `e320ba33`) as
CPU-only mechanics with native preflight BLOCKED. That branch is merged here with strategy "ours":
its commits are ancestors of the canonical branch, its implementation is **superseded and not in
this tree**, and its 87 CPU test cases are ported to the canonical API in
`tests/test_ported_e320ba33.py` (99 cases, with a per-test provenance map).

## Lineage (Titan / model soup)

Branch `research/titan-clare-sparse-composable-skills-v1`, cut from
`research/titan-model-soup-mini-v1` @ `816a1e8f4d1ab1c9d1ab847697d962a141521bac` (unmodified).

```
naive model averaging -> additive delta combination -> delta geometry / cosine
  -> sparse delta analysis -> orthogonality observations -> LEARNED SUPPORT ALLOCATION (here)
  -> functional composability testing (here)
```

Model soup asks how already-learned deltas should be combined. CLARE-RINEYA asks whether
controlling **where** each skill learns, from the start, makes later combination easier. Every
separation the soup line tested was applied *after* dense training and matched plain addition or a
matched-norm scalar. Here each skill is *trained inside* its support.

Support disjointness implies a zero parameter dot product, and nothing more. It does not imply
independent hidden states, independent logits, retention, or composition. Those are measured.

## Layout

```
clare/            package
  footprint.py    eligible coordinate space (ordered slots, global offsets, digest)
  select.py       exact-budget selection: radix select over uint64 keys, seeded tie-break
  schedule.py     CLARE capacity schedule k_t = round((1 - rho)|A_t|)
  masks.py        packed saved masks, strict AllocationState (claim, save/resume)
  deltas.py       dense/sparse/disk deltas, representation-independent checksums, compose
  training.py     DeltaHost (masked delta training on any nn.Linear footprint), run_stage, checkpoints
  allocation.py   arms A-F, CLARE discovery -> select -> RESET -> retrain -> freeze, lambda gate
  native.py       Qwen3 BF16-storage / FP32-compute patches, loss, greedy decode, teacher forcing
  recipe.py       Titan's rate_at / group_updates / compile_row (vendored)
  data.py         MAXRL-COMP-002 adapter: pinned, purpose-gated, verifier-scored
  evaluator.py    resumable per-row generation scoring; 2x2 interaction scorer
  geometry.py     cosine, dot, norms, overlap, Jaccard, sign agreement, layer energy, rank energy
  functional.py   interaction estimators (FA-001 kappa vendored), vocabulary-level interaction
  metrics.py      accuracy, retention/composition matrices, paired contrasts, damage rule, CSV
  manifests.py    per-skill manifests (git, parent, model, data, runtime, checksums)
  resources.py    host RAM (vendored hostbudget), GPU presence, VRAM, disk, GPU lease
vendor/           hostbudget.py (byte-identical to 42eb390b)
configs/          the registered configuration
cli.py            plan | native-qualify | lr-pilot | train | evaluate | interact | analyse
run_phase4.py     zero-argument tiered chain (WMI-launchable, resumable)
scripts/          synthetic_qualification.py (Phase 2 report)
tests/            154 offline tests (mask safety x14 + extras, arms, metrics, native parity,
                  and 99 cases ported from the superseded parallel implementation)
qualification/    Phase 2 and Phase 3 reports (committed evidence)
runs/             Phase 4 artifacts (gitignored: large; back up by hash)
```

## Run the offline checks

```powershell
cd experiments\clare_sparse_composition_v1
& C:\Users\clclc\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests
& C:\Users\clclc\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe -B cli.py plan
```
