# HANDOFF — Supa Long-Context Minimal Intervention Pilot

Written at a deliberate pause. **Everything below is a claim to be verified, not
a fact to be trusted.** The codebase and the JSON under `results/` are the only
authority. Where this document and the repository disagree, the repository wins.

---

## 1. Verified state at handoff

Checked by reading `results/*/run_manifest.json` at pause time:

| stage | status | canonical checkpoint unchanged |
|---|---|---|
| `phase0_conditions` | COMPLETE | true |
| `phase0_conditions_lcr_dev` | COMPLETE | true |
| `phase0_conditions_smoke` | COMPLETE | true |
| `phase0_eval` | COMPLETE | true |
| `phase0_eval_a11_ungated` | COMPLETE | true |
| `phase0_eval_a1_gated` | COMPLETE | true |
| `phase0_eval_a1_ungated` | COMPLETE | true |
| `phase0_eval_lcr_dev` | COMPLETE | true |
| `phase0_eval_smoke` | COMPLETE | true |
| `phase1_adapter_verification` | PASS | true |
| `phase1_smoke` | **partial — see below** | true |

Nothing is committed. `git status` shows the whole lab as untracked (84 files).

### Phase 1 smoke is INCOMPLETE

`results/phase1_smoke/` holds only `arm_A_*`. The session ended during Arm B.
Its `run_manifest.json` says COMPLETE, and **that is misleading** — see Known
Defect D1.

---

## 2. Known defects in the harness (fix before trusting new Phase 1 output)

**D1 — per-arm manifest collision.** Arms now run as separate processes
(`--arms A`, `--arms B`, `--arms C`) that all write `run_manifest.json`,
`started_manifest.json` and `resource.json` into the *same* stage directory.
Each overwrites the last, so the manifest describes only the final arm, and
`resource.json` under-reports total cost. Fix: write per-arm manifests
(`run_manifest_<arm>.json`) or give each arm its own subdirectory, and have
`--combine` aggregate resource totals.

**D2 — stale artefacts survive a re-run.** `results/phase1_smoke/summary.json`
predates the current code. Delete a stage directory before re-running it, or
make `Run.__enter__` refuse to start into a non-empty stage dir without an
explicit `--force`.

**D3 — noisy warning.** `training.py` does `float(out.loss)` on a tensor with
`requires_grad=True`. Harmless, but should be `float(out.loss.detach())`.

**D4 — `state_hash` changed mid-experiment.** It originally upcast to float32
and called `.numpy().tobytes()`; it now streams in native dtype. Digests are
comparable *within* a run (before vs after) but **not across** runs recorded
before and after that change. Do not compare stored hashes between stages.

---

## 3. What still has to be run

1. Finish the **Phase 1 smoke** — arms A, B, C, one process each, `--smoke`.
2. **Bounded Phase 1 on `a11`** (canonical fp16): arms A, B, C, `--max-steps 300`,
   then `--combine`.
3. **Bounded Phase 1 on `a1`** (bf16 variant): same, to answer whether the
   checkpoint changes the adapter's effect.
4. **Phase 2 only if the Phase 1 gate passes** (>= 2 absolute points on the
   long-context composite OR >= 5% relative error reduction, with preservation
   regression <= 1 point). A null or negative Phase 1 **stops the programme** —
   it does not license a layer sweep.
5. The **2x2 composition check is NOT licensed** and must not be run: it
   requires Phase 0 query selection to have won independently, and it lost.

Driver script (verify it matches the current CLI before using):
`<scratch>/run_phase1.sh`, or run the commands in `README.md` directly.

---

## 4. Claims to audit — do not take any of these on trust

Each is something the previous session asserted. Re-derive each from the
repository and the cached JSON. Several are known to be shaky.

### 4.1 Headline numbers
- [ ] Condition A on `phase0_eval` reproduces the published 45 / 125 / 170.
- [ ] `phase0_eval_a11_ungated` reproduces 41 / 121 / 162.
- [ ] `phase0_eval_a1_gated` reproduces 43 / 125 / 168.
- [ ] Every contrast quoted in `RESULT.md` matches `results/*/summary.json`.
      Recompute from `per_example.jsonl` rather than re-reading `summary.json`.

### 4.2 Design weaknesses already identified (confirm severity)
- [ ] **Format confound.** Condition A is the original context; B, C and D are
      chunk-joined with a blank-line gap marker and lose ~55% of tokens. A−B and
      A−D therefore conflate *content removal* with *format change*. Only
      comparisons among B, C and D are format-matched. Does any headline claim
      rest on A−B or A−D alone?
- [ ] **The oracle is answer-informed.** The gold answer string sits inside
      `supporting_facts` on ~90% of demo oracle-eligible rows and ~92% of LCR
      dev answerable rows. D is therefore not a clean evidence ceiling. Verify
      those percentages.
- [ ] **Selector was revised once** (anchor changed from a whole chunk to one
      sentence; budget 512→384; ceiling 768→576) after a CPU dry run, before any
      model ran. Confirm from `preregistrations/phase0_context_refinement.md`
      that this is disclosed, and judge whether it is a material researcher
      degree of freedom.
- [ ] **Preservation classes were chosen by probing the frozen model** and
      keeping the two it scored ~40% on, discarding classes where it scored 0%.
      This is selection on the baseline being measured. Is it disclosed, and
      does it bias the regression gate?
- [ ] **Multiple comparisons.** Two suites, four core/gate cells, several
      contrasts each, all on the same 256 rows with the same cached contexts.
      No correction is applied. Confirm the report says agreement across cells
      is robustness, not independent confirmation.
- [ ] **LCR dev has no published baseline**, so no validity gate. Its absolute
      level rests entirely on a scoring rule invented in this lab (below).

### 4.3 The LCR short-answer rule — highest-risk invented component
`slcmi/training_data.short_target` defines the target as the single
`must_include` span that does not appear in the question.
- [ ] Verify it resolves to exactly one span on all 6000 answerable rows.
- [ ] Verify the claim that strict scoring (32.7%) vs lenient "gold span appears
      anywhere" (34.7%) differ by only ~2 points on `phase0_eval_lcr_dev`.
- [ ] Judge independently: is this a fair rendering of the task, or does it
      depress LCR's absolute level enough to distort D−A?

### 4.4 Phase 1 premise — **not yet verified, and it matters**
- [ ] Supa-A1.1 was produced by merging the LoRA at
      `artifacts/halo_long_context_sft_v1/stage_a/checkpoint-200`
      (see `Rineya-Supa-A1.1-1.1.0-devfreeze/CORE.json`). **Determine what data
      that adapter was trained on.** If it was the LCR SFT corpus — the same
      corpus Phase 1 now trains on — then Phase 1 is continuing training on a
      distribution the checkpoint has already absorbed, and a null result would
      be uninformative about the *method* rather than evidence against late-layer
      adaptation. This was flagged in the previous session and never confirmed.
      It is the single most important open question for Phase 1's
      interpretability.
- [ ] Re-verify train/eval disjointness: 0 shared contexts, questions and titles
      between `lcr_sft_v1.jsonl` and the demo suite.
- [ ] Confirm the demo suite's `intended_use` ("do not train on this exact
      suite") is respected everywhere — it must appear only as an eval set.

### 4.5 Mechanical correctness
- [ ] `peft` 0.19.1 really restricts LoRA to layers 17/18/19 attention only.
      Re-run `scripts/phase1_verify_adapters.py` and read the receipt; do not
      trust the stored one.
- [ ] Layers 0, 1, 2 are never trainable, and `build_lora_config` raises if asked.
- [ ] The abstention gate is OFF during training and ON for every evaluation.
- [ ] `MAX_LENGTH = 2560` retains ~100% of the curriculum. Re-measure with the
      real tokenizer; an earlier value of 2048 silently dropped ~half.
- [ ] Gradient actually reaches the adapter (finite non-zero grad norms), and
      scaler-skipped steps are counted, not silently treated as trained.

---

## 5. Conclusions currently on the record

State of `RESULT.md` at pause. Both are **length-dependent**, and the second
already superseded an earlier over-general claim.

- **Primary suite (demo, ~1.3K tokens, validity-gated): interference NOT
  SUPPORTED.** B−A = −8.98 [−14.06, −3.91]; on answerable rows −15.62. B−C =
  +19.53 (selector works). D−A = −14.38. Robust across all four core/gate cells.
- **Confirmatory suite (LCR dev, ~2.4K tokens, exact oracle): INCONCLUSIVE.**
  On answerable rows B−A = −2.67 [−10.67, +5.33] and D−A = +5.33
  [−2.67, +13.33] — the oracle flips positive, but no interval excludes zero.
- **Cross-cutting:** with an exactly localised gold-evidence subset the model
  reaches only 38.0% on LCR dev. Perfect retrieval leaves 62% wrong.

An earlier draft asserted a flat "NOT SUPPORTED" from the primary suite alone.
That was withdrawn. Check `RESULT.md` still carries the withdrawal.

---

## 6. Hard constraints that remain in force

Read `README.md` for the full non-go list. The ones most easily violated:

- The canonical checkpoints are **read-only**. Both live only in the primary
  checkout at `../Chat-Bubble/Rineya_Published_Cores/`, not in this worktree.
- No pruning, no recurrence, no RoPE changes, no KV-cache work, no MoE, no full
  finetune, no training all 28 layers, no MLP training in Phase 1, no rank or
  layer or LR sweep, no second LLM, no large embedding model.
- Phase 2 compares only L17–19 against L17,18,19,25,27. No other layers.
- No claims of statistical significance on this corpus.

---

## 7. Environment

- Interpreter: `../Chat-Bubble/.venv/Scripts/python.exe` (torch 2.12.1+cu126,
  transformers 5.14.1, peft 0.19.1). The default `python` is CPU-only.
- GPU: RTX 4070 Laptop, 8 GB, **shared with other research**. Observed an 8x
  slowdown and one OOM when a neighbour held ~7 GB. Check free VRAM before
  starting, run one arm per process, and keep gradient checkpointing on.
- Run every command from the repository root.
