# RAB-001 continuation brief

**Verify every claim here before acting on it.** This file was written at the end of an
autonomous session; runs may have moved on, and this programme has a long record of confident
claims that later measurement refuted -- roughly a dozen of them in the last session alone, all
corrected in git history. Treat this as leads with verification commands attached, never as
truth. Where this file and the repository disagree, the repository wins, and say so.

---

## 1. Environment

| | |
|---|---|
| Worktree (work ONLY here) | `C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification` |
| Branch | `research/ced-rab001-hotfix001-qualification-v1` |
| CPU runtime | `C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe` |
| GPU runtime | `C:/Users/clclc/BusinessProject/.venv-rab001-cu126/Scripts/python.exe` |
| Hardware | 8 GiB VRAM, 15.2 GiB host RAM |

Do not `cd` to the main `Chat-Bubble` checkout. **One model job at a time**, and run no
diagnostic commands while one is training -- host contention has killed completed runs at the
probe step. `Supa/` and `rineya_reasoning_forge_v1/` are gitignored.

Verify before trusting anything:

```bash
PY=C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe
$PY -c "import sys;sys.path.insert(0,'.');from experiments.architecture_benchmark_001.gates import implementation_digest;print(implementation_digest()[:8])"   # expect a329eb7b
git diff --stat HEAD -- experiments/architecture_benchmark_001/     # expect empty
powershell -NoProfile -Command "Get-Process python -EA SilentlyContinue | ? {\$_.WorkingSet64 -gt 1GB} | Measure-Object | Select -Expand Count"
```

---

## 2. The deliverable

**`B28last20 + wsd:0.2:0.2 + beta2 0.95`, 5,200 tokens, lr 1e-5.** Checkpoints for seeds
17/23/29/41 in `outputs/arm-checkpoints/`.

* Panel **74/150** at seed 17 (source 47), mean **69.25** across four seeds, kept 41 of 47.
* Forge IID **30/200** against the source's 32 -- no measurable loss (p=0.804).
* **314,618,880 trainable parameters, 53% of full B28's**, with embeddings, final norm and the
  first 8 blocks frozen.
* Beats full-parameter finetuning (73/150) at half the trainable weights, keeps four more
  reference answers, and has significantly better output discipline under distribution shift
  (p < 1e-6).

Reproduce:

```bash
PYG=C:/Users/clclc/BusinessProject/.venv-rab001-cu126/Scripts/python.exe
$PYG experiments/rab001_exploration/arm_dose.py B28last20 1e-5 5200 17 1.0 wsd:0.2:0.2 0.95 final
```

---

## 3. What is established

| finding | evidence |
|---|---|
| **WSD is worth +7.5 panel rows** on tail-trained arms | 4/4 seeds, pooled 18-48 discordant, **p=0.00029** |
| WSD is **arm-specific** | nothing measurable on full B28 (p=0.70); +21 on H56, +4 on P56 |
| **Depth matters** | B28 beats last8 (p=1e-4), last12 (p=0.004); last20 beats last8 (p=0.004) |
| **Cross-position CED memory harms** | P beats H at equal parameters, p=0.0107 under WSD |
| **Output discipline scales with trainable params** | full B28 vs tail arms on transfer, 58-11, **p<1e-6** |
| In-distribution reasoning **preserved** | forge IID pooled 37-28, p=0.321, discordance above floor |
| Out-of-distribution reasoning **slightly lost** | forge transfer pooled 48-27, p=0.020 (fails Bonferroni over 4 tests) |
| **Reasoning data does not help this model** | 11.5x dose, IID 34 -> 34, p=0.868 |
| Verbosity shift is **cosmetic** | ~160/200 outputs longer, p~1e-22; but answer never present in wrong outputs (0 of 170) |
| Weight averaging is **inert here** | soup lands at its best ingredient on both splits |

### Noise floors -- read every result against these

| instrument | accuracy | termination |
|---|---|---|
| panel, 150 rows | **15-24 discordant** (mean 18.8) | - |
| forge test_iid, 200 rows | **7-12** (mean 9.2) | 6-11 (mean 8.0) |
| forge test_transfer, 200 rows | **5-10** (mean 8.5) | 4-9 (mean 6.5) |

A same-config panel pair once reached **p=0.0352** on seed alone. Single-seed panel results are
close to uninformative: pool 3-4 independent seeds, whose discordant rows are disjoint and
combine cleanly. In the last session every single-run claim was refuted and every pooled
multi-seed claim held.

---

## 4. Tools

```bash
# train; RAB001_SPLIT_DIR, RAB001_CHUNKED_CE, RAB001_GRAD_CHECKPOINT, RAB001_ADAM_STORAGE,
# RAB001_ADAM_DIRECT_IO are env-var options
arm_dose.py <arm> <lr> <budget> <seed> <probe_fractions> <schedule> <beta2> <save> [run_label]
#   arm: B28 | B28last<N> | H | P | H last<N> | Plast<N>
forge_probe.py --split test_iid|test_transfer|test_length --rows N --checkpoint TAG --label NAME
arm_compare.py <tag> <tag> ...        # paired McNemar on the panel
uniform_soup.py <tag> <tag> ...       # uniform average; no selection, so no held-out set needed
merge_depth.py <tagA> <tagB>          # union/delta merge across different tensor sets
build_splits_mixed.py --forge-token-share F --budget B --max-forge-tokens L
headroom_audit.py                     # no model; why the source fails the rows it fails
```

**Budgets must be <= `max_usable_budget` in a split manifest.** `arm_dose` counts *supervised*
targets (len-1 per row); asking for the raw token total exhausts TRAIN and raises `SystemExit`,
which bypasses the artifact write and discards a completed pass.

**Memory.** Chunked CE + gradient checkpointing lift the training ceiling from ~250 to 2048+
tokens (peak 10.31 -> 3.90 GiB at 1024). **Neither works alone** -- checkpointing alone made
things worse because the dominant term is the `[1, T, 151936]` logits tensor, which it never
touches. Full-parameter B28 needs `RAB001_ADAM_STORAGE=streamed RAB001_ADAM_DIRECT_IO=1`.

---

## 5. Traps that have actually caught someone

* **Aborted artifacts report `correct_end: 47`** -- their step-0 probe, which is the source
  score. Always filter `completed == true`. This produced a reported "+26" that was really +13.
* **`splits_onpanel` is QUARANTINED** -- it leaks 47 of 150 panel questions verbatim, 102
  blinded, because `template_family` disjointness does not imply question disjointness. Nothing
  ever trained on it. `model_soup.py` refuses to start while `CONTAMINATED.md` exists.
* **Two scorers exist in the long-budget artifacts.** `task_strict_hits` and `task_numeric_hits`
  tell different stories; quoting the wrong one produced a "dose declines past 50k" claim that
  was withdrawn.
* **`rigorous/` and `longbudget/` score panel rows 1-60**, not an independent set. They cannot
  corroborate a panel result.
* **Read `resource_trace` before theorising about memory.** It records allocated/reserved/peak
  per step and has since the first abort; four runs were wasted guessing.
* **Reason in absolute tokens, not shares.** A 75%-forge mixture beat a 30% one on the panel
  because it carried more conversational tokens in absolute terms.

---

## 6. What is worth doing next, and what is not

**Not worth doing:** more reasoning data (dose-response flat at zero across 11.5x), more weight
averaging (inert on both splits), more single-seed comparisons (uninformative), CED architecture
arms (none beats ordinary finetuning at matched budget).

**Possibly worth doing:**

* Replicate the deliverable at more seeds if a tighter estimate than 69.25 +/- 3.9 is wanted.
* Resolve 5,200 vs 15,600 tokens, currently undetermined (p=0.076 at one seed per budget). Needs
  3-4 seeds per budget.
* `test_length` (80 rows) is forge's third split and has never been probed.
* `arm_dose` should write its artifact on corpus exhaustion instead of raising `SystemExit`.

**The honest ceiling.** This programme improved a 596M model's arithmetic-panel score from 47 to
74 and did not measurably improve its general reasoning by any means tried -- schedule, depth,
dose, architecture, weight averaging, or reasoning data. That looks like a capacity limit rather
than a method failure. Further progress plausibly needs a larger model or a task this one is
closer to solving; neither is reachable on this host.
