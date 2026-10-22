# Cheap cached arithmetic complementarity, v1

This study asks whether saved Supa, last20 WSD, full B28 WSD and published Opallo
answers solve different arithmetic problems. It reads nine fixed small files and
imports only the existing pure standard-library `retention.py` parser. No model,
weight, tensor library, GPU, protected split or probability capture is accessed.

`protocol.json` freezes all input byte hashes, row inventory, precision/ancestry
metadata, analyses and limits before the new analysis. The 150 problems are
historically exposed development evidence, with one distinct family per problem;
their target overlay, bare-value permission and correction-precedence human review
remain pending. Both strict and permissive typed policies are reported. Stored
historical correctness is cross-checked only after raw answers are rescored.

The four captures are explicitly different: source Supa at initialization, trained
last20 FP32, full-parameter B28 FP32, and published Opallo **BF16 storage upcast to
FP32 compute**. Opallo is the cast descendant of the same last20 seed-17 checkpoint,
not an independent training draw. The two training runs use the same draw, 5,233
supervised tokens and 41 steps. Weight hashes in provenance are historical metadata
declarations; the present study does not open or verify weight files.

Every pair and every nonempty subset gets both/only/neither counts, an oracle union,
headroom over its best individual model and joint failure IDs. **Every oracle union
is a target-informed ceiling.** It is not an evaluated selector, sampling gain,
merged model, population efficacy claim or current native runtime qualification.
No confidence calibration is possible from these answer-only records.

Source losses and gains stay separate. Each row retains raw answer text, EOS,
typed target, exact parsed value, rejection reason, family, class and template.
The report distinguishes format/type rejection from parsed value mismatch and
nontermination. A rejected free-form answer is not automatically a numerically
wrong belief; no matching number is scraped from prose. Pairwise typed-value
comparison ignores EOS only in its explicitly named mechanism diagnostic.

Run after experiment 1, with the hardware supervisor admitting one worker at a
time. The worker budget is **512 MiB peak RSS, 900 seconds and 50 MiB outputs**.
Windows phase receipts record current/peak process working set, free/total physical
RAM, memory load and commit use. The worker checks those phase boundaries; a root
supervisor is responsible for continuous admission, polling and termination.

From this worktree's repository root, use a fresh output directory beneath `runs/`:

```powershell
python -m unittest discover -s experiments/cheap_quality_complementarity_v1/tests -v
python -m experiments.cheap_quality_complementarity_v1 --out experiments/cheap_quality_complementarity_v1/runs/run-20260929-01
```

Tests are small synthetic algorithm/scorer checks; they do not run the real study.
Test temporary files also stay beneath ignored `runs/`. Existing output directories
are refused, so failures and interrupted attempts are preserved.

`report.html` is a portable readable report. `analysis.json` carries all ID sets,
hash bindings and historical cross-checks; `rows.csv` provides binary per-family
scores; `pairs.csv` and `subset_oracles.csv` expose pair/subset headroom. `RUN.json`,
`resources.jsonl` and `COMPLETED.json` or `FAILED.json` preserve runtime evidence.
Offline analysis completion opens no model gate and licenses no training/release.
