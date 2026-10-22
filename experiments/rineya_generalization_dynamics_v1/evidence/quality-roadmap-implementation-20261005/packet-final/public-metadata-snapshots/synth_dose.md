# Matched prospective training corpus — 2026-10-03

The CPU compiler milestone has advanced to a materialized, verified and
token-matched corpus. **No student has been trained and no transfer or retention
effect is established.** The research question and complete three-arm objective
remain open.

## Authoritative artifacts

The immutable source-policy 2 bundle is
[data/crossed_source2_v1_20261003](data/crossed_source2_v1_20261003).
Its [preparation receipt](data/crossed_source2_v1_20261003/preparation_receipt.json)
records before/after source/tokenizer hashes, code snapshots, cooperative CPU
budgets and closed native flags. A separate reviewer checked the saved files,
decoded all 640 arrays and independently recounted the schedules; see
[independent audit](evidence/crossed_v2_independent_audit.json).

| Measured prospective dose | M1 surface-varied | M2 condition-diverse |
| --- | ---: | ---: |
| Input tokens including prompt, target and EOS | 130,562 | 130,574 |
| Supervised target and EOS tokens | 5,614 | 5,615 |
| Scheduled sequences | 540 | 540 |
| Full 20-sequence optimizer groups | 27 | 27 |
| Shared enrolled source parents | 20 | 20 |
| R / S scheduled sequences | 192 / 348 | 192 / 348 |
| Registered candidate cells before repetition | 320 | 320 |
| Distinct candidate student messages+target | 284 | 320 |

The 12-token between-arm difference is below the frozen 512 limit. Both doses
are inside 130,560–131,072; R/S input shares differ from half by at most 512.
Parent exposure vectors and paired occurrence order match exactly. All four
renderings have equal positive exposure within each parent/condition. Overall
S cardinality and R first-request-depth histograms also match. No padding,
packing, target truncation, one-arm top-off or post-freeze parent replacement
was used. The exact 27-step learning-rate sequence still needs its native run
registration; no optimizer operation is implied by these schedule counts.

The control has the same source worlds in crossed surface templates and
invertible item reindexings. M2 changes actual condition worlds. S targets are
selected in original item space and then remapped; neither arm selects the
lexicographically first answer after permutation. Every target is accepted by
the independent visible-task verifier, which accepts correct alternatives too.

## Negative result retained

V1 is preserved under
[data/crossed_v1_blocked_20261003](data/crossed_v1_blocked_20261003), including
its original code, catalog, enrollment, 640 arrays and records. An exact integer
certificate proves that its shared weights and required marginal matches force
M2 to consume at least **736 extra input tokens**, above the fixed 512 limit.
See [diagnostic](evidence/matching_v1_constraint_diagnostic.json).

[Amendment 002](CORPUS_AMENDMENT_002.md) and
[source registration](SOURCE_POLICY_002.json) precede the V2 solver attempt.
They change the alternate R source chain so request rebinding changes the
endpoint while preserving depth. Internal shorter aliases are excluded.
All numerical, token, position and resource checks remain fixed. This is a new
frozen source catalog; it is not falsely labeled as V1 or used to erase its block.

## Limits and next work

Input-token matching is not full difficulty equivalence. Joint item-count/
cardinality/solution-count, requested-count/maximum-depth/total-hop, target-range
and per-update distributions remain curriculum differences. S cardinality
distributions differ in **27/27** individual updates; R first-depth distributions
differ in **26/27**, although their whole-run marginals match. The maximum
between-arm position residuals are 11/576 R answer slots and 22/1,108 S slots,
both below 2%; within-arm opportunity residuals satisfy the 4% bound.
Any later result compares these two frozen curricula, not diversity isolated
from every difficulty or ordering influence.

CPU preparation used 5.799 seconds and a 166,785,024-byte peak Windows working
set, under cooperative 600-second / 512-MiB limits. This is process working-set
evidence, not a hard containment guarantee or native process-tree fit. No model
framework, model, GPU, teacher, protected bank or training worker was used by
the preparation process.

The final [137-test receipt](evidence/crossed_corpus_final_tests.json) has zero
failures, errors or skips with exact-local integrations enabled. The last two
tests run a tiny random CPU model in a separate process for compute-helper
mechanics; no published weights or GPU are used. The main tokenizer/audit
process remains free of model frameworks.

The [native qualification proposal](NATIVE_QUALIFICATION_PROPOSAL.json) records
a current-runtime candidate: unchanged BF16 resident base parameters with FP32
compute/activations and the same FP32 q/v LoRA scope. The
[precision helper](native_precision.py) has CPU toy tests only. It does not
qualify the published checkpoint or GPU. A bounded owned loader/worker/supervisor,
exact runtime/dose freeze and physical tests remain to be implemented. Retain
the failed historical likelihood gate and its 0.03125/0.01 limits. Current
backend math/restore and sustained-fit qualification, fresh prior-correct
enrollment, language/long-context preservation and fresh atomic/RS evaluation
registration still precede three-arm training. Existing CHECK/FINAL stay closed.

## Reproduce safely

From the repository root, with existing pinned local tokenizer assets and
installed CPU dependencies:

```powershell
python experiments/rineya_synth_centralized_reasoning_v1/prepare_corpus.py --output another_new_leaf --source-policy 2
$env:SYNTH_CRS_LOCAL_TOKENIZER_DIR='C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Titan-Experimental-1.0.0-devfreeze'
python -m unittest discover -s experiments/rineya_synth_centralized_reasoning_v1/tests -v
```

Use a new direct package data child. Existing bundles cannot be overwritten or
reused. Without the explicit local-tokenizer environment, two integration tests
are skipped; that default run is not evidence that tokenization/matching passed.
No package installation, model download or native launch occurs automatically.
