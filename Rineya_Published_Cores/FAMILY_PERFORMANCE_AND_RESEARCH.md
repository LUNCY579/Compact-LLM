# Rineya family: performance, construction, and research direction

Last evidence audit: 2026-08-27.

This is the protocol-aware technical report for the published Rineya family.
It separates measured checkpoint behavior from routed-system behavior, old
source-selection runs from standardized vault reruns, and completed results
from proposed research. The machine identities remain in [`CATALOG.json`](CATALOG.json)
and each `CORE.json`; this document explains what those records mean.

## Executive assessment

- **Best plain dense core:** Supa 1.7 at 40/64 Showcase, 121/192 Stability,
  161/256 combined.
- **Best routed family system:** Supa 2.0 System48 at 48/64, 128/192, 176/256,
  but it executes nine model arms and its Showcase score is selection data.
- **Best preregistered single-pass research gate:** S1 at 47/64, 130/192,
  177/256 on the BF16 base+adapter path. It is a logits processor, not a
  published plain-weight core, and **its 130 is not numerically robust** —
  FP32 and both dense merges give 128/192 with p = 0.12-0.14. The durable
  claim is +7 over the 121 baseline. See [`../artifacts/rineya_abstention_gate_v1/NUMERICAL_ROBUSTNESS.md`](../artifacts/rineya_abstention_gate_v1/NUMERICAL_ROBUSTNESS.md).
- **Best runnable one-directory abstention core:** **Supa-A1** at 43/64,
  125/192, 168/256. It uses one model execution and custom model code.
- **Best confirmed cache-compression configuration:** Halo `POOL13_SINK48`
  matches full-cache raw accuracy at 86/192 while retaining 0.293× persistent
  bytes and using 0.770× the measured local CPU wall time. The credited gain is
  structural pooling plus sink protection, not the Significance Matrix score.
- **Long-context SFT is incomplete:** Stage A stopped at checkpoint 250, has no
  final adapter, no CC0 validation/test result, and recorded NaN evaluation
  loss. The checkpoint-200 direction is useful, but it is not a completed
  long-context training result.

The current Demo evidence is controlled synthetic reading at approximately
1,119–1,556 input tokens under a 4,096-token cap. It does **not** establish
8K, 16K, 32K, or general long-document competence.

## 1. Evidence rules and protocol epochs

### 1.1 Current common protocol

The current vault baseline uses the actual published-core directories with BF16
loading, greedy decoding, seed 17, thinking disabled, 24 generated tokens, a
4,096-token context cap, and no truncation. Showcase64 was used repeatedly for
selection; Stability192 began as held-out validation but has now also been
inspected across many systems and bias sweeps. `All256` is their union and is
not an independent test.

Each scenario contributes four correlated rows: base, paraphrase,
counterfactual, and unanswerable. Reports should therefore include both row
counts and 16/48/64 scenario-pack summaries for Showcase/Stability/All256. No family core or system has run the
dual-skill sealed chat/RAG TEST.

### 1.2 Halo/Nimbus protocol refresh—not a transposition

Two result epochs are valid:

| Model | Historical source-selection run | Standardized 2026-08-26 vault rerun |
|---|---:|---:|
| Halo 1.2 | 30/64 | **29/64** |
| Nimbus 1.6 | 29/64 | **30/64** |

The historical and current own-model predictions agree on 61/64 rows for each
model, while cross-model agreement is only 47–51/64. The artifacts were not
swapped. Live `CORE.json`, `CATALOG.json`, and announcement copy use the newer
common protocol; older tables remain labeled as historical evidence.

## 2. Current long-context performance

### 2.1 Published cores and systems

| Item | Form / executions | Showcase64 | Stability192 | All256 | All-four packs |
|---|---:|---:|---:|---:|---:|
| Pulse 1.6 | dense / 1 | 31 | 89 | 120 | 0 |
| Aegis 1.1 | dense / 1 | 2 | 20 | 22 | 0 |
| Halo 1.2 | dense / 1 | 29 | 86 | 115 | 0 |
| Nimbus 1.6 (`Nimbus-A1` tracking alias) | dense / 1 | 30 | 83 | 113 | 0 |
| Lyra 1.4 | dense / 1 | 28 | 86 | 114 | 0 |
| Halo Uncensored 1.5 | dense / 1 | 31 | 96 | 127 | 0 |
| **Supa 1.7** | dense / 1 | **40** | **121** | **161** | 13 |
| Baya 1.7 | dense / 1 | 40 | 112 | 152 | 14 |
| Supa-Baya Dense Gate42 | 2 dense executions | 42 | 124 | 166 | 18 |
| Nimbus DualScale Gate45 | 2 PEFT executions | 45 | 124 | 169 | 19 |
| Baya 2.0 System46 | 3 executions | 46 | 125 | 171 | 21 |
| Supa 2.0 System48 | 9 executions | **48** | 128 | 176 | 21 |
| **Supa-A1** | 1 custom-code execution | 43 | **125** | **168** | 19 |
| S1 normalized gate (research) | 1 execution + processor | 47 | **130** | **177** | 25 |

Supa-A1 is the promoted baked checkpoint, so its **actual checkpoint** scores
are 43/64 and 125/192. The offline S4 gate scored 46/64 and 125/192; the
checkpoint must not inherit 46/64. BF16 LoRA folding changes two Showcase
predictions and the every-step implementation changes one more.

### 2.2 What the numbers show

- The long-context step-200 direction is the largest dense-core improvement:
  Nimbus 1.6 rises from 30/64 to Supa 1.7's 40/64 after PEFT transfer and dense
  materialization.
- Dense materialization is not exact. Supa's PEFT arm scores 42/64 while its
  BF16 dense form scores 40/64; 12–28% per-tensor relative delta-direction error
  was measured even though total delta magnitude was retained.
- System48 adds six Showcase hits over the 42-point dense gate, but needs nine
  executions. S1 and Supa-A1 show that most of the useful routing behavior is an
  abstention decision already visible in one model's first-step distribution.
- Supa-A1 reaches System46's 125/192 Stability score with one third of its model
  executions, but this is efficiency parity—not statistical superiority.
- Unanswerable handling remains the main gap. The oracle perfect-abstention
  ceiling on the `nimbus_low` arm is 159/192; S1 reaches 130 and System48 128.

### 2.3 Chat and grounded-RAG scope

The older 28-layer cores cluster near 0.860–0.867 on automated chat quality.
Aegis remains the measured grounded-RAG specialist: dual composite 0.8382,
oracle-supported 1.000, and real-supported 0.7358. Halo Uncensored is strongly
answer-biased: oracle abstention 0.000 with nine RAG hard failures.

Supa, Baya, the routed systems, and Supa-A1 have not received the complete
chat/RAG preservation evaluation. Their long-context gains must not be described
as general chat or grounded-RAG gains.

## 3. How each core was built

The common anchor is Qwen3-0.6B. The family changes its weights through low-rank
adapters, cross-depth block transfer, linear soups/task vectors, or custom
decode-time logic; it does not introduce a new transformer architecture.

### Pulse 1.6

Pulse places the 19 Phase-2B-kept decoder blocks into the official 28-layer
Qwen3 scaffold and blends those blocks toward the Halo donor using the
non-uniform `late_rineya` alpha shape. Alpha increases with depth and averages
0.5. Dropped blocks, embeddings, final norm, and tied language head remain from
official Qwen3. No new 28-layer fine-tuning occurred.

### Aegis 1.1

Aegis is the archived 19-layer Pulse 1.0 teacher plus a merged LoRA trained for
one epoch at learning rate `5e-4` over 16,410 examples: 12,000 short-response
rows and 4,410 grounded-RAG rows. It is the family's measured RAG specialist,
not a long-context winner.

### Halo 1.2

Halo uses the official 28-layer scaffold. Its 19 kept blocks are blended 50/50
with an equal FP32 delta soup of two Aegis-line adapters: v1 at `5e-4` and v2 at
`1e-4`. It was assembled through cross-depth weight transfer; the 28-layer
checkpoint was not directly fine-tuned.

### Nimbus 1.6 / Nimbus-A1 tracking alias

Nimbus is a uniform alpha-0.40 soup from official Qwen3 toward Halo Uncensored,
selected by the layerwise-soup lab (`cand_d1b08a8faf299ce4`). `Nimbus-A1` is a
tracking alias for this actual Nimbus lineage. It does not refer to the
weight-identical no-bias Supa control.

### Lyra 1.4

Lyra is another 50/50 cross-depth stitch, using an equal soup of v1 adapters
trained at `5e-4` and `1e-5`. It preserves strong automated chat quality but has
not surpassed the newer long-context line.

### Halo Uncensored 1.5

Halo Uncensored uses the same donor direction as Halo but fully replaces the 19
kept blocks (`alpha=1.0`). “Uncensored” here names an experimental answer-biased
pair, not a safety certification. Its zero oracle-abstention score is a material
limitation.

### Supa 1.7 and Baya 1.7

Both apply the Halo-trained Stage-A checkpoint-200 LoRA to Nimbus, even though
the adapter was not trained on Nimbus:

- Supa: scale 0.8125, FP32 merge, BF16 save.
- Baya: scale 1.25, FP32 merge, BF16 save.

They are two points along one transferred task-vector direction, not separate
fine-tuning runs. Supa is the stronger dense checkpoint on held-out Stability.

### Supa-A1

Supa-A1 has the exact Supa 1.7 weight SHA-256
`962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b`.
Its behavior changes through `config.json` and `modeling_abstain.py`, which add
5.156249985098839 to vocabulary token 51378 (`UNKNOWN`) before greedy argmax.

For the S4 signal,

```text
logP(UNKNOWN) - logP(top1) = z_UNKNOWN - z_top1
```

because the shared softmax partition function cancels. A constant logit bias is
therefore exactly the tuned two-logit decision under greedy decoding. Loading
requires `trust_remote_code=True`. The current class applies the bias at every
decode step, while the selected gate was first-token-only; no mid-answer
`UNKNOWN` occurred in the 256 evaluated rows.

## 4. Abstention findings and statistical interpretation

### 4.1 What signal worked

Generic uncertainty signals—first-token entropy, top-1/top-2 margin, answer
mean entropy, and answer log probability—did not improve Showcase accuracy.
The model is often confidently wrong. Direct probability mass on the explicit
`UNKNOWN` surfaces was predictive, with approximately 0.8 AUC across splits.

S1 thresholds normalized mass over six one-token `UNKNOWN` variants. It improves
`nimbus_low` from 121 to 130 on Stability, raising abstention recall from 0.208
to 0.479 while reducing precision from 1.000 to 0.719. This is a deliberate
risk/coverage trade, not free accuracy.

### 4.2 Row and pack statistics

The preregistered analysis used row-level exact McNemar: S1 vs `nimbus_low`
has 13 gains, four losses, and p=0.0490. Three gates reached Stability; under a
Bonferroni threshold of 0.0167 none passes, although S1 was the sole declared
primary endpoint.

**That p-value is not numerically robust.** Re-running the identical weights,
adapter and frozen gate at FP32 — strictly more accurate than the BF16 protocol
— gives 128/192 and p=0.1435. Both dense merges also give 128/192 (p=0.1185
BF16 store, p=0.1435 FP16 store). The ungated model is 121/192 in all four
realisations, so the *model* is numerically stable and the *gate* is not: an
argmax decision has margin, a threshold decision has none. The supported claim
is a stable +7 on Stability; statistical significance is not supported. Full
table in [`../artifacts/rineya_abstention_gate_v1/NUMERICAL_ROBUSTNESS.md`](../artifacts/rineya_abstention_gate_v1/NUMERICAL_ROBUSTNESS.md).

A post-outcome sensitivity audit instead used the 48 scenario packs:

| Comparison vs `nimbus_low` | Packs better / worse / tied | Exact sign p |
|---|---:|---:|
| S1 | 12 / 2 / 34 | **0.012939** |
| S2 exploratory | 12 / 2 / 34 | **0.012939** |
| S3 | 4 / 0 / 44 | 0.125 |
| Supa-A1 / baked S4 | 4 / 0 / 44 | 0.125 |
| System46 | 5 / 1 / 42 | 0.218750 |
| System48 | 8 / 2 / 38 | 0.109375 |

This pack audit is a sensitivity analysis, not retroactive confirmatory
evidence. S1 is not superior to System46 (pack p=0.1094) or System48
(p=0.5488). Supa-A1's improvement over its no-bias control is also not
significant.

### 4.3 Threshold-selection warning

Four bakeable biases tie at 46/64 on Showcase but land between 125 and 129 on
Stability. The post-hoc Stability optimum of 133/192 at bias 8.28 is a ceiling,
not a valid selected-model result. A new pack-disjoint calibration split is
required before changing Supa-A1's bias.

## 5. Long-context fine-tuning status

The Halo long-context SFT lab uses QLoRA: 4-bit NF4 with double quantization,
BF16 compute, rank 16, alpha 32, dropout 0.05, all attention and MLP projection
modules, and gradient checkpointing. It trains 10,092,544 of 606,142,464
parameters (1.665%).

The prepared mix contains 30,811 long rows plus 2,950 replay rows. Replay is
8.74%, not the intended 20%, because the short-response pool was exhausted.
Only proxy 2K/4K strata are present; Stage B 8K and Stage C 16K are unrun stubs.

Current training state:

- latest checkpoint: step 250;
- configured/resumed maximum: 800 steps;
- recorded epoch: 0.05924 (a full epoch would be about 4,221 steps);
- no final `stage_a/adapter` or training manifest;
- no CC0 validation, CC0 test, or sealed result;
- `eval_loss` is NaN at steps 100 and 200;
- the durable config does not record the resume overrides that produced the
  800-step / save-25 / eval-100 schedule.

The collator emits all `-100` labels when truncation removes the assistant
target. Because corpus “4K” lengths use proxy tokens rather than exact Qwen
tokens, some rows may contribute no supervised loss. Exact tokenizer auditing
and all-masked-batch rejection are required before resuming.

Demo-only checkpoint screening found Halo+step200 at 38/64, step225 at 35/64,
and step250 at 36/64. Transferring step200 to Nimbus produced 42/64 in PEFT
form. Step200 is therefore a useful early task vector—not proof that Stage A
completed or that CC0 performance improved.

## 6. KV-cache compression evidence

The Significance Matrix lab uses a manual chunked runtime, not the public HF
generation runner. Its B0 numbers must not be mixed with the standardized vault
table unless the runner is named.

On held-out Stability192 for Halo:

| Configuration | Hits | Persistent cache | Interpretation |
|---|---:|---:|---|
| B0 full cache | 86/192 | 155.1 MB | quality reference |
| shipped `POOL1` | 59/192 | 45.4 MB | compression deficit |
| **`POOL13_SINK48`** | **86/192** | **45.4 MB** | same raw count at 0.293× bytes; gap not detected at 48-pack power |

`POOL13_SINK48` differs from B0's predictions even though the total hits match,
so “recovers exactly the same cases” would be false. Against `POOL1`, the
predeclared cluster test gives p=0.00088.

Specificity controls score real 86, position-only 81, shuffled-score 71, and
random 62. Real versus position is not detected as different (p=1.0), so the
Significance Matrix score itself is not credited. The supported mechanism is
structural pooling plus protecting prompt/sink tokens.

A later 13-cell Showcase-only frontier screen suggests budget 192 as the next
knee: 28/64 at 40.34 MB for fixed-48 sinks. It has only 16 packs, many compared
cells, and no held-out confirmation. Only Halo has validated pooled compression;
family-wide compression transfer remains unmeasured.

## 7. Permitted claims and current limits

Safe claims:

- Supa-A1 matches System46's previously inspected Stability raw count with one model execution.
- S1 is the strongest preregistered primary single-pass gate at this operating point; exploratory S2 has the higher raw count (131/192).
- The family has a confirmed structural KV-compression recipe on Halo.
- Supa/Baya improvements arise from transferred early Stage-A LoRA geometry,
  not a completed long-context fine-tune.

Claims not supported:

- general 8K/16K/32K long-context competence;
- superiority of S1 or Supa-A1 over the routed systems;
- a completed or converged Stage-A model;
- Significance Matrix scoring as the cause of the pooling gain;
- general chat/RAG preservation for Supa, Baya, systems, or Supa-A1;
- a clean sealed-test result for any published core.

## 8. Evidence-first roadmap for stronger and cheaper models

Everything below is a proposal unless explicitly tied to a measured result.

### 8.1 Repair evaluation before optimizing again

Create a new scenario-pack-disjoint calibration set and a second untouched final
test. Generate actual-Qwen-token 2K/4K/8K/16K cases, report pack-level inference,
and add small external checks from [RULER](https://arxiv.org/abs/2404.06654)
and [LongBench](https://aclanthology.org/2024.acl-long.172/). This prevents more
selection on Showcase and repeated adaptation to Stability.

### 8.2 Calibrate abstention to explicit risk

Choose the logit bias on the new calibration split using a predeclared cost or
risk limit, not “fewest abstentions among equal-accuracy thresholds.” Evaluate
reliability, Brier/ECE, risk-coverage, answerable accuracy, and unanswerable
recall under changed class prevalence. Useful foundations are
[temperature scaling](https://proceedings.mlr.press/v70/guo17a.html) and
[Conformal Risk Control](https://research.google/pubs/conformal-risk-control/).
Conformal guarantees require exchangeability with the deployment population;
they do not survive arbitrary mixture shift.

### 8.3 Train one canonical abstention surface

Canonicalize all refusal targets to the single token `UNKNOWN`, then run
pack-balanced, refusal-aware QLoRA with an answerable-preservation loss. This
could internalize the strongest direct-logit signal and remove custom-code
serving. [R-Tuning](https://aclanthology.org/2024.naacl-long.394/) is the closest
established recipe. [Abstain-R1](https://aclanthology.org/2026.findings-acl.985/)
is a longer-term verified-reward direction, but the repository currently lacks
RLVR infrastructure and should not jump there before calibration/SFT baselines.

### 8.4 Simplify the compressed runtime

Build a position-pooled baseline with dynamic protection for the system prompt,
question delimiter, and recent suffix instead of a hard-coded 48-token sink.
Confirm budget 192 on held-out packs. Remove the Significance Matrix state and
scoring work if the simpler baseline passes parity and quality gates. This is
consistent with [SnapKV](https://arxiv.org/abs/2404.14469) and the 2026 preprint
[Protection Is (Nearly) All You Need](https://arxiv.org/abs/2605.18053).

### 8.5 Add orthogonal KV quantization

First test full-cache quantization to isolate quantization error, then combine it
with the confirmed structural selector. [KIVI](https://arxiv.org/abs/2402.02750)
uses per-channel keys, per-token values, and a recent full-precision residual.
Transformers also documents an official
[`QuantizedCache`](https://huggingface.co/docs/transformers/en/kv_cache).
Qwen3/GQA compatibility and the manual chunked path are untested locally;
memory savings may worsen latency at current short lengths.

### 8.6 Resume SFT only after eliminating silent target loss

Count exact Qwen tokens, reject/repack truncated targets, prevent all-masked
batches, record the actual resume command, and score checkpoints 200/225/250 on
CC0 validation/test. Restore the intended replay proportion through weighted
sampling rather than exhausting a small replay pool.

For genuine 8K/16K work, test exact
[FlashAttention-2](https://openreview.net/forum?id=mZn2Xyh9Ec) first. If memory
still blocks training, isolate [LongLoRA](https://openreview.net/forum?id=6PmJoRfdaK)
as a separate experiment: its shifted sparse training attention and trainable
embeddings/norms change more than the current LoRA recipe and require a new
merge/evaluation plan.

### 8.7 Scale the anchor only after protocol repair

Benchmark untouched Qwen3-1.7B under the exact Rineya protocols. The official
[Qwen3 family](https://qwenlm.github.io/blog/qwen3/) offers a larger 28-layer,
16-query/8-KV-head anchor. If its single-pass ceiling justifies the hardware
cost, port the corpus and training method—not incompatible 0.6B deltas. This is
unmeasured and must not be advertised as a Rineya capability yet.

## 9. Technology references

- [LoRA](https://openreview.net/forum?id=nZeVKeeFYf9)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [Model Soups](https://proceedings.mlr.press/v162/wortsman22a.html)
- [Task Arithmetic](https://openreview.net/forum?id=6t0Kwf8-jrj)

These sources explain the general techniques. Rineya's exact recipes, tensor
hashes, adapter scales, and measured results are local claims recorded in this
vault; the cited papers do not validate those local outcomes.

## 10. Primary local evidence

- [`../comparison.md`](../comparison.md): common-protocol and historical tables.
- [`../lineage.md`](../lineage.md): exact family build history.
- [`MODEL_REGISTRY.md`](MODEL_REGISTRY.md): weight and dependency inventory.
- [`../Rineya_Long_Context_Weight_Geometry_Lab_v1/PUBLISHED_FAMILY_SEARCH_REPORT.md`](../Rineya_Long_Context_Weight_Geometry_Lab_v1/PUBLISHED_FAMILY_SEARCH_REPORT.md): dense and routed search.
- [`../Rineya_Abstention_Gate_Lab_v1/README.md`](../Rineya_Abstention_Gate_Lab_v1/README.md): gate implementation and reproduction.
- `artifacts/rineya_abstention_gate_v1/{RESULTS.md,CHECKPOINT_RESULTS.md}`:
  frozen local abstention reports (gitignored artifacts, mirrored into Supa-A1).
- [`../rineya_significance_native_architecture_lab/VALIDATION_RESULT.md`](../rineya_significance_native_architecture_lab/VALIDATION_RESULT.md): cache-compression validation.
- Local `rineya_halo_long_context_sft_v1/README.md`: partial SFT lab and training
  plan (pre-existing untracked worktree content, intentionally not staged by
  this publication).
