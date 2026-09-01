# Cross-Scale v3.1 — The Donor Side

Protocol: `amendment_v3_1_donor_side.md` version `3.1.0-diagnostic`

Parents: v1 `preregistration.md` `1.0.0`, `amendment_v2_exploratory.md`
`2.0.0-exploratory`, `amendment_v3_transport_diagnosis.md` `3.0.0-diagnostic`.
All unchanged.

Sealed holdout: **not opened.** DEV-FULL: **not run.** Candidate checkpoints
built: **zero.**

## Abstract

v3 showed that no operator this study had run could transport a direction, and
withdrew the cross-donor agreement statistic that motivated the line. It could
not say whether cross-scale transfer was possible, because the donor checkpoints
were absent. The operator authorised a pinned re-download, and this stage
answers the question.

The answer is a clean split, and it moves the failure one level deeper.

**The basis-correspondence problem is solved.** At matched relative depth, donor
and target residual streams share strong prompt-specific structure: linear CKA
0.82–0.96 against prompt-shuffled controls of 0.03–0.21. A per-depth ridge map
carries donor representations into the target basis at held-out mean cosine
0.87–0.98 on a prompt-disjoint split. v1's only basis map was fitted on the
vocabulary embeddings and applied at every depth; it reached 0.335. Fitting per
depth is the difference between failure and near-perfect linear correspondence.

**But the map is not low-rank, and the donor is redundant.** Rank 4/8/16 give
held-out R² of 0.03–0.33 against 0.77–0.95 for the full map, so the "deliberately
tiny translator" the brief hoped for does not exist. And Qwen3-0.6B alone already
predicts Supa's representations at R² 0.946–0.985; adding donor features makes
held-out prediction slightly *worse*. High CKA meant redundancy, not extra
information.

## What was run

| ID | Experiment | Status |
|---|---|---|
| Gate 0 | Streaming forward validated against transformers | PASS, 2.8e-06 |
| E5 | Cross-scale functional alignment atlas | complete |
| E6 | Low-rank functional translator | complete |
| E6b | Ridge-penalty robustness sweep on E6's incremental test | complete |
| E7 | Answer-likelihood capability check | complete |

### Provenance

Both donors were re-downloaded at the revisions already recorded in
`01_environment/huggingface_cache_verification.md`, and every weight shard's
SHA-256 was verified against `02_architecture/qwen4b.json` and `qwen8b.json`
before any measurement:

- `Qwen/Qwen3-4B` `1cfa9a7208912126459214e8b04321603b3df60c` — 3/3 shards match.
- `Qwen/Qwen3-8B` `b968826d9c46dd6066d109eabc6255188de91218` — 5/5 shards match.

### Gate 0 — the forward pass had to be trusted first

Qwen3-8B is 15.26 GiB in bfloat16 on a host with 15.2 GiB of RAM and a CPU-only
Torch build. No donor could be loaded, so every donor measurement runs through a
hand-written streaming forward: memory-mapped shards, one weight tensor
materialised at a time, FP32 upcast, freed before the next read.

A hand-written Qwen3 forward can be subtly wrong — grouped-query attention,
per-head `q_norm`/`k_norm`, RoPE at `theta` 1e6 — so the amendment made
validation a gate. Against `transformers`' own implementation on Qwen3-0.6B,
worst relative error **2.8e-06** against a 1e-3 tolerance, on both the
per-prompt and batched paths, including a 2237-token prompt.

The gate earned its place on the first run. It caught that `transformers`
applies the final RMSNorm to its last `hidden_states` entry, so
`hidden_states[num_layers]` is `norm(last layer)` while every earlier entry is
raw. Layers 0–21 already agreed to 1.6e-06; only the final slot was wrong. Left
uncaught, every donor-to-target comparison at the deepest slot would have been
between differently-normalised quantities.

### E5 — donors and target share representations

Matched relative depth, linear CKA against a prompt-shuffled control:

| pair | CKA range | shuffled control | lift |
|---|---|---|---|
| Qwen3-4B → Qwen3-0.6B | 0.87–0.96 | 0.03–0.22 | +0.69 to +0.93 |
| Qwen3-4B → Supa | 0.82–0.96 | 0.03–0.21 | +0.67 to +0.93 |
| Qwen3-8B → Qwen3-0.6B | 0.85–0.90 | 0.04–0.19 | +0.67 to +0.86 |
| Qwen3-8B → Supa | 0.85–0.90 | 0.04–0.19 | +0.67 to +0.86 |
| Qwen3-4B ↔ Qwen3-8B | 0.91–0.95 | 0.05–0.21 | +0.71 to +0.88 |

The frozen threshold was +0.10. RSA agrees: 0.59–0.92 against a shuffled control
at approximately zero.

The contrast with v3 is the point. The same donors, transported into the
target's parameter basis by P4, aligned with Supa's own adaptation direction at
cosine **+0.0002**. In representation space at matched depth they align at CKA
**~0.9**. The information was never absent; the operator could not carry it.

Two further results:

- The depth surface is diagonal-dominant — donor layer 9 pairs with target layer
  7, 18 with 14, 26 with 21, 35 with 27. Uniform relative-depth mapping is
  functionally correct, which v1 assumed and never verified.
- The donor consensus that turned out to be operator-induced in parameter space
  is **real in function space**: Qwen3-4B and Qwen3-8B reach CKA 0.91–0.95 with
  each other against controls at 0.05–0.21.

Logit agreement is much weaker (cosine 0.34–0.55, top-1 0.39–0.73), so shared
representational geometry does not imply shared output behaviour.

### E6 — the map works; the donor does not add anything

9,013 token positions, 40 training prompts, 19 held-out prompts, prompt-disjoint.

**Can a linear map carry donor representations into the target basis?** Yes.

| statistic | range across depths and donors |
|---|---|
| ridge held-out R² | 0.774 – 0.950 |
| ridge held-out mean cosine | 0.873 – **0.982** |
| orthogonal Procrustes cosine | 0.777 – 0.932 |
| token-shuffled control R² | −0.19 to −0.98 |
| v1 embedding-fitted baseline | 0.335 ridge; 0.536 / 0.275 Procrustes |

The control is negative throughout, so this is genuine per-token structure and
not matched marginals.

**Is it low-rank?** No.

| rank | held-out R² |
|---|---|
| 4 | 0.01 – 0.19 |
| 8 | 0.03 – 0.21 |
| 16 | 0.07 – 0.33 |
| full | 0.77 – 0.95 |

The brief proposed "A is deliberately tiny: rank 4 / 8 / 16." The measurement
says the donor-to-target correspondence is high-dimensional. A rank-8 map
recovers a small fraction of it.

**Does the donor explain anything the base does not?** No.

| donor | base-only R² | + donor | + random | donor gain | gain over control |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B | 0.946 – 0.985 | 0.941 – 0.977 | 0.914 – 0.982 | −0.004 to −0.008 | −0.005 to +0.026 |
| Qwen3-8B | 0.946 – 0.985 | 0.934 – 0.974 | 0.856 – 0.976 | −0.007 to −0.016 | −0.002 to +0.078 |

Qwen3-0.6B alone predicts Supa's representations at R² up to 0.985. Adding donor
features consistently makes held-out prediction *worse*, although less so than
adding the same number of random features. The donor is better than noise and
worse than nothing.

**Limitation, stated plainly.** This tests whether the donor adds *linear*
predictive power about *Supa's representations*. Supa is a finetune of
Qwen3-0.6B, so the base is a very strong predictor almost by construction, and
the headroom being competed for is only 1.5–5.4%. The result establishes linear
redundancy of donor residual streams with target residual streams at matched
depth. It does not establish that the donors carry no capability-relevant
information under a non-linear readout, at unmatched depths, or at token
positions other than those sampled. E7 addresses the capability question
directly rather than through this proxy.

### E6b — the incremental result is not a regularisation artifact

E6's joint fit has roughly 6,000 training rows against 1,024 + 4,096 columns,
close to rank-deficient. A donor block that genuinely helped could still hurt
held-out prediction there purely through variance, which would make E6's central
conclusion an artifact. Sweeping the ridge penalty over four orders of magnitude
settles it — but only if the criterion is chosen correctly.

Asking "does the donor help at *some* penalty?" flags E6 as refuted: at
`alpha` 100 the donor increment reaches **+0.236**. That is not evidence. At that
penalty base-only R² has collapsed from 0.959 to 0.357, so the donor is patching
an over-shrunk base with correlated columns rather than supplying anything new.

The correct test gives each model its own optimal penalty:

| donor | depth | best base only | best base + donor | delta |
|---|---:|---:|---:|---:|
| Qwen3-4B | 2 / 3 / 4 | 0.9462 / 0.9594 / 0.9547 | 0.9406 / 0.9514 / 0.9504 | −0.006 / −0.008 / −0.004 |
| Qwen3-8B | 2 / 3 / 4 | 0.9462 / 0.9594 / 0.9547 | 0.9362 / 0.9433 / 0.9477 | −0.010 / −0.016 / −0.007 |

The donor never raises the best achievable held-out fit at any depth or penalty.
**E6 stands.**

### E7 — is there a capability gap to transfer at all?

If the donors are not actually better than Supa on this bank, the negative result
is not about transport. Teacher-forced log-probability of the reference answer,
per answer token, 55 of 59 calibration cases.

| model | mean | median | 20% trimmed mean | cases beating Supa | sign test |
|---|---:|---:|---:|---:|---:|
| Supa | −1.68 | −0.11 | −0.98 | — | — |
| Qwen3-8B | −4.21 | −0.00 | −2.44 | **37/55** | **p = 0.015** |
| Qwen3-0.6B | −4.17 | −2.96 | −3.93 | 11/55 | n.s. |
| Qwen3-4B | −7.16 | −4.86 | −5.79 | 23/55 | n.s. |

The metric behaves: Supa beats its own base by 2.49 nats per token on 44 of 55
cases, which is what a finetune specialised for this bank should look like.

**Qwen3-8B beats Supa on 37 of 55 cases, sign test p = 0.015.** So a real
capability advantage exists. But its shape matters more than its existence.
By family:

| family | Qwen3-8B vs Supa | mean delta |
|---|---:|---:|
| counterfactual reasoning | 4/4 | +0.92 |
| general reasoning | 4/4 | +0.19 |
| grounded QA / RAG, instruction following, multi-hop, retrieval, abstention | 4/4 each | +0.00 to +0.02 |
| paraphrase robustness | 10/12 | −0.24 |
| mathematics | 0/4 | −3.47 |
| science | 0/4 | −6.05 |
| common sense | 0/4 | −10.04 |
| short response | 0/4 | −15.67 |

The donor wins broadly and by almost nothing — often under 0.02 nats — and loses
catastrophically on the four families that demand a terse, format-specific
answer. Supa assigns high probability to "it dries"; an untuned 8B would rather
write a sentence. That is a formatting and calibration gap, not a knowledge gap,
and it runs in the target's favour.

Qwen3-4B is worse than Qwen3-0.6B on both mean and median, which is direct
evidence for v1's H7: donor size is not monotonically useful here.

**Caveat.** Teacher-forced answer likelihood is not task accuracy under greedy
generation, and the four `constraints` cases have no single reference answer and
were excluded. This bounds the claim to "probability assigned to the reference
answer", not "would score higher on the bank".

## Resource consumption

| Quantity | Value |
|---|---:|
| GPU hours | 0.0 (CUDA unavailable) |
| Peak concurrent models in memory | **0 donors, 1 target** |
| Qwen3-8B forward, wall time | 387 s |
| Qwen3-8B forward, process RSS | **0.33 GiB** against a 15.26 GiB checkpoint |
| Qwen3-4B forward, process RSS | 0.39 GiB against 7.49 GiB |
| Donor download | 22.75 GiB, all 8 shards hash-verified |
| Disk traffic avoided by loop inversion | ~1.34 TiB |
| Candidate checkpoints created | **0** |

The hardware constraint forced the method rather than merely shaping it. A
per-prompt streaming forward would have re-read each checkpoint 59 times — 442
GiB for the 4B and 900 GiB for the 8B. Inverting the loop so layers are outermost
reads each weight exactly once, at the cost of holding all 59 residual streams
simultaneously, which is 148 MiB.

## Where the cross-scale hypothesis now stands

The failure has moved twice, and it is worth being precise about where it sits.

- **v1/v2 believed:** the merge operators were insufficient.
- **v3 established:** every operator tested transmitted ~275 effective scalars
  and no directions, and the evidence for donor consensus did not survive a
  shape-matched null. The failure was the transport operator.
- **v3.1 establishes:** transport is *not* the barrier. A per-depth linear map
  achieves held-out cosine 0.98. The barrier is that the donors' representations,
  once correctly aligned, are redundant with the target's own (E6, robust under
  E6b), and the donors' remaining capability advantage is real but tiny (E7).

That is outcome **D**, but with a mechanism opposite to the one the label
suggests. "Donor consensus is scale/basis-specific and non-portable" is wrong:
it *is* portable, at cosine 0.98. What is missing is a difference worth porting.

Outcome **B** — "donor functional representations contain information that can
guide a successful target-space update" — is not supported. The functional
channel exists and is excellent; there is little distinctive signal to send
through it.

The one qualification worth preserving: E7 found a statistically significant
donor advantage (Qwen3-8B, 37/55, p = 0.015) concentrated in counterfactual and
general reasoning. It is small in magnitude and swamped by format-driven losses
elsewhere. That does not support a whole-model transfer, and v3 already showed
whole-model transfer produces a functionally unchanged Supa. It would at most
support a narrowly targeted study of those two families — which is a different
research question from the one this line has been asking, and should be
preregistered as such rather than folded in here.

## Escalation

- **DEV-FULL: not justified.** No candidate was built, and v3 already showed the
  existing candidates are functionally identical to Supa.
- **Sealed: remains CLOSED.** No frozen finalist packet exists. The sealed JSONL
  was not read at any point in v3 or v3.1.
- **Recovery LoRA: not justified.** Reserved for the strongest frozen
  training-free candidate; there is none.

## Corrections and scope notes

- E7 reads calibration reference answers, where E4–E6 read prompts only.
  Calibration is development data under the parent preregistration. The sealed
  holdout and Invisible-Limit TEST2 remain untouched.
- E5's embedding depth slot is identically zero because every prompt ends with
  the same chat-template token, leaving that slot no cross-prompt variance. The
  same degeneracy was noted in v3 E4. The layer slots are unaffected.
- E6's target depths are computed from the same relative-depth formula as the
  donors' (layers 0/7/14/20/27 of 28), which differs by one layer from the
  hard-coded slots used in v3 E4 (0/7/14/21/27). E6 is internally consistent;
  the two stages should not be compared slot-for-slot at the 75% depth.
