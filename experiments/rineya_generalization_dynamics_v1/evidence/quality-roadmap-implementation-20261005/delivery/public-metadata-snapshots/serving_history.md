# Evidence reconciliation and M1 handoff — 2026-10-05

This addendum is authoritative for the current decision where historical summaries
disagree. It preserves every historical receipt, permit and source file. Scope was
offline inspection, existing prediction reconciliation, synthetic fixtures and
preparation of one reviewed native proposal. No native model was executed, no
training or bank read occurred, and no serving prompt, bias or guard changed.

Owner: `research/supa-a1.1-nonrecurrent-adaptation-v1` at
`49212058fe2d0628227f0d3a48c70e474dd449e4`, matching the remote branch when inspected.
The checkout began clean. No competing owner chat or adapter process was found;
unrelated Python jobs were left alone. Read-only subaudits remained in this owner
lineage. The receiving M1 checkout was clean at
`3e8a3ec03881ad1ea948db6c1a62c12f73da798f`; it was not modified.

## Applied policy and endpoint ledger

Live serving policy is **3.75**, authorized 2026-09-08 at the serving layer. The
checkpoint configuration stays **5.156249985098839**. `A_deployed` in historical
prompt experiments names the prompt arm; it does not establish that the complete
served configuration was evaluated.

TRANSFER-3 has 1,920 complete unique `(policy,id)` predictions per arm: 960 items
at each of two biases. Answerable numbers below use the numeric **FINAL-VALUE**
endpoint, otherwise exact; unanswerable numbers use normalized exact UNKNOWN.
These are development outcomes, not efficacy or release qualification.

| Applied bias | Arm | Answerable /720 | Unanswerable /240 | Reference loss |
|---|---|---:|---:|---:|
| 0 | frozen | 193 | 159 | 0.823125 |
| 0 | matched_mlp | 204 | 240 | 0.483750 |
| 0 | gate_only | 187 | 173 | 0.771875 |
| 5.156249985098839 | frozen | 109 | 236 | 0.5890625 |
| 5.156249985098839 | matched_mlp | 202 | 240 | 0.485625 |
| 5.156249985098839 | gate_only | 75 | 237 | 0.616875 |
| **3.75** | **all three** | **unmeasured** | **unmeasured** | **unmeasured** |

Reference loss is `(27/40)*(1-answerable_accuracy) +
(39/40)*(1-unanswerable_accuracy)`: lambda=3, pi=13/40. Bank prevalence is 1/4.
This is an inherited benchmark proxy, not measured production economics or family
mix. Relative reductions for matched_mlp are **41.23% at bias 0** and **17.56% at
canonical bias**. Neither is a measured reduction at 3.75. Historical TRANSFER-1's
71% was cross-generator within the same 176-word vocabulary, not lexical transfer.

The +11 bias-0 answerable delta has **54 gains and 43 losses**, p=0.30993. The +93
canonical delta has **99 gains and 6 losses**, p=8.4353e-23. Positive net change does
not mean no answerable regressions. Unanswerable gains are **81/0** at bias 0
(p=8.2718e-25) and **4/0** at canonical (p=0.125): 240/240 in both observed arms,
with zero observed unanswerable regressions on this finite bank. Frozen whole-answer
exact, a different endpoint, is 126 at bias 0 and 88 at canonical; matched_mlp is
204/202 under both endpoints. Do not call +11 or +93 normalized-exact deltas.

| Bias | Family | Answerable frozen → MLP (gains/losses) | Unanswerable frozen → MLP (gains/losses) |
|---|---|---|---|
| 0 | counting_arithmetic | 77 → 81 /240 (45/41) | 6 → 80 /80 (74/0) |
| 0 | simple_retrieval | 116 → 118 /120 (4/2) | 39 → 40 /40 (1/0) |
| 0 | state_tracking | 0 → 5 /360 (5/0) | 114 → 120 /120 (6/0) |
| canonical | counting_arithmetic | 28 → 81 /240 (57/4) | 76 → 80 /80 (4/0) |
| canonical | simple_retrieval | 81 → 118 /120 (39/2) | 40 → 40 /40 (0/0) |
| canonical | state_tracking | 0 → 3 /360 (3/0) | 120 → 120 /120 (0/0) |

The published adapter **0/360** state-tracking claim copied the frozen arm. The
actual 5/360 and 3/360 are non-significant (p=0.0625 and 0.25) and still limited by
E36's variant-2 wording defect and E38's query-blind shortcut. They establish no
state-tracking capability or population generalization. Retrieval recovery and
canonical refusal lifting cannot be relabeled compositional reasoning. E25/E31
still bound BSEL-1 conditional/composed-family interpretations. The CALIB series'
neutrality claims were specific to their objective, bank and policy: CALIB-4's
dev neutrality did not survive TRANSFER-1; its canonical harm recurs in TRANSFER-3.

## T4 and the rejected candidates

**T4 at 3.75 is undetermined for both arms.** E37 already records that T4's code
ignored answerable outcomes and expected loss. Agreement of endpoint directions
does not repair that mismatch. Even monotone individual abstention curves do not
imply monotone between-arm differences or a preserved expected-loss ranking.
First-token logit translation is path-local: replay stops at the first changed
token; unseen autoregressive continuations are not in a bias-0 trace. Even a
divergence to token 51378 is not proof of a complete literal UNKNOWN answer.
Prediction-only receipts cannot reconstruct the point or the complete bias curve.
The prior claim that one logit-retaining pass necessarily prices the whole curve
requires a separately qualified continuation derivation and actual replay.

**gate_only remains DO NOT SHIP.** At bias 0 it fails T2: a 6.23% reference-loss
reduction against a 30% bar, with answerable 8 gains/14 losses. At canonical it has
1 answerable gain/35 losses, net -34 (p=1.0768e-9), and 4.72% worse reference loss.
This is measured canonical harm, not measured harm at 3.75. It still runs the same
156,225-parameter feature map and six hooks; one trainable scalar supplies no
demonstrated deployment cost advantage.

**Entity grouping remains DO NOT SHIP.** TRANSFER-2's deployed-prompt contrast is
0/360 → 0/360 on both banks **at bias 0**, not a direct 3.75 result. Historical
ADJACENCY-1 abstention costs 18→9 (A) and 83→68 (D) also belong to bias 0. On
TRANSFER-2, A's unanswerable state tracking changes 114→110 while E changes 58→65;
the effect is bank/arm dependent. The E38 “first above blind floor” claim remains
retracted: 0.4333 exactly matched the corrected floor. Off-vocabulary 0.101 was
already below the stated 0.30 floor; E38 raises the floor to roughly 0.41. E36
forbids assigning the attenuation solely to vocabulary, because domain and
phrasing changed together. These corrections do not reopen the rejected proposal.

Older STATUS wording also leaves E6–E10 unresolved after ERRATA Part 2 resolved
them: ACAL-1 uniform exact gains are 25 versus span 27; unanswerable canonical
46/48 → neutral 38/48, clustered 0/6 p=0.03125. Pooling-based invalidity and the
“simulated” characterization were refuted; missing committed raw-row custody
remains. This pass relied on the authoritative addendum, without opening its
external records. E5's “gate only amplifies contemplated abstention” remains
retracted; this gate can create an abstention over a gold argmax.

## Exact bundle, source and parent custody

The CAPABILITY-1 sidecar is one shared `matched_mlp` module, 156,225 parameters,
12 F32 tensors, hooks `[3,8,13,18,23,27]`. It is stateless and non-mergeable. The
exact bundle is present in owner and M1 checkouts and byte-identical; its presence
is not a new training or efficacy result.

| Custody object | SHA256 / identity |
|---|---|
| adapter.safetensors, 625,844 bytes | `f2ccf8faa099ba9e0dc5b07474c6a3a4ec42f5bee9141422dd795d9ae6612d79` |
| manifest.json, 928 bytes | `0274971f6b5fd08153b27c187c02824f648b4ddc68c6b4a6a5bd13c700f7613c` |
| registered_training.json | `d59750b095ce672f4b08a5a7b88e1f7aa94008dc685d2cb4e364d079b5196f17` |
| producer executable source | `8c52fc541cc1a5ee01e8dfc55f652a2467dcb42cf85439f4fa93026ad463ca88` |
| producer Git HEAD | `78a581496d9e55be32b34da39851554bc50aede9` |
| native parent Git commit | `199eed212ab3a3db487270dc6f97b16756f7bff0` |
| parent model.safetensors, 1,192,134,784 bytes | `a99f079d32212bbcbb703c2e2bad801ea32d43afea52004f38bf94eec5dc4026` |
| parent config.json | `89c4ae7398449895d044750c858614584b07db8d67130074920ef524c6eb5705` |
| parent modeling_supa_a11.py | `b77b7d709fac3690d86f5978d46de409fd7ad3548c66a9d400188c225d99fa3d` |
| M1 executable source | `4a3aa695f5f9cc31a4be7af1e06c1cfcc3dbf8d01d1c0e3bd1137c09140dce1a` |
| M1 SOURCE_MANIFEST.json | `281c7ec05e91164a7395d9201274a28fe1c4c5da28a69d303268d3f4bed62aea` |

Producer provenance is answer_lm, seed 20260903, 450 steps, not a new recipe.
Historical native receipt requal-16 binds the same source, parent, device and
historical environment. Current owner and M1 inherited executable source still
match that digest; the reconciliation tool is separately hashed outside that
17-file historical source set. All six M1 inherited raw Git-blob pins match.
M1's historical primary-record blobs stay pinned; this addendum is a separate
handoff binding, not a replacement of M1's manifest or imported permission.

The owner parent is initialized, clean and weight-hash verified offline. M1's
parent was **uninitialized at initial inspection**. A later ownership check found
the existing active chat **Complete MLP-SHIP-001 M1 mechanics**
(`01a10d63-d938-7ea0-8595-8b2ce76be38c`), whose own user prompt authorizes one
mechanical attempt. That owner is materializing local assets and an external
runtime. The final handoff records the later custody snapshot separately; no
qualification pass is inferred from asset presence. Exact-parent verification
must precede that owner's authorized call. A full native
checkpoint verifier was not executed in this pass; no sealed/model evidence bank
was read. TRANSFER-3's staged producer filename mismatch remains documented in its
failed-attempt note: do not rewrite the committed producer to disguise it.

## Handoff, verification and next decision

The destination is
`C:/Users/clclc/BusinessProject/SupaExperiment-matched-mlp-shipping-v1`, using the
existing M1 implementation and owner review. Machine-readable handoff, exact
native proposal, hashes and verification receipts live under
`C:/Users/clclc/BusinessProject/supa-run-outputs/nonrecurrent-reconciliation-20261005-01a10d63/`.
See [reconciliation_v1/README.md](reconciliation_v1/README.md) for reproducible
offline commands and receipt paths. The packet addresses M1 without modifying its
checkout. Corrections were delivered to the existing active M1 chat under the
instruction to reuse its owner; no competing worker or extra attempt was launched.

Verification covers historical checksum sets, paired identity/completeness,
agreement with committed aggregate scores, source and bundle custody, eight
synthetic fail-closed tests, and before/after preservation hashes. Fixture results
are software evidence only. No efficacy or native deployment pass is claimed.

One proposal is prepared for the existing owner: the **inference-only M1 CUDA
qualify** command, bound to exact HEAD/source/bundle/parent and external custody.
It is a reference to that owner's one attempt, not a second attempt grant. It traces
3.75 on three fixed hello/glass prompts within 420 forward tokens and 900 seconds,
with existing memory guards and exact reload/unload checks. Its direct isolation
probe compares canonical versus zero, not a fresh direct 3.75 isolation test.
These fixtures and descriptive timings do not qualify the served query-last
prompt, task outcomes, conversation retention or production latency. M1's latest
committed historical record remains 142 scoped CPU tests with **native blocked**;
the active owner's new result must be inspected independently. M2 remains closed and
unimplemented; it requires separate fresh open CAL/CHECK provenance, prompt/scorer
binding, continuation qualification and authorization before any bank exposure.

**Next decision:** reuse the active M1 owner's chat and review its one already
authorized bounded mechanical attempt against this hash-bound custody packet.
This reconciliation launches nothing and supplies no additional native grant.
Mechanical PASS would enable a later review, not
deployment, policy retuning, training or efficacy claims. Current matched_mlp
status is a historical recommendation with deployment qualification pending.

State-tracking prompt isolation is the separate sequential reconciliation in
this owner lineage. E41's four-factor attribution correction remains intact;
PREFIX-ISOLATION-1 has no surviving receipt. Transcript numbers are not evidence,
and this handoff authorizes no rerun of its spent bank or unobserved arms.
