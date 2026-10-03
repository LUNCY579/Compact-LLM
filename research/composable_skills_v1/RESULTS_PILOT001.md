# Independent old-skill anchoring: useful signal, objective not achieved

2026-09-19/20. All five supervised stages of `runs/pilot004` completed. The three
earlier attempts were engineering failures preserved separately. This is one training
draw on visible development instruments, not confirmation.

| Arm | Acquisition /384 | Family floors | Old dev /168 | Original148 retained | Lost / gained |
|---|---:|---:|---:|---:|---:|
| Frozen Supa |14|0/3|148|148|0 /0|
| Ordinary LoRA CE |194|2/3|148|144|4 /4|
| CE + independent old-input KL |193|2/3|149|146|2 /3|

**Both trained arms fail the prospective useful-acquisition criterion** (≥200 and
all3 floors). Ledger is38/128 for CE and35/128 for KL; lower95% bounds0.2194 and0.1984
do not exceed the published30/128 floor. Prerequisite scores82/84 and register74/74
clear their floors. The older prerequisite bank has no decisive guard/order examples;
these scores do not demonstrate those rules were learned. Neither candidate is a
quality-qualified or release-qualified model.

## What anchoring actually protected

The two restored source successes are missing-evidence binding answers from **one
world**. Literal UNKNOWN is56/56 for base,54/56 for CE and56/56 for KL; all three
have0/56 false UNKNOWN on answerable binding questions. There are not two independent
replications. Binding overall is111→110/112, with source-success losses2→0.

Clock-time forgetting remains: each arm loses2 of9 source-correct time
answers, but **only one lost answer is shared**. KL repairs CE's failure on15:50+18min
and introduces a failure on17:52+16min; both fail10:42+12min. The first draft of this
report incorrectly called the two lost sets identical; direct row inspection corrected
that count-to-identity error. CE gains3 others; KL gains2. Arithmetic remains28/28 in both. A whole-answer
clock-format sensitivity rescore rescued no rows, so the remaining damage cannot be
explained by just leading zeros or a terminal period under that diagnostic.

The acquisition bank also contains14 original source successes. **Both arms lose5
of those14** while learning other examples. Labeling that bank "acquisition" does not
make those losses disappear. Across both disjoint banks, the original source has162
successes and the candidates lose9 (CE) or7 (KL). The4/2 losses in the table refer only
to the separately constructed old-development bank, not all existing measured behavior.

CE versus KL has5 acquisition losses and4 gains, and3 old-dev losses and4 gains.
The independent grouped paired comparison givesp=1.0 for both endpoints. This tiny
pilot does not establish an improved population frontier or equivalence. Even zero
damaged source-correct worlds would have weak one-sided95% risk bounds:10.14% for
the28 binding/arithmetic worlds and28.31% for9 eligible time worlds (unadjusted,
illustrative iid bounds, not confirmation).

## Mechanics and cost

Both use7,208,960 FP32 trainable adapter parameters,28 updates,54,172 new input tokens,
5,254 supervised new tokens,18,873 old forward tokens and identical old-row exposure
and LR at every step. Old presentations are87 binding,16 arithmetic and9 time.
This token-weighted KL is not family-balanced. The base weights remained frozen.

Training:116.672s CE versus141.344s KL, a21.1% observed training-wall increase.
The control also computed old diagnostic forwards, so this is not total overhead
against a minimal CE-only trainer. Baseline413.843s; evaluations307.719/307.141s.
Total supervised pilot wall:1,286.719s (21.45min), plus62.234s of prior engineering
attempts. Peak allocated GPU2.603/2.628GB (decimal), measured in training.

Independent BA geometry reproduces native weight-delta norms1.46723908 and1.46446008,
ratio0.998106, cosine0.993582. The best scalar mapping leaves1.27956% residual energy.
The penalty mostly changed direction rather than magnitude, but geometry is not
evidence of behavioral preservation. First-step CE, KL, gradient norm, LR and old IDs
match; logged update norms differ only3.50e-12. Whole-vector first-step equality was
not recorded and is not claimed.

Source-cache selected logits reconstruct with exactly0 maximum absolute error. The
FP32 HF eager path is matched across all arms; no blanket DenseReference/native-serving
parity, BF16 merge equivalence or general model-quality improvement is established.

## Reproducible evidence

-Local raw generations, source/adapter hashes, step logs, frozen plan, code snapshot,
  supervisor receipts and summary: `runs/pilot004/`.
-Independent audit:
  `C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification/research/continual_skill_audit_v1/PILOT004_FINAL_AUDIT.json`.
-Independent readout: same directory, `PILOT004_RESULTS.md`.
-Independent geometry: same directory, `PILOT004_ADAPTER_GEOMETRY.json`.
-Known rule-coverage limitations: same directory, `SELECT_CONSTRAINT_COVERAGE.json`.
-Prospective harder development bank/review: same directory, `rule_worlds_v2/` and
  `RULE_BANK_REVIEW.md`. No model execution or sealed-confirmation status is implied.

## Decision

Keep the hypothesis open and the goal active. The prior new-input LWF failure did not
close independent old-input protection; this pilot shows a limited binding effect but
insufficient acquisition and remaining time damage. Next test one higher LoRA learning
strength in a separate prospective development study, retaining both arms and every
failure. Then use harder fresh binding and rule-counterfactual/joint tests. Do not tune
this completed pilot, promote its checkpoints, or claim that sparse/low-rank updates
have solved continual learning.
