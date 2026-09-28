# Centralized Rineya Titan 0.6B: qualification cycle

Date: 28 September 2026. **Native work is blocked at host-RAM admission. No model
was trained, no acquisition or retention effect was measured, and no recipe was promoted.**

## Scope and exact identity

The primary student is **Rineya-Titan-Experimental-1.0.0-devfreeze**, measured at
596,049,920 parameters in 310 BF16 tensors. It has 28 layers, hidden size 1024,
MLP 3072, 16 query heads, 8 KV heads, head dimension 128 and vocabulary 151936.
Tied embeddings are declared; loaded pointer tying is still unmeasured.
40960 is a config declaration, not a qualified context envelope.

Weight SHA256: `537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7`.
Config SHA256: `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`.
Tokenizer SHA256: `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`.
Template SHA256: `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`.

Exact local checkpoint:
`C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Experimental-1.0.0-devfreeze`.
All root checkpoint assets and the complete tensor inventory are recorded in
`receipts/PRIMARY_STUDENT_RECEIPT.json` and `receipts/PARAMETER_INVENTORY.json`.

This choice follows the user's Titan preference and keeps a simple dense BF16 base.
It is not a measured superiority claim over Titan-Sparse or Titan-Transition. Their
FP32 composition/rounding factors stay outside the first causal chain. The 344M
draft is archived and no 344M GPU work occurred.

## Source and protocol

Freshly fetched parent: `03db218ceebc61a2bc04f2482cf7e094045e01c1`.
Execution branch: `research/post-training-mimo-centralized-v1`, initially pushed at
that parent. Registration commit: `577ec6d5000bad11eaa72823d354b143b4178a26`.
Corrected E0 retry source: `285529d9ad328de49a3e820ebc8fbdb0d18d693e`.
No parent/main merge or historical branch rewrite occurred. The checkout had
untracked drafts at arrival; AMENDMENT_001 explicitly records that it was not fully
clean and preserves their original bytes. Both native attempts launched with clean source.

The supplied 66-page report was read in full; its staged experimental logic was
retained and its 344M resource assumptions were superseded. Source PDF SHA256:
`a1c5b5c66e1cef0edd04631ab7c487c93731b1a207bb67ee87fddd40cf43c2d7`.

## Measured qualification results

| Run | Result | Model loads / updates | Measured cost |
|---|---|---|---|
| Static binding | Exact Titan files/header verified | 0 / 0 | Streamed hashes; wall not continuously metered |
| Final CPU controls | 80 tests passed | 0 / 0 native | 25.732s reported test execution, excluding imports |
| E0 attempt1 | WindowsPath API failure before model load | 0 / 0 | 185.313s worker;191.422s supervisor;705,085,440B peak process-tree RSS |
| E0 attempt2 | HOST_RAM_GUARD | 0 / 0 | 67.797s worker; 73.891s supervisor; 722,173,952B peak process-tree RSS |
| Bridge preparation | 4608 open parents verified | 0 / 0 | 17.009s;0 teacher/model calls |

Attempt 1 failed because Windows psutil.disk_usage rejected a pathlib.WindowsPath.
The retry changed only conversion to string and added import progress events. No
resource guard or numerical tolerance was relaxed.

Attempt 2 measured **6.911331GiB host RAM available, below the 7.4GiB admission floor**.
The worker floor remains 6.0GiB. GPU free memory was 6.890625GiB against a 4.0GiB
admission floor. Worker RSS at this point was 0.668579GiB. Runtime import took 66.391s.
The detected GPU is RTX4070 Laptop with 8188MiB. These are pre-load measurements.

Runtime: Python 3.11.15, torch 2.10.0+cu126, Transformers 4.57.1, safetensors 0.7.0,
tokenizers 0.22.2. No pretrained tensor allocation or optimizer update occurred.
Every attempted run has config, source, runtime, failure, resource, cost and process
containment receipts. The watchdog reaped its owned worker tree.

## Scientific interpretation

This rules out launching the registered run at the observed host-RAM state. It
does **not** establish that Titan cannot fit after admission, that BF16 cannot work,
that LoRA cannot learn, or that any capability regresses. Frozen-load/forward/backward
VRAM, optimizer/activation memory, throughput, checkpoint timing and resume remain
UNMEASURED on this model. There is no native zero-adapter, likelihood or gradient pass.

Historical Titan behavior used FP32 DenseReference. Its attention arithmetic differs
from stock BF16 Qwen3; those historical scores are RECORDED FROM PRIOR WORK and do
not transfer to this proposed runtime. The unchanged base remains CURRENT_BEST_RECIPE
only as the campaign starting point, with no claimed improvement.

## Data and held-out boundaries

Original bridge populations: 4096 TRAIN parents, 256 TRAIN_SELECTION and256 DEV;
each role balances arithmetic, symbolic registers, finite sets and constrained text.
512 CHECK parents have hash-only reservations; no CHECK prompts/targets were exported.
FINAL remains unopened, with its confirmation sample size unresolved.

Canonical allocation precedes rendering/solving. All 4608 open positives pass an
independent verifier and all 4608 wrong-answer mutations fail. There are zero duplicate
selected canonical parents and zero cross-parent prompt collisions. Four duplicate
candidate specifications were skipped before selection. Full source/file hashes,
generator runtime and provenance fields are retained; raw generated rows are local
and regenerate from the committed code/registration.

Limits: canonicalization covers declared commutation/order equivalences, not general
semantic equivalence or pretraining/historical decontamination. TRAIN finite-set
answers include 105/1024 empty sets and text includes 12/1024 empty strings. Arithmetic
has 428 distinct TRAIN answers across 1024 parents. Report trivial-answer baselines
and per-family/structure effects; shared answers alone do not prove contamination.
These are deterministic bridge tasks, not MiMo trajectories.

## Tokenizer measurement

The pinned stock Qwen2TokenizerFast run completed in 111.268s with no PyTorch
import, model load or model call. All 4608 open-role examples pass source hashes,
exact template/prefix/completion checks and positional masking. Maximum total length
is 157 tokens; maximum completion including EOS is 38. No truncation or oversize
rows occurred. TRAIN contains 414,554 nonpadding tokens and 49,272 supervised tokens
per pass; all open roles total 466,567 tokens. Replay is not yet enrolled, so these
are acquisition-bank counts, not the complete E1 dose. Receipt and derived-file
hashes are in receipts/BRIDGE_TOKENIZATION_RECEIPT.json; raw token arrays stay local.
CHECK/FINAL were not opened. Tokenizer fit does not establish model resource fit.

## Analysis controls and auxiliary failure

DEV analysis uses paired parents, averages seeds within parents and bootstraps
whole parents within family strata. Duplicate seeds cannot manufacture independent
evidence. Family regressions and source-correct losses remain visible; zero source
denominators return null. The registered defaults are 10000 resamples, seed 20260927.
The helper is exploratory, DEV-only, and confers no automatic promotion.

A no-model control chooses each family's most common TRAIN target (lexicographic
tie break) and scores DEV: arithmetic 2/64, text 1/64, sets 3/64, registers 0/64. These
are trivial data controls, not Titan results; no CHECK/FINAL contents were opened.

Tokenizer attempt1 selected the shell's Hermes environment with Transformers 5.14.1
instead of the pinned 4.57.1 environment. Its no-PyTorch-import guard failed before
row tokenization, with zero model loads and no derived output. Exact failure wall
time was not captured; the preserved receipt bounds it at 150-457s. Initial progress
wording suggesting all-row processing was underway was unsupported and is corrected
in the receipt. The retry pins the interpreter and checks package versions before
import; the no-PyTorch guard remains enabled.

## Preservation gate

No existing bank is yet qualified as 1024 broad-preservation replay rows for this
Titan. The proposed plan is 768 deterministic replay rows across retrieval,
instruction-following and answerability, plus 256 reviewed language replay rows.
Answerability twins share parents; rows are not independent parent counts.

The v3 train-only 14000-row preservation slice is a language replay candidate; it
cannot serve as fresh DEV. The MaxRL replay_train 1600 rows are a reasoning-only
alternative. Neither supplies a broad-preservation certificate. Fresh DEV, actual
source-correct enrollment, language scoring and a genuinely longer-context sentinel
remain necessary. Historical spent/selection/sealed banks retain their original
access boundaries. See `evidence/RETENTION_DESIGN.md` for exact metadata/exclusions.

## Research stack and next expenditure

68 mechanisms/support components were source-audited. Current Titan native
requalification remains false throughout. Separate GRPO, MaxRL-estimator and NGU
rows expose their different variance, clipping and KL conventions. GFlowRL additionally
needs canonical reward-maximizing R0; ordinary SFT is not that control. Architecture
and inference experiments remain independent tracks.

The next highest-information native experiment is the **same bounded E0**, once
adequate host headroom is measured. Preserve 7.4GiB admission and 6.0GiB throughout;
do not infer required additional headroom from the pre-load deficit alone. No
automatic identical retry, quantization, 344M fallback or paid cloud launch is justified.
While blocked, use the completed tokenizer/analysis controls to resolve historical exclusions and
preservation design. E1 seed 101 remains blocked until E0, data/replay and evaluation gates pass.
The larger multi-method research program is ongoing; this report closes only the
present source/data/preflight cycle.
