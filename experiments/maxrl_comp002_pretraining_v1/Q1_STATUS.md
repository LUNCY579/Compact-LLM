# COMP-RS-001 qualification status — 2026-09-26

The full R/S study is **in progress**. No native source response, donor training,
composition arm or retention result exists. The Q0 `training_authorized=false`
status remains in force.

## Verified so far

- Q0 preflight on this Windows host matched the registered data manifest
  `86ac8f0d…` and all 68 listed files. Atomic R and S each have 3,000 distinct
  TRAIN parents. The Q0 receipt intentionally reports
  `SOURCE_DATA_PASS_NATIVE_BLOCKED`.
- Supa 1.7 was bound as a **candidate** anchor: exact BF16 weight SHA-256
  `962abc5b…dd24b`, 310 stored tensors, seven asset hashes and a 28-layer
  Qwen3 config. All seven hashes also match the source binding of the completed
  δ_C study. This is byte identity, not loaded-model qualification.
- The `.venv-mq001` tokenizer loaded locally with no remote code. Its actual
  rendering and direct tokenization agreed on 14,200 policy prompts across
  selected atomic TRAIN, CALIBRATION, exposed DEV, guard_dev and replay_train.
  The longest was 2,307 tokens. With a candidate 128-token response, this is
  below the model config's 40,960 positions. The tokenizer reports no BOS ID
  while config/generation files list 151643; the actual chat prefix, EOS and
  PAD IDs are recorded in the tokenizer receipt. TEST/CHALLENGE were not read.
- The original v1+v2 source suite passes **462 tests** in the candidate
  Windows runtime. The Q0/Q1 synthetic suite passes 64 tests, including a loaded-state
  one-element mutation negative control, bounded strict/content parser
  negatives and parent-cluster RS repair bookkeeping. These are **synthetic
  mechanics**, not pretrained model evidence.
- Exposed DEV has 250 aligned parent worlds for each of R, S and dependent RS.
  A paired eight-world, 16-prompt CALIBRATION source-feasibility pilot has been
  hash-selected and frozen before any generation.
- The dense study protocol incorporates separate R/S coefficients, both
  no-replay sequential orders, composition-only RS repairs, strict/content
  separation and later function/activation diagnostics. Numeric training
  ceilings and effect margins remain open pending actual source competence.

## Native gate now waiting

`q1_native_wait.py` is live and records readings in
`evidence/Q1_NATIVE_RUN.log`. It has not taken the GPU lease or loaded the model.
Its bound loader probe needs two readings at ≥5.5 GiB free host RAM. Unity was
closed during this turn, but readings remained around 3.4–4.4 GiB; Windows
reported about 2.34 GiB of nonpaged pool during inspection. The waiter does
not lower the resource floor or claim a native result from a timeout.

After a Q1 loader receipt, the next gates are native completion/EOS and
logprob parity, a separately bounded unchanged-state backward probe, actual
source R/S competence, complete source-correct guard_dev enrollment, a frozen
compute/statistical registration, then matched GRPO/MaxRL R and S donors and
their dense composition. Sequential and mechanism follow-ups remain separate.
