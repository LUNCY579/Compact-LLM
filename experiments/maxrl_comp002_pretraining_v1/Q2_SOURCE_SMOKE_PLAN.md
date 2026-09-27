# COMP-RS-001 Q2: source-competence feasibility smoke, pending Q1 native pass

This is a **prospective selection and cost contract**, not a model result or a
training grant. Q2 cannot run until Q1 loader, completion/EOS and logprob parity
receipts have passed in the same runtime. It opens no TEST/CHALLENGE or guard
bank and cannot decide donor acquisition or final pair selection by itself.

Eight CALIBRATION parent worlds are selected by lexicographically sorting
`SHA256("COMP-RS-001-Q2-SOURCE-SMOKE|" + parent_id)` and taking the first eight
from the 150 shared atomic R/S parents. Each parent supplies one R and one S
policy prompt, for 16 prompts total. The exact IDs and hashes are saved in
`evidence/Q2_SOURCE_SMOKE_SELECTION.json` **before any model output**.

Freeze one BF16 Supa source snapshot and generate eight independent samples
per prompt, batch size one, `max_new_tokens=128`, `do_sample=True`, temperature
1.0, top-p 1.0, top-k 0, repetition penalty 1.0, thinking disabled, and
generation EOS IDs `[151645,151643]`. Every policy snapshot, prompt, rollout
ID, random seed, sampled token ID, stop reason and token cost must be kept,
including malformed, cutoff and verifier-failing responses. No refill, oracle
completion, other-policy cache or selection among the eight for pass@1.
The unwarped sampler settings make raw policy and sampling distributions the
same path; Q1/Q2 logprob checks still have to demonstrate that claim natively.

Ceilings: 16 prompts × 8 samples = 128 attempted completions, at most 16,384
generated tokens, at most 225,664 prompt-prefill tokens under the 1,763-token
CALIBRATION maximum, one model load and no gradient/optimizer/checkpoint.
Launch only with the resource/lease gate and a bounded wall plan. Report R and S
pass@1 separately from pass@8, all-failure/mixed/all-success group counts,
UNKNOWN/format/cutoff rates and actual total costs. The eight-parent sample is
a feasibility pilot; it cannot establish population competence, noninferiority
or training readiness. Full CALIBRATION and complete guard_dev enrollment remain
subsequent bounded work.
