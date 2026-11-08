# MAXRL-COMP-002 / Q0: source-bound pre-training handoff

Status: **SOURCE/DATA BINDING PASSED; NATIVE QUALIFICATION BLOCKED.**
Base: `research/maxrl-composable-skills-data-v2@a97df24c2c7dd062985942ead8c3375bf20c66b9`.
This is an additive sibling package. It does not amend the registration, corpus,
verifiers, estimator, bank manifests, comparisons, statistical interpretation or
existing evidence. No training entrypoint, native loader, generation loop, optimizer,
remote workflow dispatch, grant consumption or automatic promotion is provided.

## What was actually established

The delivered Library archive was materialized unchanged. Its SHA-256 is
`79b8171e83388ff0bf406328f90b5d74ab4cb6d94f29d0ba66af559cbd9b75da`.
Its full data manifest SHA-256 matches the registered audit:
`86ac8f0d851f2178752357f62249eab5a23fd3de8fa032e7373614891df4cd44`.
All 68 manifest-listed data files match their recorded lengths and hashes. Both
Python source inventories match that manifest; the ten v2 Python hashes also match
the GitHub validation record. The original suite was rerun: **462 passed**.
This recheck is file integrity plus source tests, not another 41,500-row independent
re-solve and not model execution. The initial source-suite invocation timed out at
30 seconds; a separate complete invocation passed in 24.97 seconds. Two later
combined-suite invocations hit their 45-second execution limits without a final
pytest summary. They are recorded as incomplete, not failures or passing combined
runs. The 462 original and 46 new results below are separate completed runs.

The new suite has **46 passing synthetic tests**: left/right padding, PAD=EOS,
first-EOS inclusion, real length-capped completions, exactly one causal shift,
selected-logprob/cross-entropy agreement, direct masked-logit gradient coverage,
tied-parameter deduplication, rejection of unsupported shared-storage views,
protected outputs, and write-once receipts. Synthetic gradients do not establish
pretrained backward correctness. Prompt masking does not imply that contextual
prompt representations should receive no gradient.

Do not put this Python package inside `maxrl_composable_skills_v2`: that package's
`source_hashes()` recursively fingerprints its Python files and `audit()` requires
exact equality with the delivered manifest. Even an additive helper there would
invalidate the audited source inventory.

## First comparison: preserve the distinction between registration and recommendation

`EXPERIMENT.md` registers selecting ONE pair; it does not name an immutable pair,
checkpoint, tokenizer, precision, parameter scope, or native compute ceiling.
`STUDY_GUIDE.md` recommends R plus S, explicitly as a candidate, not a measured best
pair. Q0 binds the existing candidate banks without claiming prior pair registration:
`atomic_R.json` and `atomic_S.json` contain 3,000 TRAIN rows each; CALIBRATION contains
150 core rows per skill. `guard_dev` is an entire 1,200-row exposed preservation bank.
Bank availability is not authorization to generate on those rows.

The qualification candidate is R/S from one unchanged anchor. A later explicit
pair selection and bounded host plan are still needed before native collection.
RS, bridge training, warmstart, replay, projection, pruning, quantization changes,
architecture changes and sequential adaptation are not opened here. No candidate
model comparison or calibration-driven model search is authorized. Optional common
warmstart in the original registration is not removed; it remains a separate,
unselected intervention rather than being silently used to change the anchor.

## Exact native bindings required before any bounded native probe

`binding.template.json` records unresolved values as null, never plausible example
hashes. Filling it is not attestation; no function consumes it to unlock training.
The host must prospectively supply the following and retain raw receipts:

**Model.** An explicitly selected local checkpoint, exact revision/provenance,
SHA-256 and length of every loaded weight shard, shard index and configuration,
loader implementation hash, architecture/configuration fingerprint, dtype/cast and
adapter policy. Identify the actual instantiated class and fingerprint loaded state
before/after probing. Fail on missing, extra, renamed or reinitialized tensors,
unexpected adapters, remote-code loading, automatic download or silent fallback.
A model brand, parameter count, adjacent experiment's anchor or model-path string
is not a binding. Do not assume upstream Qwen3 weights equal Supa/Titan weights.

**Tokenizer and prompting.** Hash all loaded vocabulary/merges/tokenizer, added-token,
special-token/config and template files, the actual selected template string, and
the exact implementation/runtime. Bind BOS/EOS/PAD IDs, generation EOS IDs,
`enable_thinking=False`, generation-prefix behavior, decoding cleanup and special-
token removal. Use only the unchanged policy messages; never oracle, verifier,
world, expected-answer, diagnostic or reward columns. Prove actual rendered token
prefixes and context lengths, with no truncation or duplicate special tokens.

**Runtime and scope.** Bind Python/PyTorch/Transformers/tokenizers/safetensors and any
trainer/PEFT/attention backend versions or commits; exact loader, device, hardware,
CUDA/driver when applicable, storage/compute/accumulation precision, eager/compiled
mode, cache policy, padding/position IDs, RNGs and deterministic settings. The local
inspection host is Python 3.13.5, PyTorch 2.10.0+cpu, without Transformers; it is NOT
selected as the experiment runtime. `parameter_inventory(model)` observes every
actual parameter name, shape, dtype, device, requires_grad flag and tied alias, and
counts unique/trainable elements once. Freeze the reviewed exact inventory, not a
regex or a guessed layer count. Both skills and both objectives must use it.

**Host envelope.** Bind experiment/Q ID, exact source head, clean worktree or explicit
source file allowlist, selected asset hashes, host identity, lease owner, output
journal, operator authorization and expiration, prefill/forward/backward/generation
counts, generated-token/wall-time/RAM/VRAM/disk ceilings and stop conditions. Zero
optimizer steps, training launches, checkpoint mutations, retries, network fetches,
lease theft and process killing. This handoff creates no numerical native allowance.
An incomplete or wrong experiment lease label must fail before acquiring a lease.

## Completion, EOS and logprob evidence: native requirements, not yet passed

For a prompt with P real tokens and C sampled tokens, label only the sampled suffix.
The logit at position t-1 predicts token t: apply one causal shift, never two.
Include the first genuinely generated EOS in the suffix and exclude padding and
anything after it. Use sequence boundaries/attention masks, not token-ID inequality,
because PAD can equal EOS. Length-capped generation keeps its actual sampled tokens;
never append a fictitious EOS, silently discard costs or change the binary verifier.
Record stop reason separately. Do not strip malformed extra content into a success.

Use raw token IDs retained by the generator, not decode/re-encode to reconstruct the
training trajectory. Cross-check native selected-token logprobs against independent
teacher-forced log-softmax/gather and cross-entropy on the same tokens, including EOS.
Check unpadded versus supported padded execution, cache versus full-prefix execution,
position IDs and the first answer token. Freeze dtype-specific numerical tolerances
before outputs; the float64 synthetic 1e-12 tolerance is NOT a native threshold.

Keep raw model-policy probabilities separate from temperature/top-k/top-p/penalty-
processed sampler probabilities. Bind which distribution generated trajectories and
which is differentiated or importance-corrected. Equal-path agreement is required;
do not silently equate warped sampler scores with raw policy logprobs. Freeze the
sequence/token and prompt/group reduction and epsilon/std conventions. The inherited
v1 estimator uses GRPO sample standard deviation (ddof=1), mean-normalized MaxRL, and
zero advantages for uniform-success or uniform-failure groups. Do not change it here.

A separately bounded native backward-only probe, when explicitly permitted, must
show finite correctly scoped gradients and byte-identical parameter/buffer state
before/after, with no optimizer construction or step. An inference-only hash check
cannot establish this. Any different trainer backend needs its own adapter parity;
these tensor helpers do not certify a future verl/TRL collector or loss integration.

## Source competence and source-correct membership

After mechanical identity checks, preregister the exact CALIBRATION parent/row IDs,
fixed group size, sampling settings, RNG seeds and hard costs before collecting.
Use actual independent draws from one frozen source snapshot. Keep every attempted
rollout, including failure, cutoff and malformed responses. No refill-until-success,
selective row dropping, oracle substitution or another policy's cached trajectories.
Report per-skill pass@1, pass@k with its correct interpretation, all-failure/mixed/
all-success group counts, format/cutoff/UNKNOWN diagnostics and actual token costs.
No observed mixed groups is a feasibility warning, not permission to reshape data,
expand budgets or substitute a source. Numerical competence thresholds remain open;
this document adds no effect-size, power or success-rate cutoff.

For the smallest existing preservation instrument, enroll ALL `guard_dev` rows with
one prospectively bound pass@1 decoding protocol using unchanged `behaviour.enroll`.
A partial response bank cannot be labeled full guard_dev enrollment. Bind the raw
response/token receipts, actual model/tokenizer/decoding hashes and bank hash; retain
source-correct IDs and per-family denominators. Zero correct rows means undefined
retention, not 100%. Hash-shaped strings alone do not attest native generation.
No TEST/CHALLENGE, guard_test/guard_challenge, historical scored instrument,
coefficient tuning, candidate comparison or later acquired-R enrollment runs here.
Guard-dev enrollment is development-scoped, not final retention certification.

## Run only the implemented Q0 check

From the exact successor checkout, with the unchanged delivered `data/` directory:

```bash
python -m pytest -q experiments/maxrl_comp002_pretraining_v1/tests
python -m experiments.maxrl_comp002_pretraining_v1.preflight \
  --repo . --data /absolute/path/to/data --out /new/path/q0_receipt.json
```

The receipt is write-once and cannot be written into the data or original packages.
A successful integrity check still exits **2**, writes
`SOURCE_DATA_PASS_NATIVE_BLOCKED`, and leaves `native_qualified=false` and
`training_authorized=false`. There is no `--train`, `--force`, `--resume` or promotion
command. `binding.template.json` is a checklist, not a native runner or signed grant.

Stop here until exact assets, reviewed trainable inventory, host/resource contract,
native mechanical receipts and actual source competence/enrollment exist. Human
review, historical-exclusion limitations, final statistical margins/power, independent
evaluation and an explicit training authorization remain later separate gates.
No nonsignificance-to-equivalence, no net-gain cancellation of lost source-correct
rows, no orthogonality/no-forgetting benefit, and no quality/release claim is added.

## Primary implementation references (checked September 22, 2026)

- Original immutable `EXPERIMENT.md`, `STUDY_GUIDE.md`, `VALIDATION.json`, `export.py`,
  `behaviour.py`, `corpus.py`, and inherited `maxrl_verifiable_data_v1/maxrl.py`.
- https://huggingface.co/docs/transformers/main/en/chat_templating
- https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.cross_entropy.html
- https://arxiv.org/html/2602.02710v3 (sections 4.2-4.3; practical estimator caveats).

Documentation versions are references, not a selection of native runtime versions.
