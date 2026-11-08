# Two-track framework delivery and execution status

The current-session implementation deliverables are complete and validated.
The scientific programme remains open: neither fresh native efficacy nor new
real donor-alignment results have been produced. Both registered execution
attempts failed physical-memory admission before model loading.

## Repository and audit

Repository: `C:/Users/clclc/BusinessProject/ChatBubble-cross-scale`, remote
`https://github.com/LUNCY579/ChatBubble.git`, branch
`codex/supa-adaptive-compute-v2`. Implementation and both preregistrations are
committed at `067fa33b2ed78fc2a0f9361d313629549f214f08`. Subsequent report/evidence
commits do not alter registered source. Live remote symbolic HEAD at inspection
is `a6ee824584a9e28881826ab92a11db48e124d90c`; this new branch has no remote ref
and has not been pushed. Final local HEAD is reported in the task response.

The checkout's pre-existing untracked `experiments/cross_scale_layer_graft_v1/`
remains untracked and unchanged. New work is committed separately. Its historical
source cannot honestly be attributed to a Git commit; reuse is bound by exact
hashes. A clone alone needs that preserved local dependency for Track B.

[Existing-work inventory](00_audit/existing_work_inventory.md) records exact
paths/branches/SHAs, classifications and repairs. [Historical byte inventory](00_audit/historical_bytes.json)
covers 198 files, including prior local artifacts, with unchanged root SHA256
`b2b4986d6f3abc151ab0c7ec8748c421c919dbcf648afa526e0005d4bdcbfa8f`.
The [requirement audit](00_audit/requirements_audit.md) explicitly distinguishes
implementation, tests, missing measurements and gated future work.

Reused: native scoped repetition/whole-block wrappers; canonical loader resolution;
logit/hidden/Unicode diagnostics; streamed official checkpoint readers and actual
HF donor layers; strict portions of historical scoring; canonical hashing; exact
paired and scenario-cluster statistics. New layers repair deadline coverage,
sealed/role guards, immutable shard ownership, independent rescoring, manifests,
token-balanced/delta-aware bridges and complete-matrix selection.

## Shared architecture and data

Both callers use shared model identity, capture, generation, NLL/logit/validity,
scoring, paired analysis, cost frontier, hashes, manifests and resource exclusion.
Operators, calibration, selection, protocols and mutable evidence remain separate.
Track B's bridge API returns only its own calibration prompts and identity fields.

Each track has 1,024 primary scenarios, 128 linked diagnostics, 128 architecture
cases and 128 calibration cases (96 fit/32 holdout), with eight families and 32
templates. Eight templates are excluded from engineering. Independent semantic
oracles and tokenizer validation pass. All calibration prompts fit the frozen
256-token Track B budget; no truncation is used for final-token metrics.

Canonical bank-manifest digests:

- Native: `7ebdac9f8a5ce14277d6c74a138b96303495c82b9a98bcbaf74a498e8fedd4a2`
- Cross-scale: `6857158ee7a360bdf2f72a21c5ee78041ab0a5ecda264c0387e1e5facf638a46`
- Generator source: `a36051907e6bd246211eeeddb37acdec22144b74573f744300338749b58f6a01`
- Strict scorer identity: `62937db0530db2f8b43672ff1ea3e9ffa96dce5e8b7c9cf5bac6996b3d29c11e`

These data/scorer sources are frozen. A changed question or scorer requires a
prospective new version; no model output can be used to repair its own test item.

## Track A

[Protocol](native_armr/protocol/preregistration.md): A0, A1=L12 repeated once,
A3=L12-L13 block repeated once, A16=L16 repeated once, AM=fixed independent
SwiGLU residual after L13. AM has 31,457,280 added distinct parameters and
62,914,560 added dominant matrix FLOPs/token, matching A3's two decoder-layer
matrix work. It is an untrained generic control; attention/norm costs are excluded.

Registration: `a2bb36ff710ba95b371e6e4dbb1c74840477beca050226d8678f5c334d43964f`.
It binds textual protocol, executable configuration, transitive source and the
independent bank; commitment was verified before the admission attempt.

Execution: A0 architecture shard 0/32 attempted at the committed implementation;
it failed at 1.70 GiB available RAM versus registered 3.00 GiB. Zero examples were
generated. Complete 128-case/all-five-arm architecture admission is still required
before confirmation. No raw accuracy, paired win/loss, NLL or latency comparison
can be reported for v2. Historical A0/A1/A3 12/12/13 of 16 remains unchanged and
hypothesis-generating. **Scientific GO/NO-GO: pending, not a negative result.**

The implemented final analysis requires complete compatible evidence, independent
rescoring, A3>A0 followed by A3>AM, meaningful size (≥2pp or ≥5% RER), paired CI/p,
preservation noninferiority and no extra invalid output. A1/A16 form a secondary
Holm family. Family/held-out-template and linked-original robustness accompany
the descriptive quality/cost frontier.

## Track B

[Protocol](cross_scale/protocol/transformation_alignment_v2.md): B0 historical
absolute ridge, B1 token-normalized state fit, B2 delta-aware fit, B3 both,
BS cross-scenario shuffled pairs, BR seeded random bridge. The 16 depth pairs are
Supa post {8,12,16,19} × donor {12,16,20,24}: 96 objective cells. B2/B3 train only
upward affine parameters through the frozen actual donor transformation; native
backbones and downward maps remain fixed.

Registration: `99d73b899bdc057ab8d3e61e528841a2254714348b1e2b8f1ed79eacc36df9b8`.
The complete held-out mechanism gate is ordinary delta cosine ≥.60, improvement
≥.10 over B0 and null controls, token-relative MSE ≤.8 B0, plus all-token/scale
checks. Selection is entirely calibration-based and requires all controls/pairs.

Execution: capture admission failed at 1.80 GiB available RAM versus 3.00 GiB.
No activations, fitted bridges, new held-out metrics or selected bridge exist.
Historical delta cosine .443202525 remains only a historical baseline.
The future dose grid [.03,.10,.20,.40] is prospective; dose generation is not
licensed or executed. Stage 1 caller exists; Stages 2/3 remain gated future work.
**Scientific GO/NO-GO: pending.** Qwen3-8B remains closed.

## Power and integrity

[Power planning](data/generated/v2/power_planning.json) includes all requested
sample sizes and effect scenarios. At N=1,024, a two-point gain has approximate
82%/53%/30% power for assumed paired discordance 5%/10%/20%; 80%-power MDEs are
about 1.96/2.77/3.92 points. RER planning assumes a hypothetical 75% baseline.
No observed discordance or empirical power exists yet. Correlated diagnostic
variants contribute no extra primary N; multiplicity weakens sensitivity.

Canonical checkpoint SHA256 remains
`a99f079d32212bbcbb703c2e2bad801ea32d43afea52004f38bf94eec5dc4026`.
Tokenizer/config/code sidecars and the installed Qwen implementation are bound to
the qualified historical runtime. CPU FP32/eager, four threads, no cache and
deterministic decoding are mandatory. Model/control trainables, gradients,
before/after state and file hashes are guarded, with independent audit checks.
Both real attempts stopped before backbone loading; new real-model state
preservation measurements therefore cannot be claimed. No checkpoint was saved.

No sealed/TEST2/private dataset was read. Native and donor evidence roots are
separate. Failed attempts, their source/config hashes and traceback receipts are
retained. Neither process remains running and the machine lease was released.
Routing, MLA/MHA/MTP, PEFT, backbone training and distillation remain closed.

## Evidence and tests

All paths below are under `experiments/supa_adaptive_compute_v2/`:

- Audit/test receipt: `00_audit/validation_20260905T1727/`
- Native registration: `native_armr/evidence/00_preregistration/a2bb36ff710ba95b371e6e4dbb1c74840477beca050226d8678f5c334d43964f/`
- Native input validation: `native_armr/evidence/01_dataset_validation/20260905T171537_568883Z_e2e20e72/`
- Native failed admission: `native_armr/evidence/01_architecture_dev/20260905T173248_727906Z_9ed863b1/`
- Donor registration: `cross_scale/evidence/00_preregistration/99d73b899bdc057ab8d3e61e528841a2254714348b1e2b8f1ed79eacc36df9b8/`
- Donor input validation: `cross_scale/evidence/01_dataset_validation/20260905T171720_931623Z_4247ee5b/`
- Donor failed admission: `cross_scale/evidence/01_alignment_matrix/alignment-v2-admission-001/`

Exact combined test command from the repository root:

```powershell
& 'C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe' -m pytest -q experiments/supa_adaptive_compute_v2/tests experiments/supa_adaptive_compute_v2/native_armr/tests experiments/supa_adaptive_compute_v2/cross_scale/tests experiments/anchor_routed_middle_recurrence_v1/tests experiments/cross_scale_layer_graft_v1/tests --junitxml=experiments/supa_adaptive_compute_v2/00_audit/validation_20260905T1727/pytest.xml
```

**234 tests passed, 148 subtests passed, 55.44 seconds.** This includes 149 new
tests and 85 historical tests. Compilation passed. The XML and receipt bind the
tested source. The command is the historical invocation: use a new XML path for
any future verification so this saved evidence is not overwritten.

## Limitations and exactly one next causal experiment per track

All new successful execution to date is testing, bank validation and tokenizer
inspection. These cannot establish efficacy or real donor fidelity. Banks are
synthetic bounded English tasks, with repeated structures and modest robustness
strata; grounded QA is not broad factual knowledge. AM is untrained. There is no
runtime qualification for GPU/FP16 and no source-controlled copy of the user's
pre-existing untracked graft dependency. No result has been published or pushed.

- **Track A:** execute the frozen five-arm native campaign, starting with the
  required architecture qualification and then its unchanged 1,024-case paired
  confirmation. Resume using `scripts/run_native_campaign.py` once available RAM
  meets the registered threshold. The failed shard is retained and retried in a
  fresh directory.
- **Track B:** execute the frozen calibration-only 16-pair/six-objective matrix,
  using fresh run ID `alignment-v2-001`, then complete-matrix alignment selection.
  Exact capture and resumable pair commands are in `cross_scale/README.md`.

These remain separate causal experiments. The complete research goal stays open.
