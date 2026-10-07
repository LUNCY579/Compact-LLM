# RHF-001 execution receipt — 2026-10-07

## What actually executed

**76 tests passed, zero failed** in the isolated new package. `compileall` also
completed. Runtime: Linux, Python 3.13.5, PyTorch 2.10.0+cpu, CPU-only, one intra-op
thread. No GPU, pretrained checkpoint, native Transformers forward, full-repo
suite, CI workflow, or user-host run was executed. Native-wrapper tests used an
explicit toy interface, not Qwen3. Confirmation seeds remain unrun.

Executed final code: smoke; 8-cell development sweep with 4 construction variants
and controls; tagged fact-adaptive SSFR at seeds 17 and 29, 400 training steps.
`validation.json` pins source/config/raw-report hashes and environment.
`development_results.csv` includes all 32 construction arms, including failures.
Full raw JSON is in the companion delivery ZIP under `raw_runs/`; timings are
machine-dependent and make regenerated raw-byte hashes differ.

## Selected observations, not native claims

| Setting | Data-dependent strict clean recall | Random full-whitened strict clean recall |
| --- | ---: | ---: |
| 64 facts, D=16, M=32, seed 17 | 64/64 | 52/64 |
| 64 facts, D=16, M=32, seed 29 | 64/64 | 50/64 |

At relative query noise 0.3, those data-dependent memories scored 56/64 and
52/64; random full-whitened memories scored 37/64 and 32/64. This is a small
mechanism observation on synthetic keys, not a confidence-qualified superiority
claim. It does not compensate for the numerical failures below.

More width did not make retrieval uniformly robust: with 64 facts and M=64,
random full whitening fitted 64/64 on each seed but scored only 23/64 and 15/64
at noise 0.3. In the correlated N=32/M=32 cells, clean full-whitened recall was
32/32 on both seeds, while noise-0.3 recall was 5/32 and 4/32. **Exact fit is not
robust retrieval.** No hyperparameter was selected using confirmation results.

For the tagged SSFR test, each seed scored 128/128 on fresh-context A retrieval
and 128/128 after swapping to mapping B while freezing the reader. Stale A
memory scored 0/128 against B. Each 128-context panel contains only **16 facts**
with 8 contexts apiece. Maximum reader/key L2 error was approximately 0.01169
and 0.00661. The explicit tags simplify routing; this is not language understanding.

## Failures and implementation corrections retained

Data-dependent construction reached the checked CG iteration limit in **6 of
8 development cells**. Both successful data-dependent cells also had blocked
sequential first-bank factor fits. They are marked `BLOCKED_NUMERICS`, never
counted as accurate memories, and cannot qualify an all-arms confirmation claim.
No convergence threshold or maximum-iteration budget was weakened to obtain gains.

During development, a blocked sequential fit initially escaped the nested study
and aborted the first sweep. That failure was preserved locally; the runner now
records the subexperiment failure while retaining independent measurements. A
regression test covers it. Review also caught and corrected a readout-swap dtype
mutation and a readout-variant mismatch before final validation. All final runs
were regenerated against the corrected source hashes.

## Decision

Proceed with numerical-conditioning diagnostics and controlled confirmation,
not a production MLP replacement. Preserve both vector and upstream-score
objectives, explicitly sweep projection regularization on development only,
and inspect singular spectra/solve residuals before interpreting factor-fit gains.
Prioritize query robustness and native hidden-state geometry, not just clean fit.

The native bridge is implemented and mock-tested, but its local Qwen3 loader,
actual checkpoint behavior, CUDA/Windows execution, acquired-fact language
quality, broad retention, and long-generation/cache behavior remain unqualified.
Nothing in this receipt licenses a new Rineya capability or a release.
