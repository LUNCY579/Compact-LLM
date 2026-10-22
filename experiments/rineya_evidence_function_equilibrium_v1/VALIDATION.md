# EFE-001 validation receipt

Date: September 29, 2026. Scope: newly added standalone reference package only.

## Executed locally

Environment: Python 3.13.5, NumPy 2.3.5, Linux CPU.

```text
python -m unittest discover -s experiments/rineya_evidence_function_equilibrium_v1 -p 'test_*.py' -v
Ran 40 tests in 2.186s
OK
```

`compileall` completed without errors. The `plan` command emitted parseable JSON.
The standalone smoke completed and explicitly marked its output
`synthetic_cpu_mechanics`, `native_model_loads=0`, `native_training_steps=0`,
`promotion_eligible=false`. The native command's blocked exit and refusal to
overwrite a smoke output are covered by subprocess tests.

Tests cover protected prompt/recent indices, budget infeasibility, deterministic
head-specific random streams, scored and recency controls, no resurrection of
evicted tokens, the decode-only saving bound, exact/rank-deficient/missing-direction
fits, preservation tradeoffs, zero-target ratios, finite-value/shape rejection,
weight/output orthogonality distinction, affine fixed points, deceptive damping,
returned-state residuals, singular/ill-conditioned adjoints, finite-difference
adjoint parity, and declared split/parent/hash/spent guards.

## Not executed or established

No native Rineya checkpoint load, forward, training, generation, JVP extraction,
merge/export, physical cache compaction, sidecar training, GPU measurement, or
benchmark evaluation. No adaptive-FGD theorem or paper benchmark reproduced.
No comprehensive inherited repository test suite run. This package was tested
standalone; cross-branch/native integration remains future work.

No GitHub Actions success is claimed by this receipt. Remote checks, if triggered
by the PR, are separate from these local results. No new workflow was added.

This receipt qualifies only reference mechanics. It does not establish learned
capability, memory/latency improvement, noninferiority, or promotion eligibility.
