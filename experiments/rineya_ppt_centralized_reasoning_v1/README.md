# PPT-001: centralized reasoning through interacting search

**Dedicated branch:** `research/rineya-ppt-centralized-reasoning-v1`  
**Initial target:** frozen Rineya 0.6B; no checkpoint has been selected or loaded.  
**Current state:** CPU enumerable sampler qualified; native scoring unqualified.

This package records an enduring, complementary research direction inspired by
Parallel Power Tempering. A tiny CPU sampler now accompanies the founding design;
the native sampler and experiment remain unimplemented. No Rineya accuracy or
efficiency result is claimed. See the dated
[CPU amendment and next scoring gate](evidence/CPU_SAMPLER_2026-10-03.md).
The [October 9 review](evidence/REVIEW_2026-10-09.md) records archive recovery,
two validation fixes, the preserved original producer, and fresh qualification.

## Read in this order

1. [Owner intent and preservation charter](../../RINEYA_PPT_CENTRALIZED_REASONING.md)
2. [Scoped continuation instructions](AGENTS.md)
3. [System design and method boundaries](DESIGN.md)
4. [Controlled experiment proposal](EXPERIMENT_PLAN.md)
5. [Sources and pinned dependencies](SOURCES.md)
6. [Current status and next handoff](STATUS.md)

[registration.json](registration.json) is a historical design record, not a
launch configuration or permission grant. [VALIDATION.md](VALIDATION.md) records
the checks performed on this founding package.

## Local checks

From the repository root, using only the Python standard library:

```sh
python -m unittest discover -s experiments/rineya_ppt_centralized_reasoning_v1/tests -p 'test_*.py' -v
```

This command runs 34 checks: the original 12 registration checks, 17 independent
exact sampler checks, and five qualification controls. They exercise the
seven-state CPU fixture and 49-state joint
kernel without loading a model, accessing the network, opening a dataset, or
exercising inherited native code. Sampler mechanics and registration checks
remain separate evidence; this command cannot authorize a model launch.

For a measured receipt, run `qualify_cpu.py` with a new package-relative receipt
filename as documented in the [CPU amendment](evidence/CPU_SAMPLER_2026-10-03.md).
The historical [registration.json](registration.json) remains unchanged.
Qualification requires every named check to load and execute successfully;
unrelated tests cannot replace missing mechanics checks. Skips and expected
failures prevent qualification.

## Why keep it alongside other tracks?

PPT-001 asks how interacting searches use a frozen model. CRS-001 asks about
shared evidence and reusable reasoning procedures. Resampling, recurrence,
training, and skill composition each address different interventions. A future
controlled integration may combine them; a positive or negative result in one
track does not automatically validate, retire, or replace another.
