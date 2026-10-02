# PPT-001: centralized reasoning through interacting search

**Dedicated branch:** `research/rineya-ppt-centralized-reasoning-v1`  
**Initial target:** frozen Rineya 0.6B; no checkpoint has been selected or loaded.  
**Current state:** `DESIGN_RECORDED_NATIVE_NOT_IMPLEMENTED`

This package records an enduring, complementary research direction inspired by
Parallel Power Tempering. It is not a completed sampler or a runnable native
experiment. No Rineya accuracy or efficiency result is claimed.

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

The checks read only this package and its root charter. They do not load a model,
access the network, run a sampler, open a dataset, or exercise inherited native
code. An implementation successor must supply separate mathematical and native
qualification tests; this command cannot authorize a model launch.

## Why keep it alongside other tracks?

PPT-001 asks how interacting searches use a frozen model. CRS-001 asks about
shared evidence and reusable reasoning procedures. Resampling, recurrence,
training, and skill composition each address different interventions. A future
controlled integration may combine them; a positive or negative result in one
track does not automatically validate, retire, or replace another.
