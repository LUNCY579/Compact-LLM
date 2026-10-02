# Founding-package validation

**Date:** 2026-10-02  
**Scope:** local design/registration checks only; not sampler or native qualification.  
**Result:** 12 tests passed locally using Python 3.13.5 (standard library only).

The standard-library test command is:

```sh
python -m unittest discover -s experiments/rineya_ppt_centralized_reasoning_v1/tests -p 'test_*.py' -v
```

The checks cover identity, pinned provenance and paper version, complementary
preservation and owner-controlled disposition, reversible parking, closed
execution permissions, absence of native result claims, unset launch
prerequisites, external-reference boundaries, comparison arms, document
presence and relative-link resolution.

This receipt does not measure or qualify a Markov chain, language model,
checkpoint, cache, dataset, verifier, host memory, model accuracy or latency.
No native or inherited repository test suite is run by this command. Passing
these checks does not grant permission to execute future experimental stages.

The founding change is limited to the new root PPT charter and this new package.
No inherited file is intentionally modified or deleted. Remote publication and
comparison must be verified separately; local checks alone do not establish
that a branch was pushed.
