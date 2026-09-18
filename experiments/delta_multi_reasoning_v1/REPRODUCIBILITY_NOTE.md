# Two ways an arm's hash moves while the model does not

**Measured 2026-09-18. Relevant to every screen in this programme, because all of them verify
an arm by rebuilding it and comparing `sha256(model.safetensors)` — and that test conflates
"the model changed" with "the file changed".**

Prompted by two open defects: RANK-001 Erratum 2 (r=4 does not rebuild to its recorded hash)
and a COMPOSE-001 arm built by **pure elementwise addition, no SVD**, also failing rebuild
verification. The second cannot be an SVD problem, so the file format itself was worth testing.

## Finding 1 — metadata is hashed, so a computed statistic in it moves the hash

```
save_file(t, p, metadata={"format":"pt"})                             -> aa41eb3910193120
save_file(t, p, metadata={"format":"pt","norm":"0.8122499254369390"}) -> 7e4b4183ab3c2774
save_file(t, p, metadata={"format":"pt","norm":"0.8122499254369391"}) -> 6ddc26cec6e9d3c8
tensor values identical across all three:                                True
```

A one-digit change in a metadata string changes the file sha256 with every tensor bitwise
identical.

## Finding 2 — float32 reductions are thread-order dependent

```
float32 sum of 3e6 elements:   1 thread -> 2999321.25
                              16 threads -> 2999321.5
```

## The two composed

**Putting a float32-reduced statistic in `metadata=` makes the arm's hash depend on thread
count and memory pressure while the weights stay identical.** That is a complete explanation
for a pure-addition arm failing rebuild verification, and it is a *false alarm* when it
happens: the model is the same, only the header differs.

The builders in this programme are safe by luck rather than design — `build_rank.py` writes
`{"format":"pt","rank001_r": str(r)}` and `build_n16.py` writes
`{"format":"pt","dmrf_n16_match_rank":"16"}`, both constant strings. Any builder that starts
recording a realised norm in `metadata=` acquires this defect silently.

**Discipline:** computed statistics belong in `MATERIALIZATION.json`, which is not hashed into
the weights. Keep `metadata=` to constants.

## What this does NOT explain

**Dict insertion order — tested and refuted.** String hashing here *is* randomized per process
(three different `hash()` values for one tensor name across three interpreters), so a builder
iterating a `set` of tensor names gets a per-process-random order. But safetensors normalizes
key order internally: the same tensors written sorted and reversed give byte-identical files.
Ordering is not a hash-stability risk.

**RANK-001's r=4 is a different defect** and this note must not be read as absorbing it.
`build_rank.py`'s metadata is constant, and the concurrent session reproduced the original
`e4f7452e…` twice by holding 3 GiB while building — so r=4 is genuine SVD non-determinism,
sensitive to both thread count and free memory. My own spectral measurement separately
falsified the registered cause (an ill-conditioned rank-4 subspace: σ₄/σ₅ relative gaps are
**2.6× wider** than σ₁₆/σ₁₇, and r=64 with the tightest gaps rebuilds fine).

One refinement that follows: **this builder calls `svdvals` only, never the singular vectors,
and it rebuilds byte-identical.** Singular values are unique; singular subspaces are not. If
`svdvals` output matches bitwise across thread counts while reconstructions differ, the defect
localises to the vector phase and pinning threads is a sufficient fix rather than a workaround.

## Triage for any future hash mismatch

Before diagnosing arithmetic, separate the two cases:

```python
from safetensors import safe_open
import torch
with safe_open(A, framework="pt") as a, safe_open(B, framework="pt") as b:
    print("metadata A:", a.metadata()); print("metadata B:", b.metadata())
    diff = [n for n in a.keys() if not torch.equal(a.get_tensor(n), b.get_tensor(n))]
    print("tensors differing:", len(diff), diff[:5])
```

- **0 tensors differ, metadata differs** → header-only. Same model; the measured scores stand.
- **Tensors differ** → real. Report `max|Δ|` and the count: last-bit drift across many tensors
  reads as accumulation order; a large delta on few tensors reads as a device or dtype branch.

## Conditions under which this experiment's arm is reproducible

`502260c9b28994a4335e9a9ff9f8c954da42032a2818eee3cec16430a9055c36` was built twice,
byte-identical, at `host_free_gib_at_start` **5.6396** and this host's torch default of **16
threads**. `build_n16.py` now pins `torch.set_num_threads` and records the count; the default
is 16 rather than 1 so the shipped default still reproduces the measured hash. The measured
arm's `MATERIALIZATION.json` predates the pin and records `host_free_gib_at_start` but not the
thread count — it is left unedited, and this paragraph is the amendment.
