# Vendored modules (byte-identical copies; do not edit)

| file | source | sha256 |
|---|---|---|
| hostbudget.py | `git show 42eb390b:experiments/rab001_exploration/hostbudget.py` (commit 42eb390b8ffde8cb61eb1bd2c47af2ad994d3426, not an ancestor of this branch) | 9f9dcf351f6c9ebfe00bf4e470296cd48db424585f268c205801bd9b0f38acd1 |

`hostbudget` is the programme's authoritative free-physical-RAM gate (GlobalMemoryStatusEx).
It is vendored rather than re-implemented so every lane measures the same resource the same way.
The GPU lease is NOT vendored: it is imported from `experiments/fresh_dual_skill_instrument_v1/gpu_lease.py`
so the lock semantics stay identical to every peer session on this machine.
