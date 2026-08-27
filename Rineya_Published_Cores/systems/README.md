# Rineya published family systems

These are versioned, runnable inference systems backed up alongside the dense
core vault. They are not single checkpoints, because their measured scores
depend on answer-blind routing across multiple model predictions.

| System | Form | Showcase | Stability | All 256 |
|---|---|---:|---:|---:|
| **Supa 2.0 System48** | 9-arm ordered consensus cascade | **48/64** | **128/192** | **176/256** |
| **Baya 2.0 System46** | 3-arm UNKNOWN cascade | **46/64** | **125/192** | **171/256** |
| Nimbus DualScale PEFT Gate45 | 2 PEFT executions | **45/64** | **124/192** | **169/256** |
| Supa-Baya Dense Gate42 | 2 dense-core executions | **42/64** | **124/192** | **166/256** |

Supa Core 1.7 and Baya Core 1.7 remain immutable 40/64 dense dependency
anchors. Their weights are not relabeled with system scores. Each system bundle
contains `SYSTEM.json`, adapter assets where needed, frozen evidence, hashes,
and `run_family_system.py` for prediction-only replay.

Showcase64 was selection data. Stability192 was frozen validation. Sealed TEST
was not run.
