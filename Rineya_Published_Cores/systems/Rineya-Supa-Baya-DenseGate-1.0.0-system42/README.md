# Rineya Supa-Baya Dense Gate 42

Status: **Showcase-selected, Stability-validated family system**.

This directory is a runnable multi-model system bundle. It is **not** a single
`model.safetensors` checkpoint. The measured score depends on the exact
answer-blind routing rule across 2 inference arms.

- Showcase (selection): **42/64**
- Stability (frozen validation): **124/192**
- Complete public Demo: **166/256**

Run `python run_family_system.py --replay <split>` to replay bundled prediction
arms and verify the canonical output hash. See `SYSTEM.json` for live arm base,
adapter, scale, hashes, and routing order.

This 42 result is the answer-blind gate over the two preserved dense 1.7 cores. It is distinct from the coincidental 42-point single PEFT arm.
