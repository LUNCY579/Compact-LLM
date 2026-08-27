# Rineya Baya 46

Status: **Showcase-selected, Stability-validated family system**.

This directory is a runnable multi-model system bundle. It is **not** a single
`model.safetensors` checkpoint. The measured score depends on the exact
answer-blind routing rule across 3 inference arms.

- Showcase (selection): **46/64**
- Stability (frozen validation): **125/192**
- Complete public Demo: **171/256**

Run `python run_family_system.py --replay <split>` to replay bundled prediction
arms and verify the canonical output hash. See `SYSTEM.json` for live arm base,
adapter, scale, hashes, and routing order.

Showcase selected the three-arm cascade; Stability was frozen validation. This is a routed system, not one dense checkpoint.
