# Rineya Nimbus DualScale PEFT Gate 45

Status: **Showcase-selected, Stability-validated family system**.

This directory is a runnable multi-model system bundle. It is **not** a single
`model.safetensors` checkpoint. The measured score depends on the exact
answer-blind routing rule across 2 inference arms.

- Showcase (selection): **45/64**
- Stability (frozen validation): **124/192**
- Complete public Demo: **169/256**

Run `python run_family_system.py --replay <split>` to replay bundled prediction
arms and verify the canonical output hash. See `SYSTEM.json` for live arm base,
adapter, scale, hashes, and routing order.

The 45 score requires two PEFT executions. Dense materialization changes predictions and does not inherit this result.
