# Cross-scale composable skill study, CSC-001

This directory implements an offline, NumPy-only mechanics study. No training stack,
model loader, download helper, native execution grant, or automatic stage advancement
is included. Read the root prior-work/binding audits and PREREGISTRATION.md first.

From the dedicated worktree in PowerShell:

```powershell
uv venv .venv-cross-scale --python 'C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe'
uv pip install --python .venv-cross-scale/Scripts/python.exe -r experiments/cross_scale_composable_001/requirements-offline.txt
.venv-cross-scale/Scripts/python.exe -m unittest discover -s experiments/cross_scale_composable_001 -t . -v
.venv-cross-scale/Scripts/python.exe -m experiments.cross_scale_composable_001.fixture_run --out C:/absolute/new-fixture-receipt.json
```

Use a fresh output path; existing evidence is never overwritten. Run tests before
staging a source commit; source cleanliness is checked separately after committing.
`mechanics.py` is pure algebra and cannot authenticate caller provenance. Use the
`receipts.py` gates before fitting, composing or opening endpoint payloads. The
fixture runner demonstrates that sequence on synthetic data only.

Reused designs: XSG's affine transport/delta-without-bias distinction; layerwise soup
strict tensor inventory; LoRA B@A semantics; existing paired exact discordance and
byte-bound provenance patterns. New code is isolated to avoid importing native
dependencies or altering historical lanes. This is a focused implementation of the
new residual mechanism and its controls, not a claim that those mathematical
primitives or representation transport were invented here.

Evidence copies live under `evidence/` and a new namespace in the existing records
repository. Byte identity is preserved with `-text`. Source binding inventories
exclude generated receipts to avoid circular hashes; final source commit/cleanliness
is recorded in an append-only external receipt after committing.

Native capture, residual-loss adapter training, and behavioral evaluation are future
work behind explicit authority. Native calls remain zero. A fixture's synthetic
scores or planted linear relationship are never model evidence.
