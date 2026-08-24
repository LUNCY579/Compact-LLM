# Figures index

The reviewed Halo visualization program contains **50 manifest-verified PNGs**
in two collections:

- [`halo_controls_v1`](halo_controls_v1): 28 engineering and early-control
  figures;
- [`halo_deep_architecture_v1`](halo_deep_architecture_v1): 22 panels covering
  prospective endpoint gates, the locally registered endpoint result,
  eligible-pool count v2, corrected R7 v2, and Experiments B-F.

Both [`figure_manifest.json`](halo_controls_v1/figure_manifest.json) and the
[deep manifest](halo_deep_architecture_v1/figure_manifest.json) report `PASS`.
Every panel has an exported source-data table, source hashes, an image hash,
and an individual claim limit.

These are visualizations of existing artifacts, not 50 independent
experiments. The early controls mostly reuse 11 B0-selected R1/R3/R4 clusters;
the deep B-D panels reuse 12 difficulty-calibrated endpoint-v2 clusters; the F
panels have only three held-out clusters and two positive-negative AUC pairs.
Local registration, exploratory reuse, runtime-version boundaries, and
underpowered-controller limits continue to apply to the underlying evidence.

## Coverage and limits

The combined collections now visualize:

- family runtime compatibility and B0 capability;
- endpoint calibration, failed prospective v1 B0 gating, difficulty-calibrated
  v2 gating, and the 12-cluster local registered contrast;
- count identity, clean eligible-pool controls, boundary evidence, and held-
  constant schedule interactions;
- open-loop and closed-loop adaptive allocation, including padded bytes;
- attention-prior retention, answers, and entropy;
- memory-slot retention, answer accuracy, slot count, and matched total bytes;
- deterministic recovery risk/coverage and recorded cost;
- tiny-controller AUC, balanced accuracy, and accepted coverage;
- corrected R7 v2 answer-side capability.

The collections still do not supply broad KV-budget, context-length, evidence-
depth, or exponent sweeps; complete evidence/distractor/stale rank trajectories;
fully registered long-horizon layer/head concentration; a memory-slot-count
sweep beyond M0 versus M4; an attention-prior-strength sweep; synchronized GPU
latency/VRAM; or deployment quality-versus-cost evidence. CPU timing is local
instrumentation, and the padded heterogeneous runtime is not a ragged-storage
kernel.

The old Tier-A figures remain under methodology review and are not active Halo
evidence. Always cite the collection manifest and source artifact rather than
using a plot to upgrade an exploratory result.
