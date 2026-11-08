# Track B alignment matrix figures

The two final figures in this directory show all 96 frozen held-out calibration
cells: sixteen Supa/donor depth pairs and B0, B1, B2, B3, BS, BR. They visualize
ordinary-token transformation fidelity, not generation efficacy. Each figure is
available as a 2574 × 1826 PNG and a vector PDF with embedded TrueType fonts.

- `ordinary_delta_cosine`: mean per-token transformation cosine; one common
  linear color scale from 0 to 1, covering every observed value.
- `ordinary_delta_relative_mse`: mean per-token transformation relative MSE;
  one common logarithmic color scale from 0.25 to 128, covering every observed
  value. Cell annotations are the untransformed metric, rounded to two decimals.
- `ordinary_token_metrics.csv`: all 96 cells at full stored precision.
- `input_sha256_receipt.json`: input and output hashes, rendering versions,
  coverage checks, selected cell, and committed-input verification.

The outlined cell is the selection already recorded in the committed
`alignment-cpu-v2-002/aggregate/attempt_000/selection.json`: B3 at Supa L19 and
Qwen3-4B L24. It has ordinary-token cosine 0.8491775842112589 and relative MSE
0.2820784626027358. Selection was not recomputed or altered for these figures.
All panels use 32 held-out calibration scenarios and 2,395 ordinary tokens.
Token observations are not independent efficacy examples. These figures do not
summarize special-token or final-token fidelity; those strata remain in the
complete source matrix, and the selected B3 has worse special-token fidelity
than the same-pair B0.

Source matrix SHA256:
`ad91c4776179ab9a047d8b2acbc888cede0701ec72b7664a08076dec71898d00`.
Source selection SHA256:
`e05a8a8f4f6807f60b0f00809107f2ab2df753a601119f09b92802a0688f619b`.
Both input files were compared byte-for-byte against Git HEAD before rendering.

Visual QA: both final PNGs were inspected using `view_image`. The initial draft
had an x-axis/caption collision; the final layout increases the lower margin.
The initial script, receipt and outputs are preserved in `draft_001/`. The final
PNG inspection found no clipped titles, missing labels, overlapping text or
illegible cell annotations. PDFs are emitted from the same Matplotlib figures.

The renderer imports only plotting/standard libraries; it does not import Torch,
load models, access evaluation answers or modify any experimental input. Its
outputs use exclusive file creation. To reproduce, run the script in an empty
copy of this artifact directory at the same programme-relative location, or
preserve existing outputs elsewhere before running it again.
