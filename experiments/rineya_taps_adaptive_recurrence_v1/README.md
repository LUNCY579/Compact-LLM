# TAPS-RINEYA-001 — trajectory-adaptive recurrent update scaling

**Research recommendation: BLOCKED for native qualification.** The frozen
inference apparatus and scoped middle-block bridge are implemented and CPU-tested.
No Rineya weights were loaded; no native accuracy, speedup or training result is
claimed. Start with [RESULTS.md](RESULTS.md), [VALIDATION.md](VALIDATION.md), and
[RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md).

## The three controls are different

Whether to recur, how many iterations to run, and how much of each update to apply
are separate controls. This package tests the last one against no extra recurrence,
unit recurrence, a DEV-selected global fixed alpha, a retrospective oracle and a
DEV-distribution random control. It does not replace prior recurrence/gating work.

For the audited ARMR location, keep the normal L12–L13 block pass unchanged. Use
that whole-block update to initialize trajectory history. Then optionally apply
`old + alpha * (proposal - old)` to ONE extra block application. This provides two
observations without adding an extra warmup loop. The first native lane does not
license the previously failed three-total-application ARMR arm.

## Implementation map

| File | Responsibility |
|---|---|
| `scheduler.py` | Validated config/state; per-sequence bias-corrected moments; bounded TAPS/fixed/unit/random scaling; FP32 statistics; explicit reset and padding contract. |
| `integration.py` | Scoped one/two-layer no-cache middle-block wrapper; native first pass unchanged; exact unit passthrough; restoration and overlap guards. |
| `experiment.py` | Live analytic fixtures, DEV freeze, TEST firewall, complete fixed grid, bounded exhaustive schedule oracle, streamed evidence and generated tables. |
| `native.py` | Exact-source CPU Supa loader/ARMR scorer bridge; fresh identity and all-arm smoke gates; complete MERGEDEV-fast DEV matrix; no native TEST entry point. |
| `diagnostics.py` | Disables noncausal full-gold NLL without changing inherited generation/scoring; prevents label-conditioned alphas entering the random pool. |
| `safety.py` | Exclusive journals, resource guards, checksums and no-download preflight. |
| `tests/test_taps.py` | CPU numerical, controller, firewall, host-wrapper and end-to-end apparatus checks. |
| `scripts/` | Generated research report and hash-checked execution of three unchanged inherited numerical tests. |

Paper review: full arXiv v1 HTML, mathematical appendices, HF paper/model pages,
Sudoku author `trm.py` and YAML. An official HF implementation exists. No separate
author GitHub repository was found; no author checkpoint or code was executed or
vendored. See [PAPER_NOTES.md](PAPER_NOTES.md) for exact sources and differences.

## Commands, from the repository root

Python with PyTorch, pytest and psutil is enough for the safe apparatus lane.
These commands never download a model. Every output directory must be new.

```sh
python -m pytest -q experiments/rineya_taps_adaptive_recurrence_v1/tests
python -m experiments.rineya_taps_adaptive_recurrence_v1 fixture --out /tmp/taps-fixture-new
python -m experiments.rineya_taps_adaptive_recurrence_v1 native-preflight --out /tmp/taps-preflight-new
python experiments/rineya_taps_adaptive_recurrence_v1/scripts/report.py \
  --run /tmp/taps-fixture-new --out /tmp/taps-report-new.md
```

Native preflight is expected to exit **2** when blocked. To inspect the real host,
provide `--source /exact/local/Supa-A1.1-directory`. After metadata readiness,
`native-dev` uses the same arguments and performs fresh native identity and smoke
qualification before the DEV factorial. It requires the full inherited checkout,
Python 3.11.15, Torch 2.12.1+cu126, Transformers 5.14.1, exact Qwen implementation,
checkpoint/tokenizer/custom-code hashes, and at least 7.4 GiB available host RAM.
The new process RSS ceiling is 6 GiB; all inference is CPU FP32/eager and cache-free.
The supplied native runner has not been executed against those real dependencies.

## Evidence and limits

`results/fixture_v3/` contains generated decision/compute/trajectory tables and
source/config/runtime manifests for the final instrumented fixture run.
`results/native_preflight_v1/` records the actual native blockers.
The full original JSONL journals, earlier apparatus run and validation logs are
retained in the handoff bundle; [artifacts/EVIDENCE_MANIFEST.json](artifacts/EVIDENCE_MANIFEST.json)
records their hashes. They are not replaced by invented or manually entered data.
Repository tables are review artifacts, not a claim that every full raw journal
is hosted in Git. Re-run into a fresh directory to regenerate full JSONL locally.

TAPS does not beat the best global fixed-alpha control on these fixtures. One DEV
oracle contrast has headroom that this controller does not capture; the fixture
TEST contrast has no such headroom. These are apparatus results only. Do not
promote, train, select a native checkpoint or open protected data on their basis.
The next substantive experiment is the exact-source native DEV grid and trajectory
screen, not a larger architecture rewrite or a training run.
