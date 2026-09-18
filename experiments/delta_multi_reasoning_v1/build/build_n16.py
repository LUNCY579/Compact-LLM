"""Build the N16 arm: the norm-matched magnitude control for the rank-16 truncation.

WHY THIS ARM EXISTS
-------------------
RANK-001 measured that theta_supa + P16(delta) scores 235/384, exactly the full delta's 235,
while storing 2.31% of the delta. The tempting reading is "the 16 leading singular directions
ARE the skill". But P16 does two things at once:

  1. it discards all but 16 directions per matrix  (direction selection), and
  2. it shrinks the per-tensor update norm to 81.2% of the full delta  (magnitude reduction).

INT-001 already established that plain magnitude reduction alone moves these endpoints:
alpha=0.50 scored 219/384 and alpha=0.25 lost a family floor. So without a control that
matches (2) while keeping the full delta's directions, an r16 gain cannot be attributed to
direction selection rather than to the smaller update.

N16 supplies that control. For each eligible 2D matrix:

    s_t      = ||P16(delta_t)||_F / ||delta_t||_F
    delta_t' = s_t * delta_t

so the per-tensor update Frobenius norm of N16 is EXACTLY that of the r16 arm, while every
singular direction of the original delta is retained in its original proportion. Comparing
L (=r16) against N16 isolates direction selection. Comparing either against D (the promoted
k=0.25 mask core) and B (the frozen source) gives the rest of the attribution.

This is ONE algebraically defined arm, not an alpha sweep: every s_t is determined by the
delta itself, with no free parameter.

NONMATRIX POLICY
----------------
Identical to RANK-001's, and declared here rather than discovered later: the 80 one-dimensional
norm deltas are kept IN FULL (s_t = 1), because rank truncation is undefined for a vector and
RANK-001 keeps them in full. They are 46,080 parameters = 0.0146% of the delta's parameters and
0.0146% of its norm (measured, receipts/DELTA_MEASUREMENT.json), so they cannot carry a result
either way. Keeping the policy identical is what makes L and N16 comparable at all.

FP32 throughout, never BF16: INT-001 measured that a BF16 round erases 60.61% of this update's
elements, and SPARSE-001 measured that it would zero 28.49% of the kept elements of a pruned
arm. A BF16 copy of this arm would be a different, measurably worse model.

No model is run. No optimizer step is taken. No existing checkpoint is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

SUPA = Path(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze")
TRAINED = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/int001-20260917/trained_fp32.safetensors")
SUPA_SHA = "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b"
TRAINED_SHA = "07156fb5aa4a8c510c63e8af2118a08a8b6bf004549532d497c2e89e2deec24d"
SIDECARS = ("config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "vocab.json", "merges.txt")

# The rank whose norm profile this arm matches. Fixed at 16 to pair with the registered
# primary rank; --match-rank exists so the r4/r64 controls can be built under the same code
# path, not so that a rank can be chosen after seeing scores.
DEFAULT_MATCH_RANK = 16


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def free_gib() -> float:
    """Free physical memory, by the same GlobalMemoryStatusEx call hostbudget.py uses."""
    import ctypes
    import sys as _sys
    if not _sys.platform.startswith("win"):
        return float("inf")

    class _M(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    m = _M()
    m.dwLength = ctypes.sizeof(_M)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / (1024 ** 3)


def require_free(min_free_gib: float) -> float:
    """Refuse to start when the host cannot take a whole-model FP32 buffer.

    This build accumulates the complete output state dict before `save_file`, which is
    596,049,920 float32 parameters = 2.38 GiB resident, plus the source tensor and SVD
    workspace. The host has 15.22 GiB total and this programme runs ONE model job at a time;
    a concurrent probe battery commits enough that starting here would page. hostbudget.py
    records a measured 10x degradation from exactly that mistake, so this refuses instead --
    matching the probes' own convention rather than inventing a laxer one.
    """
    got = free_gib()
    if got < min_free_gib:
        raise SystemExit(
            f"REFUSING TO START N16 build: needs {min_free_gib:.1f} GiB free physical memory, "
            f"host has {got:.2f} GiB. This build holds a 2.38 GiB FP32 state dict; starting "
            f"while another model job is resident would page the host and slow both. Wait for "
            f"the other job to finish, then re-run. Nothing has been written.")
    return got


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match-rank", type=int, default=DEFAULT_MATCH_RANK, dest="match_rank")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and report the scale factors without writing weights")
    ap.add_argument("--min-free-gib", type=float, default=4.0, dest="min_free_gib",
                    help="refuse to start below this much free physical memory")
    # Default 16 because that is this host's torch default and therefore the count the
    # MEASURED arm 502260c9... was actually built at. Defaulting to 1 would have been tidier
    # and would have silently stopped the shipped default from reproducing the measured hash.
    ap.add_argument("--threads", type=int, default=16,
                    help="torch thread count, PINNED and recorded; see the note below")
    a = ap.parse_args(argv)
    if a.match_rank not in (4, 16, 64):
        raise SystemExit(f"match-rank {a.match_rank} outside the registered grid (4, 16, 64)")
    if a.out.exists() and not a.dry_run:
        raise SystemExit("refusing to overwrite an existing arm directory")
    # The dry run holds one tensor at a time, so it is exempt from the whole-model floor.
    free_at_start = free_gib() if a.dry_run else require_free(a.min_free_gib)

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Thread count is PINNED and recorded in the receipt. RANK-001 Erratum 2 measured that
    # `torch.linalg.svd` on this build is deterministic at a fixed thread count and NOT across
    # thread counts, which is why its r=4 arm did not rebuild to its recorded hash. This arm
    # calls only `svdvals`, and singular values are unique where singular subspaces are not --
    # but a last-bit change in a singular value still moves s_t and therefore the FP32 output,
    # so the hash is only meaningful alongside the count that produced it.
    torch.set_num_threads(a.threads)

    if sha256_file(SUPA / "model.safetensors") != SUPA_SHA:
        raise SystemExit("source weight hash mismatch")
    if sha256_file(TRAINED) != TRAINED_SHA:
        raise SystemExit("trained tensor hash mismatch")

    out: dict = {}
    scales: list[dict] = []
    n_scaled = n_full1d = n_src = n_zero_delta = 0
    kept_sq = full_sq = 0.0

    with safe_open(str(SUPA / "model.safetensors"), framework="pt") as fs, \
         safe_open(str(TRAINED), framework="pt") as ft:
        trained_names = set(ft.keys())
        for name in fs.keys():
            src = fs.get_tensor(name).to(torch.float32).contiguous()
            if name not in trained_names:
                out[name] = src
                n_src += 1
                continue
            trn = ft.get_tensor(name)
            if trn.dtype != torch.float32 or list(trn.shape) != list(src.shape):
                raise SystemExit(f"invalid trained tensor: {name}")
            d = (trn - src).contiguous()
            full_sq += float(d.to(torch.float64).pow(2).sum())

            if d.dim() != 2:
                # 1D norm deltas kept in full, exactly as RANK-001 does. s_t = 1.
                out[name] = trn.contiguous()
                n_full1d += 1
                kept_sq += float(d.to(torch.float64).pow(2).sum())
                scales.append({"name": name, "shape": list(d.shape), "scale": 1.0,
                               "policy": "1d_kept_full"})
                continue

            r = min(a.match_rank, min(d.shape))
            sv = torch.linalg.svdvals(d).to(torch.float64)
            tot = float(sv.pow(2).sum())
            if tot <= 0.0:
                # A zero delta has no direction and no magnitude: the control is the source.
                # Measured count on this delta is 0 (DELTA_MEASUREMENT.json), but handled so
                # the algebra is defined rather than incidentally safe.
                n_zero_delta += 1
                out[name] = src
                scales.append({"name": name, "shape": list(d.shape), "scale": 0.0,
                               "policy": "zero_delta_source_returned"})
                continue
            s_t = (float(sv[:r].pow(2).sum()) / tot) ** 0.5
            scaled = (d.to(torch.float64) * s_t).to(torch.float32).contiguous()
            kept_sq += float(scaled.to(torch.float64).pow(2).sum())
            v = (src + scaled).contiguous()
            if not bool(torch.isfinite(v).all()):
                raise SystemExit(f"non-finite arm tensor: {name}")
            out[name] = v
            n_scaled += 1
            scales.append({"name": name, "shape": list(d.shape), "scale": s_t,
                           "policy": "scaled_to_rank_norm",
                           "delta_fro": tot ** 0.5,
                           "matched_fro": (float(sv[:r].pow(2).sum())) ** 0.5})

    realised = (kept_sq / full_sq) ** 0.5 if full_sq else None
    summary = {
        "match_rank": a.match_rank,
        "tensors_scaled": n_scaled,
        "tensors_1d_kept_full": n_full1d,
        "tensors_zero_delta": n_zero_delta,
        "source_tensors_untouched_count": n_src,
        "delta_norm_fraction_retained": realised,
        "scale_min": min((s["scale"] for s in scales if s["policy"] == "scaled_to_rank_norm"), default=None),
        "scale_max": max((s["scale"] for s in scales if s["policy"] == "scaled_to_rank_norm"), default=None),
    }

    if a.dry_run:
        print(json.dumps({**summary, "dry_run": True}, indent=2))
        return 0

    a.out.mkdir(parents=True)
    save_file(out, str(a.out / "model.safetensors"),
              metadata={"format": "pt", "dmrf_n16_match_rank": str(a.match_rank)})
    for f in SIDECARS:
        shutil.copy2(SUPA / f, a.out / f)
        if sha256_file(a.out / f) != sha256_file(SUPA / f):
            raise SystemExit(f"sidecar copy differs from source: {f}")

    wsha = sha256_file(a.out / "model.safetensors")
    receipt = {
        "schema": 1, "record": "CHECKPOINT_MATERIALIZATION", "layout": "huggingface",
        "what_this_is": (
            "DMRF N16 arm. The frozen source plus the full trained update, rescaled PER TENSOR "
            "so that each tensor's update Frobenius norm equals that of its rank-"
            f"{a.match_rank} truncation. Matches the truncated arm's per-tensor norm profile "
            "while retaining every singular direction. FP32; never routed through BF16. No "
            "model was run to produce these bytes."),
        "experiment": "DMRF-001", "arm": f"N{a.match_rank}",
        "role": ("magnitude control for the rank-"
                 f"{a.match_rank} arm; isolates direction selection from magnitude reduction"),
        "formula": "theta_supa + s_t * delta_t, s_t = ||P_r(delta_t)||_F / ||delta_t||_F per 2D tensor",
        "nonmatrix_policy": "80 one-dimensional norm deltas kept in full (s_t = 1), as RANK-001 does",
        "zero_denominator_policy": "||delta_t|| == 0 returns the source tensor and records scale 0.0",
        **summary,
        "source": str(SUPA), "source_weight_sha256": SUPA_SHA,
        "trained_fp32_sha256": TRAINED_SHA,
        "checkpoint": ("C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/"
                       "rrp015-20260917/control-journal/runs/train-001/microfit/checkpoint-00028"),
        "checkpoint_manifest_sha256": "3d5cef318a578c71e3d0fe10bdeb27657a31c21d4826f26a6a276e07cacd50cc",
        "out_weight_sha256": wsha, "stored_dtype": "float32",
        "tensors_in_checkpoint": 220,
        "sidecars_copied": list(SIDECARS),
        "analysis_dtype": "float64 for norms and scale factors; float32 stored",
        "tf32": False,
        "host_free_gib_at_start": free_at_start,
        "host_free_gib_floor": a.min_free_gib,
        "torch_num_threads_pinned": a.threads,
        "hash_reproducibility_note": (
            "out_weight_sha256 is reproducible only at torch_num_threads_pinned. RANK-001 "
            "Erratum 2 measured thread-count-dependent SVD output on this build; this arm pins "
            "the count rather than relying on a default."),
        "model_calls": 0, "optimizer_steps": 0, "training_runs": 0,
        "quality_claim": False, "release_permitted": False,
        "per_tensor_scales": scales,
    }
    (a.out / "MATERIALIZATION.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    (a.out / "ARM.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**summary, "out_weight_sha256": wsha}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
