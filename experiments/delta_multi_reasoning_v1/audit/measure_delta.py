"""DMRF audit: measure the RRP-015 control delta exactly, in every scope that matters.

This exists because three separate report claims depend on numbers whose SCOPE was never
stated, and the scope changes the answer:

  1. "the norm ratio is approximately 0.000558" -- ratio of what to what, over which tensors?
     The same delta has a different relative norm over the 220 trained tensors than over all
     310, and a different one again per block.
  2. "per-block ||dW|| is 0.169-0.199, near-uniform" -- absolute or relative?
  3. "the task delta is low-rank; effective rank 16-72" -- established from FOUR sampled
     tensors in the RANK-001 prereg. This measures ALL 2D trained tensors so the claim can be
     checked against the population rather than a sample.

It also emits the per-tensor scale factors needed to build the N16 norm-matched control:
  s_t = ||P16(delta_t)||_F / ||delta_t||_F      (per eligible 2D matrix)

Read-only with respect to every existing artifact. Writes one JSON receipt.
No model is run; no checkpoint is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from collections import defaultdict
from pathlib import Path

SUPA = Path(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze")
TRAINED = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/int001-20260917/trained_fp32.safetensors")
SUPA_SHA = "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b"
TRAINED_SHA = "07156fb5aa4a8c510c63e8af2118a08a8b6bf004549532d497c2e89e2deec24d"

# Ranks whose retained-energy share we tabulate per tensor. 16 is the primary registered rank.
RANK_GRID = (4, 16, 64)
BLOCK_RE = re.compile(r"model\.layers\.(\d+)\.")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def sumsq(t, chunk_rows: int = 4096) -> float:
    """Sum of squares in float64, accumulated in row chunks.

    Chunked because the tied embedding is 151936x1024: a whole-tensor float64 cast would
    spike 1.24 GiB, and this host's retention probes refuse below 5.0 GiB free. Bounded
    memory here is the difference between this script running and the guard refusing.
    """
    import torch
    if t.dim() == 0:
        return float(t.to(torch.float64).pow(2).sum())
    total = 0.0
    for i in range(0, t.shape[0], chunk_rows):
        total += float(t[i:i + chunk_rows].to(torch.float64).pow(2).sum())
    return total


def effective_rank(sv) -> float:
    """Participation-ratio effective rank: (sum s^2)^2 / sum s^4, on singular values.

    This is the same quantity the RANK-001 prereg tabulates (it reports 16.2 for L27 o_proj
    against a 1024 ambient dimension), stated explicitly here so the number is reproducible
    rather than inherited.
    """
    e = sv.pow(2)
    tot = float(e.sum())
    if tot <= 0.0:
        return 0.0
    return float(tot ** 2 / float(e.pow(2).sum()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-hash", action="store_true",
                    help="skip the 3.6 GB of hashing when identity was verified this session")
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite an existing receipt: {a.out}")

    import torch
    from safetensors import safe_open

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    identity = {"source_sha256_expected": SUPA_SHA, "trained_sha256_expected": TRAINED_SHA}
    if a.skip_hash:
        identity["verified"] = "SKIPPED_BY_FLAG"
    else:
        got_s = sha256_file(SUPA / "model.safetensors")
        got_t = sha256_file(TRAINED)
        if got_s != SUPA_SHA or got_t != TRAINED_SHA:
            raise SystemExit(f"identity mismatch: source={got_s} trained={got_t}")
        identity["verified"] = "RECOMPUTED_AND_MATCHED"

    per_tensor: list[dict] = []
    # Accumulators. Everything is summed as squared Frobenius norms in float64.
    sq = defaultdict(float)   # scope -> sum of squared delta entries
    theta_sq = defaultdict(float)
    n_params = defaultdict(int)
    block_sq: dict[int, float] = defaultdict(float)
    block_theta_sq: dict[int, float] = defaultdict(float)
    counts = {"tensors_total": 0, "tensors_trained": 0, "trained_2d": 0, "trained_1d": 0,
              "trained_other_dim": 0, "tensors_untrained": 0, "zero_delta_trained": 0}

    with safe_open(str(SUPA / "model.safetensors"), framework="pt") as fs, \
         safe_open(str(TRAINED), framework="pt") as ft:
        trained_names = set(ft.keys())
        for name in fs.keys():
            counts["tensors_total"] += 1
            src = fs.get_tensor(name)
            t_sq = sumsq(src)
            theta_sq["all"] += t_sq
            n_params["all"] += src.numel()
            blk = BLOCK_RE.search(name)
            bi = int(blk.group(1)) if blk else -1

            if name not in trained_names:
                counts["tensors_untrained"] += 1
                theta_sq["untrained"] += t_sq
                n_params["untrained"] += src.numel()
                continue

            counts["tensors_trained"] += 1
            trn = ft.get_tensor(name)
            if list(trn.shape) != list(src.shape):
                raise SystemExit(f"shape mismatch on trained tensor {name}")
            if trn.dtype != torch.float32:
                raise SystemExit(f"trained tensor {name} is {trn.dtype}, expected float32")
            # Trained tensors live in blocks 8-27 only; the largest is 3072x1024, so a
            # whole-tensor float64 delta is 25 MB and needs no chunking.
            d = trn.to(torch.float64) - src.to(torch.float64)
            d_sq = float(d.pow(2).sum())
            sq["all"] += d_sq
            sq["trained"] += d_sq
            theta_sq["trained"] += t_sq
            n_params["trained"] += d.numel()
            block_sq[bi] += d_sq
            block_theta_sq[bi] += t_sq
            if d_sq == 0.0:
                counts["zero_delta_trained"] += 1

            rec = {"name": name, "shape": list(d.shape), "block": bi,
                   "delta_fro": d_sq ** 0.5, "theta_fro": t_sq ** 0.5,
                   "relative_fro": (d_sq / t_sq) ** 0.5 if t_sq > 0 else None,
                   "numel": int(d.numel())}

            if d.dim() == 2:
                counts["trained_2d"] += 1
                sq["trained_2d"] += d_sq
                n_params["trained_2d"] += d.numel()
                sv = torch.linalg.svdvals(d.to(torch.float32)).to(torch.float64)
                tot = float(sv.pow(2).sum())
                rec["effective_rank"] = effective_rank(sv)
                rec["ambient_rank"] = int(min(d.shape))
                rec["top1_energy"] = float(sv[0].pow(2)) / tot if tot > 0 else None
                for r in RANK_GRID:
                    rr = min(r, int(min(d.shape)))
                    kept = float(sv[:rr].pow(2).sum())
                    # energy share = ||P_r(delta)||_F^2 / ||delta||_F^2
                    rec[f"energy_r{r}"] = kept / tot if tot > 0 else None
                    # the N16-style per-tensor scale factor is the sqrt of the energy share
                    rec[f"normscale_r{r}"] = (kept / tot) ** 0.5 if tot > 0 else None
                    rec[f"factored_params_r{r}"] = int(rr * (d.shape[0] + d.shape[1]))
            elif d.dim() == 1:
                counts["trained_1d"] += 1
                sq["trained_1d"] += d_sq
                n_params["trained_1d"] += d.numel()
            else:
                counts["trained_other_dim"] += 1
            per_tensor.append(rec)

    def ratio(num_key, den_key):
        n, dd = sq[num_key], theta_sq[den_key]
        return (n / dd) ** 0.5 if dd > 0 else None

    scopes = {
        "delta_over_whole_model": {
            "definition": "||delta||_F / ||theta_supa||_F with delta zero on untrained tensors, "
                          "denominator over ALL 310 tensors",
            "value": ratio("all", "all"),
        },
        "delta_over_trained_tensors_only": {
            "definition": "||delta||_F / ||theta_supa restricted to the 220 trained tensors||_F",
            "value": ratio("trained", "trained"),
        },
        "delta_2d_over_trained_tensors": {
            "definition": "||delta restricted to 2D trained matrices||_F / ||theta_supa trained||_F",
            "value": ratio("trained_2d", "trained"),
        },
    }

    per_block = []
    for bi in sorted(b for b in block_sq if b >= 0):
        per_block.append({
            "block": bi,
            "delta_fro": block_sq[bi] ** 0.5,
            "theta_fro": block_theta_sq[bi] ** 0.5,
            "relative_fro": (block_sq[bi] / block_theta_sq[bi]) ** 0.5 if block_theta_sq[bi] else None,
        })
    abs_norms = [b["delta_fro"] for b in per_block]
    rel_norms = [b["relative_fro"] for b in per_block if b["relative_fro"] is not None]

    twod = [r for r in per_tensor if r["shape"] and len(r["shape"]) == 2]
    eff = sorted(r["effective_rank"] for r in twod)
    e16 = sorted(r["energy_r16"] for r in twod)

    def pct(xs, q):
        if not xs:
            return None
        i = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
        return xs[i]

    receipt = {
        "schema": 1,
        "record": "DMRF_DELTA_MEASUREMENT",
        "experiment": "DMRF-001",
        "what_this_is": (
            "An exact, whole-population measurement of the RRP-015 control delta. Computed in "
            "float64 from the frozen source and the consolidated FP32 trained tensors. No model "
            "was run and no checkpoint was modified."),
        "identity": identity,
        "source": str(SUPA),
        "trained": str(TRAINED),
        "analysis_dtype": "float64 accumulation; float32 inputs to svdvals",
        "tf32": False,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "counts": counts,
        "param_counts": dict(n_params),
        "norm_scopes": scopes,
        "delta_fro_absolute": sq["all"] ** 0.5,
        "theta_fro_absolute": theta_sq["all"] ** 0.5,
        "delta_share_of_norm_in_1d_tensors": (sq["trained_1d"] / sq["trained"]) if sq["trained"] else None,
        "delta_share_of_params_in_1d_tensors": (n_params["trained_1d"] / n_params["trained"]) if n_params["trained"] else None,
        "per_block": per_block,
        "per_block_absolute_range": [min(abs_norms), max(abs_norms)] if abs_norms else None,
        "per_block_relative_range": [min(rel_norms), max(rel_norms)] if rel_norms else None,
        "per_block_absolute_spread_ratio": (max(abs_norms) / min(abs_norms)) if abs_norms and min(abs_norms) > 0 else None,
        "spectrum_population": {
            "n_2d_tensors_measured": len(twod),
            "effective_rank_min": eff[0] if eff else None,
            "effective_rank_p50": pct(eff, 0.5),
            "effective_rank_max": eff[-1] if eff else None,
            "energy_r16_min": e16[0] if e16 else None,
            "energy_r16_p50": pct(e16, 0.5),
            "energy_r16_max": e16[-1] if e16 else None,
            "n_tensors_with_energy_r16_below_0p59": sum(1 for x in e16 if x < 0.59),
            "note": ("The RANK-001 prereg tabulated FIVE named tensors and described the range as "
                     "'effective rank 16-72, 16 directions carry 59-88% of the energy'. These are "
                     "the same statistics over ALL 2D trained matrices."),
        },
        "per_tensor": per_tensor,
        "quality_claim": False,
        "release_permitted": False,
        "model_calls": 0,
        "optimizer_steps": 0,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "counts": counts,
        "norm_scopes": {k: v["value"] for k, v in scopes.items()},
        "per_block_absolute_range": receipt["per_block_absolute_range"],
        "per_block_relative_range": receipt["per_block_relative_range"],
        "spectrum": {k: v for k, v in receipt["spectrum_population"].items() if k != "note"},
        "delta_share_of_norm_in_1d_tensors": receipt["delta_share_of_norm_in_1d_tensors"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
