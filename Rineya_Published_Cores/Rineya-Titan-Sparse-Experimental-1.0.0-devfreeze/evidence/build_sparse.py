"""Build one SPARSE-001 arm: theta_supa + M_k (x) delta, FP32, per-tensor top-k by |delta|.

M_k keeps the largest k share of |delta| entries within EACH trained tensor, ties broken by
flat index ascending, so the mask is deterministic. FP32 throughout; never BF16, because
INT-001 measured that a BF16 round erases 60.61% of this update's elements.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil
from pathlib import Path

GRID = (0.10, 0.25, 0.50)
SUPA = Path(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze")
TRAINED = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/int001-20260917/trained_fp32.safetensors")
TRAINED_SHA = "07156fb5aa4a8c510c63e8af2118a08a8b6bf004549532d497c2e89e2deec24d"
SUPA_SHA = "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b"
SIDECARS = ("config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "vocab.json", "merges.txt")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while c := f.read(8 * 1024 * 1024):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.keep not in GRID:
        raise SystemExit(f"keep {a.keep} outside the registered grid {GRID}")
    if a.out.exists():
        raise SystemExit("refusing to overwrite an existing arm directory")

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if sha256_file(SUPA / "model.safetensors") != SUPA_SHA:
        raise SystemExit("source weight hash mismatch")
    if sha256_file(TRAINED) != TRAINED_SHA:
        raise SystemExit("trained tensor hash mismatch")

    a.out.mkdir(parents=True)
    out = {}
    kept_total = elems_total = 0
    delta_sq_kept = delta_sq_all = 0.0
    n_masked = n_src = 0

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
            flat = d.view(-1)
            n = flat.numel()
            k = int(n * a.keep)          # floor; realised sparsity is reported, not assumed
            elems_total += n
            delta_sq_all += float(flat.pow(2).sum())
            if k <= 0:
                out[name] = src
                n_masked += 1
                continue
            # Deterministic tie-break. torch.topk(sorted=False) does NOT define which of
            # several equal magnitudes it keeps, and this delta has many exact ties, so two
            # builds selected different tied positions and produced different bytes with an
            # identical retained norm. A STABLE descending argsort breaks ties by ascending
            # original index, which is what the registration actually specified.
            mag = flat.abs()
            order = torch.argsort(mag, descending=True, stable=True)
            keep = torch.zeros(n, dtype=torch.bool)
            keep[order[:k]] = True
            masked = torch.where(keep, flat, torch.zeros((), dtype=torch.float32))
            kept_total += int(keep.sum())
            delta_sq_kept += float(masked.pow(2).sum())
            v = (src.view(-1) + masked).view(src.shape).contiguous()
            if not bool(torch.isfinite(v).all()):
                raise SystemExit(f"non-finite arm tensor: {name}")
            out[name] = v
            n_masked += 1

    save_file(out, str(a.out / "model.safetensors"),
              metadata={"format": "pt", "sparse001_keep": str(a.keep)})
    for f in SIDECARS:
        shutil.copy2(SUPA / f, a.out / f)
        if sha256_file(a.out / f) != sha256_file(SUPA / f):
            raise SystemExit(f"sidecar copy differs from source: {f}")

    wsha = sha256_file(a.out / "model.safetensors")
    realised_elems = kept_total / elems_total
    realised_norm = (delta_sq_kept / delta_sq_all) ** 0.5

    receipt = {
        "schema": 1, "record": "CHECKPOINT_MATERIALIZATION", "layout": "huggingface",
        "what_this_is": ("SPARSE-001 arm. The frozen source plus a per-tensor magnitude-masked "
                         "slice of the checkpoint's update, in FP32. Never routed through BF16. "
                         "No model was run to produce these bytes."),
        "experiment": "SPARSE-001", "keep_fraction_nominal": a.keep,
        "keep_fraction_realised_elements": realised_elems,
        "delta_norm_fraction_retained": realised_norm,
        "mask": "per-tensor top-k by |delta|, ties by ascending flat index",
        "source": str(SUPA), "source_weight_sha256": SUPA_SHA,
        "trained_fp32_sha256": TRAINED_SHA,
        "checkpoint": ("C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/"
                       "rrp015-20260917/control-journal/runs/train-001"),
        "checkpoint_experiment": "MQ-001",
        "checkpoint_manifest_sha256": "3d5cef318a578c71e3d0fe10bdeb27657a31c21d4826f26a6a276e07cacd50cc",
        "out_weight_sha256": wsha, "stored_dtype": "float32",
        "tensors_in_checkpoint": 220, "tensors_applied": n_masked,
        "sidecars_copied": list(SIDECARS),
        "source_tensors_untouched_count": n_src,
        "quality_claim": False, "release_permitted": False,
    }
    (a.out / "MATERIALIZATION.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (a.out / "ARM.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in (
        "keep_fraction_nominal", "keep_fraction_realised_elements",
        "delta_norm_fraction_retained", "out_weight_sha256", "tensors_applied")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
