"""CONFIRM-002 — build H25, TH25 and THFULL (PREREG.md §2, 779cd431). FP32, HF (Qwen) namespace.

    H25     = BASE   + 0.25 * P16(delta_H)
    TH25    = SPARSE + 0.25 * P16(delta_H)
    THFULL  = SPARSE + 1.00 * delta_H          (unrounded)

- BASE (Supa cast to FP32) and SPARSE (Supa + dSparse, FP32) are the existing, preflight-hashed arms in
  titan_opallo_skill_vector_v1/arms. Adding H to exactly those files makes TH25 - SPARSE differ by the
  H term and nothing else.
- delta_H = trained_H_fp32 - BASE on the 220 trained tensors (compose001, sha a1103060...).
- P16 matches indep001-20260921/build_indep.py exactly: float64 SVD, rank 16 for 2-D tensors, and 1-D
  tensors passed through whole. The sum is formed in float64, then cast to float32.

CANARY, required before anything is written: sqrt(sum ||P16(delta_H)||^2) must reproduce INDEP's
recorded 0.7098943555451119 within 2e-6 (INDEP's TOL), and ||SPARSE - BASE|| must reproduce ||dSparse||
= 0.6940550846962167 within 1e-6 relative.
VERIFY, after writing: untouched tensors are bitwise equal to their base arm, and each H tensor's
||out - base|| equals its intended contribution within float32 rounding.
Create-only outputs under composition_confirmation_v2/arms/. Sources are read-only.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, shutil, struct, sys
from pathlib import Path
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
SV = EXP / "titan_opallo_skill_vector_v1" / "arms"
TRAINED_H = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/compose001-20260918/trained_H_fp32.safetensors")
TRAINED_H_SHA = "a1103060591a91ee7d9ceb2ad7450621be352b087cd19188f2c383a0dbbc7b90"
RANK = 16
P16_H_NORM_RECORDED = 0.7098943555451119          # indep001 MATERIALIZATION norm_P16_delta_H
DSPARSE_NORM = 0.6940550846962167
ARMS = {"H25": ("BASE", 0.25, "p16"), "TH25": ("SPARSE", 0.25, "p16"), "THFULL": ("SPARSE", 1.00, "full")}
ASSETS = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json", "config.json")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def p16(delta: torch.Tensor) -> torch.Tensor:
    if delta.ndim != 2:
        return delta.double()
    U, S, Vh = torch.linalg.svd(delta.double(), full_matrices=False)
    r = min(RANK, S.numel())
    return (U[:, :r] * S[:r]) @ Vh[:r]


def check_sources():
    for a in ("BASE", "SPARSE"):
        want = json.loads((SV / a / "BUILD.json").read_text(encoding="utf-8"))["weight_sha256"]
        if sha256(SV / a / "model.safetensors") != want:
            raise SystemExit(f"source arm {a} does not match its BUILD.json hash")
    if sha256(TRAINED_H) != TRAINED_H_SHA:
        raise SystemExit("trained_H_fp32 does not match its backup-manifest hash")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--outdir", type=Path, default=HERE / "arms")
    ap.add_argument("--smoke-keys", nargs="*", default=None, help="restrict to these tensors; skips the canary (smoke test only)")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    for arm in a.arms:
        if (a.outdir / arm).exists():
            raise SystemExit(f"refusing to overwrite {a.outdir / arm}")
    if not a.smoke_keys:
        check_sources()

    with safe_open(str(SV / "BASE" / "model.safetensors"), "pt") as B, safe_open(str(SV / "SPARSE" / "model.safetensors"), "pt") as S, \
         safe_open(str(TRAINED_H), "pt") as H:
        keys = sorted(B.keys()) if not a.smoke_keys else sorted(a.smoke_keys)
        hkeys = set(H.keys())
        # pass 1: canary norms (and keep nothing but scalars)
        nh16 = nhf = nds = 0.0
        for k in keys:
            b = B.get_tensor(k).float()
            nds += float(((S.get_tensor(k).double() - b.double()) ** 2).sum())
            if k in hkeys:
                dh = H.get_tensor(k).float() - b
                nh16 += float((p16(dh) ** 2).sum()); nhf += float((dh.double() ** 2).sum())
        nh16, nhf, nds = math.sqrt(nh16), math.sqrt(nhf), math.sqrt(nds)
        print(f"||P16(delta_H)|| = {nh16:.13f} (recorded {P16_H_NORM_RECORDED:.13f}) | ||delta_H|| = {nhf:.10f} | "
              f"||dSparse|| = {nds:.13f} (recorded {DSPARSE_NORM:.13f})", flush=True)
        if not a.smoke_keys:
            if abs(nh16 - P16_H_NORM_RECORDED) > 2e-6:
                raise SystemExit("CANARY FAILED: ||P16(delta_H)|| does not reproduce INDEP's recorded norm")
            if abs(nds / DSPARSE_NORM - 1) > 1e-6:
                raise SystemExit("CANARY FAILED: ||SPARSE - BASE|| does not reproduce ||dSparse||")

        # headers (all F32, same layout for every arm)
        header, off = {}, 0
        for k in keys:
            shape = list(B.get_slice(k).get_shape()); nb = math.prod(shape) * 4
            header[k] = {"dtype": "F32", "shape": shape, "data_offsets": [off, off + nb]}; off += nb
        outs, fhs, per_tensor = {}, {}, {arm: {} for arm in a.arms}
        for arm in a.arms:
            h = dict(header); h["__metadata__"] = {"format": "pt", "experiment": "CONFIRM-002", "arm": arm}
            enc = json.dumps(h, separators=(",", ":"), ensure_ascii=True).encode(); enc += b" " * (-len(enc) % 8)
            d = a.outdir / arm; d.mkdir(parents=True, exist_ok=False)
            (d / "INCOMPLETE.json").write_bytes(b'{"building": true}\n')
            fhs[arm] = open(d / "model.safetensors.partial", "xb"); fhs[arm].write(struct.pack("<Q", len(enc))); fhs[arm].write(enc)
            outs[arm] = (d, len(enc))
        # pass 2: write
        for k in keys:
            b = B.get_tensor(k).float(); s = S.get_tensor(k).float()
            if k in hkeys:
                dh = H.get_tensor(k).float() - b
                d16 = p16(dh)
            for arm in a.arms:
                base_name, coef, kind = ARMS[arm]
                base = b if base_name == "BASE" else s
                if k in hkeys:
                    upd = coef * (d16 if kind == "p16" else dh.double())
                    w = (base.double() + upd).float()
                    per_tensor[arm][k] = float(torch.linalg.norm(upd))
                else:
                    w = base
                if not torch.isfinite(w).all():
                    raise RuntimeError(f"nonfinite in {arm}/{k}")
                fhs[arm].write(w.contiguous().numpy().astype("<f4", copy=False).tobytes(order="C"))
        for arm in a.arms:
            fhs[arm].flush(); os.fsync(fhs[arm].fileno()); fhs[arm].close()
            d, enclen = outs[arm]
            if (d / "model.safetensors.partial").stat().st_size != 8 + enclen + off:
                raise RuntimeError(f"{arm}: byte count mismatch")
            for f in ASSETS:
                shutil.copyfile(SV / "BASE" / f, d / f)
            (d / "model.safetensors.partial").rename(d / "model.safetensors")

    # verify each arm independently of the writer
    with safe_open(str(SV / "BASE" / "model.safetensors"), "pt") as B, safe_open(str(SV / "SPARSE" / "model.safetensors"), "pt") as S:
        for arm in a.arms:
            base_name = ARMS[arm][0]; d = a.outdir / arm
            bad_untouched, worst, n_out_of_tol = 0, 0.0, 0
            with safe_open(str(d / "model.safetensors"), "pt") as O:
                for k in keys:
                    o = O.get_tensor(k); base = (B if base_name == "BASE" else S).get_tensor(k).float()
                    if k in per_tensor[arm]:
                        got = float(torch.linalg.norm(o.double() - base.double()))
                        want = per_tensor[arm][k]
                        # float32 rounding floor: casting base+update to float32 perturbs each element by
                        # <= eps32*|w|/2, so the norm of the realised update can move by <= eps32*||w||_F
                        tol = max(1e-3 * want, 2 * 1.1920929e-07 * float(torch.linalg.norm(base.double())))
                        n_out_of_tol += abs(got - want) > tol
                        worst = max(worst, abs(got - want) / max(want, 1e-30))
                    else:
                        bad_untouched += int((o.view(torch.int32) != base.view(torch.int32)).sum())
            upd_norm = math.sqrt(sum(v * v for v in per_tensor[arm].values()))
            ok = bad_untouched == 0 and n_out_of_tol == 0
            print(f"verify {arm}: untouched-tensor bit mismatches {int(bad_untouched)}, H tensors outside the rounding "
                  f"tolerance {n_out_of_tol} (worst relative norm error {worst:.2e}) -> {'OK' if ok else 'FAILED'}", flush=True)
            if not ok:
                raise SystemExit(f"VERIFY FAILED for {arm}")
            rec = {"experiment": "CONFIRM-002", "arm": arm, "base_arm": base_name, "coefficient_H": ARMS[arm][1], "H_form": ARMS[arm][2],
                   "construction": f"{base_name} + {ARMS[arm][1]} * {'P16(delta_H)' if ARMS[arm][2] == 'p16' else 'delta_H (unrounded)'}, float64 sum, cast to float32",
                   "rank_r": RANK, "trained_H_sha256": TRAINED_H_SHA if not a.smoke_keys else None,
                   "norm_P16_delta_H": nh16, "norm_delta_H": nhf, "norm_dSparse": nds, "norm_H_update": upd_norm,
                   "canary": None if a.smoke_keys else {"P16_norm_recorded": P16_H_NORM_RECORDED, "abs_error": abs(nh16 - P16_H_NORM_RECORDED)},
                   "smoke_subset": a.smoke_keys, "weight_sha256": sha256(d / "model.safetensors"),
                   "status": "MATERIALIZED_NOT_EVALUATED", "quality_claim": False}
            (d / "BUILD.json").write_bytes((json.dumps(rec, indent=2) + "\n").encode())
            (d / "INCOMPLETE.json").unlink()
            print(json.dumps({k: rec[k] for k in ("arm", "norm_H_update", "weight_sha256")}), flush=True)


if __name__ == "__main__":
    main()
