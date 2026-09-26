"""TRISKILL-001 arm builder (PREREG §3). CPU only, no model calls, create-only outputs.

    python build_arm.py consolidate --checkpoint <journal/runs/train-001/microfit/checkpoint-N>
    python build_arm.py canary
    python build_arm.py build --arm TTC25

`consolidate` extracts delta_C's trained FP32 tensors from a completed native checkpoint. It is
consolidate() from clockfix001's build_clockfix_arm.py (itself verbatim from compose001), and it
writes trained_C_fp32.safetensors once.

`canary` runs two checks, both of which must pass before any arm is built or scored:
  (a) Supa + 1.00*delta_C equals the trained tensors EXACTLY on every trained key, and Supa elsewhere;
  (b) TT + 0.00*delta_C equals TT bit for bit on every tensor.
It also records ||delta_C||.

`build` writes FP32( base + beta*delta_C ), one tensor at a time, as a float64 sum cast to float32:
  - base is Supa (upcast) for C25/C50/C100, and Titan-Transition for TTC25/TTC50/TTC100/TTR;
  - delta_C = trained_C - Supa(FP32 upcast), against the same source as every delta in this programme;
  - TTR adds a Gaussian direction (seed 20260926), rescaled per tensor to ||beta* * delta_C||, and
    needs --beta-star (fixed by Stage A).
Non-trained tensors are copied from the base. BUILD.json records the weight sha256 and the norms.
"""
from __future__ import annotations

import argparse, ctypes, hashlib, json, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORES = Path(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores")
SUPA = CORES / "Rineya-Supa-1.7.0-devfreeze"
SUPA_SHA = "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b"
TT = CORES / "Rineya-Titan-Transition-1.0.0-devfreeze"
TT_SHA = "a4d73353db9a91cbba7b4cd7c8fe4fe6822670d10fe523b26a6d651bb9509567"
TRAINED = HERE / "trained_C_fp32.safetensors"
CONSOLIDATION = HERE / "DELTA_C_CONSOLIDATION.json"
CANARY = HERE / "BUILD_CANARY.json"
ARMS_DIR = HERE / "arms"
ARMS = {"C25": ("supa", 0.25), "C50": ("supa", 0.50), "C100": ("supa", 1.00),
        "TTC25": ("tt", 0.25), "TTC50": ("tt", 0.50), "TTC100": ("tt", 1.00), "TTR": ("tt", None)}
SIDECARS = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
RAND_SEED = 20260926
RAM_FLOOR_GIB = 5.0


def free_ram_gib() -> float:
    class M(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = M(); m.dwLength = ctypes.sizeof(M)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 30


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def consolidate(checkpoint: Path) -> int:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    if TRAINED.exists() or CONSOLIDATION.exists():
        raise SystemExit("REFUSED: trained_C_fp32 already consolidated (create-only)")
    man = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    if not man.get("completed", True):
        raise SystemExit("checkpoint did not complete; refusing to consolidate")
    trained = {}
    for e in man["entries"]:
        with safe_open(str(checkpoint / e["file"]), framework="pt") as f:
            t = f.get_tensor("parameter")
        if t.dtype != torch.float32 or list(t.shape) != e["shape"]:
            raise SystemExit(f"bad checkpoint tensor: {e['name']}")
        trained[e["name"]] = t.contiguous()
    save_file(trained, str(TRAINED), metadata={"source": str(checkpoint)})
    rec = {"record": "DELTA_C_CONSOLIDATION", "experiment": "TRISKILL-001", "checkpoint": str(checkpoint),
           "checkpoint_binding": man.get("binding", {}), "trained_tensors": len(trained),
           "trained_C_fp32_sha256": sha256_file(TRAINED), "tier": "A (trained delta; NOT rebuildable -- back up before any deletion)"}
    CONSOLIDATION.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(rec, indent=1))
    return 0


def _inputs_ok():
    if sha256_file(SUPA / "model.safetensors") != SUPA_SHA:
        raise SystemExit("REFUSED: Supa weight hash mismatch")
    if sha256_file(TT / "model.safetensors") != TT_SHA:
        raise SystemExit("REFUSED: Titan-Transition weight hash mismatch")
    rec = json.loads(CONSOLIDATION.read_text(encoding="utf-8"))
    if sha256_file(TRAINED) != rec["trained_C_fp32_sha256"]:
        raise SystemExit("REFUSED: trained_C_fp32 changed since consolidation")
    return rec


def canary() -> int:
    import torch
    from safetensors import safe_open
    if CANARY.exists():
        raise SystemExit("REFUSED: canary already recorded (create-only)")
    rec = _inputs_ok()
    sq, n_a, n_b, bad_a, bad_b = 0.0, 0, 0, [], []
    with safe_open(str(SUPA / "model.safetensors"), framework="pt") as fs, \
         safe_open(str(TT / "model.safetensors"), framework="pt") as ft, \
         safe_open(str(TRAINED), framework="pt") as fc:
        tkeys = set(fc.keys())
        if not tkeys <= set(fs.keys()) or not tkeys <= set(ft.keys()):
            raise SystemExit("REFUSED: trained tensors absent from Supa or TT")
        for k in fs.keys():
            s = fs.get_tensor(k).double()
            if k in tkeys:
                c = fc.get_tensor(k)
                d = c.double() - s
                sq += float(d.pow(2).sum())
                built = (s + 1.0 * d).float()
                ok_a = torch.equal(built, c)
                if n_a == 0 or not hasattr(canary, "_neg_done"):
                    # negative control: a one-element perturbation must be detected
                    probe = built.clone(); probe.view(-1)[0] = torch.nextafter(probe.view(-1)[0], probe.view(-1)[0] + 1)
                    if torch.equal(probe, c):
                        raise SystemExit("CANARY IS VACUOUS: a one-ulp perturbation was not detected")
                    canary._neg_done = True
            else:
                d = None
                ok_a = torch.equal(s.float(), fs.get_tensor(k).float())
            n_a += 1
            if not ok_a:
                bad_a.append(k)
            t = ft.get_tensor(k)
            built_b = (t.double() + 0.0 * (d if d is not None else 0.0)).float()
            n_b += 1
            if not torch.equal(built_b, t):
                bad_b.append(k)
    out = {"record": "BUILD_CANARY", "experiment": "TRISKILL-001",
           "a_supa_plus_delta_equals_trained": {"tensors": n_a, "mismatches": len(bad_a), "examples": bad_a[:3]},
           "b_tt_plus_zero_equals_tt": {"tensors": n_b, "mismatches": len(bad_b), "examples": bad_b[:3]},
           "delta_C_norm_fro": sq ** 0.5, "trained_C_fp32_sha256": rec["trained_C_fp32_sha256"],
           "pass": not bad_a and not bad_b}
    CANARY.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))
    return 0 if out["pass"] else 2


def build(arm: str, beta_star: float | None) -> int:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    if arm not in ARMS:
        raise SystemExit(f"unregistered arm {arm}")
    out_dir = ARMS_DIR / arm
    if out_dir.exists():
        raise SystemExit(f"REFUSED: {out_dir} exists (create-only)")
    can = json.loads(CANARY.read_text(encoding="utf-8")) if CANARY.exists() else None
    if not can or not can["pass"]:
        raise SystemExit("REFUSED: the build canary has not passed")
    free = free_ram_gib()
    if free < RAM_FLOOR_GIB:
        raise SystemExit(f"REFUSED: {free:.2f} GiB free host RAM, need {RAM_FLOOR_GIB}")
    base_kind, beta = ARMS[arm]
    if arm == "TTR":
        if beta_star not in (0.25, 0.50, 1.00):
            raise SystemExit("TTR needs --beta-star from Stage A (one of 0.25, 0.50, 1.00)")
        beta = beta_star
    elif beta_star is not None:
        raise SystemExit("--beta-star is only for TTR")
    rec = _inputs_ok()
    base_dir = SUPA if base_kind == "supa" else TT
    gen = torch.Generator().manual_seed(RAND_SEED)
    out, sq_upd, sq_mod = {}, 0.0, {"attn": 0.0, "mlp": 0.0, "other": 0.0}
    with safe_open(str(base_dir / "model.safetensors"), framework="pt") as fb, \
         safe_open(str(SUPA / "model.safetensors"), framework="pt") as fs, \
         safe_open(str(TRAINED), framework="pt") as fc:
        tkeys = set(fc.keys())
        for k in sorted(fb.keys()):
            b = fb.get_tensor(k).double()
            if k in tkeys:
                d = fc.get_tensor(k).double() - fs.get_tensor(k).double()
                if arm == "TTR":
                    r = torch.randn(d.shape, generator=gen, dtype=torch.float64)
                    dn, rn = float(d.norm()), float(r.norm())
                    upd = r * ((beta * dn) / rn) if rn > 0 else torch.zeros_like(d)
                else:
                    upd = beta * d
                s = float(upd.pow(2).sum()); sq_upd += s
                sq_mod["attn" if ".self_attn." in k else "mlp" if ".mlp." in k else "other"] += s
                t = (b + upd).float().contiguous()
            else:
                t = b.float().contiguous()
            if not torch.isfinite(t).all():
                raise SystemExit(f"REFUSED: non-finite tensor {k}")
            out[k] = t
    out_dir.mkdir(parents=True)
    save_file(out, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    del out
    for f in SIDECARS:
        src = (base_dir if (base_dir / f).exists() else SUPA) / f
        shutil.copyfile(src, out_dir / f)
    cfg = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    if cfg.get("torch_dtype") not in ("float32", None) and base_kind == "supa":
        cfg["torch_dtype"] = "float32"      # the arm IS FP32; auto-dtype must not round it (Titan-Sparse's rule)
        (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8", newline="\n")
    build_rec = {"experiment": "TRISKILL-001", "arm": arm, "base": base_kind, "base_sha256": SUPA_SHA if base_kind == "supa" else TT_SHA,
                 "beta": beta, "update": "gaussian_norm_matched_per_tensor" if arm == "TTR" else "beta*delta_C",
                 "rand_seed": RAND_SEED if arm == "TTR" else None,
                 "construction": "FP32(float64(base) + update), per tensor; non-trained tensors copied from base",
                 "trained_C_fp32_sha256": rec["trained_C_fp32_sha256"], "delta_C_norm_fro": can["delta_C_norm_fro"],
                 "update_norm_fro": sq_upd ** 0.5,
                 "update_sq_norm_share": {m: (v / sq_upd if sq_upd else None) for m, v in sq_mod.items()},
                 "weight_sha256": sha256_file(out_dir / "model.safetensors"),
                 "status": "MATERIALIZED_NOT_EVALUATED", "quality_claim": False, "release_permitted": False, "model_calls": 0}
    (out_dir / "BUILD.json").write_text(json.dumps(build_rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: build_rec[k] for k in ("arm", "beta", "update_norm_fro", "weight_sha256")}), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("consolidate"); c.add_argument("--checkpoint", type=Path, required=True)
    sub.add_parser("canary")
    b = sub.add_parser("build"); b.add_argument("--arm", required=True); b.add_argument("--beta-star", type=float, default=None)
    a = ap.parse_args()
    if a.cmd == "consolidate":
        return consolidate(a.checkpoint)
    if a.cmd == "canary":
        return canary()
    return build(a.arm, a.beta_star)


if __name__ == "__main__":
    raise SystemExit(main())
