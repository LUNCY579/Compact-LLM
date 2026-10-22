"""COMPOSE4B-001: build TH_s and SCALED_T from delta_T and delta_H.

    TH_s     = s_t * ( delta_T + delta_H )
    SCALED_T = s_t * delta_T
    s_t      = ||delta_T|| / ||delta_T + delta_H||     Frobenius, PER 2D TENSOR

Both deltas are natively rank 16, so SCALED-T-001's P16 truncation is the IDENTITY here
and is omitted rather than simulated — on the 596M the deltas were full-rank and had to be
projected; here they are not. That removes a confound rather than introducing one.

Exactness: for a LoRA module, effective dW = (alpha/r) * B @ A.
  delta_T + delta_H is represented EXACTLY at rank 32 by concatenation:
      A' = [A_T ; A_H]  (32, in)      B' = [B_T | B_H]  (out, 32)
      with r'=32, alpha'=64 so alpha'/r' == alpha/r == 2
  so (alpha'/r') B' A' == (alpha/r)(B_T A_T + B_H A_H). No approximation.
  Scaling by s_t is applied to B'.

s_t is computed ONCE here, written to PER_TENSOR_SCALES.json, and the SCALED_T build READS
it back rather than recomputing — SCALED-T-001 adopted this after a recomputed scale diverged
at the ninth decimal.
"""
from pathlib import Path
import json
import shutil

import torch
from safetensors.torch import load_file, save_file

BASE = Path("C:/Users/clclc/BusinessProject")
RECORDS = BASE / "rineya-dataset-retention-records"
OUT = RECORDS / "compose4b001-20260921"
T_DIR = RECORDS / "qwenft001-20260918/run/adapter"
H_DIR = OUT / "run-h/adapter"


def modules(sd):
    return sorted({k.rsplit(".lora_", 1)[0] for k in sd if ".lora_" in k})


def eff(sd, m, scale):
    A = sd[f"{m}.lora_A.weight"].to(torch.float64)
    B = sd[f"{m}.lora_B.weight"].to(torch.float64)
    return scale * (B @ A)


def main():
    cfgT = json.loads((T_DIR / "adapter_config.json").read_text(encoding="utf-8"))
    cfgH = json.loads((H_DIR / "adapter_config.json").read_text(encoding="utf-8"))
    for k in ("r", "lora_alpha", "lora_dropout", "bias", "peft_type"):
        if cfgT[k] != cfgH[k]:
            raise SystemExit(f"adapters differ on {k}: {cfgT[k]} vs {cfgH[k]}")
    if set(cfgT["target_modules"]) != set(cfgH["target_modules"]):
        raise SystemExit("target module sets differ")
    r, alpha = cfgT["r"], cfgT["lora_alpha"]
    scale = alpha / r

    sdT = load_file(str(T_DIR / "adapter_model.safetensors"))
    sdH = load_file(str(H_DIR / "adapter_model.safetensors"))
    if set(sdT) != set(sdH):
        raise SystemExit("adapter key sets differ")
    mods = modules(sdT)

    scales, checks = {}, []
    th_sd, st_sd = {}, {}
    for m in mods:
        dT = eff(sdT, m, scale)
        dH = eff(sdH, m, scale)
        nT = torch.linalg.matrix_norm(dT, ord="fro").item()
        nS = torch.linalg.matrix_norm(dT + dH, ord="fro").item()
        if not (nS > 0):
            raise SystemExit(f"zero-norm sum at {m}")
        s = nT / nS
        scales[m] = s

        A = torch.cat([sdT[f"{m}.lora_A.weight"], sdH[f"{m}.lora_A.weight"]], dim=0)
        B = torch.cat([sdT[f"{m}.lora_B.weight"], sdH[f"{m}.lora_B.weight"]], dim=1) * s
        th_sd[f"{m}.lora_A.weight"] = A.contiguous()
        th_sd[f"{m}.lora_B.weight"] = B.contiguous()

        st_sd[f"{m}.lora_A.weight"] = sdT[f"{m}.lora_A.weight"].clone().contiguous()
        st_sd[f"{m}.lora_B.weight"] = (sdT[f"{m}.lora_B.weight"] * s).contiguous()

        # verify the rank-32 concatenation is EXACT, and the magnitudes are what we claim
        th_eff = eff(th_sd, m, (2 * alpha) / (2 * r))
        st_eff = eff(st_sd, m, scale)
        checks.append({
            "module": m, "s_t": s,
            "norm_T": nT, "norm_T_plus_H": nS,
            "rel_err_TH_vs_s_times_sum": (torch.linalg.matrix_norm(th_eff - s * (dT + dH), ord="fro")
                                          / max(nT, 1e-30)).item(),
            "rel_err_SCALED_T_vs_s_times_T": (torch.linalg.matrix_norm(st_eff - s * dT, ord="fro")
                                              / max(nT, 1e-30)).item(),
            "norm_TH_s": torch.linalg.matrix_norm(th_eff, ord="fro").item(),
            "norm_SCALED_T": torch.linalg.matrix_norm(st_eff, ord="fro").item(),
        })

    worst_th = max(c["rel_err_TH_vs_s_times_sum"] for c in checks)
    worst_st = max(c["rel_err_SCALED_T_vs_s_times_T"] for c in checks)
    # ||TH_s|| must equal ||delta_T|| by construction of s_t; that IS the magnitude match
    worst_match = max(abs(c["norm_TH_s"] - c["norm_T"]) / max(c["norm_T"], 1e-30) for c in checks)
    # float32 adapter storage: eps 1.19e-7, so a 1e-9 tolerance is not achievable in
    # principle and the first run correctly refused at 6.1e-8. Adopting SCALED-T-001's
    # OWN registered numerical tolerance (2e-06, its amended value) rather than inventing
    # one to fit: this is the parent construction's tolerance, and the observed error is
    # ~33x inside it. Recorded in the receipt so the change is visible, not silent.
    TOL = 2e-6
    if worst_th > TOL or worst_st > TOL or worst_match > TOL:
        raise SystemExit(f"BUILD GATE FAILED: th {worst_th:.3e} st {worst_st:.3e} match {worst_match:.3e}")

    for name, sd, rr, aa in (("arm-th-s", th_sd, 2 * r, 2 * alpha),
                             ("arm-scaled-t", st_sd, r, alpha)):
        d = OUT / name / "adapter"
        d.mkdir(parents=True)
        cfg = dict(cfgT); cfg["r"] = rr; cfg["lora_alpha"] = aa
        (d / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        save_file(sd, str(d / "adapter_model.safetensors"))

    (OUT / "PER_TENSOR_SCALES.json").write_text(json.dumps({
        "what": "s_t per module, computed ONCE at TH_s build; SCALED_T reads this, never recomputes",
        "formula": "s_t = ||dW_T||_F / ||dW_T + dW_H||_F, float64",
        "p16_truncation": "IDENTITY — both deltas are natively rank 16, unlike the 596M full-rank case",
        "modules": len(mods),
        "s_t_min": min(scales.values()), "s_t_median": sorted(scales.values())[len(scales) // 2],
        "s_t_max": max(scales.values()), "fallbacks": 0,
        "worst_rel_err_TH_s": worst_th, "worst_rel_err_SCALED_T": worst_st,
        "worst_magnitude_match_err": worst_match, "tolerance": TOL,
        "tolerance_provenance": "SCALED-T-001 amended numerical tolerance 2e-06; an initial 1e-9 was unachievable against float32 adapter storage (eps 1.19e-7) and the build correctly refused at 6.1e-8",
        "per_tensor_scales": scales, "checks": checks,
    }, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "modules": len(mods),
        "s_t": {"min": min(scales.values()),
                "median": sorted(scales.values())[len(scales) // 2],
                "max": max(scales.values())},
        "worst_rel_err_TH_s": worst_th,
        "worst_rel_err_SCALED_T": worst_st,
        "worst_magnitude_match_err": worst_match,
        "build_gate": "PASS",
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
