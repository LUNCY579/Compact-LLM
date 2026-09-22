"""COMPOSE4B-003: build SCALED_H and INDEP, completing the 3x2 factorial.

    scaled_h = s_t * delta_H                  the SCALED-H-001 control, never run on this base
    indep    = 1.0 * delta_T + s_t * delta_H  independent coefficients; the INDEP-001 cell

`s_t` is the PER-MODULE vector recorded by COMPOSE4B-001. It is READ BACK from
PER_TENSOR_SCALES.json and gated on that file's sha256 -- never recomputed. SCALED-T-001 adopted
read-back after a recomputed scale diverged at the ninth decimal; a silent drift would make
`th_s` and `indep` incomparable, which is the whole point of the factorial.

BESTARM-001's rule, adopted verbatim: "a third implementation is a third chance to disagree."
So `modules()` and `eff()` are IMPORTED from build_compose.py rather than rewritten.

The distinction that IS the experiment: `th_s` multiplies the WHOLE of B' by s. Here only the
H half is scaled. One column block.

GATE 6 IS A POSITIVE CONTROL AND IT RUNS FIRST. A deliberately mutated arm -- s applied to the
T half instead of the H half, and s omitted from scaled_h -- must make gates 4/5 FAIL before any
correct arm is written. An absence check and a passing check return the same answer; this is the
only way to know the tolerance is not vacuous.
"""
from pathlib import Path
import hashlib
import json
import sys

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).parent))
from build_compose import modules, eff  # noqa: E402  -- deliberately not reimplemented

BASE = Path("C:/Users/clclc/BusinessProject")
RECORDS = BASE / "rineya-dataset-retention-records"
SRC = RECORDS / "compose4b001-20260921"
OUT = RECORDS / "compose4b003-20260921"
T_DIR = RECORDS / "qwenft001-20260918/run/adapter"
H_DIR = SRC / "run-h/adapter"
SCALES = SRC / "PER_TENSOR_SCALES.json"

SCALES_SHA = "ebabde31a6d8af6b3d52ae133bc0f9cc653de7afd9a2af66489ee6c2e8288d12"
N_MODULES = 252
# SCALED-T-001's OWN amended tolerance, adopted unchanged. float32 adapter storage has
# eps 1.19e-7, so COMPOSE4B-001's first attempt at 1e-9 was unachievable in principle and
# refused at 6.1e-8 -- the guard being too strict, not the arithmetic being wrong.
TOL = 2e-6


def build_scaled_h(sdH, m, s):
    """s * delta_H at rank r: scale B only. (alpha/r)(s B_H) A_H == s * delta_H."""
    return {
        f"{m}.lora_A.weight": sdH[f"{m}.lora_A.weight"].clone().contiguous(),
        f"{m}.lora_B.weight": (sdH[f"{m}.lora_B.weight"] * s).contiguous(),
    }


def build_indep(sdT, sdH, m, s, mutate=False):
    """delta_T + s * delta_H at rank 2r, alpha 2*alpha.

    A' = [A_T ; A_H], B' = [B_T | s*B_H]
    (2a/2r) B'A' == (a/r)(B_T A_T) + s*(a/r)(B_H A_H) == delta_T + s*delta_H

    mutate=True applies s to the T half instead -- the deliberate error gate 6 must catch.
    """
    BT = sdT[f"{m}.lora_B.weight"]
    BH = sdH[f"{m}.lora_B.weight"]
    if mutate:
        BT, BH = BT * s, BH
    else:
        BH = BH * s
    return {
        f"{m}.lora_A.weight": torch.cat(
            [sdT[f"{m}.lora_A.weight"], sdH[f"{m}.lora_A.weight"]], dim=0).contiguous(),
        f"{m}.lora_B.weight": torch.cat([BT, BH], dim=1).contiguous(),
    }


def main():
    # ---- gate 1: the scales file is the recorded one -------------------------------------
    got = hashlib.sha256(SCALES.read_bytes()).hexdigest()
    if got != SCALES_SHA:
        raise SystemExit(f"GATE 1 FAILED: PER_TENSOR_SCALES.json sha256 {got} != {SCALES_SHA}")
    scales_doc = json.loads(SCALES.read_text(encoding="utf-8"))
    s_t = scales_doc["per_tensor_scales"]
    if len(s_t) != N_MODULES or scales_doc["modules"] != N_MODULES:
        raise SystemExit(f"GATE 1 FAILED: {len(s_t)} scales, expected {N_MODULES}")

    # ---- gate 2/3: the two adapters are the same object shape ----------------------------
    cfgT = json.loads((T_DIR / "adapter_config.json").read_text(encoding="utf-8"))
    cfgH = json.loads((H_DIR / "adapter_config.json").read_text(encoding="utf-8"))
    for k in ("r", "lora_alpha", "lora_dropout", "bias", "peft_type"):
        if cfgT[k] != cfgH[k]:
            raise SystemExit(f"GATE 2 FAILED: adapters differ on {k}: {cfgT[k]} vs {cfgH[k]}")
    if set(cfgT["target_modules"]) != set(cfgH["target_modules"]):
        raise SystemExit("GATE 2 FAILED: target module sets differ")
    r, alpha = cfgT["r"], cfgT["lora_alpha"]
    scale, scale2 = alpha / r, (2 * alpha) / (2 * r)

    sdT = load_file(str(T_DIR / "adapter_model.safetensors"))
    sdH = load_file(str(H_DIR / "adapter_model.safetensors"))
    if set(sdT) != set(sdH):
        raise SystemExit("GATE 3 FAILED: adapter key sets differ")
    mods = modules(sdT)
    if len(mods) != N_MODULES:
        raise SystemExit(f"GATE 3 FAILED: {len(mods)} modules, expected {N_MODULES}")
    if sorted(mods) != sorted(s_t):
        raise SystemExit("GATE 3 FAILED: module names disagree with the recorded scales")

    # ---- gate 6: POSITIVE CONTROL, run BEFORE any correct arm is written -----------------
    # Two deliberate errors on the first module. Each must be caught by gate 4 or gate 5.
    m0 = mods[0]
    s0 = s_t[m0]
    dT0, dH0 = eff(sdT, m0, scale), eff(sdH, m0, scale)
    nH0 = torch.linalg.matrix_norm(dH0, ord="fro").item()

    bad_indep = build_indep(sdT, sdH, m0, s0, mutate=True)
    bad_err = (torch.linalg.matrix_norm(eff(bad_indep, m0, scale2) - (dT0 + s0 * dH0), ord="fro")
               / max(nH0, 1e-30)).item()

    bad_sh = {f"{m0}.lora_A.weight": sdH[f"{m0}.lora_A.weight"],
              f"{m0}.lora_B.weight": sdH[f"{m0}.lora_B.weight"]}  # s omitted entirely
    bad_sh_err = (torch.linalg.matrix_norm(eff(bad_sh, m0, scale) - s0 * dH0, ord="fro")
                  / max(nH0, 1e-30)).item()

    control = {
        "module": m0, "s_t": s0,
        "mutation_indep_scale_applied_to_T_half": bad_err,
        "mutation_scaled_h_scale_omitted": bad_sh_err,
        "tolerance": TOL,
        "both_detected": bool(bad_err > TOL and bad_sh_err > TOL),
    }
    if not control["both_detected"]:
        raise SystemExit(f"GATE 6 FAILED -- THE GATE IS VACUOUS: a deliberately wrong arm passed "
                         f"at {TOL}. indep_err {bad_err:.3e} scaled_h_err {bad_sh_err:.3e}")
    print(json.dumps({"gate_6_positive_control": control}, indent=2))

    # ---- build both arms, gating 4 and 5 per module ---------------------------------------
    sh_sd, ind_sd, checks = {}, {}, []
    for m in mods:
        s = s_t[m]
        dT, dH = eff(sdT, m, scale), eff(sdH, m, scale)
        nT = torch.linalg.matrix_norm(dT, ord="fro").item()
        nH = torch.linalg.matrix_norm(dH, ord="fro").item()
        if not (nH > 0):
            raise SystemExit(f"zero-norm delta_H at {m}")

        sh = build_scaled_h(sdH, m, s)
        ind = build_indep(sdT, sdH, m, s)
        sh_eff = eff(sh, m, scale)
        ind_eff = eff(ind, m, scale2)

        # gate 4: exactness against the target
        e_sh = (torch.linalg.matrix_norm(sh_eff - s * dH, ord="fro") / nH).item()
        e_ind = (torch.linalg.matrix_norm(ind_eff - (dT + s * dH), ord="fro") / max(nT, nH)).item()
        # gate 5: the magnitude identity -- s landed on the H block and only the H block
        e_mag_sh = abs(torch.linalg.matrix_norm(sh_eff, ord="fro").item() - s * nH) / nH
        e_mag_ind = (torch.linalg.matrix_norm(ind_eff - dT, ord="fro").item() - s * nH) / nH

        if max(e_sh, e_ind, e_mag_sh, abs(e_mag_ind)) > TOL:
            raise SystemExit(f"GATE 4/5 FAILED at {m}: exact_sh {e_sh:.3e} exact_indep {e_ind:.3e} "
                             f"mag_sh {e_mag_sh:.3e} mag_indep {e_mag_ind:.3e}")

        sh_sd.update(sh)
        ind_sd.update(ind)
        checks.append({"module": m, "s_t": s, "norm_T": nT, "norm_H": nH,
                       "exact_err_scaled_h": e_sh, "exact_err_indep": e_ind,
                       "mag_err_scaled_h": e_mag_sh, "mag_err_indep": e_mag_ind,
                       "norm_scaled_h": torch.linalg.matrix_norm(sh_eff, ord="fro").item(),
                       "norm_indep": torch.linalg.matrix_norm(ind_eff, ord="fro").item()})

    worst = {k: max(abs(c[k]) for c in checks) for k in
             ("exact_err_scaled_h", "exact_err_indep", "mag_err_scaled_h", "mag_err_indep")}

    for name, sd, rr, aa in (("arm-scaled-h", sh_sd, r, alpha),
                             ("arm-indep", ind_sd, 2 * r, 2 * alpha)):
        d = OUT / name / "adapter"
        d.mkdir(parents=True)  # create-only: refuses if the arm already exists
        cfg = dict(cfgT)
        cfg["r"], cfg["lora_alpha"] = rr, aa
        (d / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        save_file(sd, str(d / "adapter_model.safetensors"))

    receipt = {
        "experiment": "COMPOSE4B-003",
        "arms": {"scaled_h": "s_t * delta_H", "indep": "1.0 * delta_T + s_t * delta_H"},
        "s_t_source": str(SCALES.relative_to(RECORDS)),
        "s_t_sha256": SCALES_SHA, "s_t_recomputed": False,
        "s_t_min": scales_doc["s_t_min"], "s_t_median": scales_doc["s_t_median"],
        "s_t_max": scales_doc["s_t_max"],
        "modules": len(mods), "r": r, "lora_alpha": alpha,
        "tolerance": TOL,
        "tolerance_provenance": "SCALED-T-001 amended value, adopted unchanged; 1e-9 is below "
                                "float32 eps 1.19e-7 and is unachievable in principle",
        "worst_errors": worst,
        "gate_6_positive_control": control,
        "build_gate": "PASS",
        "quality_claim": False, "release_permitted": False, "model_execution": False,
    }
    (OUT / "BUILD_003.json").write_text(json.dumps({**receipt, "checks": checks}, indent=2) + "\n",
                                        encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
