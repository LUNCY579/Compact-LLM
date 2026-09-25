"""SMALLVD-001 Amendment 5 design record -- empirical CALIBRATION of the H-LC verdict, per band of the
OBSERVABLE per-row log-ratio spread (lc_verdict.diagnostics -> s_ratio_hat).

Why banded: null calibration of every resampling procedure tried (percentile, basic, sign-flip)
failed with the spread of per-row log(LC/plain) -- not with effective n per se (sign-flip at
effective n 1.8 was calibrated when the spread was 0.1). The spread is estimable after the fit, so the
verdict's one-sided level alpha* is calibrated per spread band, at the band's UPPER edge (the null rate
rises with the spread, so the upper edge is the conservative point).

Stores RAW per-simulation outputs of the COMMITTED lc_verdict.verdict (bootstrap tail fractions,
r-hat, diagnostics) so any threshold is evaluated after the fact, not re-simulated.
Models (as design/hlc_power_sim.py): per-row plain KL lognormal(0, s_kl); ratio lognormal
  N1: mean (1 - r_true)   -- the MEAN-ratio null/alternatives for r (the estimand)
  N2: log-mean log(1 - r_true), i.e. median (1 - r_true) -- at r_true = 0 a HARMFUL alternative
usage: python hlc_calibration_sim.py <band_upper_s_ratio> <n_null> <n_boundary> <n_power> <n_n2> <out.jsonl>
"""
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lc_verdict as lv  # noqa: E402

SIZES = {"abstention_unknown": (14, "chat"), "concise_reasoning": (16, "chat"), "conversation": (18, "chat"),
         "grounded_extraction": (13, "chat"), "instruction_writing": (14, "chat"), "long_context": (15, "chat"),
         "text_english_prose": (9, "text"), "text_javascript": (8, "text"), "text_markdown": (10, "text"),
         "text_python": (7, "text")}
S_KL = (0.5, 1.0, 1.5)


def data(rng, r_true, s_kl, s_ratio, model):
    plain, comp, ids = [], [], []
    for c, (k, kind) in SIZES.items():
        for j in range(k):
            i = f"{c}_{j}"
            n = rng.randint(10, 65) if kind == "chat" else rng.randint(100, 511)
            kl = math.exp(rng.gauss(0, s_kl))
            mu = math.log(1 - r_true) - (s_ratio ** 2 / 2 if model == "N1" else 0.0)
            plain.append({"id": i, "category": c, "n": n, "kl": kl})
            comp.append({"id": i, "category": c, "n": n, "kl": kl * math.exp(rng.gauss(mu, s_ratio))})
            ids.append(i)
    return plain, comp, ids


def main():
    band, n_null, n_bnd, n_pow, n_n2, out = float(sys.argv[1]), *map(int, sys.argv[2:6]), sys.argv[6]
    rng = random.Random(int(band * 1000) + 5)
    keep = ("relative_reduction_r", "boot_frac_le_0", "boot_frac_le_precise", "s_ratio_hat", "s_kl_hat",
            "min_eff_n", "verdict")
    with open(out, "w", encoding="utf-8", newline=chr(10)) as f:
        for s_kl in S_KL:
            for kind, r_true, model, n in (("null", 0.0, "N1", n_null), ("boundary", 0.10, "N1", n_bnd),
                                           ("power", 0.30, "N1", n_pow), ("n2_harm", 0.0, "N2", n_n2)):
                for _ in range(n):
                    v = lv.verdict(*data(rng, r_true, s_kl, band, model))
                    f.write(json.dumps({"band_s_ratio": band, "s_kl": s_kl, "kind": kind, "r_true": r_true,
                                        "model": model, **{k: v[k] for k in keep}}) + chr(10))
                f.flush()


if __name__ == "__main__":
    main()
