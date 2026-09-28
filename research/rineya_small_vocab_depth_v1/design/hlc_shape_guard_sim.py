"""SMALLVD-001 Amendment 5 (A5.8) design record -- thresholds for the OUT-OF-FAMILY shape guard.

The calibration family draws per-row log(LC/plain) NORMAL and INDEPENDENT of the plain KL, at the real
REPORT sizes (124 rows). The guard flags a realised REPORT set whose shape departs grossly from that:
skewness, excess kurtosis of the per-row log-ratio, and corr(log kl_plain, log-ratio). Thresholds are
chosen so that the guard fires on at most ~1 % of IN-FAMILY sets (all three together), i.e. it flags
gross departures only. Pure Python; statistics computed exactly as lc_verdict.shape_stats does.
usage: python hlc_shape_guard_sim.py <n_sims> <out.json>
"""
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lc_verdict as lv  # noqa: E402

N_ROWS = sum(lv.CALIBRATED_FOR["rows_per_category"].values())


def main():
    n_sims, out = int(sys.argv[1]), sys.argv[2]
    rng = random.Random(8080)
    stats = []
    for i in range(n_sims):
        s_kl = (0.5, 1.0, 1.5)[i % 3]
        s_ratio = (0.2, 0.45, 0.8)[(i // 3) % 3]
        lk = [rng.gauss(0, s_kl) for _ in range(N_ROWS)]
        lr = [rng.gauss(-s_ratio ** 2 / 2, s_ratio) for _ in range(N_ROWS)]
        stats.append(lv.shape_stats(lr, lk))
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]
    abs_skew = [abs(s["skew"]) for s in stats]
    kurt = [s["excess_kurtosis"] for s in stats]
    abs_corr = [abs(s["corr_logkl_logratio"]) for s in stats]
    res = {"n_sims": n_sims, "rows": N_ROWS,
           "quantiles": {p: {"abs_skew": q(abs_skew, p), "excess_kurtosis": q(kurt, p), "abs_corr": q(abs_corr, p)}
                         for p in (0.99, 0.995, 0.999)},
           "thresholds_tested": {}}
    for sk, ku, co in ((0.75, 1.5, 0.3), (0.8, 1.75, 0.3), (1.0, 2.0, 0.35)):
        fire = sum(1 for s in stats if abs(s["skew"]) > sk or s["excess_kurtosis"] > ku
                   or abs(s["corr_logkl_logratio"]) > co) / n_sims
        res["thresholds_tested"][f"skew>{sk}|kurt>{ku}|corr>{co}"] = fire
    json.dump(res, open(out, "w", encoding="utf-8", newline=chr(10)), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
