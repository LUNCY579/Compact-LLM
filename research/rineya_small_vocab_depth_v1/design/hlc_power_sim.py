"""SMALLVD-001 Amendment 4 design record. H-LC power at the REAL REPORT category sizes, using the COMMITTED verdict (lc_verdict.verdict).

Per row: positions n (chat U[10,65], text U[100,511]); plain KL ~ lognormal(0, s_kl); the LC/plain
ratio ~ lognormal with mean (1 - r_true) and log-sd s_ratio (per-row heterogeneity of the benefit).
For each cell: P(lower bound > 0), P(SUPPORTED), mean point estimate, mean interval width.
"""
import json
import math
import random
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import lc_verdict as lv

SIZES = {"abstention_unknown": (14, "chat"), "concise_reasoning": (16, "chat"), "conversation": (18, "chat"),
         "grounded_extraction": (13, "chat"), "instruction_writing": (14, "chat"), "long_context": (15, "chat"),
         "text_english_prose": (9, "text"), "text_javascript": (8, "text"), "text_markdown": (10, "text"),
         "text_python": (7, "text")}
SIMS = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def one(rng, r_true, s_kl, s_ratio):
    plain, comp, ids = [], [], []
    for c, (k, kind) in SIZES.items():
        for j in range(k):
            i = f"{c}_{j}"
            n = rng.randint(10, 65) if kind == "chat" else rng.randint(100, 511)
            kl = math.exp(rng.gauss(0, s_kl))
            ratio = math.exp(rng.gauss(math.log(1 - r_true) - s_ratio ** 2 / 2, s_ratio))
            plain.append({"id": i, "category": c, "n": n, "kl": kl})
            comp.append({"id": i, "category": c, "n": n, "kl": kl * ratio})
            ids.append(i)
    return lv.verdict(plain, comp, ids)


rng = random.Random(20260924)
out = []
for s_kl in (0.5, 1.0, 1.5):
    for s_ratio in (0.1, 0.3, 0.6, 1.0):
        for r_true in (0.1, 0.2, 0.3, 0.4):
            vs = [one(rng, r_true, s_kl, s_ratio) for _ in range(SIMS)]
            row = {"s_kl": s_kl, "s_ratio": s_ratio, "r_true": r_true, "sims": SIMS,
                   "P_lower_gt_0": sum(v["bootstrap_95"][0] > 0 for v in vs) / SIMS,
                   "P_SUPPORTED": sum(v["verdict"] == "SUPPORTED" for v in vs) / SIMS,
                   "mean_r_hat": sum(v["relative_reduction_r"] for v in vs) / SIMS,
                   "mean_ci_width": sum(v["bootstrap_95"][1] - v["bootstrap_95"][0] for v in vs) / SIMS}
            out.append(row)
            print(json.dumps(row), flush=True)
json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else "HLC_POWER_SIM.json", "w", encoding="utf-8", newline=chr(10)), indent=1)
