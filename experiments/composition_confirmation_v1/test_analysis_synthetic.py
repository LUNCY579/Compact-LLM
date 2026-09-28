"""Synthetic validation of the amended analyse_confirm.py: every verdict must be reachable both ways,
and the input guards must refuse bad files. Uses a synthetic 864-row bank; never touches real data."""
import hashlib, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

CC = Path(__file__).resolve().parent
sys.path.insert(0, str(CC))
import analyse_confirm as A

tmp = Path(tempfile.mkdtemp())
fams = ["ledger_replay", "prerequisite_plan", "register_machine"]
rows = [{"id": f"CF1-{fams[i % 3]}-{i:04d}", "family": fams[i % 3], "reference": f"R{i}"} for i in range(288)]
rows += [{"id": f"CF1X-{fams[i % 3]}-{i:04d}", "family": fams[i % 3], "reference": f"R{i}"} for i in range(288, 864)]
bank_bytes = (json.dumps({"rows": rows}) + "\n").encode()
(tmp / "bank.json").write_bytes(bank_bytes)
A.BANK = tmp / "bank.json"; A.BANK_SHA = hashlib.sha256(bank_bytes).hexdigest()
ids = sorted(r["id"] for r in rows); ref = {r["id"]: r["reference"] for r in rows}
rng = np.random.default_rng(0)


def arm(correct_ids, period=False, wrong="nope"):
    return {i: (ref[i] + ("." if period else "")) if i in correct_ids else wrong for i in ids}


def runfile(name, arms, mnt=48):
    d = {"bank_sha256": A.BANK_SHA, "runtime": {"max_new_tokens": mnt},
         "arms": [{"arm": k, "predictions": [{"id": i, "generation": g[i]} for i in g]} for k, g in arms.items()]}
    p = tmp / name; p.write_text(json.dumps(d)); return str(p)


def pick(pool, k):
    return set(rng.choice(sorted(pool), k, replace=False))


def scenario(n_sparse=450, n_avg=310, add_lost=10, add_gained=0, sa_minus_rm=0, rm_minus_sp=0, base=0, period=True):
    S = pick(ids, n_sparse)
    B = pick(sorted(set(ids) - S), base) if base else set()
    AV = pick(S, n_avg)
    AD = (S - pick(S, add_lost)) | pick(sorted(set(ids) - S), add_gained)
    RM = pick(ids, 300)
    SA = (RM - pick(RM, max(0, -sa_minus_rm))) | pick(sorted(set(ids) - RM), max(0, sa_minus_rm))
    SP = (RM - pick(RM, max(0, rm_minus_sp))) | pick(sorted(set(ids) - RM), max(0, -rm_minus_sp))
    arms = {"BASE": arm(B), "SPARSE": arm(S), "AVG": arm(AV), "ADD-100": arm(AD, period=period),
            "SPARSE_ATTN": arm(SA), "SCALAR_MATCHED": arm(pick(ids, 350)), "RAND_MASK": arm(RM), "SCALAR_P": arm(SP)}
    return arms


def run(arms, **kw):
    f = runfile(f"r{rng.integers(1e9)}.json", arms, **kw)
    bank, loaded = A.load([f])
    t = A.thresholds(864, original=False)
    g = A.gates(bank, sorted(bank), loaded, t)
    if not g["ALL_PASS"]:
        return {"gates": g}
    r = A.analyse(bank, sorted(bank), loaded, t)
    v3 = [i for i in sorted(bank) if i.startswith(A.V3_PREFIX)]
    r["secondary"] = A.analyse(bank, v3, loaded, A.thresholds(len(v3), original=True))
    return r


def verdicts(r):
    c = r["confirmatory"]
    return (c["C1_attenuation"]["PASS"], c["C2_add100_beats_avg"]["PASS"], c["C3_interference_small"]["PASS"],
            r["C4_convention_descriptive"]["meets_registered_pattern"], r["C5_location"]["verdict"].split(" (")[0],
            r["C6_mask_vs_expectation"]["verdict"].split(" (")[0])


checks = []
def check(name, cond):
    checks.append((name, bool(cond))); print(("PASS " if cond else "FAIL ") + name)

print("thresholds 864:", A.thresholds(864, False)); print("thresholds 288:", A.thresholds(288, True))
check("thresholds 864 exact", A.thresholds(864, False) == {"G1_max": 172, "G2_min": 346, "G3_min": 260, "C5_effect": 30, "C5_null": 18})

r = run(scenario()); v = verdicts(r); print("all-pass:", v, r["confirmatory"]["C3_interference_small"]["retained_fraction"])
check("all three pass", v[:3] == (True, True, True))
check("C4 pattern met", v[3])
check("C5 null", v[4] == "NO LOCATION EFFECT"); check("C6 null", v[5] == "MASK ACTS LIKE ITS EXPECTATION")
check("secondary uses 288 rows", r["secondary"]["n"] == 288 and r["secondary"]["gates"]["thresholds"]["G1_max"] == 57)

r = run(scenario(add_lost=70)); v = verdicts(r); print("C3 fail:", v)
check("C3 fails when ADD loses 70 of 450", v[2] is False and v[0] and v[1])
r = run(scenario(add_lost=30)); v = verdicts(r); print("C3 borderline 30:", v, r["confirmatory"]["C3_interference_small"]["lower95"])
r = run(scenario(n_avg=430, add_lost=0, add_gained=20)); v = verdicts(r); print("C1 fail:", v)
check("C1 fails alone when AVG loses only 20", v == (False, True, True, True, v[4], v[5]))
r = run(scenario(n_avg=440, add_lost=10)); v = verdicts(r); print("C2 fail:", v)
check("C2 fails when ADD == AVG", v[1] is False)
r = run(scenario(period=False)); check("C4 not met without periods", verdicts(r)[3] is False)
r = run(scenario(n_sparse=300, n_avg=200)); check("gates reject SPARSE 300", "confirmatory" not in r and not r["gates"]["G2_donor_ceiling"])
r = run(scenario(base=180)); check("gates reject BASE 180", "confirmatory" not in r and not r["gates"]["G1_base_floor"])
r = run(scenario(n_sparse=346, n_avg=200, base=87)); check("gates reject separation 260-1", "confirmatory" not in r and not r["gates"]["G3_separation"])
r = run(scenario(n_sparse=346, n_avg=200, base=86)); check("gates accept separation exactly 260", r["gates"]["ALL_PASS"])

for d, want in ((40, "LOCATION EFFECT REPLICATES"), (-40, "REVERSED"), (24, "INCONCLUSIVE"), (18, "NO LOCATION EFFECT")):
    r = run(scenario(sa_minus_rm=d)); got = verdicts(r)[4]; check(f"C5 diff {d} -> {want}", got == want)
for d, want in ((40, "MASK EXCEEDS ITS EXPECTATION"), (-40, "MASK FALLS SHORT OF ITS EXPECTATION"), (24, "INCONCLUSIVE")):
    r = run(scenario(rm_minus_sp=d)); got = verdicts(r)[5]; check(f"C6 diff {d} -> {want}", got == want)

# guards
def refuses(fn):
    try:
        fn(); return False
    except SystemExit:
        return True
arms = scenario()
check("refuses max_new_tokens 96", refuses(lambda: A.load([runfile("m96.json", arms, mnt=96)])))
check("refuses duplicate arm", refuses(lambda: A.load([runfile("d1.json", {"BASE": arms["BASE"]}), runfile("d2.json", {"BASE": arms["BASE"]})])))
bad = dict(arms); bad["BASE"] = {i: g for i, g in list(arms["BASE"].items())[:-1]}
check("refuses missing row", refuses(lambda: A.load([runfile("mr.json", {"BASE": bad["BASE"]})])))
sha = A.BANK_SHA; A.BANK_SHA = "0" * 64
check("refuses wrong bank sha", refuses(lambda: A.load([runfile("ws.json", arms)])))
A.BANK_SHA = sha
check("primary strips exactly one period", A.primary("R1.", "R1") and not A.primary("R1..", "R1") and A.primary(" R1 ", "R1") and not A.primary("R1 .", "R1"))
print(f"\n{sum(ok for _, ok in checks)}/{len(checks)} checks passed")
