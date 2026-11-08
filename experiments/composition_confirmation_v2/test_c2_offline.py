"""Offline tests for CONFIRM-002's scorer, Stage A and Stage B analyses. No model is loaded and no fresh-bank
output is read. Fake run files use only the banks' row ids and references.
"""
from __future__ import annotations

import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyse_stage_b as B  # noqa: E402
import analyse_stage_a as A  # noqa: E402
import score_boundary as SB  # noqa: E402

PY = sys.executable
tmp = Path(tempfile.mkdtemp())
checks = []


def check(name, cond):
    checks.append((name, bool(cond))); print(("PASS " if cond else "FAIL ") + name, flush=True)


# ---------------- scorer ----------------
from transformers import AutoTokenizer  # noqa: E402
sys.path.insert(0, str(HERE.parent / "fresh_dual_skill_instrument_v1"))
from score_arms import render as t_render  # noqa: E402
tok = AutoTokenizer.from_pretrained(str(HERE.parent / "titan_opallo_skill_vector_v1" / "arms" / "BASE"), local_files_only=True)
row = json.loads(A.VIEWS.read_text(encoding="utf-8").splitlines()[0])
sysm = next(m["content"] for m in row["prompt"] if m["role"] == "system"); user = next(m["content"] for m in row["prompt"] if m["role"] == "user")
check("scorer render == the transition scorer's byte-exact template", SB.render(tok, row["prompt"]) == t_render(sysm, user))
rows = A.views_rows()
rec = json.loads((A.RECORDED["BASE"]).read_text(encoding="utf-8"))
dec = {p["id"]: SB.score_reply(rows[p["id"]], p["prediction"]) for p in rec["predictions"]}
b = sum(d["credited"] for i, d in dec.items() if rows[i]["row_type"] == "boundary")
at = sum(d["attempt"] for i, d in dec.items() if rows[i]["row_type"] == "answerable")
co = sum(d["credited"] for i, d in dec.items() if rows[i]["row_type"] == "answerable")
check(f"score_reply reproduces recorded Supa 71/111/40 (got {b}/{at}/{co})", (b, at, co) == (71, 111, 40))
sealed = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/hskill-views-002/HSKILL_CONFIRM_V2.jsonl")
r = subprocess.run([PY, str(HERE / "score_boundary.py"), "--bank", str(sealed), "--arms-root", ".", "--arms", "X",
                    "--output", str(tmp / "never.json")], capture_output=True, text=True)
check("scorer refuses the sealed bank by name", r.returncode != 0 and "REFUSING" in (r.stderr + r.stdout))

# ---------------- Stage A canary ----------------
def fake_views_run(path, arms_preds):
    d = {"bank_sha256": A.VIEWS_SHA, "runtime": {"max_new_tokens": 48}, "arms": []}
    for arm, preds in arms_preds.items():
        d["arms"].append({"arm": arm, "predictions": [{"id": i, "prediction": g, **SB.score_reply(rows[i], g)} for i, g in preds.items()]})
    path.write_text(json.dumps(d)); return path

rec_s = {p["id"]: p["prediction"] for p in json.loads(A.RECORDED["BASE"].read_text(encoding="utf-8"))["predictions"]}
rec_t = {p["id"]: p["prediction"] for p in json.loads(A.RECORDED["SPARSE"].read_text(encoding="utf-8"))["predictions"]}
good = fake_views_run(tmp / "good.json", {"BASE": rec_s, "SPARSE": rec_t})
c = A.canary(A.mine([good], rows), rows)
check("canary PASSES on the recorded predictions themselves", c["PASS"] and c["BASE"]["per_row_credited_agreement"] == 224)
bad_s = dict(rec_s)
for i in [i for i in rows if rows[i]["row_type"] == "boundary"][:10]:
    bad_s[i] = "Sure, here it is: 42"                       # flip ten boundary abstentions
bad = fake_views_run(tmp / "bad.json", {"BASE": bad_s, "SPARSE": rec_t})
c2 = A.canary(A.mine([bad], rows), rows)
check("canary FAILS when 10 boundary decisions are perturbed", not c2["PASS"])

# ---------------- Stage B decision functions and gate mapping ----------------
rng = np.random.default_rng(0)
def arr(n, k, seed):
    a = np.zeros(n, bool); a[np.random.default_rng(seed).choice(n, k, replace=False)] = True; return a

nT, nB, nA = 864, 336, 109
def scenario(th_t_loss=5, h_gain=45, th_gain=45, att_loss=0, thf_loss=150, spar_t=447, base_t=4, h25_gain=45):
    S = arr(nT, spar_t, 1); Bt = np.zeros(nT, bool); Bt[np.where(~S)[0][:base_t]] = True
    TH = S.copy(); TH[np.where(S)[0][:th_t_loss]] = False
    THF = S.copy(); THF[np.where(S)[0][:thf_loss]] = False
    Bb = arr(nB, 150, 2); Sb = Bb.copy()
    H25 = Bb.copy(); H25[np.where(~Bb)[0][:h25_gain]] = True
    THb = Sb.copy(); THb[np.where(~Sb)[0][:th_gain]] = True
    Sa = np.ones(nA, bool); THa = Sa.copy(); THa[:att_loss] = False
    T = {"BASE": Bt, "SPARSE": S, "TH25": TH, "THFULL": THF}
    Hb = {"BASE": Bb, "SPARSE": Sb, "H25": H25, "TH25": THb, "THFULL": Sb}
    Ha = {"BASE": Sa, "SPARSE": Sa, "H25": Sa, "TH25": THa, "THFULL": Sa}
    return T, Hb, Ha

def run_b(**kw):
    T, Hb, Ha = scenario(**kw)
    gt = B.t_gates(T["BASE"], T["SPARSE"]); gt["ALL"] = gt["G1"] and gt["G2"] and gt["G3"]
    gh = B.g_h(Hb["BASE"], Hb["H25"])
    r = B.confirmatory(T, Hb, Ha, gt, gh, np.random.default_rng(1))
    return gt, gh, r, B.c5(T["SPARSE"], T["THFULL"])

gt, gh, r, c5 = run_b()
check("all four confirmatory limbs pass in the good scenario", all(r[k]["PASS"] for k in ("C1", "C2", "C3", "C4")))
check("C5 passes when THFULL loses 150 of 447", c5["PASS"])
gt, gh, r, c5 = run_b(th_t_loss=40)
check("C1 fails when TH25 loses 40 of 447 (margin 22)", not r["C1"]["PASS"] and r["C2"]["PASS"])
gt, gh, r, c5 = run_b(th_gain=10)
check("C2 fails when the composite gains only 10 boundary rows (< 17)", not r["C2"]["PASS"])
gt, gh, r, c5 = run_b(th_gain=20, h25_gain=45)
check("C3 fails when the composite keeps 20/45 of H's solo gain", not r["C3"]["PASS"])
gt, gh, r, c5 = run_b(h25_gain=10)
check("G_H failure makes C3 NOT EVALUABLE and removes it from Holm", r["C3"].get("status") == "NOT_EVALUABLE_G_H_FAILED" and "holm" in r["C2"])
gt, gh, r, c5 = run_b(att_loss=9)
check("C4 fails when the composite refuses 9 more answerable rows", not r["C4"]["PASS"])
gt, gh, r, c5 = run_b(spar_t=300)
check("T gate failure rejects C1", r["C1"].get("status") == "REJECTED_BY_T_GATES")
gt, gh, r, c5 = run_b(thf_loss=20)
check("C5 fails when THFULL loses only 20", not c5["PASS"])
check("min_gain scales: 17 at 336, 34 at 672", B.min_gain(336) == 17 and B.min_gain(672) == 34)

# ---------------- Stage B end to end on fake run files (bank ids/references only) ----------------
tb = json.loads(B.T_BANK.read_text(encoding="utf-8"))["rows"]; tref = {x["id"]: x["reference"] for x in tb}; tids = sorted(tref)
hrows = [json.loads(l) for l in B.H_BANK.read_text(encoding="utf-8").splitlines() if l.strip()]
T, Hb, Ha = scenario()
def t_file(name, arms):
    d = {"bank_sha256": B.T_SHA, "runtime": {"max_new_tokens": 48}, "arms": [
        {"arm": a, "predictions": [{"id": i, "generation": (tref[i] if T[a][j] else "nope")} for j, i in enumerate(tids)]} for a in arms]}
    (tmp / name).write_text(json.dumps(d)); return str(tmp / name)
bids = sorted(x["id"] for x in hrows if x["row_type"] == "boundary"); aids = sorted(x["id"] for x in hrows if x["row_type"] == "answerable")
def h_file(name, arms):
    arms_out = []
    for a in arms:
        preds = [{"id": i, "row_type": "boundary", "credited": bool(Hb[a][j]), "prediction": ""} for j, i in enumerate(bids)]
        preds += [{"id": i, "row_type": "answerable", "credited": False, "attempt": bool(Ha[a][j]), "prediction": ""} for j, i in enumerate(aids)]
        arms_out.append({"arm": a, "predictions": preds})
    (tmp / name).write_text(json.dumps({"bank_sha256": B.H_SHA, "runtime": {"max_new_tokens": 48}, "arms": arms_out})); return str(tmp / name)
tg = t_file("tg.json", ["BASE", "SPARSE"]); tr = t_file("tr.json", ["TH25", "THFULL"])
hg1 = h_file("hg1.json", ["BASE"]); hg2 = h_file("hg2.json", ["H25"]); hs = h_file("hs.json", ["SPARSE"]); hr = h_file("hr.json", ["TH25", "THFULL"])
rc = subprocess.call([PY, str(HERE / "analyse_stage_b.py"), "--gates-only", "--t-runs", tg, "--h-runs", hg1, hg2, "--output", str(tmp / "g.json")],
                     stdout=subprocess.DEVNULL)
check("end-to-end gates-only run exits 0 with both gate sets passing", rc == 0 and json.loads((tmp / "g.json").read_text())["gate_H"]["PASS"])
rc = subprocess.call([PY, str(HERE / "analyse_stage_b.py"), "--t-runs", tg, tr, "--h-runs", hg1, hg2, hs, hr, "--output", str(tmp / "full.json")],
                     stdout=subprocess.DEVNULL)
full = json.loads((tmp / "full.json").read_text())
check("end-to-end full run: C1-C4 PASS and C5 PASS", rc == 0 and all(full["confirmatory"][k]["PASS"] for k in ("C1", "C2", "C3", "C4")) and full["C5_full_strength"]["PASS"])
dup = subprocess.run([PY, str(HERE / "analyse_stage_b.py"), "--t-runs", tg, tg, "--h-runs", hg1, hg2, "--output", str(tmp / "d.json")], capture_output=True, text=True)
check("analysis refuses a duplicated arm across run files", dup.returncode != 0)

# ---------------- Stage A power smoke (small) ----------------
pw = A.power({k: v for k, v in T.items() if k != "THFULL"} | {"THFULL": T["THFULL"]}, Hb, Ha, sims=20, inner=200)
check("power projection runs and reports every limb", set(pw) >= {"C1", "C2", "C3", "C4", "ALL4", "C5"})
print(f"\n{sum(ok for _, ok in checks)}/{len(checks)} checks passed")
