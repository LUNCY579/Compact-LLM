"""TRISKILL-001 held-out Stage B analysis (PREREG §7, AMENDMENT_1).

Run once, after all five registered arms have been scored. C3's world-level p,
not its row-level p, enters the four-limb Holm family. The C4 bootstrap samples
whole C worlds alongside individual T rows. Geometry and retention are separate
descriptive endpoints and cannot alter this decision.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

from stage_b_guard import selected

HERE = Path(__file__).resolve().parent
SB = HERE / "stage_b"
T_BANK = HERE / "T_BANK_V5.json"
H_BANK = HERE / "H_FRESH.jsonl"
H_SHA = "471c7a48a3821038e9e401f710526d01f0cef4b3d70b9c8625cff15ad8ed9cc4"
BOOT = 10_000
SEED = 20260926
ALPHA = 0.05
sys.path.insert(0, str(HERE))
sys.path.insert(0, r"C:/Users/clclc/BusinessProject/Rineya-Dataset/Rineya_Skill_Integration_Foundry_v1/src")
sys.path.insert(0, r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/hskill-views-002")
from sif.scoring import score_records  # noqa: E402
import boundary_rule as BR  # noqa: E402
from score_c import load as load_c_bank  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refusal(text: str) -> bool:
    return bool(BR.abstains(BR.first_line(text)))


def primary_t(text: str, reference: str) -> bool:
    stripped = text.strip()
    if stripped.endswith("."):
        stripped = stripped[:-1]
    return stripped == reference


def one_sided(gain: int, loss: int) -> float:
    n = gain + loss
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(gain, n + 1)) / 2**n


def percentile(values: list[int], q: float) -> float:
    ordered = sorted(values)
    at = q * (len(ordered) - 1)
    lo = math.floor(at)
    hi = math.ceil(at)
    return ordered[lo] + (at - lo) * (ordered[hi] - ordered[lo])


def bootstrap(diffs: list[int], seed: int) -> list[int]:
    rng = random.Random(seed)
    n = len(diffs)
    return [sum(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(BOOT)]


def paired(ref: dict, arm: dict, field: str) -> dict:
    if set(ref) != set(arm):
        raise SystemExit(f"paired ID mismatch: {field}")
    gained = sum(bool(arm[i][field]) and not bool(ref[i][field]) for i in ref)
    lost = sum(bool(ref[i][field]) and not bool(arm[i][field]) for i in ref)
    return {"reference": sum(bool(r[field]) for r in ref.values()),
            "arm": sum(bool(r[field]) for r in arm.values()),
            "gained": gained, "lost": lost, "net": gained - lost}


def load_c(name: str, keys: list[dict]) -> dict:
    gold = {r["id"]: r["answer"] for r in keys}
    path = SB / "c_preds" / f"{name}.jsonl"
    preds = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(preds) != 336 or len({p["id"] for p in preds}) != 336:
        raise SystemExit(f"{name}: C predictions missing or duplicated")
    _, scored = score_records(keys, preds)
    completions = {p["id"]: p["completion"] for p in preds}
    rows = {r["id"]: {**r, "reference": gold[r["id"]],
                       "completion": completions[r["id"]], "refusal": refusal(completions[r["id"]])}
            for r in scored}
    if len(rows) != 336:
        raise SystemExit(f"{name}: C scored row count mismatch")
    return rows


def load_t(name: str, bank: dict, bank_sha: str) -> dict:
    path = SB / "t" / f"{name}.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("bank_sha256") != bank_sha or result.get("bank_n") != 864 or result.get("runtime", {}).get("max_new_tokens") != 48:
        raise SystemExit(f"{name}: T scoring receipt mismatch")
    if len(result.get("arms", [])) != 1:
        raise SystemExit(f"{name}: expected one T arm")
    refs = {r["id"]: r for r in bank["rows"]}
    preds = result["arms"][0]["predictions"]
    if len(preds) != 864 or {p["id"] for p in preds} != set(refs):
        raise SystemExit(f"{name}: T predictions mismatch")
    return {p["id"]: {"primary": primary_t(p["generation"], refs[p["id"]]["reference"]),
                       "strict": p["generation"].strip() == refs[p["id"]]["reference"],
                       "refusal": refusal(p["generation"]), "generation": p["generation"],
                       "reference": refs[p["id"]]["reference"], "family": refs[p["id"]]["family"],
                       "trailing_period": p["generation"].strip().endswith(".")}
            for p in preds}


def load_h(name: str, bank: dict) -> dict:
    path = SB / "h" / f"{name}.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("bank_sha256") != H_SHA or result.get("runtime", {}).get("max_new_tokens") != 48 or not str(result.get("rule_module_sha256", "")).startswith("416cfb6a"):
        raise SystemExit(f"{name}: H scoring receipt mismatch")
    if len(result.get("arms", [])) != 1:
        raise SystemExit(f"{name}: expected one H arm")
    preds = result["arms"][0]["predictions"]
    if len(preds) != 336 or {p["id"] for p in preds} != set(bank):
        raise SystemExit(f"{name}: H predictions mismatch")
    if any(p["row_type"] != "boundary" for p in preds):
        raise SystemExit(f"{name}: H_FRESH contains non-boundary rows")
    return {p["id"]: {"credited": bool(p["credited"]), "prediction": p["prediction"],
                       "reference": bank[p["id"]]["reference"], "family": bank[p["id"]]["family"]}
            for p in preds}


def world_test(ref: dict, arm: dict) -> dict:
    nets = defaultdict(int)
    for row_id, row in ref.items():
        nets[row["parent_id"]] += int(arm[row_id]["semantic_correct"]) - int(row["semantic_correct"])
    up = sum(x > 0 for x in nets.values())
    down = sum(x < 0 for x in nets.values())
    if len(nets) != 112:
        raise SystemExit(f"C_FRESH world count {len(nets)} != 112")
    return {"worlds": 112, "up": up, "down": down, "tied": 112 - up - down,
            "p_one_sided": one_sided(up, down)}


def holm(pvalues: dict[str, float]) -> dict:
    order = sorted(pvalues, key=pvalues.get)
    out = {}
    prior_pass = True
    for index, limb in enumerate(order):
        threshold = ALPHA / (len(order) - index)
        prior_pass = prior_pass and pvalues[limb] < threshold
        out[limb] = {"p": pvalues[limb], "threshold": threshold, "significant": prior_pass}
    return out


def lost_samples(ref: dict, arm: dict, field: str, group: str, prediction: str, limit: int = 5) -> dict:
    groups = defaultdict(list)
    for row_id in ref:
        if ref[row_id][field] and not arm[row_id][field]:
            key = ref[row_id][group]
            if len(groups[key]) < limit:
                groups[key].append({"id": row_id, "reference": ref[row_id].get("reference", ref[row_id].get("answer")),
                                    "parent": ref[row_id].get("parent_id"),
                                    "TT": ref[row_id][prediction], "TTC": arm[row_id][prediction]})
    return dict(groups)


def new_refusal_samples(ref: dict, arm: dict, group: str, prediction: str, limit: int = 5) -> dict:
    groups = defaultdict(list)
    for row_id in ref:
        if not ref[row_id]["refusal"] and arm[row_id]["refusal"]:
            key = ref[row_id][group]
            if len(groups[key]) < limit:
                groups[key].append({"id": row_id, "reference": ref[row_id]["reference"],
                                    "TT": ref[row_id][prediction], "TTC": arm[row_id][prediction]})
    return dict(groups)


def main() -> int:
    composite, solo, beta = selected()
    out = HERE / "STAGE_B_RESULT.json"
    if out.exists():
        raise SystemExit("Stage B result exists (create-only)")
    names = ("BASE", "TT", composite, solo, "TTR")
    for name in names:
        for path in (SB / "c_preds" / f"{name}.jsonl", SB / "t" / f"{name}.json", SB / "h" / f"{name}.json"):
            if not path.exists():
                raise SystemExit(f"Stage B incomplete: {path}")
    t_raw = T_BANK.read_bytes()
    t_sha = hashlib.sha256(t_raw).hexdigest()
    t_bank = json.loads(t_raw)
    if len(t_bank["rows"]) != 864:
        raise SystemExit("T_BANK_V5 length mismatch")
    h_raw = H_BANK.read_bytes()
    if hashlib.sha256(h_raw).hexdigest() != H_SHA:
        raise SystemExit("H_FRESH sha256 mismatch")
    h_bank = {r["id"]: r for r in (json.loads(line) for line in h_raw.decode("utf-8").splitlines() if line.strip())}
    if len(h_bank) != 336:
        raise SystemExit("H_FRESH length mismatch")
    _, c_keys = load_c_bank("cfresh")
    C = {name: load_c(name, c_keys) for name in names}
    T = {name: load_t(name, t_bank, t_sha) for name in names}
    H = {name: load_h(name, h_bank) for name in names}
    tt_t, x_t = T["TT"], T[composite]
    tt_h, x_h = H["TT"], H[composite]
    tt_c, x_c = C["TT"], C[composite]

    t_pair = paired(tt_t, x_t, "primary")
    h_pair = paired(tt_h, x_h, "credited")
    c_pair = paired(tt_c, x_c, "semantic_correct")
    t_diffs = [int(x_t[i]["primary"]) - int(tt_t[i]["primary"]) for i in tt_t]
    h_diffs = [int(x_h[i]["credited"]) - int(tt_h[i]["credited"]) for i in tt_h]
    bt = bootstrap(t_diffs, SEED + 1)
    bh = bootstrap(h_diffs, SEED + 2)
    t_margin = -0.05 * t_pair["reference"]
    h_margin = -0.05 * h_pair["reference"]
    c1 = {**t_pair, "margin": t_margin, "lower95": percentile(bt, 0.05),
          "p_bootstrap": sum(v <= t_margin for v in bt) / BOOT}
    c2 = {**h_pair, "margin": h_margin, "lower95": percentile(bh, 0.05),
          "p_bootstrap": sum(v <= h_margin for v in bh) / BOOT}
    c_gain, c_loss = c_pair["gained"], c_pair["lost"]
    world = world_test(tt_c, x_c)
    c3 = {**c_pair, "required_net_rows": 34, "p_row_one_sided": one_sided(c_gain, c_loss),
          "world_level": world, "holm_p_source": "world_level.p_one_sided"}

    t_refusal = [int(x_t[i]["refusal"]) - int(tt_t[i]["refusal"]) for i in tt_t]
    c_world_refusal = defaultdict(int)
    for row_id, row in tt_c.items():
        c_world_refusal[row["parent_id"]] += int(x_c[row_id]["refusal"]) - int(row["refusal"])
    if len(c_world_refusal) != 112:
        raise SystemExit("C4 world count mismatch")
    rng = random.Random(SEED + 4)
    worlds = list(c_world_refusal.values())
    b4 = [sum(t_refusal[rng.randrange(864)] for _ in range(864)) +
          sum(worlds[rng.randrange(112)] for _ in range(112)) for _ in range(BOOT)]
    c4_point = sum(t_refusal) + sum(c_world_refusal.values())
    c4 = {"TT_refusals": sum(r["refusal"] for r in tt_t.values()) + sum(r["refusal"] for r in tt_c.values()),
          "TTC_refusals": sum(r["refusal"] for r in x_t.values()) + sum(r["refusal"] for r in x_c.values()),
          "net_extra": c4_point, "upper95": percentile(b4, 0.95), "max_extra": 5,
          "p_bootstrap": sum(v >= 5 for v in b4) / BOOT,
          "bootstrap": "T rows individually plus C worlds as clusters (Amendment 1)"}

    adjusted = holm({"C1": c1["p_bootstrap"], "C2": c2["p_bootstrap"],
                     "C3": world["p_one_sided"], "C4": c4["p_bootstrap"]})
    c1["holm"] = adjusted["C1"]
    c2["holm"] = adjusted["C2"]
    c3["holm"] = adjusted["C3"]
    c4["holm"] = adjusted["C4"]
    c1["pass"] = c1["lower95"] > t_margin and c1["holm"]["significant"]
    c2["pass"] = c2["lower95"] > h_margin and c2["holm"]["significant"]
    c3["pass"] = c3["net"] >= 34 and c3["p_row_one_sided"] < ALPHA and c3["holm"]["significant"]
    c4["pass"] = c4["upper95"] < 5 and c4["holm"]["significant"]
    limbs = {"C1": c1, "C2": c2, "C3": c3, "C4": c4}

    table = {}
    for name in names:
        table[name] = {"C_semantic": sum(r["semantic_correct"] for r in C[name].values()),
                       "C_strict": sum(r["strict_correct"] for r in C[name].values()),
                       "C_refusals": sum(r["refusal"] for r in C[name].values()),
                       "T_primary": sum(r["primary"] for r in T[name].values()),
                       "T_strict": sum(r["strict"] for r in T[name].values()),
                       "T_trailing_periods": sum(r["trailing_period"] for r in T[name].values()),
                       "T_refusals": sum(r["refusal"] for r in T[name].values()),
                       "H_boundary": sum(r["credited"] for r in H[name].values())}
    solo_gain = table[solo]["C_semantic"] - table["BASE"]["C_semantic"]
    f_comp = table[composite]["C_strict"] / table[composite]["C_semantic"] if table[composite]["C_semantic"] else None
    f_solo = table[solo]["C_strict"] / table[solo]["C_semantic"] if table[solo]["C_semantic"] else None
    passed = all(x["pass"] for x in limbs.values())
    if passed and f_comp is not None and f_comp >= 0.90:
        headline = "a third skill arrives, with its output contract, with no measurable loss of the first two."
    elif passed:
        headline = ("C's content arrives with no measurable loss of the first two; its "
                    f"{{\"answer\": N}} output contract is overridden by Titan-Transition's bare-answer convention (f = {f_comp:.3f}).")
    else:
        failed = ", ".join(k for k, v in limbs.items() if not v["pass"])
        if f_comp is None:
            contract = "No semantic-correct composite C row exists for a contract fraction."
        elif f_comp >= 0.90:
            contract = f"C's output contract held on {f_comp:.3f} of its correct C rows (descriptive)."
        else:
            contract = ("C's content and output contract are reported separately: the "
                        f"{{\"answer\": N}} contract held on {f_comp:.3f} of correct C rows (descriptive).")
        headline = f"{failed} failed. {contract}"
    samples = {"T_lost_by_family": lost_samples(tt_t, x_t, "primary", "family", "generation"),
               "H_lost_by_family": lost_samples(tt_h, x_h, "credited", "family", "prediction"),
               "C_lost_by_variant": lost_samples(tt_c, x_c, "semantic_correct", "variant", "completion"),
               "T_new_refusals_by_family": new_refusal_samples(tt_t, x_t, "family", "generation"),
               "C_new_refusals_by_variant": new_refusal_samples(tt_c, x_c, "variant", "completion")}
    result = {"schema": 1, "experiment": "TRISKILL-001", "stage": "B",
              "registered_in": ["PREREG.md (0d2b64b4) §7", "AMENDMENT_1.md (6f594ba8)"],
              "selected": {"beta_star": beta, "composite": composite, "solo": solo},
              "bank_sha256": {"T_BANK_V5": t_sha, "H_FRESH": H_SHA,
                              "C_FRESH": "52f921fc713898e06edf0ebf0241d89e1dd29e331d4c2ad65a094a79497d7ae3"},
              "bootstrap": {"draws": BOOT, "seed": SEED, "one_sided_percentiles": [5, 95]},
              "limbs": limbs, "table": table, "holm": adjusted,
              "P_C": (c_pair["net"] / solo_gain if solo_gain else None),
              "P_C_denominator_solo_gain": solo_gain,
              "contract_fraction": {"composite": f_comp, "solo": f_solo},
              "samples": samples,
              "verdict": "PASS" if passed else "FAIL", "headline": headline,
              "quality_claim": False, "release_permitted": False}
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"verdict": result["verdict"], "headline": headline, "limbs": limbs,
                      "contract_fraction": result["contract_fraction"], "P_C": result["P_C"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
