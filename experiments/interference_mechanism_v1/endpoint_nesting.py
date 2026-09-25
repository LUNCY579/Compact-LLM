"""INTERFERENCE-MECH-001 step 0c (exploratory, zero forwards): is the Titan-skill loss a property of the
ROWS and the COEFFICIENT, or of the donor's representation?

Data, all on T_BANK_V3X (864 rows; exposed after CONFIRM-001):
  - CONFIRM-001: SPARSE, AVG, ADD-100 (reconstructed FP32 dOpallo);
  - the audit lane's TITAN-OPALLO-EXACT-ENDPOINT-DEV-V4: DEV-A100-B100-STORED and DEV-A100-B075-STORED
    (stored BF16 dOpallo, which differs from the reconstruction by 50.7% of its norm, cosine 0.876).
Questions:
  Q1 do the two full-strength composites lose the SAME rows relative to SPARSE?
  Q2 is the damage NESTED in beta: is every row lost at b = 0.75 also lost at b = 1.00?
  Q3 how much of it overlaps the rows averaging loses?
Writes ENDPOINT_NESTING.json.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
C1 = EXP / "composition_confirmation_v1"
DEV = EXP / "titan_opallo_exact_dev_v4" / "runs"


def prim(g, r):
    s = g.strip()
    return (s[:-1] if s.endswith(".") else s) == r


def arms(f):
    d = json.loads(Path(f).read_text(encoding="utf-8"))
    assert d["bank_sha256"] == "173aa372466d5e3cc8efbebe4cc7e7ba09a8eb34de4598ce8f9d96a2b8e6557c", f
    return {a["arm"]: {p["id"]: p["generation"] for p in a["predictions"]} for a in d["arms"]}


def main():
    bank = {r["id"]: r["reference"] for r in json.loads((C1 / "T_BANK_V3X.json").read_text(encoding="utf-8"))["rows"]}
    ids = sorted(bank)
    g = arms(C1 / "CONF_GATES.json") | arms(C1 / "CONF_COMPOSE.json")
    for f in sorted(DEV.glob("*.json")):
        g |= arms(f)
    V = {k: np.array([prim(v[i], bank[i]) for i in ids]) for k, v in g.items()}
    S = V["SPARSE"]
    lost = {k: set(map(int, np.where(S & ~V[k])[0])) for k in ("AVG", "ADD-100", "DEV-A100-B100-STORED", "DEV-A100-B075-STORED")}
    gained = {k: set(map(int, np.where(~S & V[k])[0])) for k in ("ADD-100", "DEV-A100-B100-STORED", "DEV-A100-B075-STORED")}
    a, b, c = lost["ADD-100"], lost["DEV-A100-B100-STORED"], lost["DEV-A100-B075-STORED"]
    out = {
        "Q1_same_rows": {"lost_reconstructed": len(a), "lost_stored": len(b), "both": len(a & b), "jaccard": len(a & b) / len(a | b),
                         "gained_both": len(gained["ADD-100"] & gained["DEV-A100-B100-STORED"]),
                         "identical_generations": sum(g["ADD-100"][i].strip() == g["DEV-A100-B100-STORED"][i].strip() for i in ids)},
        "Q2_nested_in_beta": {"lost_at_0.75": len(c), "also_lost_at_1.00": len(c & b), "lost_only_at_0.75": len(c - b),
                              "lost_only_at_1.00": len(b - c)},
        "Q3_overlap_with_AVG": {"reconstructed": len(a & lost["AVG"]), "stored": len(b & lost["AVG"]), "AVG_lost": len(lost["AVG"])},
    }
    (HERE / "ENDPOINT_NESTING.json").write_bytes((json.dumps(out, indent=1) + "\n").encode())
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
