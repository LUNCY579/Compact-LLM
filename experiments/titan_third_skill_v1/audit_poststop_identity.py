"""Compare TT and the norm-matched random arm's raw Stage A generations by ID."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "POSTSTOP_PREDICTION_IDENTITY.json"


def compare(a: dict, b: dict) -> dict:
    if set(a) != set(b):
        raise SystemExit("TT and TTR25 prediction IDs differ")
    changed = [{"id": i, "TT": a[i], "TTR25": b[i]} for i in sorted(a) if a[i] != b[i]]
    return {"n": len(a), "identical": len(a) - len(changed), "changed": changed}


def main() -> int:
    if OUT.exists():
        raise SystemExit("identity audit exists (create-only)")
    results = {}
    for bank in ("dev", "cal"):
        a = [json.loads(line) for line in (HERE / "stage_a" / "c_preds" / f"TT__{bank}.jsonl").read_text(encoding="utf-8").splitlines()]
        b = [json.loads(line) for line in (HERE / "poststop_null" / "c_preds" / f"TTR25__{bank}.jsonl").read_text(encoding="utf-8").splitlines()]
        results[f"C_{bank}"] = compare({x["id"]: x["completion"] for x in a},
                                       {x["id"]: x["completion"] for x in b})
    for bank, field in (("t", "generation"), ("h", "prediction")):
        a = json.loads((HERE / "stage_a" / bank / "TT.json").read_text(encoding="utf-8"))["arms"][0]["predictions"]
        b = json.loads((HERE / "poststop_null" / bank / "TTR25.json").read_text(encoding="utf-8"))["arms"][0]["predictions"]
        results[bank.upper()] = compare({x["id"]: x[field] for x in a}, {x["id"]: x[field] for x in b})
    OUT.write_text(json.dumps({"experiment": "TRISKILL-POSTSTOP-001", "identity": results}, indent=1,
                              ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: {"n": v["n"], "identical": v["identical"]} for k, v in results.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
