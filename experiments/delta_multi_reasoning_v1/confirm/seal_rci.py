"""Seal RCI-001's confirmation bank, and provide the check that proves it is still unspent.

A confirmation instrument is worth exactly the discipline around it. This writes the seal
record — content hash, read budget, the conditions of the single authorised read, and the
calibration verdict the sealed bank inherits — and installs a verifier that recomputes the
hash and reports whether the budget has been drawn down.

WHAT THE SEAL DOES AND DOES NOT DO
-----------------------------------
It does not make the file unreadable; nothing here can. What it does is make a read
**detectable and attributable**: the hash proves the content is the content that was
calibrated, and `reads_used` is a single integer that any later reader must increment
honestly. That is the same standard the rest of this programme runs on — `QUARANTINE.json` is
"built to be checked against by code, not read by a human at decision time" — and it fails the
same way, by someone choosing to lie rather than by accident.

THE ONE RULE
------------
The sealed bank may be read ONCE, AFTER a candidate is frozen, and NEVER to choose among arms,
thresholds, prompts, seeds or checkpoints. Using it to select spends it permanently and the
programme is back where `CONFIRMATION_INSTRUMENT_GAP.md` found it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while c := f.read(8 * 1024 * 1024):
            h.update(c)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--verify", action="store_true",
                    help="recompute the hash and report whether the seal is intact and unspent")
    a = ap.parse_args(argv)

    sealed = a.root / "RCI_SEALED.jsonl"
    seal_path = a.root / "RCI_SEAL.json"

    if a.verify:
        if not seal_path.exists():
            raise SystemExit("no seal record found")
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        got = sha_file(sealed)
        intact = got == seal["sealed_sha256"]
        unspent = seal["read_budget"]["reads_used"] == 0
        print(json.dumps({
            "sealed_file": str(sealed),
            "recorded_sha256": seal["sealed_sha256"], "recomputed_sha256": got,
            "content_intact": intact,
            "reads_authorised": seal["read_budget"]["reads_authorised"],
            "reads_used": seal["read_budget"]["reads_used"],
            "unspent": unspent,
            "verdict": ("INTACT AND UNSPENT" if intact and unspent else
                        "CONTENT CHANGED" if not intact else "ALREADY READ"),
        }, indent=2))
        return 0 if (intact and unspent) else 1

    build = json.loads((a.root / "RCI_BUILD.json").read_text(encoding="utf-8"))
    calib = json.loads((a.root / "RCI_CALIBRATION_REPORT.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in sealed.read_text(encoding="utf-8").splitlines() if l.strip()]

    seal = {
        "schema": 1, "record": "RCI_001_SEAL", "instrument": "RCI-001",
        "sealed_file": sealed.name,
        "sealed_sha256": sha_file(sealed),
        "content_sha256_of_rows": build["banks"]["SEALED"]["sha256"],
        "rows": len(rows),
        "packs": len({r["scenario_id"] for r in rows}),
        "unanswerable_rows": sum(1 for r in rows if not r["answerable"]),
        "admissible_uses": ["CONFIRMATION"],
        "inadmissible_uses": ["TRAIN", "SELECTION", "DEV"],
        "read_budget": {
            "reads_authorised": 1, "reads_used": 0,
            "condition": ("after a candidate is frozen; never to choose among arms, thresholds, "
                          "prompts, seeds or checkpoints"),
            "spent_if": "used for selection in any form",
            "how_to_draw_it_down": ("increment reads_used here, then render with "
                                    "render_rci.py --i-am-spending-the-seal"),
        },
        "floors_published_before_any_model_call": True,
        "binding_floor_rate": build["floors"]["SEALED"]["binding_floor_rate"],
        "binding_predictor": build["floors"]["SEALED"]["binding_predictor"],
        "floor_rule": build["floor_rule"],
        "calibration_inherited": {
            "verdict": calib["verdict"],
            "scored_bank": "CALIBRATION twin (disjoint entity partition, same generator)",
            "systems": {s: {"rate": v["rate"], "ci_lower_95": v["ci_lower_95"],
                            "correct_abstention": f"{v['correct_abstention']}/{v['unanswerable_n']}",
                            "false_abstention_on_answerable": v["false_abstention_on_answerable"]}
                        for s, v in calib["systems"].items()},
            "what_it_licenses": ("the sealed bank is EXPECTED to discriminate, on the evidence "
                                 "of its twin. That is an inherited expectation, not a "
                                 "measurement of the sealed bank, and it is the strongest thing "
                                 "that can be said without spending the seal."),
        },
        "calibration_bank_is_now_spent": {
            "file": "RCI_CALIBRATION.jsonl",
            "status": "SPENT for confirmation — it has been scored",
            "remains_admissible_for": ["DEV", "SELECTION"],
        },
        "verification": "python confirm/seal_rci.py --root <this dir> --verify",
        "what_this_seal_cannot_do": (
            "It cannot prevent a read. It makes one detectable and attributable: the hash proves "
            "the content is what was calibrated, and reads_used is an integer a later reader "
            "must increment honestly. It fails by someone choosing to lie, not by accident."),
        "why_this_instrument_exists": (
            "CONFIRMATION_INSTRUMENT_GAP.md: every long-context and literal-UNKNOWN instrument "
            "in the repository is spent, reserved or locked to another programme, so seven "
            "screens terminate in 'measured, never confirmed'. This is the missing instrument."),
    }
    seal_path.write_text(json.dumps(seal, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: seal[k] for k in
                      ("sealed_sha256", "rows", "packs", "unanswerable_rows",
                       "binding_floor_rate", "read_budget")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
