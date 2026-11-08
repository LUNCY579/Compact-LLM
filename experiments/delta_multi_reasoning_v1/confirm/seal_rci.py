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
    ap.add_argument("--instrument", default="RCI-001")
    ap.add_argument("--claim-read", metavar="REASON", dest="claim_read",
                    help="DRAW DOWN THE BUDGET NOW, before rendering. See --help notes.")
    ap.add_argument("--verify", action="store_true",
                    help="recompute the hash and report whether the seal is intact and unspent")
    a = ap.parse_args(argv)

    sealed = a.root / "RCI_SEALED.jsonl"
    seal_path = a.root / "RCI_SEAL.json"

    if a.claim_read:
        # THE FIX. RCI-001 incremented `reads_used` AFTER rendering, so between the render
        # starting and the counter moving the record read "unspent" while a read was in flight.
        # A peer audit arrived in exactly that window with the words "reads_used is still 0, so
        # this is free right now and impossible to fix afterwards". It was not still 0. The
        # floor correction it carried could no longer be applied.
        #
        # The budget is now claimed FIRST and the render refuses to run until it has been.
        # Claiming is irreversible by design: if the read then fails for a technical reason,
        # that is a spent instrument, and the honest record of that is more valuable than a
        # counter that can go backwards.
        if not seal_path.exists():
            raise SystemExit("no seal record found; seal the bank before claiming a read")
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        got = sha_file(sealed)
        if got != seal["sealed_sha256"]:
            raise SystemExit(f"REFUSING: content changed since sealing\n"
                             f"  recorded   {seal['sealed_sha256']}\n  recomputed {got}")
        rb = seal["read_budget"]
        if rb["reads_used"] >= rb["reads_authorised"]:
            raise SystemExit(f"REFUSING: budget already spent "
                             f"({rb['reads_used']}/{rb['reads_authorised']}). "
                             f"This instrument is finished. Build a successor.")
        rb["reads_used"] += 1
        rb.setdefault("claims", []).append({
            "reason": a.claim_read,
            "sealed_sha256_at_claim": got,
            "note": ("claimed BEFORE rendering; any audit arriving after this point is too "
                     "late to change the instrument and must be published as an erratum"),
        })
        seal_path.write_text(json.dumps(seal, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"claimed": True, "reason": a.claim_read,
                          "reads_used": rb["reads_used"],
                          "reads_authorised": rb["reads_authorised"],
                          "next": "render with render_rci.py --i-am-spending-the-seal"},
                         indent=2))
        return 0

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
    # RCI-001 put one floor at the top of `floors.SEALED`; RCI-002 nests per-endpoint and
    # per-stratum reports under it. Accept both rather than forking the sealer.
    fl = build["floors"]["SEALED"]
    fl = fl if "binding_floor_rate" in fl else fl["aggregate"]
    rows = [json.loads(l) for l in sealed.read_text(encoding="utf-8").splitlines() if l.strip()]

    seal = {
        "schema": 1, "record": f"{a.instrument.replace(chr(45), chr(95))}_SEAL",
        "instrument": a.instrument,
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
            "how_to_draw_it_down": ("seal_rci.py --claim-read '<reason>' FIRST, then "
                                    "render_rci.py --i-am-spending-the-seal. The render "
                                    "refuses until the claim is recorded; this closes the "
                                    "window in which the record reads unspent while a read is "
                                    "in flight, which is how RCI-001's floor correction "
                                    "arrived too late to apply."),
        },
        "floors_published_before_any_model_call": True,
        "binding_floor_rate": fl["binding_floor_rate"],
        "binding_predictor": fl["binding_predictor"],
        "endpoints": build.get("endpoints"),
        "floor_rule": build["floor_rule"],
        "calibration_inherited": {
            "verdict": calib["verdict"],
            "scored_bank": "CALIBRATION twin (disjoint entity partition, same generator)",
            "systems": calib.get("systems"),
            "endpoints": {k: {kk: v[kk] for kk in ("n", "B", "D", "direction") if kk in v}
                          for k, v in calib.get("endpoints", {}).items()},
            "registered_bands": calib.get("bands_to_register_for_the_sealed_read"),
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
