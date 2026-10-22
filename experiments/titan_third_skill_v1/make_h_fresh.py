"""TRISKILL-001 Stage B: build H_FRESH, 48 rows per boundary class from the unused pool (PREREG §5).

The pool, the VIEWS and sealed-CONFIRM derivations, and the SOUP_V3 take are reproduced by IMPORTING
composition_confirmation_v2/make_hskill_soup_v3.py (module import only; its main() is never called).
From what SOUP_V3 left, this bank orders each boundary class by SALT and takes 48. No answerable rows
remain in the pool (measured 2026-09-26), so none are taken; C4 counts refusals on the T and C banks.

Gates, each with a positive control:
  - the re-derived VIEWS parents equal HSKILL_VIEWS_V2.jsonl;
  - the re-derived SOUP_V3 ids equal HSKILL_SOUP_V3.jsonl's source ids (proves the take is reproduced);
  - 0 shared parents with VIEWS, CONFIRM or SOUP_V3 (control: the check fires on an injected SOUP_V3 id);
  - 0 overlap with delta_H's 648 training rows by id or user text (control: all 648 are fingerprinted);
  - 48 rows in each of the 7 classes.
The sealed HSKILL_CONFIRM_V2.jsonl is never opened. Writes H_FRESH.jsonl and H_FRESH_BUILD.json. Create-only, LF.
"""
from __future__ import annotations

import hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
C2 = HERE.parent / "composition_confirmation_v2"
sys.path.insert(0, str(C2))
import make_hskill_soup_v3 as S  # noqa: E402  (module import only)
B, HV, HTRAIN = S.B, S.HV, S.HTRAIN

SALT = "TRISKILL-001-H-FRESH"
PER_CLASS = 48
OUT = HERE / "H_FRESH.jsonl"
REC = HERE / "H_FRESH_BUILD.json"
SOUP = C2 / "HSKILL_SOUP_V3.jsonl"
SOUP_SHA = "686df45b94fa000be8df8d15e1f026a7f74e22deadfa4aacda96f5c6244c8ad5"


def main() -> int:
    if OUT.exists() or REC.exists():
        raise SystemExit("create-only: H_FRESH already exists")
    if S.sha256_file(SOUP) != SOUP_SHA:
        raise SystemExit("HSKILL_SOUP_V3.jsonl does not match its recorded sha256")
    by_class, (tids, tusers, tpairs), src_sha, panel_path = S.pool()

    views, confirm, soup, chosen = set(), set(), set(), []
    for c in sorted(by_class):
        k = B.PER_BOUNDARY_CLASS if c in B.BR.BOUNDARY_CLASSES else B.N_ANSWERABLE
        ordered = sorted(by_class[c], key=lambda r: (B.digest([B.SALT_SELECT, r["id"]]), r["id"]))
        views |= {r["id"] for r in ordered[:k]}
        rest = sorted(ordered[k:], key=lambda r: (B.digest([B.SALT_CONFIRM, r["id"]]), r["id"]))
        confirm |= {r["id"] for r in rest[:k]}
        remaining = sorted(rest[k:], key=lambda r: (B.digest([S.SALT_SOUP, r["id"]]), r["id"]))
        took = remaining[:S.PER_BOUNDARY] if c in B.BR.BOUNDARY_CLASSES else remaining
        soup |= {r["id"] for r in took}
        if c not in B.BR.BOUNDARY_CLASSES:
            if remaining[len(took):]:
                raise SystemExit(f"{c}: answerable rows remain, contrary to the registration")
            continue
        left = sorted(remaining[S.PER_BOUNDARY:], key=lambda r: (B.digest([SALT, r["id"]]), r["id"]))
        if len(left) < PER_CLASS:
            raise SystemExit(f"{c}: only {len(left)} rows remain, need {PER_CLASS}")
        chosen += left[:PER_CLASS]

    views_file = {json.loads(l)["parent_id"] for l in (HV / "HSKILL_VIEWS_V2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    if views != views_file:
        raise SystemExit("POSITIVE CONTROL FAILED: re-derived VIEWS parents differ from the file")
    soup_file = {json.loads(l)["source_id"] for l in SOUP.read_text(encoding="utf-8").splitlines() if l.strip()}
    if soup != soup_file:
        raise SystemExit("POSITIVE CONTROL FAILED: re-derived SOUP_V3 take differs from the file")
    ids = {r["id"] for r in chosen}
    shared = ids & (views | confirm | soup)
    if shared:
        raise SystemExit(f"GATE FAILED: {len(shared)} parents shared with VIEWS/CONFIRM/SOUP_V3")
    probe = next(iter(soup))
    if not ({probe} & (views | confirm | soup)):
        raise SystemExit("CONTROL FAILED: the shared-parent check cannot fire")
    htrain = [json.loads(l) for l in HTRAIN.read_text(encoding="utf-8").splitlines() if l.strip()]
    h_ids = {r["parent_id"] for r in htrain}
    h_users = {[m["content"] for m in r["prompt"] if m["role"] == "user"][-1].strip() for r in htrain}
    ctl = sum(r["parent_id"] in tids for r in htrain)
    if ctl != len(htrain):
        raise SystemExit(f"CONTROL FAILED: only {ctl}/{len(htrain)} delta_H training rows are fingerprinted")
    users = {next(m["content"] for m in r["messages"] if m["role"] == "user").strip() for r in chosen}
    ov_id, ov_user = len(ids & h_ids), len(users & h_users)
    if ov_id or ov_user:
        raise SystemExit(f"GATE FAILED: overlap with delta_H training rows (ids {ov_id}, user texts {ov_user})")

    rows = []
    for r in chosen:
        rc = (r.get("metadata") or {})["reply_class"]
        rows.append({"family": rc, "row_type": "boundary", "gold_verified_by": "corpus_reference_reply",
                     "id": f"HFRESH:{r['id']}", "parent_id": r["id"], "source_id": r["id"],
                     "prompt": [m for m in r["messages"] if m["role"] in ("system", "user")],
                     "reference": next(m["content"] for m in r["messages"] if m["role"] == "assistant"),
                     "source_split": "train", "view_role": "TRISKILL-001-H-FRESH"})
    rows.sort(key=lambda x: (x["family"], x["id"]))
    counts = {}
    for x in rows:
        counts[x["family"]] = counts.get(x["family"], 0) + 1
    if sorted(counts.values()) != [PER_CLASS] * 7:
        raise SystemExit(f"class counts wrong: {counts}")
    payload = "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows).encode("utf-8")
    with open(OUT, "xb") as f:
        f.write(payload)
    rec = {"instrument": "H_FRESH", "experiment": "TRISKILL-001", "registered_in": "PREREG.md (0d2b64b4) section 5",
           "salt": SALT, "per_boundary_class": PER_CLASS, "counts": counts, "n": len(rows), "n_answerable": 0,
           "sha256": hashlib.sha256(payload).hexdigest(), "source_sha256": src_sha, "panel_path": str(panel_path),
           "builder_module_sha256": S.sha256_file(HV / "build_hskill_v2.py"),
           "soup_builder_sha256": S.sha256_file(C2 / "make_hskill_soup_v3.py"),
           "rule_module_sha256": S.sha256_file(HV / "boundary_rule.py"),
           "gates": {"views_rederivation_matches_file": True, "soup_v3_rederivation_matches_file": True,
                     "shared_parents_with_views_confirm_or_soup": 0,
                     "delta_h_training_overlap": {"ids": 0, "user_texts": 0},
                     "controls": {"shared_parent_check_fires_on_injected_soup_id": True,
                                  "delta_h_training_rows_in_fingerprint_set": f"{ctl}/{len(htrain)}"}},
           "sealed_confirm_file_opened": False, "read_budget": "ONE", "quality_claim": False}
    REC.write_bytes((json.dumps(rec, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({k: rec[k] for k in ("n", "counts", "sha256", "gates")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
