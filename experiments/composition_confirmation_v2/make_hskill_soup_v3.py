"""CONFIRM-002 — build HSKILL_SOUP_V3, the fresh boundary + answerable instrument (PREREG.md §3, 779cd431).

Pool construction is replicated VERBATIM from rineya-dataset-retention-records/hskill-views-002/
build_hskill_v2.py (lines 108-160: sources, class filter, single-turn, trained-row fingerprints, panel
text contact). That builder's own ordering then gives:
  VIEWS   = first k by SALT_SELECT                        (unsealed file; used as a positive control)
  CONFIRM = first k of the rest by SALT_CONFIRM           (sealed; RE-DERIVED here, the file is never opened)
This bank takes, from what remains, 48 rows per boundary class ordered by SALT_SOUP, plus EVERY remaining
arithmetic_short_answer row (the answerable half).

Gates, each with a positive control:
  - the re-derived VIEWS parents equal the parents in HSKILL_VIEWS_V2.jsonl;
  - 0 shared parents with VIEWS or CONFIRM (control: the check fires on an injected VIEWS parent);
  - 0 overlap with delta_H's 648 training rows by id, user text or (user, reply) pair (control: all 648
    training rows are flagged by the same check);
  - class counts are exact.
Writes HSKILL_SOUP_V3.jsonl and HSKILL_SOUP_V3_BUILD.json here. Create-only, LF. No model is run.
"""
from __future__ import annotations

import hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HV = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/hskill-views-002")
HTRAIN = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/compose001-20260918/H/H-prepared/microfit-train.jsonl")
sys.path.insert(0, str(HV))
import build_hskill_v2 as B  # noqa: E402  (module import only; its main() is never called)

SALT_SOUP = "CONFIRM-002-SOUP-mixed-boundary"
PER_BOUNDARY = 48
OUT = HERE / "HSKILL_SOUP_V3.jsonl"
REC = HERE / "HSKILL_SOUP_V3_BUILD.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pool():
    sys.path.insert(0, str(B.LANE))
    from experiments.reasoning_posttraining_v1.text_contact import audit_and_filter
    from experiments.reasoning_posttraining_v1 import process_mix
    panel = [json.loads(l) for l in process_mix.PANEL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    tids, tusers, tpairs = B.trained_fingerprints()
    wanted = set(B.CLASS_SOURCE)
    by_class = {c: [] for c in wanted}
    src_sha = {}
    for path in sorted(set(B.CLASS_SOURCE.values())):
        src_sha[path.name] = sha256_file(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rc = (row.get("metadata") or {}).get("reply_class")
            if rc in B.PANEL_REPLY_CLASSES or rc not in wanted or B.CLASS_SOURCE[rc] != path:
                continue
            if [m["role"] for m in row["messages"]] != ["system", "user", "assistant"]:
                continue
            user = next(m["content"] for m in row["messages"] if m["role"] == "user").strip()
            reply = next(m["content"] for m in row["messages"] if m["role"] == "assistant").strip()
            if row["id"] in tids or user in tusers or (user, reply) in tpairs:
                continue
            by_class[rc].append(row)
    for c in sorted(wanted):
        kept, _ = audit_and_filter(by_class[c], panel)
        by_class[c] = kept
    return by_class, (tids, tusers, tpairs), src_sha, process_mix.PANEL_PATH


def main() -> int:
    if OUT.exists() or REC.exists():
        raise SystemExit("create-only: HSKILL_SOUP_V3 already exists")
    by_class, (tids, tusers, tpairs), src_sha, panel_path = pool()

    views, confirm, chosen = set(), set(), []
    for c in sorted(by_class):
        k = B.PER_BOUNDARY_CLASS if c in B.BR.BOUNDARY_CLASSES else B.N_ANSWERABLE
        ordered = sorted(by_class[c], key=lambda r: (B.digest([B.SALT_SELECT, r["id"]]), r["id"]))
        views |= {r["id"] for r in ordered[:k]}
        rest = sorted(ordered[k:], key=lambda r: (B.digest([B.SALT_CONFIRM, r["id"]]), r["id"]))
        confirm |= {r["id"] for r in rest[:k]}
        remaining = sorted(rest[k:], key=lambda r: (B.digest([SALT_SOUP, r["id"]]), r["id"]))
        take = remaining[:PER_BOUNDARY] if c in B.BR.BOUNDARY_CLASSES else remaining
        if c in B.BR.BOUNDARY_CLASSES and len(take) != PER_BOUNDARY:
            raise SystemExit(f"{c}: only {len(take)} rows remain, need {PER_BOUNDARY}")
        chosen += take

    # positive control: re-derived VIEWS == the unsealed selection file
    views_file = {json.loads(l)["parent_id"] for l in (HV / "HSKILL_VIEWS_V2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    if views != views_file:
        raise SystemExit("POSITIVE CONTROL FAILED: re-derived VIEWS parents differ from the file")
    ids = {r["id"] for r in chosen}
    # shared-parent gate, with a control that must fire
    shared = ids & (views | confirm)
    if shared:
        raise SystemExit(f"GATE FAILED: {len(shared)} parents shared with VIEWS/CONFIRM")
    probe = next(iter(views))
    if not ({probe} & (views | confirm)):
        raise SystemExit("CONTROL FAILED: the shared-parent check cannot fire")
    # delta_H training overlap gate, with a control
    htrain = [json.loads(l) for l in HTRAIN.read_text(encoding="utf-8").splitlines() if l.strip()]
    h_ids = {r["parent_id"] for r in htrain}
    h_users = {[m["content"] for m in r["prompt"] if m["role"] == "user"][-1].strip() for r in htrain}
    ctl = sum(r["parent_id"] in tids for r in htrain)
    if ctl != len(htrain):
        raise SystemExit(f"CONTROL FAILED: only {ctl}/{len(htrain)} delta_H training rows are in the fingerprint set")
    users = {next(m["content"] for m in r["messages"] if m["role"] == "user").strip() for r in chosen}
    ov_id, ov_user = len(ids & h_ids), len(users & h_users)
    if ov_id or ov_user:
        raise SystemExit(f"GATE FAILED: overlap with delta_H training rows (ids {ov_id}, user texts {ov_user})")

    rows = []
    for r in chosen:
        rc = (r.get("metadata") or {})["reply_class"]
        rows.append({"family": rc, "row_type": "answerable" if rc in B.BR.ANSWERABLE_CLASSES else "boundary",
                     "gold_verified_by": "corpus_reference_reply", "id": f"HSKILL3S:{r['id']}",
                     "parent_id": r["id"], "source_id": r["id"],
                     "prompt": [m for m in r["messages"] if m["role"] in ("system", "user")],
                     "reference": next(m["content"] for m in r["messages"] if m["role"] == "assistant"),
                     "source_split": "train", "view_role": "CONFIRM-002-SOUP"})
    rows.sort(key=lambda x: (x["row_type"], x["family"], x["id"]))
    counts = {}
    for x in rows:
        counts[x["family"]] = counts.get(x["family"], 0) + 1
    payload = "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows).encode("utf-8")
    with open(OUT, "xb") as f:
        f.write(payload)
    rec = {"instrument": "HSKILL_SOUP_V3", "experiment": "CONFIRM-002 (soup lane)", "registered_in": "PREREG.md 779cd431",
           "salt": SALT_SOUP, "per_boundary_class": PER_BOUNDARY, "counts": counts, "n": len(rows),
           "n_boundary": sum(x["row_type"] == "boundary" for x in rows), "n_answerable": sum(x["row_type"] == "answerable" for x in rows),
           "sha256": hashlib.sha256(payload).hexdigest(), "source_sha256": src_sha, "panel_path": str(panel_path),
           "builder_module_sha256": sha256_file(HV / "build_hskill_v2.py"), "rule_module_sha256": sha256_file(HV / "boundary_rule.py"),
           "gates": {"views_rederivation_matches_file": True, "shared_parents_with_views_or_confirm": 0,
                     "delta_h_training_overlap": {"ids": 0, "user_texts": 0},
                     "controls": {"shared_parent_check_fires_on_injected_views_parent": True,
                                  "delta_h_training_rows_in_fingerprint_set": f"{ctl}/{len(htrain)}"}},
           "sealed_confirm_file_opened": False, "read_budget": "ONE"}
    REC.write_bytes((json.dumps(rec, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({k: rec[k] for k in ("n", "n_boundary", "n_answerable", "counts", "sha256", "gates")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
