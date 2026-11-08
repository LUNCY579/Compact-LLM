"""ABST-001 -- the fresh RTG1-dev panel (ABST001_PREREG.md §3), CPU only, built BEFORE registration.

Pool: RTG1 dev (the programme's accepted reasoning corpus; its holdout split is RESERVED and never read,
its train split is training data). Eligible rows:
  * id not in LAYER_PANELS (either split), CONFIRM_PANEL (HLCC-001, spent) or TOKENIZER_CORPUS;
  * id not in the designer-seen list (rows whose content appeared in the design agents' transcripts);
  * normalised last user turn not equal to that of any used row or any RTG1 TRAIN row
    (ID-disjoint is not content-disjoint) -- positive control: an injected used row must be flagged;
  * not flagged by the content check against the SIX non-RCI evaluation instruments (RCI-001/002 are
    spent and never opened; HSKILL-CONFIRM-002 is sealed and never opened), with HLCC's exact-path,
    sentence-path and negative controls;
  * reference 'UNKNOWN' (UNKNOWN stratum) or a digit 0-3 (known stratum).
Sampling: per family (sorted), bc.ordered(ids, 'abst001:<stratum>:<family>') -- a content-independent
hash order -- first 64: a stratified simple random sample without replacement from fixed pools. The
unsampled eligible UNKNOWN rows are listed as a SEALED RESERVE (never measured by ABST-001).
The output directory is created only after every check passes (HLCC's early-mkdir flaw not repeated).
usage: python build_abst_panel.py --teacher <dir> --smallvd <smallvd001 dir> --hlcc <hlcc001 dir>
       --exclude-ids <json list> --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_calibration as bc  # noqa: E402
from build_calibration import hostbudget  # noqa: E402

PER_STRATUM_FAMILY = 64
FAMILIES = ("counterfactual_state", "evidence_chain", "prerequisite_schedule", "register_machine")
RCI = ("rci001_calibration", "rci002_calibration")


def user_turn(msgs):
    u = [m["content"] for m in msgs if m["role"] == "user"]
    return u[-1] if u else ""


def uhash(s: str) -> str:
    return bc.h(bc.norm(s))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, type=Path)
    ap.add_argument("--smallvd", required=True, type=Path)
    ap.add_argument("--hlcc", required=True, type=Path)
    ap.add_argument("--exclude-ids", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists (create-only)")
    hostbudget.require(bc.RAM_REQUIRE, "ABST-001 panel build")
    from transformers import AutoTokenizer
    import transformers
    if transformers.__version__ != "4.51.3":
        raise SystemExit("render under the probes' pinned transformers 4.51.3")
    tok = AutoTokenizer.from_pretrained(str(a.teacher), local_files_only=True)

    # the six non-RCI instruments only: RCI files are spent and are never opened here
    six = {k: v for k, v in bc.INSTRUMENT_FILES.items() if k not in RCI}
    if len(six) != 6:
        raise SystemExit(f"expected 6 non-RCI instruments, got {sorted(six)}")
    bc.INSTRUMENT_FILES = six
    idx = bc.instrument_index()
    ctrl = [bc.flagged(c["last_user"], c["last_user"], idx) for c in idx["controls"]]
    if not ctrl or not all(ctrl):
        raise SystemExit(f"POSITIVE CONTROL FAILED: {sum(ctrl)}/{len(ctrl)}")
    sent_ctrl = [bc.flagged(None, "An unrelated preface sentence. " + c["last_user"] + " A closing remark.",
                            {"users": set(), "sentences": idx["sentences"]})
                 for c in idx["controls"] if bc.sentences(c["last_user"])]
    if not sent_ctrl or not all(sent_ctrl):
        raise SystemExit(f"SENTENCE-PATH CONTROL FAILED: {sum(sent_ctrl)}/{len(sent_ctrl)}")
    if bc.flagged("A brand-new question about lighthouses and tides.",
                  "A brand-new question about lighthouses and tides that no instrument has ever asked anyone.", idx):
        raise SystemExit("NEGATIVE CONTROL FAILED")

    def jl(p):
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    layer = jl(a.smallvd / "calibration" / "LAYER_PANELS.jsonl")
    corpus = jl(a.smallvd / "calibration" / "TOKENIZER_CORPUS.jsonl")
    confirm = jl(a.hlcc / "panel" / "CONFIRM_PANEL.jsonl")
    used = {r["id"] for r in layer} | {r["id"] for r in corpus} | {r["id"] for r in confirm}
    seen = set(json.loads(a.exclude_ids.read_text(encoding="utf-8")))
    dev = jl(bc.RTG1 / "dev.jsonl")
    train = jl(bc.RTG1 / "train.jsonl")
    by_id = {x["id"]: x for x in dev}
    used_users = {uhash(user_turn(by_id[i]["messages"])) for i in used if i in by_id}
    train_users = {uhash(user_turn(x["messages"])) for x in train}
    banned_users = used_users | train_users
    # positive control for the content-duplicate check: a used RTG1 row must be caught by it
    used_rtg = sorted(i for i in used if i in by_id)
    if not used_rtg or uhash(user_turn(by_id[used_rtg[0]]["messages"])) not in banned_users:
        raise SystemExit("CONTENT-DUP POSITIVE CONTROL FAILED")

    excluded = {"used_id": 0, "designer_seen": 0, "content_duplicate": 0, "instrument_flagged": 0, "other_reference": 0}
    eligible = {("unk", f): [] for f in FAMILIES} | {("known", f): [] for f in FAMILIES}
    for x in dev:
        if x["id"] in used:
            excluded["used_id"] += 1
            continue
        if x["id"] in seen:
            excluded["designer_seen"] += 1
            continue
        if uhash(user_turn(x["messages"])) in banned_users:
            excluded["content_duplicate"] += 1
            continue
        if bc.flagged(user_turn(x["messages"]), user_turn(x["messages"]) + x["completion"], idx):
            excluded["instrument_flagged"] += 1
            continue
        c = x["completion"].strip()
        st = "unk" if c == "UNKNOWN" else "known" if c in ("0", "1", "2", "3") else None
        if st is None or x["family"] not in FAMILIES:
            excluded["other_reference"] += 1
            continue
        eligible[(st, x["family"])].append(x["id"])
    panel, reserve, counts = [], {}, {}
    for (st, f), ids in sorted(eligible.items()):
        order = bc.ordered(ids, f"abst001:{st}:{f}")
        if len(order) < PER_STRATUM_FAMILY:
            raise SystemExit(f"stratum {st}/{f} has only {len(order)} eligible rows")
        take, rest = order[:PER_STRATUM_FAMILY], order[PER_STRATUM_FAMILY:]
        counts[f"{st}/{f}"] = {"eligible": len(ids), "sampled": len(take), "reserve": len(rest)}
        reserve[f"{st}/{f}"] = rest
        for rid in take:
            x = by_id[rid]
            panel.append({"id": rid, "stratum": st, "family": f, "tier": x.get("tier"), "split": "abst",
                          "kind": "chat", "source": "rtg1:dev", "reference": x["completion"].strip(),
                          "prompt_text": bc.render(tok, x["messages"], True)})
    # evidence recomputed from the rows to be written
    ids = {p["id"] for p in panel}
    ev = {"id_overlap_with_used": len(ids & used), "designer_seen_in_panel": len(ids & seen),
          "reserve_overlap_with_panel": len(ids & {i for v in reserve.values() for i in v}),
          "content_duplicates_in_panel": sum(uhash(user_turn(by_id[i]["messages"])) in banned_users for i in ids),
          "duplicate_ids": len(panel) - len(ids)}
    if any(ev.values()) or len(panel) != 2 * len(FAMILIES) * PER_STRATUM_FAMILY:
        raise SystemExit(f"PANEL CHECK FAILED: {ev}, rows {len(panel)}")
    a.out.mkdir(parents=True, exist_ok=False)
    with open(a.out / "ABST_PANEL.jsonl", "x", encoding="utf-8", newline="\n") as f:
        for r in panel:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    man = {"schema": 1, "record": "ABST_PANEL_MANIFEST", "experiment": "ABST-001",
           "builder_sha256": sha(Path(__file__)), "calibration_builder_sha256": sha(HERE / "build_calibration.py"),
           "rtg1_dev_sha256": sha(bc.RTG1 / "dev.jsonl"), "rtg1_train_sha256": sha(bc.RTG1 / "train.jsonl"),
           "layer_panels_sha256": sha(a.smallvd / "calibration" / "LAYER_PANELS.jsonl"),
           "tokenizer_corpus_sha256": sha(a.smallvd / "calibration" / "TOKENIZER_CORPUS.jsonl"),
           "confirm_panel_sha256": sha(a.hlcc / "panel" / "CONFIRM_PANEL.jsonl"),
           "designer_seen_ids": sorted(seen), "renderer": {"transformers": transformers.__version__, "enable_thinking": False},
           "instruments_used": sorted(six), "instrument_controls": {"exact_path": f"{sum(ctrl)}/{len(ctrl)}",
                                                                   "sentence_path": f"{sum(sent_ctrl)}/{len(sent_ctrl)}",
                                                                   "negative_fresh_text_flagged": False},
           "content_dup_positive_control": "passed", "excluded": excluded, "strata": counts,
           "disjointness_evidence": ev, "sealed_reserve_ids": reserve,
           "rows": len(panel), "panel_sha256": sha(a.out / "ABST_PANEL.jsonl"),
           "quality_claim": False, "release_permitted": False}
    with open(a.out / "ABST_MANIFEST.json", "x", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
    print(json.dumps({"strata": counts, "excluded": excluded, "evidence": ev,
                      "controls": man["instrument_controls"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
