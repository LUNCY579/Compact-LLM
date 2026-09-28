"""HLCC-001 -- a FRESH confirmation panel for linear compensation, disjoint from SMALLVD-001.

SMALLVD-001's HOLDOUT rows have been seen (the H-LC raw estimate is known), so confirming on them would
not be a confirmation. This builds new rows from the SAME sources and categories as LAYER_PANELS, 32 per
category (320), with:
  * no row id shared with LAYER_PANELS (either split) or with the tokenizer corpus (chat rows);
  * no v3 TEMPLATE FAMILY shared with any LAYER_PANELS or tokenizer-corpus v3 row, one row per family;
  * the same content-disjointness check against the eight evaluation instruments, with the same
    positive (exact path, sentence path) and negative controls, refusing to build if any control fails.
Text rows come from the tokenizer corpus's HELD-OUT split, from rows not used by any panel: they were
touched only by the tokenizer's held-out token-ratio measurement, never by layer selection or the
compensation fit. Written create-only; the manifest records the overlap evidence.
usage: python build_confirm_panel.py --teacher <dir> --smallvd <smallvd001 records dir> --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_calibration as bc  # noqa: E402
from build_calibration import hostbudget  # noqa: E402

PER_CATEGORY = 32


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, type=Path)
    ap.add_argument("--smallvd", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    hostbudget.require(bc.RAM_REQUIRE, "HLCC-001 confirm panel build")
    a.out.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer
    import transformers
    if transformers.__version__ != "4.51.3":
        raise SystemExit("render under the probes' pinned transformers 4.51.3")
    tok = AutoTokenizer.from_pretrained(str(a.teacher), local_files_only=True)
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

    layer = [json.loads(l) for l in (a.smallvd / "calibration" / "LAYER_PANELS.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    corpus = [json.loads(l) for l in (a.smallvd / "calibration" / "TOKENIZER_CORPUS.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    used_ids = {r["id"] for r in layer}
    corpus_ids = {r["id"] for r in corpus}
    panel_text_src = {r["id"][len("panel:"):] for r in layer if r["kind"] == "text"}

    all_classes = [c for v in bc.PANEL_CLASSES.values() for c in v]
    fam_of = {}
    rows_by_id = {}
    for c, r in bc.v3_rows(all_classes):
        fam_of[r["id"]] = r.get("metadata", {}).get("template_family") or r["id"]
        rows_by_id[r["id"]] = (c, r)
    used_families = {fam_of[i] for i in (used_ids | corpus_ids) if i in fam_of}
    fresh = bc.one_per_family((c, r) for i, (c, r) in rows_by_id.items()
                              if fam_of[i] not in used_families and i not in used_ids and i not in corpus_ids)
    by_id = {r["id"]: (c, r) for c, r in fresh}
    cls_group = {c: g for g, cs in bc.PANEL_CLASSES.items() for c in cs}

    panel, flags = [], {}

    def add(cat, rid, kind, src, prompt=None, reference=None, scorer=None, text=None, user=None):
        if bc.flagged(user, (prompt or "") + (reference or "") + (text or ""), idx):
            flags.setdefault(cat, []).append(rid)
            return False
        panel.append({"id": rid, "category": cat, "kind": kind, "source": src, "split": "confirm",
                      "prompt_text": prompt, "reference": reference, "scorer": scorer, "text": text})
        return True

    for g in bc.PANEL_CLASSES:
        k = 0
        for rid in bc.ordered([i for i in by_id if cls_group[by_id[i][0]] == g], f"hlcc:panel:{g}"):
            c, r = by_id[rid]
            msgs = r["messages"]
            k += add(g, rid, "chat", f"v3:{c}", bc.render(tok, msgs[:-1], True), msgs[-1]["content"],
                     "exact" if g in ("concise_reasoning", "grounded_extraction") else None, user=msgs[-2]["content"])
            if k >= PER_CATEGORY:
                break
    with open(bc.RTG1 / "dev.jsonl", encoding="utf-8") as f:
        dev = {x["id"]: x for x in map(json.loads, f)}
    order = [i for i in bc.ordered(dev, "hlcc-rtg") if i not in used_ids and i not in corpus_ids]
    for want_unknown in (True, False):
        k = 0
        for rid in order:
            x = dev[rid]
            if (x["completion"] == "UNKNOWN") != want_unknown:
                continue
            k += add("abstention_unknown", rid, "chat", "rtg1:dev", bc.render(tok, x["messages"], True),
                     x["completion"], "exact", user=x["messages"][-1]["content"])
            if k >= PER_CATEGORY // 2:
                break
    lc = {}
    for p in sorted(glob.glob(str(bc.CC0 / "validation" / "*.jsonl"))):
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if len(r["messages"]) == 3 and len(r["messages"][1]["content"]) <= bc.LC_MAX_CHARS:
                    lc[r["id"]] = r
    k = 0
    for rid in bc.ordered([i for i in lc if i not in used_ids and i not in corpus_ids], "hlcc-lc"):
        r = lc[rid]
        k += add("long_context", rid, "chat", "cc0:validation", bc.render(tok, r["messages"][:-1], True),
                 r["messages"][-1]["content"], "exact", user=r["messages"][1]["content"])
        if k >= PER_CATEGORY:
            break
    text_src = {"english_prose": lambda r: r["source"] == "storyforge", "python": lambda r: r["category"] == "python",
                "javascript": lambda r: r["category"] == "javascript", "markdown": lambda r: r["category"] == "markdown"}
    for cat, pred in text_src.items():
        pool = {r["id"]: r for r in corpus if r["split"] == "heldout" and pred(r) and r["id"] not in panel_text_src}
        k = 0
        for rid in bc.ordered(pool, f"hlcc-text:{cat}"):
            k += add("text_" + cat, "confirm:" + rid, "text", pool[rid]["source"], text=pool[rid]["text"])
            if k >= PER_CATEGORY:
                break

    counts = {}
    for p in panel:
        counts[p["category"]] = counts.get(p["category"], 0) + 1
    short = {c: n for c, n in counts.items() if n < PER_CATEGORY}
    if short or len(counts) != 10:
        raise SystemExit(f"confirm panel short: {counts}")
    # overlap evidence, recomputed from the written rows
    ids = {p["id"] for p in panel}
    chat_ids = {p["id"] for p in panel if p["kind"] == "chat"}
    ev = {"id_overlap_with_layer_panels": len(ids & used_ids),
          "chat_id_overlap_with_tokenizer_corpus": len(chat_ids & corpus_ids),
          "text_rows_from_panel_sources": len({p["id"][len("confirm:"):] for p in panel if p["kind"] == "text"} & panel_text_src),
          "v3_family_overlap_with_used": len({fam_of[i] for i in chat_ids if i in fam_of} & used_families),
          "v3_rows_one_per_family": len({fam_of[i] for i in chat_ids if i in fam_of}) == sum(1 for i in chat_ids if i in fam_of)}
    if any(v for k, v in ev.items() if k != "v3_rows_one_per_family") or not ev["v3_rows_one_per_family"]:
        raise SystemExit(f"DISJOINTNESS FAILED: {ev}")
    with open(a.out / "CONFIRM_PANEL.jsonl", "x", encoding="utf-8", newline="\n") as f:
        for r in panel:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = {"schema": 1, "record": "HLCC_CONFIRM_PANEL", "experiment": "HLCC-001",
           "builder_sha256": bc.sha256_file(Path(__file__)), "calibration_builder_sha256": bc.sha256_file(HERE / "build_calibration.py"),
           "smallvd_layer_panels_sha256": bc.sha256_file(a.smallvd / "calibration" / "LAYER_PANELS.jsonl"),
           "smallvd_tokenizer_corpus_sha256": bc.sha256_file(a.smallvd / "calibration" / "TOKENIZER_CORPUS.jsonl"),
           "renderer": {"transformers": transformers.__version__, "enable_thinking": False},
           "per_category": counts, "rows": len(panel), "disjointness_evidence": ev,
           "instrument_controls": {"exact_path": f"{sum(ctrl)}/{len(ctrl)}", "sentence_path": f"{sum(sent_ctrl)}/{len(sent_ctrl)}",
                                   "negative_fresh_text_flagged": False},
           "instrument_flagged_and_dropped": {k: len(v) for k, v in flags.items()},
           "panel_sha256": bc.sha256_file(a.out / "CONFIRM_PANEL.jsonl"),
           "quality_claim": False, "release_permitted": False}
    with open(a.out / "CONFIRM_MANIFEST.json", "x", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
    print(json.dumps({"per_category": counts, "disjointness": ev, "flagged": man["instrument_flagged_and_dropped"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
