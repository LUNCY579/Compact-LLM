"""HLCC-001 R2 -- pin the ALL-AXIS unit map for the registered sensitivity analysis.

The provenance recovery is the cold reviewer's (rineya-dataset-retention-records/hlcc001-20260925/
HLCC_PREREG_COLD_REVIEW_sim.py, records commit cdbf844), reused here so the map is built by committed
lane code and pinned by hash BEFORE the run. Axes: prose book and author, code package, markdown
document, long-context generator task and source packet (+ shared anchor cards), RTG1 generator
family, v3 generator class. Units = connected components of rows sharing ANY axis value.

CANARY: the component counts must equal the review's (KL 127 / bpb 42 over the eight axes; 121 / 36
with anchor cards), or the builder refuses -- a map that disagrees with the review is not the map the
review measured. Every row must receive its category's axis (a row with no provenance is an error).
CPU only; reads panel METADATA and source files; loads no model. Create-only output.
usage: python design/hlcc_allaxis_units.py --panel <CONFIRM_PANEL.jsonl> --out <HLCC_ALLAXIS_UNIT_MAP.json>
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import build_calibration as bc  # noqa: E402

BP = "C:/Users/clclc/BusinessProject"
AUTHORS = ["Arthur_Conan_Doyle", "Charles_Dickens", "Fyodor_Dostoevsky", "George_Bernard_Shaw", "H._G._Wells",
           "Henrik_Ibsen", "Jane_Austen", "Jules_Verne", "Mary_Shelley", "O._Henry", "Shakespeare"]
AXES = ["book", "author", "pkg", "doc", "task", "packet", "rtg_family", "v3_class"]
EXPECT = {"kl": 127, "bpb": 42, "kl_anchor": 121, "bpb_anchor": 36}      # the cold review's counts


def provenance(panel):
    rtg = {}
    with open(bc.RTG1 / "dev.jsonl", encoding="utf-8") as f:
        for x in map(json.loads, f):
            rtg[x["id"]] = x["family"]
    lcmeta = {}
    for p in glob.glob(str(bc.CC0 / "validation" / "*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for x in map(json.loads, f):
                sp = x["metadata"]["source_provenance"]
                lcmeta[x["id"]] = (re.search(r"source_family_id=([^;]+)", sp).group(1),
                                   sorted(re.search(r"anchor_card_ids=([^;]+)", sp).group(1).split(",")))
    paths = {}
    for root in [bc.SITE / d for d in ("requests", "urllib3", "jinja2", "click", "rich", "httpx", "packaging",
                                       "pydantic", "yaml", "networkx")] + [bc.STDLIB] + [bc.NPM / "npm" / "lib", bc.NPM / "@nestjs"]:
        for p in bc.walk(root):
            paths[bc.h(p)[:16]] = p.replace("\\", "/")

    def pypkg(p):
        if "/site-packages/" in p:
            return p.split("/site-packages/")[1].split("/")[0]
        t = p.split("/Lib/")[1].split("/")
        return "stdlib:" + (t[0] if len(t) > 1 else "<top>")

    def jspkg(p):
        rel = p.split("node_modules/")[-1].split("/")
        return "/".join(rel[:2]) if rel[0].startswith("@") else rel[0]

    out = []
    for r in panel:
        ax, rid = {}, r["id"]
        if rid.startswith("confirm:story:"):
            b = rid.split(":")[2]
            ax["book"], ax["author"] = b, next(a for a in AUTHORS if b.startswith(a))
        elif rid.startswith("confirm:py:"):
            ax["pkg"] = "py:" + pypkg(paths[rid.split(":")[2]])
        elif rid.startswith("confirm:js:"):
            ax["pkg"] = "js:" + jspkg(paths[rid.split(":")[2]])
        elif rid.startswith("confirm:md:"):
            ax["doc"] = rid.split(":")[2]
        elif r["source"] == "cc0:validation":
            ax["task"] = rid.split("__")[1]
            ax["packet"], ax["_anchors"] = lcmeta[rid]
        elif r["source"] == "rtg1:dev":
            ax["rtg_family"] = rtg[rid]
        elif r["source"].startswith("v3:"):
            ax["v3_class"] = r["source"]
        if not ax:
            raise SystemExit(f"no provenance axis for {rid}")
        out.append({"id": rid, "bpb": r["kind"] == "chat" and bool(r.get("reference")), "ax": ax})
    return out


def components(rs, anchors: bool) -> dict[str, str]:
    """row id -> component label (the smallest row id in its component; deterministic)."""
    parent = list(range(len(rs)))

    def f(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(i, j):
        a, b = f(i), f(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    keys = AXES + (["_anchors"] if anchors else [])
    for k in keys:
        first = {}
        for i, r in enumerate(rs):
            vals = r["ax"].get(k)
            if vals is None:
                continue
            for v in (vals if k == "_anchors" else [vals]):
                if v in first:
                    join(i, first[v])
                else:
                    first[v] = i
    members: dict[int, list[str]] = {}
    for i, r in enumerate(rs):
        members.setdefault(f(i), []).append(r["id"])
    label = {root: min(ids) for root, ids in members.items()}
    return {r["id"]: "comp:" + label[f(i)] for i, r in enumerate(rs)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists (create-only)")
    panel = [json.loads(l) for l in a.panel.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = provenance(panel)
    bpb = [r for r in rows if r["bpb"]]
    maps = {"kl": components(rows, False), "bpb": components(bpb, False),
            "kl_anchor": components(rows, True), "bpb_anchor": components(bpb, True)}
    counts = {k: len(set(v.values())) for k, v in maps.items()}
    if counts != EXPECT:
        raise SystemExit(f"CANARY FAILED: component counts {counts} != the cold review's {EXPECT}")
    rec = {"schema": 1, "record": "HLCC_ALLAXIS_UNIT_MAP", "experiment": "HLCC-001", "revision": "R2",
           "panel_sha256": hashlib.sha256(a.panel.read_bytes()).hexdigest(),
           "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "axes": AXES, "registered_sensitivity_maps": ["kl_anchor", "bpb_anchor"],
           "counts": counts, "canary": {"expected_from_cold_review": EXPECT, "status": "reproduced"},
           "axis_values": {r["id"]: {k: v for k, v in r["ax"].items()} for r in rows},
           "maps": maps}
    with open(a.out, "x", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False, sort_keys=True)
    print(json.dumps({"counts": counts, "canary": "reproduced"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
