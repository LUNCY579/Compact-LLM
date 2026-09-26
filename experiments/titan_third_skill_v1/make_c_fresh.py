"""TRISKILL-001 Stage B: build C_FRESH, a fresh atomic-C draw from SIF-001's own generator (PREREG §5).

112 parent worlds x (full, counterfactual, distractor) = 336 JSON rows. It uses SIF-001's make_world,
counterfactual, row_for, render and solve unchanged; only the seed tuple is new. A world is rejected
if any of its programs collides with ANY SIF-001 split (train, dev, cal, check_iid, check_ood) on:
  K1 world_signature (SIF's own content key: stage-name- and table-order-invariant, includes the seed);
  K2 the rules table + default alone, ignoring the seed (stricter: no reused table under a new seed);
  K3 the rendered prompt's sha256.
POSITIVE CONTROLS: K1 and K2 must flag all 128 SIF DEV atomic_C rows; K3 must flag all 128 DEV prompts.
Floors (majority answer, echo-seed) are computed from the keys and published in the build record.
Writes C_FRESH.jsonl (model-facing prompts + keys) and C_FRESH_BUILD.json. Create-only, LF. No model.
"""
from __future__ import annotations

import collections, gzip, hashlib, json, platform, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIF = Path(r"C:/Users/clclc/BusinessProject/Rineya-Dataset/Rineya_Skill_Integration_Foundry_v1")
sys.path.insert(0, str(SIF / "src"))
from sif.core import SYSTEM, canonical, digest, rng_for, make_world, solve, counterfactual  # noqa: E402
from sif.build import row_for, world_signature  # noqa: E402

SEED = ("TRISKILL-001-C-FRESH", 20260927)
N_WORLDS = 112
OUT = HERE / "C_FRESH.jsonl"
REC = HERE / "C_FRESH_BUILD.json"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rules_key(program: dict) -> str:
    """K2: the C node's rules table and default, ignoring the seed and stage names."""
    nodes = [n for n in program["nodes"] if n["skill"] == "C"]
    return digest([{"default": n["data"]["default"], "rules": sorted(n["data"]["rules"], key=canonical)} for n in nodes])


def sif_keys():
    k1, k2, k3, dev_c = set(), set(), set(), []
    for gz in sorted((SIF / "private_oracles").rglob("*.keys.jsonl.gz")):
        for line in gzip.open(gz, "rt", encoding="utf-8"):
            r = json.loads(line)
            prog = r.get("program")
            if prog is not None:
                k1.add(world_signature(prog))
                if any(n["skill"] == "C" for n in prog["nodes"]):
                    k2.add(rules_key(prog))
            k3.add(r["prompt_sha256"])
            if r["split"] == "dev" and r["bank"] == "atomic_C":
                dev_c.append(r)
    return k1, k2, k3, dev_c


def main() -> int:
    if OUT.exists() or REC.exists():
        raise SystemExit("create-only: C_FRESH already exists")
    k1, k2, k3, dev_c = sif_keys()
    # positive controls: each key must flag every SIF DEV atomic_C row
    if len(dev_c) != 128:
        raise SystemExit(f"expected 128 DEV atomic_C keys, got {len(dev_c)}")
    ctl = {"K1": sum(world_signature(r["program"]) in k1 for r in dev_c),
           "K2": sum(rules_key(r["program"]) in k2 for r in dev_c),
           "K3": sum(r["prompt_sha256"] in k3 for r in dev_c)}
    if any(v != 128 for v in ctl.values()):
        raise SystemExit(f"POSITIVE CONTROL FAILED: {ctl}")

    rows, rejected, seen1, seen2 = [], collections.Counter(), set(), set()
    i = 0
    while len(rows) < 3 * N_WORLDS:
        if i > 50 * N_WORLDS:
            raise SystemExit("could not find enough disjoint worlds")
        parts = SEED + (i,)
        p = make_world(("C",), parts, 7, 2, "chain")
        style = rng_for("render_style", *parts).randrange(6)
        parent = "cfresh_" + digest(parts)[:24]
        base = row_for(p, "triskill_c_fresh", "atomic_C", parent, "full", ("C",), style)
        cf, kind = counterfactual(p, i)
        if solve(cf)[0] == base["answer"]:
            raise SystemExit("counterfactual did not change the answer")
        cr = row_for(cf, "triskill_c_fresh", "atomic_C", parent, "counterfactual", ("C",), style)
        cr["intervention"] = kind; cr["paired_with"] = base["id"]
        inv = row_for(p, "triskill_c_fresh", "atomic_C", parent, "distractor", ("C",), style, irrelevant=5)
        inv["paired_with"] = base["id"]
        fam = [base, cr, inv]
        i += 1
        why = None
        for r in fam:
            s1, s2 = world_signature(r["program"]), rules_key(r["program"])
            ph = hashlib.sha256(r["prompt"].encode()).hexdigest()
            r["prompt_sha256"] = ph
            if s1 in k1: why = "K1_world_signature_in_SIF"
            elif s2 in k2: why = "K2_rules_table_in_SIF"
            elif ph in k3: why = "K3_prompt_in_SIF"
            elif s2 in seen2 and r["variant"] == "full": why = "duplicate_rules_table_in_bank"
            if why: break
        if why:
            rejected[why] += 1
            continue
        for r in fam:
            seen1.add(world_signature(r["program"])); seen2.add(rules_key(r["program"]))
        rows.extend(fam)

    out = []
    for r in rows:
        out.append({"id": r["id"], "parent_id": r["parent_id"], "bank": "atomic_C", "variant": r["variant"],
                    "response_format": "json", "required_skills": r["required_skills"],
                    "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["prompt"]}],
                    "prompt": r["prompt"], "prompt_sha256": r["prompt_sha256"],
                    "answer": r["answer"], "completion": r["completion"],
                    "world_sha256": world_signature(r["program"]), "paired_with": r.get("paired_with")})
    ans = collections.Counter(json.dumps(r["answer"]) for r in out)
    maj, maj_n = ans.most_common(1)[0]
    echo = sum(1 for r in out if (m := re.search(r"seed=(\d+)", r["prompt"])) and r["answer"] == int(m.group(1)))
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    rec = {"instrument": "C_FRESH", "experiment": "TRISKILL-001", "registered_in": "PREREG.md (0d2b64b4) section 5",
           "generator": "SIF-001 src/sif (make_world, counterfactual, row_for), seed tuple " + json.dumps(SEED),
           "sif_source_sha256": {n: sha256_file(SIF / "src/sif" / n) for n in ("core.py", "build.py", "scoring.py")},
           "python": platform.python_version(),
           "n": len(out), "parents": len({r["parent_id"] for r in out}),
           "variants": dict(collections.Counter(r["variant"] for r in out)),
           "answer_distribution": dict(ans), "floor": {"majority_answer": maj, "majority_rows": maj_n,
                                                        "echo_seed_rows": echo, "floor_rows": max(maj_n, echo)},
           "exclusion": {"sif_world_signatures": len(k1), "sif_rules_tables": len(k2), "sif_prompt_hashes": len(k3),
                         "rejected_families": dict(rejected), "candidate_worlds_drawn": i},
           "positive_controls": {k: f"{v}/128" for k, v in ctl.items()},
           "collisions_after_generation": {"K1": sum(r["world_sha256"] in k1 for r in out),
                                           "K3": sum(r["prompt_sha256"] in k3 for r in out)},
           "sha256": sha256_file(OUT), "read_budget": "ONE", "quality_claim": False}
    if rec["collisions_after_generation"] != {"K1": 0, "K3": 0}:
        OUT.unlink(); raise SystemExit(f"GATE FAILED: {rec['collisions_after_generation']}")
    REC.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: rec[k] for k in ("n", "parents", "variants", "floor", "exclusion", "positive_controls", "sha256")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
