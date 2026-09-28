"""COMP-RS-001 Amendment 1 section 11 data gates, computed on specs only.

python -m experiments.maxrl_comp002_pretraining_v1.data_gates --data DATA --out NEW.json [--roles dev calibration]

Floors, shortcut rates, content keys and triple structure are computations on specs, not reads (section 1).
No model output is opened. Endpoints use tag == 'core' rows only. Role 'test' is admitted only with
--at-freeze, and then only the two gates section 11 recomputes on TEST specs are computed: the lookup-blind
floor and the triple structure. TEST oracles are never opened. Challenge banks are never admitted.
Only the exact data paths this module builds are read, each checked against manifest['files'] first.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import random

from experiments.maxrl_composable_skills_v2 import contracts, corpus, verifier
from experiments.maxrl_composable_skills_v2.contracts import canonical, strict_json
from experiments.maxrl_composable_skills_v2.corpus import reward
from experiments.maxrl_composable_skills_v2.verifier import concrete, score
from experiments.maxrl_comp002_pretraining_v1 import preflight
from experiments.maxrl_comp002_pretraining_v1.preflight import MANIFEST, digest

HELD_OUT_ROLES = ("dev", "calibration")
FREEZE_ROLES = ("test",)
SEALED_ROLES = ("test", "challenge", "guard_test", "guard_challenge")
ATOMIC_S_MANIFEST = "manifests/atomic_S.json"
SHORTCUT_DEFINITION = ("Return first k indices in numeric order, or all indices without selection; "
                       "and always UNKNOWN. No source model.")
UNKNOWN_RESPONSE = '{"answer":"UNKNOWN"}'
OVERLAP_SIGNATURES = ("S", "RS")
OVERLAP_CONTROL_SEED = 20260926
MAX_LINE = 131072
FREEZE_GATES = ("rs_floor", "triple_structure")  # section 11: the gates recomputed on TEST specs at freeze
SEALED_FILES = frozenset(r + ".jsonl" for r in SEALED_ROLES)
# The only data paths read: exact manifest['files'] keys, built by role_file (canonical/test.jsonl at freeze only).
READABLE = frozenset({ATOMIC_S_MANIFEST, "canonical/train.jsonl"}
                     | {d + "/" + r + ".jsonl" for d in ("canonical", "oracles") for r in HELD_OUT_ROLES})
FREEZE_READABLE = frozenset({"canonical/test.jsonl"})
# Every source file the gate values depend on: hashed at import, re-checked before the receipt is written.
CODE_FILES = {"data_gates.py": Path(__file__).resolve(),
              "preflight.py": Path(preflight.__file__).resolve(),
              "maxrl_composable_skills_v2/verifier.py": Path(verifier.__file__).resolve(),
              "maxrl_composable_skills_v2/corpus.py": Path(corpus.__file__).resolve(),
              "maxrl_composable_skills_v2/contracts.py": Path(contracts.__file__).resolve()}


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def code_sha256() -> dict:
    """sha256 of every file in CODE_FILES, keyed by name."""
    return {name: sha256_bytes(path.read_bytes()) for name, path in CODE_FILES.items()}


_IMPORT_CODE = code_sha256()


def signature(row: dict) -> str:
    return "".join(row["skills"])


def core_rows(rows, signatures=None) -> list[dict]:
    """Composition rows with tag == 'core', optionally restricted to the named signatures."""
    out = [r for r in rows if r.get("kind") == "composition" and r.get("tag") == "core"]
    if signatures is not None:
        out = [r for r in out if signature(r) in signatures]
    return out


def row_spec(row: dict) -> dict:
    verifier = row["verifier"]
    if verifier.get("kind") != "composition":
        raise ValueError("not a composition row: " + str(row.get("id")))
    return verifier["spec"]


def resolved_set(spec: dict) -> set[int]:
    candidates, _ = concrete(spec)
    if candidates is None:
        raise ValueError("unresolvable lookup")
    return candidates


def blind_subsets(spec: dict) -> list[tuple[int, ...]]:
    """Every k-subset of ALL indices whose initial values sum to target, in lexicographic order."""
    if "updates" in spec or "selection" not in spec:
        raise ValueError("blind subsets need a selection over initial values")
    values = spec["initial"]
    k, target = spec["selection"]["k"], spec["selection"]["target"]
    return [c for c in combinations(range(len(values)), k) if sum(values[i] for i in c) == target]


def _mean(values: list[float]):
    return math.fsum(values) / len(values) if values else None


def rs_floor(rows) -> dict:
    """Lookup-blind floor on core RS rows: p_blind = share of blind sum-valid k-subsets inside the resolved set."""
    per_row = []
    for row in core_rows(rows, ("RS",)):
        spec = row_spec(row)
        inside_set = resolved_set(spec)
        blind = blind_subsets(spec)
        if not blind:
            raise ValueError("RS row without a sum-valid subset: " + row["id"])
        inside = sum(set(c) <= inside_set for c in blind)
        first = bool(score(spec, canonical({"answer": list(blind[0])})))
        if first != (set(blind[0]) <= inside_set):
            raise ValueError("verifier disagrees with resolved-set inclusion: " + row["id"])
        per_row.append({"id": row["id"], "parent_id": row["parent_id"], "k": spec["selection"]["k"],
                        "blind_subsets": len(blind), "inside": inside, "p_blind": inside / len(blind),
                        "first_lex_passes": first})
    per_row.sort(key=lambda r: r["id"])
    by_k = {}
    for k in sorted({r["k"] for r in per_row}):
        chosen = [r for r in per_row if r["k"] == k]
        by_k[str(k)] = {"rows": len(chosen), "mean_p_blind": _mean([r["p_blind"] for r in chosen]),
                        "first_lex_pass": sum(r["first_lex_passes"] for r in chosen)}
    return {"rows": len(per_row), "status": "COMPUTED" if per_row else "NO_ROWS",
            "mean_p_blind": _mean([r["p_blind"] for r in per_row]), "by_k": by_k,
            "first_lex_pass": sum(r["first_lex_passes"] for r in per_row) if per_row else None, "per_row": per_row,
            "definition": "p_blind = #(k-subsets of all indices summing to target inside the resolved set) / "
                          "#(k-subsets of all indices summing to target); first_lex = the lexicographically first such subset"}


def _resolve(links: dict, alias: str) -> int:
    seen = set()
    node = alias
    while type(node) is str:
        if node in seen or node not in links:
            raise ValueError("unresolvable alias")
        seen.add(node)
        node = links[node]
    return node


def chi_square_uniform(counts) -> float | None:
    """Pearson chi-square of `counts` against equal expected counts (df = len(counts) - 1); None when empty."""
    total = sum(counts)
    if total == 0:
        return None
    expected = total / len(counts)
    return math.fsum((c - expected) ** 2 for c in counts) / expected


def triple_structure(rows) -> dict:
    """Every value class of 'initial' is a triple with exactly two resolved members (core R and RS rows).

    Status is PASS/FAIL on that check alone. 'No ordering cue' (section 11) rests on code facts, not on a
    spec-level test (Clarification C9): canonical() sorts keys and the generator shuffles 'requested'. The
    ordering statistics below, including the chi-square of the excluded member's rank, are descriptive only.
    """
    by_signature = defaultdict(lambda: {"rows": 0, "pass": 0})
    failures = []
    ranks = [0, 0, 0]
    requested_in_order = 0
    for row in core_rows(rows, ("R", "RS")):
        spec = row_spec(row)
        inside = resolved_set(spec)
        classes = defaultdict(list)
        for i, value in enumerate(spec["initial"]):
            classes[value].append(i)
        ok = all(len(members) == 3 and len(inside.intersection(members)) == 2 for members in classes.values())
        sig = signature(row)
        by_signature[sig]["rows"] += 1
        by_signature[sig]["pass"] += int(ok)
        if not ok:
            failures.append(row["id"])
            continue
        for members in classes.values():
            excluded = next(i for i in members if i not in inside)
            ranks[sorted(members).index(excluded)] += 1
        order = [_resolve(spec["lookup"]["links"], a) for a in spec["lookup"]["requested"]]
        requested_in_order += int(order == sorted(order))
    total = sum(v["rows"] for v in by_signature.values())
    status = "NO_ROWS" if total == 0 else "FAIL" if failures else "PASS"
    return {"rows": total, "status": status, "by_signature": {k: dict(v) for k, v in sorted(by_signature.items())},
            "failures": sorted(failures),
            "descriptive": {"excluded_member_index_rank": dict(zip(("lowest", "middle", "highest"), ranks)),
                            "excluded_member_rank_chi_square": {
                                "statistic": chi_square_uniform(ranks), "df": 2,
                                "note": "descriptive only (C9): against equal counts per rank; not a gate"},
                            "requested_aliases_in_resolved_index_order": requested_in_order}}


def s_rs_leak(rows, oracles) -> dict:
    """Core RS rows whose core S sibling shares (k, target), and whose S first oracle reference passes RS."""
    oracle_by_id = oracles if isinstance(oracles, dict) else {o["id"]: o for o in oracles}
    siblings = {}
    for row in core_rows(rows, ("S",)):
        if row["parent_id"] in siblings:
            raise ValueError("two core S rows on one parent")
        siblings[row["parent_id"]] = row
    same, passes = [], []
    rs = core_rows(rows, ("RS",))
    for row in rs:
        sibling = siblings.get(row["parent_id"])
        if sibling is None:
            raise ValueError("core RS row without a core S sibling: " + row["id"])
        a, b = row_spec(sibling)["selection"], row_spec(row)["selection"]
        if (a["k"], a["target"]) == (b["k"], b["target"]):
            same.append(row["id"])
        oracle = oracle_by_id.get(sibling["id"])
        if oracle is None or oracle["parent_id"] != sibling["parent_id"] or not oracle["references"]:
            raise ValueError("missing or misaligned S oracle: " + sibling["id"])
        first = oracle["references"][0]
        if score(row_spec(sibling), first) != 1:
            raise ValueError("S oracle reference fails its own row: " + sibling["id"])
        if score(row_spec(row), first) == 1:
            passes.append(row["id"])
    return {"rs_rows": len(rs), "status": "COMPUTED" if rs else "NO_ROWS",
            "same_k_target": len(same), "s_first_reference_passes_rs": len(passes),
            "same_k_target_ids": sorted(same), "s_first_reference_passes_rs_ids": sorted(passes)}


def shortcut_response(spec: dict) -> str:
    n = len(spec["initial"])
    answer = list(range(spec["selection"]["k"])) if "selection" in spec else list(range(n))
    return canonical({"answer": answer})


def shortcut(rows) -> dict:
    """The dataset's own shortcut diagnostic, scored with the unchanged corpus.reward, per core signature."""
    by_signature = defaultdict(lambda: {"total": 0, "first_indices_correct": 0, "unknown_correct": 0,
                                        "first_indices_correct_ids": []})
    for row in core_rows(rows):
        entry = by_signature[signature(row)]
        entry["total"] += 1
        if reward(row, shortcut_response(row_spec(row))):
            entry["first_indices_correct"] += 1
            entry["first_indices_correct_ids"].append(row["id"])
        entry["unknown_correct"] += reward(row, UNKNOWN_RESPONSE)
    for entry in by_signature.values():
        entry["first_indices_correct_ids"].sort()
    return {"definition": SHORTCUT_DEFINITION, "by_signature": {k: dict(v) for k, v in sorted(by_signature.items())}}


def content_key(spec: dict) -> tuple:
    return (tuple(sorted(spec["initial"])), spec["selection"]["k"], spec["selection"]["target"])


def atomic_pool(train_rows, manifest_ids) -> list[dict]:
    """The TRAIN rows named by manifests/atomic_S.json, each verified as a core atomic-S train row."""
    ids = list(manifest_ids)
    wanted = set(ids)
    if len(ids) != len(wanted):
        raise ValueError("duplicate atomic_S manifest id")
    pool = [r for r in train_rows if r["id"] in wanted]
    if len(pool) != len(wanted) or {r["id"] for r in pool} != wanted:
        raise ValueError("atomic_S manifest does not match TRAIN rows")
    if any(r["role"] != "train" or r["tag"] != "core" or r["skills"] != ["S"] for r in pool):
        raise ValueError("atomic_S bank contamination")
    return pool


def _permuted(spec: dict, rng: random.Random) -> dict:
    initial = list(spec["initial"])
    if len(set(initial)) < 2:
        raise ValueError("cannot permute a constant initial vector")
    permuted = initial[:]
    while permuted == initial:
        rng.shuffle(permuted)
    return dict(spec, initial=permuted)


def overlap(train_rows, heldout_rows, signatures=OVERLAP_SIGNATURES, seed: int = OVERLAP_CONTROL_SEED) -> dict:
    """TRAIN atomic-S -> held-out content overlap on key (sorted(initial), k, target), with both controls."""
    index = defaultdict(list)
    for row in train_rows:
        if row["tag"] != "core" or row["skills"] != ["S"]:
            raise ValueError("overlap pool must be core atomic S rows")
        index[content_key(row_spec(row))].append(row["id"])
    by_signature = {}
    for sig in signatures:
        chosen = core_rows(heldout_rows, (sig,))
        matches = []
        for row in chosen:
            hits = index.get(content_key(row_spec(row)))
            if hits:
                matches.append({"id": row["id"], "train_ids": sorted(hits)})
        matches.sort(key=lambda m: m["id"])
        by_signature[sig] = {"rows": len(chosen), "exposed": len(matches),
                             "exposed_ids": [m["id"] for m in matches], "matches": matches}
    rng = random.Random(seed)
    positive = negative = foreign = 0
    for row in sorted(train_rows, key=lambda r: r["id"]):
        spec = row_spec(row)
        positive += int(row["id"] in index.get(content_key(_permuted(spec, rng)), ()))
        shifted = dict(spec, selection={"k": spec["selection"]["k"], "target": spec["selection"]["target"] + 1})
        hits = index.get(content_key(shifted), ())
        negative += int(row["id"] in hits)
        foreign += int(bool(hits))
    ok = positive == len(train_rows) and negative == 0 and len(train_rows) > 0
    return {"key": "(sorted(initial), k, target)", "pool_rows": len(train_rows), "pool_distinct_keys": len(index),
            "by_signature": by_signature,
            "controls": {"status": "PASS" if ok else "FAIL", "seed": seed,
                         "permuted_train_spec_recovers_its_row": positive,
                         "target_plus_one_recovers_its_row": negative,
                         "target_plus_one_matches_another_train_row": foreign,
                         "rule": "PASS iff every permuted TRAIN spec recovers its own row and no target+1 spec does; "
                                 "target+1 collisions with OTHER pool rows are descriptive"}}


def _sealed_name(relative: str) -> bool:
    """The final name is a sealed bank, compared as Windows opens it (case, trailing dots/spaces, ':stream')."""
    name = relative.replace("\\", "/").rsplit("/", 1)[-1]
    return name.split(":", 1)[0].rstrip(" .").casefold() in SEALED_FILES


def verified_bytes(data: Path, relative: str, manifest: dict, record: dict, at_freeze: bool = False) -> bytes:
    """Read one data file once, check its sha against manifest['files'], record it.

    Before any byte is read: `relative` must be one of the exact paths this module builds (READABLE;
    canonical/test.jsonl only when at_freeze is the literal True) and an exact manifest['files'] key.
    Any other sealed-bank name, however spelled, is refused; challenge banks and TEST oracles always.
    """
    if type(relative) is not str:
        raise SystemExit("data path must be a str")
    freeze = at_freeze is True
    if _sealed_name(relative) and not (freeze and relative in FREEZE_READABLE):
        raise SystemExit("sealed file refused: " + relative)
    if relative not in READABLE and not (freeze and relative in FREEZE_READABLE):
        raise SystemExit("data file not admitted: " + relative)
    files = manifest.get("files") if type(manifest) is dict else None
    if type(files) is not dict or relative not in files:
        raise SystemExit("data file not in the manifest: " + relative)
    raw = (data / relative).read_bytes()
    digest_value = sha256_bytes(raw)
    if digest_value != files[relative]["sha256"]:
        raise SystemExit("data file drift: " + relative)
    record[relative] = digest_value
    return raw


def jsonl_rows(raw: bytes) -> list[dict]:
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    return [strict_json(line.decode("utf-8"), MAX_LINE) for line in lines]


def role_file(directory: str, role: str, at_freeze: bool) -> str:
    if role in FREEZE_ROLES and at_freeze is not True:
        raise SystemExit("role test requires --at-freeze")
    if role not in HELD_OUT_ROLES + FREEZE_ROLES:
        raise SystemExit("role not admitted: " + role)
    if role in FREEZE_ROLES and directory != "canonical":
        raise SystemExit("only TEST specs are read at freeze: " + directory)
    return directory + "/" + role + ".jsonl"


def role_gates(rows: list[dict], oracles: dict, pool: list[dict]) -> dict:
    return {"rs_floor": rs_floor(rows), "triple_structure": triple_structure(rows),
            "s_rs_leak": s_rs_leak(rows, oracles), "shortcut": shortcut(rows), "overlap": overlap(pool, rows)}


def freeze_gates(rows: list[dict]) -> dict:
    """Section 11 at freeze, on TEST specs: the lookup-blind floor and the triple structure, nothing else.

    The S->RS leak is a DEV-only gate (the reason for D4), so no TEST oracle is needed or opened.
    """
    return {"rs_floor": rs_floor(rows), "triple_structure": triple_structure(rows)}


def write_receipt(receipt: dict, out: Path) -> None:
    """Create-only LF write, refused if any CODE_FILES source changed since import."""
    if code_sha256() != _IMPORT_CODE:
        raise SystemExit("data-gate source changed since import")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--roles", nargs="+", choices=HELD_OUT_ROLES + FREEZE_ROLES, default=list(HELD_OUT_ROLES))
    ap.add_argument("--at-freeze", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    roles = list(dict.fromkeys(args.roles))
    if any(role in FREEZE_ROLES for role in roles) and not args.at_freeze:
        raise SystemExit("role test requires --at-freeze")
    if args.out.exists():
        raise SystemExit("data gate receipt exists (create-only)")
    data = args.data.resolve(strict=True)
    if args.out.resolve().is_relative_to(data):
        raise SystemExit("protected output path")
    files: dict[str, str] = {}
    manifest_raw = (data / "manifest.json").read_bytes()
    if sha256_bytes(manifest_raw) != MANIFEST:
        raise SystemExit("dataset manifest drift")
    manifest = json.loads(manifest_raw.decode("utf-8"))
    plan = json.loads(verified_bytes(data, ATOMIC_S_MANIFEST, manifest, files).decode("utf-8"))
    train = jsonl_rows(verified_bytes(data, "canonical/train.jsonl", manifest, files))
    pool = atomic_pool(train, plan["ids"])
    del train
    pool_shortcut = shortcut(pool)["by_signature"]["S"]
    gates = {}
    for role in roles:
        canonical_file = role_file("canonical", role, args.at_freeze)
        rows = jsonl_rows(verified_bytes(data, canonical_file, manifest, files, args.at_freeze))
        if len({r["id"] for r in rows}) != len(rows) or any(r["role"] != role for r in rows):
            raise SystemExit("duplicate id or role drift: " + role)
        if role in FREEZE_ROLES:
            gates[role] = freeze_gates(rows)
            continue
        wanted = {r["id"] for r in core_rows(rows, ("S",))}
        oracle_file = role_file("oracles", role, args.at_freeze)
        oracles = {o["id"]: o for o in jsonl_rows(verified_bytes(data, oracle_file, manifest, files, args.at_freeze))
                   if o["id"] in wanted}
        gates[role] = role_gates(rows, oracles, pool)
    receipt = {"experiment": "COMP-RS-001", "stage": "S0PRE_DATA_GATES",
               "amendment": "COMP_RS_001_AMENDMENT_1.md section 11", "dataset_manifest_sha256": MANIFEST,
               "code_sha256": dict(_IMPORT_CODE), "code_bundle_sha256": digest(_IMPORT_CODE),
               "files_sha256": files,
               "roles": roles, "at_freeze": bool(args.at_freeze), "core_filter": "tag == 'core'",
               "freeze_role_gates": list(FREEZE_GATES),
               "train_atomic_S": {"manifest": ATOMIC_S_MANIFEST, "rows": len(pool),
                                  "ids_sha256": digest(sorted(r["id"] for r in pool)),
                                  "shortcut": {k: v for k, v in pool_shortcut.items() if k != "first_indices_correct_ids"}
                                  | {"first_indices_correct_ids_sha256": digest(pool_shortcut["first_indices_correct_ids"])}},
               "gates": gates, "test_challenge_read": any(r in FREEZE_ROLES for r in roles),
               "model_outputs_read": False,
               "note": "Spec-only computations; not reads under Amendment 1 section 1."}
    write_receipt(receipt, args.out)
    summary = {}
    for role, g in gates.items():
        entry = {"rs_floor_mean": g["rs_floor"]["mean_p_blind"], "first_lex": g["rs_floor"]["first_lex_pass"],
                 "triple_structure": g["triple_structure"]["status"]}
        if "s_rs_leak" in g:
            entry["leak"] = [g["s_rs_leak"]["same_k_target"], g["s_rs_leak"]["s_first_reference_passes_rs"]]
            entry["shortcut_S"] = g["shortcut"]["by_signature"].get("S", {}).get("first_indices_correct")
            entry["overlap"] = {s: v["exposed"] for s, v in g["overlap"]["by_signature"].items()}
        summary[role] = entry
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
