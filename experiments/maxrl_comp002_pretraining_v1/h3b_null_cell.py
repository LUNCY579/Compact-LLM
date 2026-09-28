"""H3b null cell on core RS specs (COMP-RS-001 Amendment 1, sections 3 and 11; clarification C7). Spec-only.

Two pairs of deterministic lookup-blind policies answer every core RS row with
a sum-valid k-subset over ALL item indices: lex-first vs lex-last, and
lex-first vs a fixed random choice. For each design re-draw (excluded member
of every triple re-drawn uniformly) e is recomputed for each policy and the
registered sign-flip test is run on the paired difference. The false-rejection
rate, its Monte Carlo SE and its one-sided 95% CP upper bound decide the
section 3 rule: an upper bound above 0.075 switches H3b's reference to the
design-randomization distribution.

Registered counts (C7): >= 2,000 re-draws and exactly 10^6 flips per sign-flip
test. Only a run at those counts emits the rule outcome; any other run records
outcome NOT_A_REGISTERED_RUN with rule_compliant false. The freeze run passes
H3b's registered flip seed as --flip-seed, so the null cell tests against the
sign patterns H3b will use.

No model output is generated or scored. Role test reads canonical/test.jsonl
and therefore requires --at-freeze, the registered counts and --flip-seed, all
checked before any file is touched; it is never run before arm freeze.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

from experiments.maxrl_comp002_pretraining_v1 import content_scoring, preflight, stats_exact
from experiments.maxrl_comp002_pretraining_v1.preflight import digest
from experiments.maxrl_comp002_pretraining_v1.stats_exact import (
    cp_upper, design, e_value, redraw_e, signflip_test_columns)
from experiments.maxrl_composable_skills_v2 import contracts, corpus, verifier
from experiments.maxrl_composable_skills_v2.contracts import strict_json

MANIFEST_SHA = "86ac8f0d851f2178752357f62249eab5a23fd3de8fa032e7373614891df4cd44"
EXPECTED_ROWS = {"dev": 250, "test": 250}
FREEZE_ROLES = ("test",)
MAX_LINE = 131072
MAX_MANIFEST = 1 << 20
LEVEL = 0.05
SWITCH_BOUND = 0.075
REGISTERED_MIN_REDRAWS = 2000        # section 3 Rule: ">= 2,000 re-draws"
REGISTERED_FLIPS = 1_000_000         # section 3 H3b: "Monte Carlo with 10^6 flips"
SIGNFLIP_REFERENCE = "SIGNFLIP_REFERENCE"
DESIGN_RANDOMIZATION_REFERENCE = "DESIGN_RANDOMIZATION_REFERENCE"
NOT_A_REGISTERED_RUN = "NOT_A_REGISTERED_RUN"
PAIRS = (("lex_first", "lex_last"), ("lex_first", "fixed_random"))
PAIR_NAMES = tuple(f"{first}_vs_{second}" for first, second in PAIRS)
POLICIES = ("lex_first", "lex_last", "fixed_random")
HERE = Path(__file__).resolve().parent
# Every source file the null cell's numbers depend on, hashed at import and re-checked before writing.
CODE_FILES = {"h3b_null_cell.py": Path(__file__).resolve(),
              "stats_exact.py": Path(stats_exact.__file__).resolve(),
              "content_scoring.py": Path(content_scoring.__file__).resolve(),
              "preflight.py": Path(preflight.__file__).resolve(),
              "maxrl_composable_skills_v2/verifier.py": Path(verifier.__file__).resolve(),
              "maxrl_composable_skills_v2/corpus.py": Path(corpus.__file__).resolve(),
              "maxrl_composable_skills_v2/contracts.py": Path(contracts.__file__).resolve()}
AMENDMENT_REFERENCE = {"redraws": 400, "mean_e_blind": -0.0015, "sd_across_redraws": 0.030,
                       "false_rejection": {"lex_first_vs_lex_last": 0.062,
                                           "lex_first_vs_fixed_random": 0.050},
                       "mc_se": 0.011, "lex_first_passes_dev": 83, "floor_dev": 0.344,
                       "note": "revision 1 figures at 400 re-draws: not a registered run (C7)"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def code_sha256() -> dict:
    """sha256 of every file in CODE_FILES, keyed by name."""
    return {name: sha256(path) for name, path in CODE_FILES.items()}


_IMPORT_CODE = code_sha256()


def load_core_rs(data: Path, role: str, *, at_freeze: bool = False) -> tuple[list[dict], str]:
    """Core RS rows of canonical/<role>.jsonl, sorted by id, after manifest and file sha checks.

    Role test is refused unless at_freeze is True, before any path is touched.
    Each file is read once: its bytes are hashed and that same buffer is parsed
    with contracts.strict_json (duplicate keys and non-finite constants refused).
    """
    if type(role) is not str or role not in EXPECTED_ROWS:
        raise ValueError("role must be dev or test")
    if role in FREEZE_ROLES and at_freeze is not True:
        raise ValueError("role " + role + " requires at_freeze=True")
    root = Path(data)
    manifest_raw = (root / "manifest.json").read_bytes()
    if sha256_bytes(manifest_raw) != MANIFEST_SHA:
        raise SystemExit("dataset manifest drift")
    manifest = strict_json(manifest_raw.decode("utf-8"), MAX_MANIFEST)
    name = f"canonical/{role}.jsonl"
    raw = (root / name).read_bytes()
    file_sha = sha256_bytes(raw)
    if name not in manifest["files"] or file_sha != manifest["files"][name]["sha256"]:
        raise SystemExit(f"file drift: {name}")
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    rows, ids, parents = [], set(), set()
    for line in lines:
        row = strict_json(line.decode("utf-8"), MAX_LINE)
        if row["tag"] != "core" or row["skills"] != ["R", "S"]:
            continue
        if row["kind"] != "composition" or row["verifier"]["kind"] != "composition":
            raise SystemExit("core RS row is not a composition row")
        if row["id"] in ids or row["parent_id"] in parents:
            raise SystemExit("core RS rows must be one per id and one per parent")
        ids.add(row["id"])
        parents.add(row["parent_id"])
        rows.append({"id": row["id"], "parent_id": row["parent_id"], "spec": row["verifier"]["spec"]})
    if len(rows) != EXPECTED_ROWS[role]:
        raise SystemExit(f"expected {EXPECTED_ROWS[role]} core RS rows, found {len(rows)}")
    return sorted(rows, key=lambda row: row["id"]), file_sha


def blind_subsets(spec: dict) -> list[list[int]]:
    """Lookup-blind sum-valid k-subsets over ALL indices, in lexicographic order (section 11 floor)."""
    values = spec["initial"]
    k, target = spec["selection"]["k"], spec["selection"]["target"]
    subsets = [list(c) for c in combinations(range(len(values)), k) if sum(values[i] for i in c) == target]
    if not subsets:
        raise ValueError("no lookup-blind sum-valid subset")
    return subsets


def floor_canaries(specs: list[dict]) -> dict:
    """Section 11 spec-only facts: lex-first passes and the lookup-blind floor (overall, by k)."""
    by_k: dict = {}
    lex_first = 0
    for spec in specs:
        _, resolved = design(spec)
        subsets = blind_subsets(spec)
        passing = [set(c) <= resolved for c in subsets]
        lex_first += passing[0]
        by_k.setdefault(spec["selection"]["k"], []).append(sum(passing) / len(passing))
    fractions = [f for values in by_k.values() for f in values]
    return {"lex_first_passes": lex_first, "rows": len(specs),
            "floor_mean": math.fsum(fractions) / len(fractions),
            "floor_by_k": {str(k): {"rows": len(v), "mean": math.fsum(v) / len(v)}
                           for k, v in sorted(by_k.items())}}


def policies(specs: list[dict], rng: np.random.Generator) -> dict:
    """Deterministic blind policies; the random choice is drawn once per row and then held fixed."""
    out = {name: [] for name in POLICIES}
    for spec in specs:
        subsets = blind_subsets(spec)
        out["lex_first"].append(subsets[0])
        out["lex_last"].append(subsets[-1])
        out["fixed_random"].append(subsets[int(rng.integers(len(subsets)))])
    return out


def derive_seeds(seed: int, flip_seed: int | None = None) -> dict:
    """Re-draw, policy and flip seeds from --seed; flip_seed, if given, is H3b's registered seed."""
    children = np.random.SeedSequence(seed).spawn(3)
    names = ("redraw_seed", "policy_seed", "flip_seed")
    seeds = {"seed": seed, **{name: int(child.generate_state(1, np.uint64)[0])
                              for name, child in zip(names, children)}}
    if flip_seed is None:
        seeds["flip_seed_source"] = "derived: SeedSequence(seed).spawn(3)[2]"
    else:
        if isinstance(flip_seed, bool) or not isinstance(flip_seed, int) or flip_seed < 0:
            raise ValueError("flip_seed must be a nonnegative integer")
        seeds["flip_seed"] = flip_seed
        seeds["flip_seed_source"] = "given: H3b's registered flip seed (--flip-seed)"
    return seeds


def rule_compliant(redraws: int, flips: int) -> bool:
    """Clarification C7: >= 2,000 re-draws and exactly 10^6 flips per sign-flip test."""
    return redraws >= REGISTERED_MIN_REDRAWS and flips == REGISTERED_FLIPS


def count_rejections(pvalues) -> int:
    """Null rejections at the registered nominal level: p <= 0.05."""
    return int((np.asarray(pvalues, dtype=np.float64) <= LEVEL).sum())


def pair_bound(rejections: int, redraws: int) -> dict:
    """One pair's false-rejection rate, its Monte Carlo SE and its one-sided 95% CP upper bound."""
    rate = rejections / redraws
    upper = cp_upper(rejections, redraws, 0.95)
    return {"rejections": rejections, "false_rejection_rate": rate,
            "mc_se": math.sqrt(rate * (1.0 - rate) / redraws),
            "upper95": upper, "exceeds_switch_bound": upper > SWITCH_BOUND}


def switch_rule(rejections: dict, redraws: int, flips: int) -> dict:
    """The section 3 rule on per-pair rejection counts (pure).

    The decision switches iff ANY registered pair's one-sided 95% CP upper
    bound on its null rejection rate exceeds 0.075: the bound, never the rate.
    Only a run at the registered counts emits it as the outcome (C7); any
    other run's outcome is NOT_A_REGISTERED_RUN, with the decision it would
    have indicated kept as switch_indicated.
    """
    if type(rejections) is not dict or set(rejections) != set(PAIR_NAMES):
        raise ValueError("rejections must name exactly the registered pairs")
    if isinstance(redraws, bool) or not isinstance(redraws, int) or redraws < 1:
        raise ValueError("redraws must be a positive integer")
    if isinstance(flips, bool) or not isinstance(flips, int) or flips < 1:
        raise ValueError("flips must be a positive integer")
    for name in PAIR_NAMES:
        count = rejections[name]
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= redraws:
            raise ValueError(f"rejections for {name} must be an integer in [0, redraws]")
    pairs = {name: pair_bound(rejections[name], redraws) for name in PAIR_NAMES}
    switch = any(pair["exceeds_switch_bound"] for pair in pairs.values())
    compliant = rule_compliant(redraws, flips)
    if not compliant:
        outcome = NOT_A_REGISTERED_RUN
    else:
        outcome = DESIGN_RANDOMIZATION_REFERENCE if switch else SIGNFLIP_REFERENCE
    rule = {"level": LEVEL, "switch_bound": SWITCH_BOUND,
            "basis": "switch if ANY registered pair's one-sided 95% CP upper bound exceeds the bound",
            "registered_counts": {"min_redraws": REGISTERED_MIN_REDRAWS, "flips": REGISTERED_FLIPS},
            "redraws": redraws, "flips": flips, "rule_compliant": compliant,
            "switch_indicated": switch, "outcome": outcome}
    return {"pairs": pairs, "rule": rule}


def null_cell(specs: list[dict], redraws: int, flips: int, seed: int, *, flip_seed: int | None = None) -> dict:
    """Re-draw the design, recompute e per policy, sign-flip each paired difference."""
    if redraws < 1 or flips < 1:
        raise ValueError("redraws and flips must be positive")
    seeds = derive_seeds(seed, flip_seed)
    chosen = policies(specs, np.random.default_rng(seeds["policy_seed"]))
    matrices = redraw_e([chosen[name] for name in POLICIES], specs, redraws,
                        np.random.default_rng(seeds["redraw_seed"]))
    e = dict(zip(POLICIES, matrices))
    report = {"seeds": seeds, "redraws": redraws, "flips": flips, "rows": len(specs), "policies": {},
              "pairs": {}}
    for name in POLICIES:
        per_redraw = e[name].mean(axis=1)
        observed = [e_value(c, s) for c, s in zip(chosen[name], specs)]
        report["policies"][name] = {
            "mean_e": float(per_redraw.mean()),
            "sd_across_redraws": float(per_redraw.std(ddof=1)) if redraws > 1 else None,
            "observed_mean_e": math.fsum(observed) / len(observed)}
    counts, extras = {}, {}
    for (first, second), name in zip(PAIRS, PAIR_NAMES):
        difference = e[first] - e[second]
        pvalues = signflip_test_columns(difference.T, flips, seed=seeds["flip_seed"])
        counts[name] = count_rejections(pvalues)
        extras[name] = {"rows_differing": int((np.abs(difference) > 0).any(axis=0).sum()),
                        "mean_p": float(np.mean(pvalues))}
    decision = switch_rule(counts, redraws, flips)
    report["pairs"] = {name: {**decision["pairs"][name], **extras[name]} for name in PAIR_NAMES}
    report["rule"] = decision["rule"]
    return report


def write_receipt(receipt: dict, out: Path) -> None:
    """Create-only LF write, refused if any CODE_FILES source changed since import."""
    if code_sha256() != _IMPORT_CODE:
        raise SystemExit("null-cell source changed since import")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")


def default_data() -> Path:
    return Path(os.environ.get("COMP_RS_DATA", r"C:\Users\clclc\BusinessProject\Rineya-Dataset\data"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--role", choices=tuple(EXPECTED_ROWS), required=True)
    ap.add_argument("--redraws", type=int, required=True)
    ap.add_argument("--flips", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--flip-seed", type=int, default=None,
                    help="H3b's registered flip seed; required for --role test")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--at-freeze", action="store_true",
                    help="required for --role test: arms and analysis plan are frozen")
    args = ap.parse_args(argv)
    if args.role == "test" and not args.at_freeze:
        raise SystemExit("role test requires --at-freeze")
    if args.redraws < 1 or args.flips < 1 or args.seed < 0 or (args.flip_seed is not None and args.flip_seed < 0):
        raise SystemExit("redraws/flips must be positive and seeds nonnegative")
    if args.role == "test" and not rule_compliant(args.redraws, args.flips):
        raise SystemExit(f"role test requires the registered counts: >= {REGISTERED_MIN_REDRAWS} re-draws "
                         f"and exactly {REGISTERED_FLIPS} flips")
    if args.role == "test" and args.flip_seed is None:
        raise SystemExit("role test requires --flip-seed (H3b's registered flip seed)")
    if args.out.exists():
        raise SystemExit("null-cell receipt exists (create-only)")
    data = (args.data or default_data()).resolve(strict=True)
    rows, file_sha = load_core_rs(data, args.role, at_freeze=args.at_freeze)
    specs = [row["spec"] for row in rows]
    for spec in specs:
        design(spec)
    report = null_cell(specs, args.redraws, args.flips, args.seed, flip_seed=args.flip_seed)
    receipt = {"experiment": "COMP-RS-001", "stage": "S0PRE_H3B_NULL_CELL", "role": args.role,
               "at_freeze": bool(args.at_freeze), "dataset_manifest_sha256": MANIFEST_SHA,
               "canonical_file": f"canonical/{args.role}.jsonl", "canonical_file_sha256": file_sha,
               "core_rs_rows": len(rows), "parents": len({row["parent_id"] for row in rows}),
               "ids_sha256": digest(sorted(row["id"] for row in rows)),
               "triple_structure_rows_verified": len(specs),
               "canaries": floor_canaries(specs), **report,
               "signflip_chunk": stats_exact.SIGNFLIP_CHUNK,
               "code_sha256": dict(_IMPORT_CODE), "code_bundle_sha256": digest(_IMPORT_CODE),
               "code_hash_rule": "hashed at import, re-checked unchanged before the receipt is written",
               "runtime": {"python": sys.version.split()[0], "numpy": np.__version__},
               "amendment_reference": AMENDMENT_REFERENCE,
               "test_challenge_read": args.role == "test", "model_outputs_read": False}
    write_receipt(receipt, args.out)
    print(json.dumps({"role": args.role, "redraws": args.redraws, "flips": args.flips,
                      "rule": receipt["rule"]["outcome"], "rule_compliant": receipt["rule"]["rule_compliant"],
                      "pairs": {k: [v["false_rejection_rate"], v["upper95"]]
                                for k, v in receipt["pairs"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
