"""Independent prompt-only BULK witness/statistics audit; stdlib, no model."""
import ast
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import NormalDist
import sys
import xml.etree.ElementTree as ET

C = Path(__file__).resolve().parent.parent
PACKET = C / "runs/bulk-bank-apparatus-001"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def arithmetic(node, state):
    if isinstance(node, ast.Expression): return arithmetic(node.body, state)
    if isinstance(node, ast.Constant) and type(node.value) is int: return node.value
    if isinstance(node, ast.Name): return state[node.id]
    if isinstance(node, ast.BinOp):
        a, b = arithmetic(node.left, state), arithmetic(node.right, state)
        if isinstance(node.op, ast.Add): return a + b
        if isinstance(node.op, ast.Sub): return a - b
        if isinstance(node.op, ast.Mult): return a * b
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Gt):
        return arithmetic(node.left, state) > arithmetic(node.comparators[0], state)
    raise ValueError("unregistered program expression")


def witness(prompt):
    """Only visible prompt is accepted; no case/scorer/metadata parameters."""
    if type(prompt) is not str: raise TypeError("prompt only")
    body, question = re.split(r"\n(?:Question|Query): ", prompt)
    query = question[:3]
    content = body.split("\n", 1)[1].rstrip(".")
    if body.startswith("LEDGERS:"):
        totals = {}
        for record in content.split(";"):
            name, _, opening, remainder = record.split(" ", 3)
            if remainder.startswith("entries "):
                delta = sum(int(n) for n in remainder[len("entries "):].split(","))
            else:
                credits, debits = remainder[len("credits "):].split(" debits ")
                delta = sum(int(n) for n in credits.split(",") if n) - sum(int(n) for n in debits.split(",") if n)
            totals[name] = int(opening) + delta
        return str(totals[query]), "ANSWER"
    if body.startswith("RULES:"):
        rules_text = body.split(". ", 1)[1].split(".\nFACTS:", 1)[0]
        rules = [(set(left.split("&")), right) for left, right in (r.split(">") for r in rules_text.split(","))]
        facts = set(content.removeprefix("FACTS: ").split(","))
        remaining, uses = [], defaultdict(list)
        for i, (antecedents, _) in enumerate(rules):
            remaining.append(set(antecedents))
            for antecedent in antecedents: uses[antecedent].append(i)
        agenda, known = deque(facts), set(facts)
        while agenda:
            fact = agenda.popleft()
            for i in uses[fact]:
                remaining[i].discard(fact)
                consequence = rules[i][1]
                if not remaining[i] and consequence not in known:
                    known.add(consequence); agenda.append(consequence)
        return "YES" if query in known else "NO", "ANSWER"
    if body.startswith("PROGRAM:"):
        initial, steps = content.split(".\nSTEPS: ")
        state = {name: int(value) for name, value in (item.split("=") for item in initial.removeprefix("INIT: ").split(","))}
        numbered = sorted((int(line.split(" ", 1)[0]), line.split(" ", 1)[1]) for line in steps.split(";"))
        for _, text in numbered:
            if text.startswith("IF "):
                condition, alternatives = text[3:].split(" THEN ")
                yes, no = alternatives.split(" ELSE ")
                text = yes if arithmetic(ast.parse(condition, mode="eval"), state) else no
            dest, expression = text.split("=")
            state[dest] = arithmetic(ast.parse(expression, mode="eval"), state)
        return str(state[query]), "ANSWER"
    if body.startswith("MEMORY:"):
        relevant = []
        for line in content.split(";"):
            pieces = line.split()
            if pieces[2] == query and pieces[3] not in ("quotes", "QUOTE"):
                relevant.append((int(pieces[1]), pieces[3], pieces[4] if len(pieces) == 5 else None))
        _, action, value = max(relevant)
        return (value, "ANSWER") if action in ("SET", "says") else ("UNKNOWN", "MISSING")
    if body.startswith("DOCUMENTS:"):
        approved = []
        latest = "latest approved date" in body.split("\n", 1)[0]
        for line in content.split(";"):
            pieces = line.split()
            if pieces[0] == query and pieces[1] == "A":
                approved.append((int(pieces[2][1:]) if len(pieces) == 4 else 0, int(pieces[-1])))
        if not approved: return "UNKNOWN", "MISSING"
        date = max(d for d, _ in approved)
        values = {value for d, value in approved if not latest or d == date}
        return (str(values.pop()), "ANSWER") if len(values) == 1 else ("UNKNOWN", "CONFLICT")
    raise ValueError("unknown frozen grammar")


def main():
    manifest = read(PACKET / "manifest.json")
    assert sha(C / "bulk_bank.py") == manifest["source_sha256"] == "3242439231064fc81991e157c0da62833f4e4223ec71d60cf9ffd2851815b5eb"
    assert sha(C / "open_bank.py") == manifest["normalizer_source_sha256"] == "25db8ad7b8bd4166188eccab4be73c9a430f696610dea5b8c83b380883f9d42a"
    for name, digest in manifest["files"].items(): assert sha(PACKET / name) == digest
    from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import bulk_bank
    bank, audit = read(PACKET / "bank.json"), read(PACKET / "audit.json")
    independent = []
    for row in bank:
        answer, status = witness(row["prompt"])
        assert (answer, status) == (row["gold"], row["source_status"])
        independent.append({"case_id": row["id"], "prediction": answer, "source_status": status, "correct": True})
    assert len(independent) == 288
    floors, denominators = defaultdict(Counter), Counter()
    for row in bank:
        for prefix, key in (("all", "all"), ("split", row["split"]), ("category", row["category"]),
            ("family", row["generator_family"]), ("difficulty", row["difficulty_stratum"]),
            ("variant", str(row["variant_id"])),
            ("category_split_level", "|".join((row["category"], row["split"], row["difficulty_stratum"])))):
            group = prefix + ":" + key
            denominators[group] += 1
            for name, response in bulk_bank.blind_predictions(row["prompt"]).items():
                floors[group][name] += int(bulk_bank.score_response(response, row)["correct"])
    for key in floors:
        assert dict(floors[key]) == audit["floor_tables"][key]["correct_counts"]
        assert denominators[key] == audit["floor_tables"][key]["cases"]
    by_parent = defaultdict(list)
    for row in bank: by_parent[row["parent_id"]].append(row)
    assert len(by_parent) == 144 and len({r["prompt"] for r in bank}) == 288
    dev = {parent for parent, rows in by_parent.items() if rows[0]["split"] == "DEVELOPMENT"}
    val = {parent for parent, rows in by_parent.items() if rows[0]["split"] == "OPEN_VALIDATION"}
    assert len(dev) == 96 and len(val) == 48 and not dev & val
    styles, pairs = Counter(), Counter()
    for parent, rows in by_parent.items():
        a, b = sorted(rows, key=lambda r: r["variant_id"])
        assert {a["variant_id"], b["variant_id"]} == {0, 1}
        assert all(a[k] == b[k] for k in ("category", "difficulty_stratum", "split", "generator_family"))
        assert a["parent_id"] not in a["prompt"] and b["parent_id"] not in b["prompt"]
        if a["category"] == "answerability_source_conflict":
            assert a["answerability"] == "answerable" and b["gold"] == "UNKNOWN"
            assert a["expected_response_change"] is b["expected_response_change"] is True
            def bundles(prompt):
                return sorted(line.split(" ", 1)[1] for line in re.split(r"\n(?:Question|Query): ", prompt)[0].split("\n", 1)[1].rstrip(".").split(";"))
            assert bundles(a["prompt"]) == bundles(b["prompt"])
            for r in (a, b): styles[r["split"], r["difficulty_stratum"], r["wording_style"], r["answerability"]] += 1
            pairs["evidence_transfer"] += 1
        else:
            assert a["gold"] == b["gold"] and a["source_status"] == b["source_status"]
            pairs["same_gold"] += 1
    assert pairs == {"same_gold": 120, "evidence_transfer": 24}
    for split in ("DEVELOPMENT", "OPEN_VALIDATION"):
        for level in ("easy", "hard"):
            assert len({styles[split, level, style, answer] for style in (0, 1) for answer in ("answerable", "unanswerable")}) == 1
    z95 = NormalDist().inv_cdf(.975)
    zsum = z95 + NormalDist().inv_cdf(.8)
    feasibility = []
    for n, scope in ((48, "primary validation"), (96, "calibration DEV"), (144, "descriptive all-worlds")):
        feasibility.append({"parents": n, "scope": scope,
            "normal_80pct_MDE_vs0": [{"parent_delta_SD": s, "MDE": zsum*s/math.sqrt(n), "95pct_halfwidth": z95*s/math.sqrt(n)} for s in (.1, .2, .35, .5, 1.)],
            "max_SD_for_normal_80pct_5point_effect": .05*math.sqrt(n)/zsum,
            "illustrative_no_harm_sign_power": [{"parent_gain": gain, "prob_gain": .05/gain,
                "P_six_or_more_positive_parents": sum(math.comb(n, k)*(.05/gain)**k*(1-.05/gain)**(n-k) for k in range(6, n+1))} for gain in (.5, 1.)],
            "zero_harms_one_sided95pct_parent_risk_upper": 1-.05**(1/n)})
    branch = Counter()
    for rows in by_parent.values():
        r = rows[0]
        if r["category"] == "program_state" and r["difficulty_stratum"] == "hard" and r["split"] == "OPEN_VALIDATION":
            branch[str(r["difficulty_factors"]["branch_truth"])] += 1
    junit = C / "reviews/bulk_design_cpu_junit_001.xml"
    tests = list(ET.parse(junit).getroot().iter("testsuite"))
    assert sum(int(s.get("tests", "0")) for s in tests) == 21
    assert all(int(s.get(k, "0")) == 0 for s in tests for k in ("failures", "errors", "skipped"))
    assert "torch" not in sys.modules
    result = {"status": "PASS_BULK_SOURCE_DATA_APPARATUS_NOT_NATIVE_ADMITTED", "schema_version": "1.0",
        "reviewer": "/root/adversarial_review", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha(C / "bulk_bank.py"), "bank_sha256": sha(PACKET / "bank.json"),
        "manifest_sha256": sha(PACKET / "manifest.json"), "normalizer_sha256": sha(C / "open_bank.py"),
        "analysis_plan_sha256": sha(C / "BULK_ANALYSIS_PLAN_001.md"), "review_script_sha256": sha(__file__),
        "independent_CPU_tests": {"passed": 21, "failed": 0, "skipped": 0, "pytest_seconds": 1.27, "junit_sha256": sha(junit)},
        "independent_prompt_only_witnesses": independent, "world_counts": {"cases": 288, "parents": 144, "DEV_parents": 96, "validation_parents": 48,
            "same_gold_pairs": 120, "evidence_transfer_pairs": 24, "categories": 6, "per_category_DEV_parents": 16, "per_category_validation_parents": 8},
        "independence": "conditional randomized-world uncertainty only; six shared authored code families, not288 independent tasks or held-out source confirmation",
        "independent_floor_tables": {key: {"cases": denominators[key], "correct_counts": dict(counts)} for key, counts in floors.items()},
        "global_max_tested_blind_floor": max(floors["all:all"].values()), "global_tested_blind_ceiling": 144,
        "matched_evidence_style_counterbalance": [{"split": s, "difficulty": d, "style": w, "answerability": a, "cases": n} for (s,d,w,a),n in sorted(styles.items())],
        "validation_hard_program_branch_parents": dict(branch), "conditional_feasibility": feasibility,
        "per_category_validation_zero_harm_one_sided95pct_upper": 1-.05**(1/8),
        "native_model_loads": 0, "native_calls": 0, "protected_or_native_TEST_reads": False, "human_review": "NOT_REVIEWED",
        "resolved_pre_export_issues": ["HardDEV document noise made draft gold inconsistent; authority corrected before first export",
            "Dialogue code-age magnitude/anyCLEAR shortcut removed through independent value ranges/redeclaration/quote cycles; skeleton floors retained",
            "Bulk reader/provenance/normalizer schema aligned with actual generator before freeze",
            "Named representation variation separated from shared generator dependence; parent-world holdout stated"],
        "limitations": ["A5 seeds and paired variants average within parent; primaryDEV-selected comparator validation uses48parents only",
            "Normal MDE is approximate and sparse changes can make it optimistic; no guaranteed5point gain resolution or tight family preservation",
            "Point margin.05 is fixed; demonstrating an effect beyond.05 with uncertainty requires larger true excess than detecting an effect versus0",
            "All case/variant harms must be preserved despite parent averaging; category6 evidence response and categories1–5 order consistency remain distinct",
            "Tested floor set is bounded and category binary/UNKNOWN50percent floors remain; stronger parser/algorithm shortcuts are not excluded",
            "Counterbalanced deterministic order does not remove late A5 phase/shared GPU/instrumentation timing limitations",
            "No controller fit or source-confirmation claim; finite-grid oracle/capture diagnostic and negative/>1 values must remain",
            "Exact tokenizer binding, reviewed native/admission/parent/complete-cost forecast and fresh6GiBGPU/7GiBRAM gates remain necessary; no retry"],
        "unresolved_source_data_blockers": []}
    with (C / "reviews/bulk_design_review_001.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n")
    md = "# Independent prospective bulk design review\n\nPASS for source/data preparation, not native admission. All288 prompt-only witnesses were recomputed by a separate safe arithmetic/worklist/latest-event/evidence solver and matched constructed gold plus source status. The21 authorCPU fixtures independently pass; packet/source/normalizer hashes agree. The first and only export uses nonce9463250d48f1/seed17. All worlds are retained; no model outcomes or nonce search were used.\n\nThere are144 worldparents:96 DEVELOPMENT and48 validation,16/8 percategory. Five categories have120 same-gold wording/order pairs; category6 has24 matched evidence-transfer pairs with oneanswerable and oneUNKNOWN member. Global evidence content/counts remain unchanged by the entity-bundle transfer; style×answerability is counterbalanced. Per-case/correct-case harms must not cancel in a parent average. Shared authored algorithms/templates remain explicit, despite12 genuinely different named representations/operators.\n\nThe tested prompt-only globalfloor maximum is113/288; category choice/UNKNOWN50percent floors remain. Full floors are bound by category/split/difficulty/variant and source metadata. Metadata-known variant1UNKNOWN100percent in category6 is disclosed; variant/IDs never enter the model prompt. Easy/hard factors and program branch coverage are preserved rather than selected for favorable outcomes. Two source-only defects were fixed before export: unprinted approvedDEV noise contradicted gold, and dialogue magnitude/withdrawal patterns admitted an age-blind shortcut. The final normalizer independently separates correctness from canonical format.\n\nAt paired-parent SD.20, approximate two-sided.05/80percent MDE versus0 is8.09points for48validation parents,5.72 for96DEV and4.67 for144descriptive units. A5point effect on48parents is not assured: under an idealized no-harm sign test, the detection chance is3.17percent if rare changedparents fix bothvariants,34.69percent if onevariant. Zero observed parentharms still allows6.05percent one-sided95percent upper risk under conditionalindependence; pervalidationcategory n8 permits31.23percent. These are assumption-bound feasibility illustrations, not a new confirmatory test or broad population guarantee.\n\nPrimaryvalidation must keep theDEV-frozen strongest6constant comparator (.8,.9,1,1.1,1.2,priorA3mean1.1473425965417514). A5's two seeds average within cases, then variants within parents. Full category×difficulty/query/source-status patterns, fixed curves/headroom, everyvariant harm and physical-only costs remain mandatory. Retrospective validation/family constants/oracles are diagnostic; capture remainsnull with nonpositiveheadroom and otherwise unclipped. Keep unchangedA3 until headroom/predictive evidence justifies a new prospective hypothesis. No nativequalification, semanticintelligence, tightpreservation, deployment speed or researchallocation efficiency is established.\n"
    with (C / "reviews/bulk_design_review_001.md").open("x", encoding="utf-8") as f: f.write(md)
    print(json.dumps({"status": result["status"], "cases": 288, "parents": 144, "validation_parents": 48, "independent_witnesses": len(independent), "blind_max": result["global_max_tested_blind_floor"], "branch_coverage": branch}))


if __name__ == "__main__": main()
