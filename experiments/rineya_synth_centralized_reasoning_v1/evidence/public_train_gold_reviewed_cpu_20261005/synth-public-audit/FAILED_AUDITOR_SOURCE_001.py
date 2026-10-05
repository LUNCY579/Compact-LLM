"""Independent audit of the actual approved640 SYNTH atomic TRAIN messages.

Only this file's standard-library parser and solvers determine public answers.
No SYNTH module is imported. Private specs, labels and verifier records are read
only after an oracle result is fixed. Mutations remain transient diagnostics;
no training corpus, admission, model call, tokenizer or bank access is produced.
"""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
from itertools import combinations
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import tracemalloc

SOURCE = Path("C:/Users/clclc/BusinessProject/ChatBubble-rineya-synth-centralized-reasoning-v1")
OUT = Path(__file__).resolve().parent
PACKAGE = "experiments/rineya_synth_centralized_reasoning_v1"
BUNDLE = PACKAGE + "/data/crossed_source2_v1_20261003/"
SOURCE_FILES = ("AGENTS.md", "RINEYA_SYNTH_CENTRALIZED_REASONING.md", PACKAGE + "/AGENTS.md",
                PACKAGE + "/rs_contract.py", PACKAGE + "/rs_verifier.py", PACKAGE + "/task_compiler.py")
DATA_PINS = {
    "candidate_records.jsonl": "90c151edb668416935608912a0d21c0425dbbd9f37a81a1796d4132dcee9f350",
    "student_pool_M1.jsonl": "5e3ab59b30ea07ae29b89c712b9eb39de1392cd9cb62c92418858c1aa003e1b9",
    "student_pool_M2.jsonl": "8fde7abe77b5479288a913034f110c5197c6a5764281b294aea6576fd7123752",
    "plan.json": "a7de5c99f772ece147b08249c332c8d3448e9747a8c2df64dcd0375ff36534d7",
    "enrollment.json": "2730f96e1e7b9b23bab6f5bfc869400d16fd42eca2071d004c68aacd2bb1defc",
}
MAX_INPUT = 32 * 1024 * 1024
SYSTEM = ('Solve using only the stated data and rules. Return exactly one JSON object '
          'with the single key "answer": a nonempty list of distinct zero-based item '
          'indices satisfying all requirements, or the string "UNKNOWN" only when '
          'no such list exists. Any valid list is accepted. No explanations or extra keys.')
INTRO = ("Complete this dispatch task.", "Find one valid plan for these items.",
         "Return a certificate meeting every requirement.", "Resolve this finite planning instance.")
R_RULE = ('Resolve EVERY requested alias through links until an integer item index is reached. '
          'Repeated resolved items count once. A missing link or cycle in any requested '
          'chain means no valid answer exists. Other aliases are distractors.')
S_RULE = ('Choose exactly k distinct eligible item indices whose INITIAL values sum to target. '
          'There is no preferred subset.')
NO_EXTRA = "Order does not matter. There are no updates, filtering, or precedence rules."
ALIAS = re.compile(r"[a-z][a-z0-9_]{0,19}")
WALL_START = time.monotonic()
CPU_START = time.process_time()
MASK_CACHE = {}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def budget():
    require(time.monotonic() - WALL_START <= 300 and time.process_time() - CPU_START <= 300,
            "300second CPU/wall audit budget exceeded; cohort remains incomplete")


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def parse(raw, limit=MAX_INPUT):
    require(type(raw) in (bytes, str) and len(raw) <= limit, "bounded JSON required")
    value = json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("ascii")).hexdigest()


def read_bound(relative, expected=None):
    budget()
    path = SOURCE / relative
    require(".." not in relative.split("/") and not path.is_symlink() and path.is_file(), "selected regular source file required")
    before = path.stat()
    require(0 < before.st_size <= MAX_INPUT, "32MiB input bound")
    raw = path.read_bytes()
    after = path.stat()
    require(before.st_size == after.st_size == len(raw) and before.st_mtime_ns == after.st_mtime_ns,
            "input changed during read")
    sha = hashlib.sha256(raw).hexdigest()
    require(expected is None or sha == expected, "approved input hash drift: " + relative)
    return raw, {"path": str(path), "relative_path": relative, "bytes": len(raw), "sha256": sha}


def git_identity():
    def call(*args):
        result = subprocess.run(["git", "-C", str(SOURCE), *args], capture_output=True, text=True, timeout=15)
        require(result.returncode == 0, "source Git identity unavailable")
        return result.stdout.strip()
    return {"head": call("rev-parse", "HEAD"), "branch": call("branch", "--show-current"),
            "status_porcelain": call("status", "--porcelain")}


def parse_public(messages):
    """Parse public text only: do not accept private spec/skill/template arguments."""
    require(type(messages) is list and len(messages) == 2 and messages[0] == {"role": "system", "content": SYSTEM},
            "unsupported public system/turn grammar")
    require(type(messages[1]) is dict and set(messages[1]) == {"role", "content"}
            and messages[1]["role"] == "user" and type(messages[1]["content"]) is str,
            "unsupported public user-turn grammar")
    text = messages[1]["content"]
    require(len(text.encode("utf-8")) <= 65536, "public task byte bound")
    lines = text.splitlines()
    require(len(lines) >= 7 and lines[0] in INTRO, "unregistered public introduction")
    if lines[4] == "SPEC" and lines[-1] == "END_SPEC":
        require(len(lines) == 7, "exact SPEC block boundary")
        problem = parse(lines[5], 65536)
        rendering = "SPEC"
    elif lines[4] == "DATA_FIELDS" and lines[-1] == "END_DATA_FIELDS":
        problem = {}
        for line in lines[5:-1]:
            match = re.fullmatch(r"(initial|lookup|selection) = (.+)", line)
            require(match is not None and match[1] not in problem, "unregistered/duplicate DATA_FIELDS entry")
            problem[match[1]] = parse(match[2], 65536)
        rendering = "DATA_FIELDS"
    else:
        raise ValueError("unsupported registered public data rendering")
    require(type(problem) is dict and set(problem) in ({"initial", "lookup"}, {"initial", "selection"}),
            "approved corpus requires explicitly atomic R or S")
    values = problem["initial"]
    require(type(values) is list and 6 <= len(values) <= 15 and all(type(v) is int and -100 <= v <= 100 for v in values),
            "invalid public initial values")
    n = len(values)
    if "lookup" in problem:
        skill = "R"
        require(lines[1:4] == [R_RULE, "Return ALL resolved items, each exactly once.", NO_EXTRA], "R instruction drift")
        lookup = problem["lookup"]
        require(type(lookup) is dict and set(lookup) == {"links", "requested"}, "public lookup grammar")
        links, requested = lookup["links"], lookup["requested"]
        require(type(links) is dict and 1 <= len(links) <= 80, "public link-map bound")
        require(all(type(k) is str and ALIAS.fullmatch(k) and
                    (type(v) is int and 0 <= v < n or type(v) is str and ALIAS.fullmatch(v)) for k, v in links.items()), "public alias/link types")
        require(type(requested) is list and 1 <= len(requested) <= n and
                all(type(v) is str and ALIAS.fullmatch(v) for v in requested), "public requested-alias types")
    else:
        skill = "S"
        require(lines[1:4] == ["All items are eligible; no reference lookup is needed.", S_RULE, NO_EXTRA], "S instruction drift")
        selection = problem["selection"]
        require(type(selection) is dict and set(selection) == {"k", "target"}
                and type(selection["k"]) is int and 1 <= selection["k"] <= min(5, n)
                and type(selection["target"]) is int and -50000 <= selection["target"] <= 50000, "public selection grammar")
    return problem, skill, rendering


def replace_problem(messages, problem):
    changed = deepcopy(messages)
    lines = changed[1]["content"].splitlines()
    if lines[4] == "SPEC":
        lines[5] = canonical(problem)
    else:
        lines = lines[:5] + [key + " = " + canonical(problem[key]) for key in ("initial", "lookup", "selection") if key in problem] + ["END_DATA_FIELDS"]
    changed[1]["content"] = "\n".join(lines)
    return changed


def walk_references(problem):
    links, requested = problem["lookup"]["links"], problem["lookup"]["requested"]
    endpoints, paths = [], []
    for root in requested:
        node, visited, path = root, set(), []
        while type(node) is str:
            if node not in links or node in visited:
                return (), paths + [{"requested": root, "path": path, "failure": "missing" if node not in links else "cycle"}]
            visited.add(node)
            path.append(node)
            node = links[node]
        endpoints.append(node)
        paths.append({"requested": root, "path": path, "endpoint": node})
    return (tuple(sorted(set(endpoints))),), paths


def closure_references(problem):
    """Independent finite directed transitive closure, not request-chain walking."""
    links, requested = problem["lookup"]["links"], problem["lookup"]["requested"]
    nodes = set(links) | set(requested) | set(links.values())
    closure = {node: ({links[node]} if node in links else set()) for node in nodes}
    for pivot in sorted(nodes, key=lambda x: (type(x).__name__, str(x))):
        budget()
        for node in nodes:
            if pivot in closure[node]:
                closure[node].update(closure[pivot])
    missing = {node for node in nodes if type(node) is str and node not in links}
    cycles = {node for node in nodes if node in closure[node]}
    endpoints = set()
    for requested_alias in requested:
        reachable = closure[requested_alias] | {requested_alias}
        terminals = {node for node in reachable if type(node) is int}
        if reachable & (missing | cycles) or len(terminals) != 1:
            return ()
        endpoints.update(terminals)
    return (tuple(sorted(endpoints)),)


def combination_certificates(problem):
    values, selection = problem["initial"], problem["selection"]
    return tuple(choice for choice in combinations(range(len(values)), selection["k"])
                 if sum(values[index] for index in choice) == selection["target"])


def mask_certificates(problem):
    """Independent bitmask/hamming-cardinality arithmetic enumeration."""
    values, selection = problem["initial"], problem["selection"]
    n, k = len(values), selection["k"]
    if (n, k) not in MASK_CACHE:
        MASK_CACHE[n, k] = tuple(mask for mask in range(1 << n) if mask.bit_count() == k)
    solutions = []
    for mask in MASK_CACHE[n, k]:
        indices, value = [], 0
        for index, number in enumerate(values):
            if mask & (1 << index):
                indices.append(index)
                value += number
        if value == selection["target"]:
            solutions.append(tuple(indices))
    return tuple(sorted(solutions))


def oracle(messages):
    budget()
    problem, skill, rendering = parse_public(messages)
    if skill == "R":
        solutions, paths = walk_references(problem)
        independent = closure_references(problem)
        difficulty = {"n": len(problem["initial"]), "requested": len(problem["lookup"]["requested"]),
                      "first_depth": len(paths[0]["path"]), "maximum_requested_depth": max(len(p["path"]) for p in paths),
                      "total_requested_hops": sum(len(p["path"]) for p in paths), "alias_nodes": len(problem["lookup"]["links"])}
    else:
        solutions = combination_certificates(problem)
        independent = mask_certificates(problem)
        paths = None
        difficulty = {"n": len(problem["initial"]), "k": problem["selection"]["k"],
                      "target": problem["selection"]["target"], "solution_count": len(solutions)}
    require(solutions == independent, "independent public solver disagreement")
    return {"problem": problem, "skill": skill, "rendering": rendering, "solutions": solutions,
            "oracle_sha256": digest({"skill": skill, "solutions": solutions}), "difficulty": difficulty,
            "R_paths": paths}


def accepts(text, result):
    try:
        value = parse(text, 65536)
        require(type(value) is dict and set(value) == {"answer"}, "exact response key")
        answer = value["answer"]
        if type(answer) is str:
            return answer == "UNKNOWN" and not result["solutions"]
        require(type(answer) is list and answer and all(type(v) is int and 0 <= v < len(result["problem"]["initial"]) for v in answer)
                and len(set(answer)) == len(answer), "distinct typed zero-based indices")
        return tuple(sorted(answer)) in result["solutions"]
    except (ValueError, TypeError, RecursionError):
        return False


def controls(messages, result, supplied_target, private):
    original_problem = result["problem"]
    changed = deepcopy(original_problem)
    if result["skill"] == "R":
        first = changed["lookup"]["requested"][0]
        node = first
        while type(changed["lookup"]["links"][node]) is str:
            node = changed["lookup"]["links"][node]
        unused = next(index for index in range(len(changed["initial"])) if index not in result["solutions"][0])
        changed["lookup"]["links"][node] = unused
        feasible_edit = "requested_terminal_rebound_to_unrequested_index"
        impossible = deepcopy(original_problem)
        impossible["lookup"]["links"][first] = first
        impossible_edit = "requested_self_cycle"
        nuisance = deepcopy(original_problem)
        nuisance["lookup"]["links"]["audit_noise"] = "audit_noise"
        nuisance_result = oracle(replace_problem(messages, nuisance))
        require(nuisance_result["oracle_sha256"] == result["oracle_sha256"], "unrequested-cycle distractor must not alter oracle")
    else:
        n, k, target = len(changed["initial"]), changed["selection"]["k"], changed["selection"]["target"]
        other_sum = next((sum(changed["initial"][index] for index in choice) for choice in combinations(range(n), k)
                          if sum(changed["initial"][index] for index in choice) != target), None)
        require(other_sum is not None, "actual S instance lacks feasible distinct-target control")
        changed["selection"]["target"] = other_sum
        feasible_edit = "different_attainable_exact_size_target"
        impossible = deepcopy(original_problem)
        impossible["selection"]["target"] = sum(sorted(impossible["initial"], reverse=True)[:k]) + 1
        impossible_edit = "target_above_maximum_k_item_sum"
    feasible_result = oracle(replace_problem(messages, changed))
    impossible_result = oracle(replace_problem(messages, impossible))
    require(feasible_result["solutions"] and feasible_result["oracle_sha256"] != result["oracle_sha256"], "public condition must change valid answers")
    require(not impossible_result["solutions"] and accepts('{"answer":"UNKNOWN"}', impossible_result)
            and not accepts(supplied_target, impossible_result), "unsupported public condition must require UNKNOWN")
    require(not accepts(supplied_target, feasible_result), "old target must fail after answer-changing public edit")
    poison = deepcopy(private)
    poison.update({"spec": {"private_poison": True}, "target": '{"answer":"UNKNOWN"}', "skill": "PRIVATE_WRONG_LABEL",
                   "verification": {"accepted": True, "outcome": {"solutions": [[999]], "status": "solved"}}})
    poisoned_result = oracle(messages)
    require(poisoned_result["oracle_sha256"] == result["oracle_sha256"] and not accepts(poison["target"], poisoned_result),
            "private spec/target/verifier edits must not influence oracle")
    surface = deepcopy(messages)
    lines = surface[1]["content"].splitlines()
    lines[0] = INTRO[(INTRO.index(lines[0]) + 1) % 4]
    surface[1]["content"] = "\n".join(lines)
    require(oracle(surface)["oracle_sha256"] == result["oracle_sha256"], "registered surface introduction must preserve answers")
    accepted = 0
    nontrivial_order = 0
    for certificate in result["solutions"]:
        text = json.dumps({"answer": list(reversed(certificate))}, indent=2)
        require(accepts(" \n" + text + "\n ", result), "valid alternative certificate/order/JSON whitespace rejected")
        accepted += 1
        nontrivial_order += len(certificate) > 1
    invalid = ('{"answer":"UNKNOWN"}', '{"answer":[true]}', '{"answer":[0.0]}', '{"answer":[0,0]}',
               '{"answer":[]}', '{"answer":[0],"extra":1}', '{"answer":[0]} explanation', '{"answer":[0],"answer":[0]}')
    require(all(not accepts(text, result) for text in invalid), "invalid response or false UNKNOWN accepted")
    return {"feasible_public_edit": feasible_edit, "feasible_changed_oracle_sha256": feasible_result["oracle_sha256"],
            "impossible_public_edit": impossible_edit, "impossible_changed_oracle_sha256": impossible_result["oracle_sha256"],
            "private_poison_oracle_unchanged": True, "surface_oracle_unchanged": True,
            "unrequested_cycle_ignored_if_R": result["skill"] == "R",
            "alternative_certificates_reversed_whitespace_accepted": accepted,
            "nontrivial_reversed_order_certificates": nontrivial_order, "malformed_or_false_UNKNOWN_responses_refused": len(invalid),
            "control_messages_or_targets_exported_as_training_data": False}


def main():
    tracemalloc.start()
    source_before = git_identity()
    refs, documents = {}, {}
    for relative in SOURCE_FILES:
        _, refs[relative] = read_bound(relative)
    for filename, pin in DATA_PINS.items():
        raw, refs[BUNDLE + filename] = read_bound(BUNDLE + filename, pin)
        documents[filename] = [parse(line) for line in raw.splitlines() if line.strip()] if filename.endswith(".jsonl") else parse(raw)
    private_rows = documents["candidate_records.jsonl"]
    enrollment, plan = documents["enrollment.json"], documents["plan.json"]
    require(len(private_rows) == 640 and len({row["row_id"] for row in private_rows}) == 640, "complete640 distinct candidate roster")
    by_arm = {arm: [row for row in private_rows if row["arm"] == arm] for arm in ("M1", "M2")}
    pools = {arm: documents["student_pool_" + arm + ".jsonl"] for arm in by_arm}
    require(all(len(by_arm[arm]) == len(pools[arm]) == 320 for arm in by_arm), "complete320-per-arm exported roster")
    require(set(row["parent_id"] for row in private_rows) == set(enrollment["parents"]) and len(enrollment["parents"]) == 20,
            "exact20 enrolled parent coverage")
    require(enrollment["plan_sha256"] == plan["sha256"], "enrollment/plan binding")
    rows, failures = [], []
    difficulty, distinct, condition_counts = {}, {}, {}
    for arm in ("M1", "M2"):
        require([row["row_id"] for row in by_arm[arm]] == enrollment["record_ids"][arm], "exact approved record enrollment/order")
        difficulty[arm], distinct[arm], condition_counts[arm] = Counter(), set(), Counter()
        for ordinal, (public, private) in enumerate(zip(pools[arm], by_arm[arm]), 1):
            row = {"row_id": private["row_id"], "arm": arm, "export_ordinal": ordinal,
                   "parent_id": private["parent_id"], "component_id": private["component_id"], "condition": private["condition"],
                   "public_messages_sha256": digest(public["messages"]), "provided_target_sha256": digest(public["target"])}
            try:
                require(set(public) == {"messages", "target"}, "actual student export contains private sidecar fields")
                result = oracle(public["messages"])
                # Only now compare metadata, labels, private world, or supplied target.
                require(public["messages"] == private["messages"] and public["target"] == private["target"], "candidate/student export byte-content join")
                require(private["split"] == "train" and result["skill"] == private["skill"], "public atomic TRAIN skill declaration")
                require(result["problem"] == private["spec"], "private spec disagrees with independently parsed public task")
                require(accepts(public["target"], result), "supplied actual training target false under public task")
                require(result["solutions"], "UNKNOWN supervision absent from registered first atomic TRAIN dose")
                check = controls(public["messages"], result, public["target"], private)
                row.update({"status": "PASS_ACTUAL_PUBLIC_TARGET", "public_skill": result["skill"], "rendering": result["rendering"],
                            "oracle_sha256": result["oracle_sha256"], "all_certificate_count": len(result["solutions"]),
                            "certificate_set_sha256": digest(result["solutions"]), "difficulty": result["difficulty"],
                            "R_requested_chain_certificate": result["R_paths"], "controls": check,
                            "provided_target_accepted": True, "private_spec_matches_public": True})
                d = result["difficulty"]
                joint = ("R", d["n"], d["requested"], d["first_depth"], d["maximum_requested_depth"], d["total_requested_hops"]) if result["skill"] == "R" else ("S", d["n"], d["k"], d["solution_count"])
                difficulty[arm][canonical(joint)] += 1
                distinct[arm].add(digest(public))
                condition_counts[arm][private["condition"]] += 1
            except (ValueError, KeyError, TypeError, RecursionError) as error:
                row.update({"status": "BLOCKED_PUBLIC_ROW_AUDIT", "error": str(error)[:1000], "native_permission": False})
                failures.append({"row_id": row["row_id"], "arm": arm, "ordinal": ordinal, "error": str(error)[:1000]})
            rows.append(row)
    require(len(rows) == 640, "full cohort retained without dropped rows")
    stable = True
    for relative, ref in refs.items():
        _, after = read_bound(relative, ref["sha256"])
        stable &= after == ref
    source_after = git_identity()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    pass_rows = [row for row in rows if row["status"] == "PASS_ACTUAL_PUBLIC_TARGET"]
    report = {"schema": "synth-actual640-student-visible-independent-audit-v1",
        "scope": "APPROVED_ATOMIC_TRAIN_PUBLIC_MESSAGES_ONLY_NO_SFT_EXECUTION_OR_NEW_DATA_ADMISSION",
        "status": "PASS_ALL640_ACTUAL_PUBLIC_TARGETS_CPU_ONLY" if not failures and stable else "BLOCKED_ACTUAL_CORPUS_AUDIT",
        "source_before": source_before, "source_after": source_after, "file_refs": refs, "all_selected_bytes_stable": stable,
        "oracle_implementation": {"path": str(Path(__file__)), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                  "SYNTH_module_imports": [], "primary_R": "per-request functional graph walk", "independent_R": "finite directed transitive closure with active-cycle/missing-node detection",
                                  "primary_S": "all exact-size itertools combinations", "independent_S": "bitmask cardinality and arithmetic sums"},
        "roster": {"rows": 640, "arms": {arm: [row["row_id"] for row in by_arm[arm]] for arm in by_arm},
                   "enrolled_parents": enrollment["parents"], "completed_rows": len(pass_rows), "blocked_rows": len(failures)},
        "rows": rows, "failures_and_counterexamples": failures,
        "observed_counts": {"public_skills": dict(Counter(row.get("public_skill", "BLOCKED") for row in rows)),
                            "by_arm_skill": {arm: dict(Counter(row.get("public_skill", "BLOCKED") for row in rows if row["arm"] == arm)) for arm in by_arm},
                            "renderings": dict(Counter(row.get("rendering", "BLOCKED") for row in rows)),
                            "distinct_messages_plus_targets": {arm: len(distinct[arm]) for arm in distinct},
                            "condition_counts": {arm: dict(condition_counts[arm]) for arm in condition_counts},
                            "nonunique_solution_rows": sum(row.get("all_certificate_count", 0) > 1 for row in rows),
                            "alternative_certificates_checked": sum(row.get("controls", {}).get("alternative_certificates_reversed_whitespace_accepted", 0) for row in rows),
                            "nontrivial_order_certificates_checked": sum(row.get("controls", {}).get("nontrivial_reversed_order_certificates", 0) for row in rows),
                            "public_feasible_condition_controls": len(pass_rows), "public_UNKNOWN_condition_controls": len(pass_rows),
                            "private_poison_invariance_controls": len(pass_rows), "registered_surface_invariance_controls": len(pass_rows),
                            "unrequested_R_cycle_invariance_controls": sum(row.get("public_skill") == "R" for row in pass_rows),
                            "malformed_and_false_UNKNOWN_response_refusals": sum(row.get("controls", {}).get("malformed_or_false_UNKNOWN_responses_refused", 0) for row in rows)},
        "candidate_pool_joint_difficulty_histograms": {arm: dict(sorted(difficulty[arm].items())) for arm in difficulty},
        "declared_preexisting_schedule_facts_not_recomputed_here": {"sequences_per_trained_arm": 540, "updates": 27, "seed": 101,
            "M1_input_tokens": 130562, "M2_input_tokens": 130574, "M1_supervised_tokens": 5614, "M2_supervised_tokens": 5615,
            "between_arm_input_residual": 12, "S_cardinality_update_distribution_differences": "27/27", "R_first_depth_update_distribution_differences": "26/27",
            "reference": "existingCPU_DATA_MILESTONE and matched schedules; neither schedule nor token pools read by this restricted audit"},
        "interpretation": "Public gold truth for these640actual approvedexamples is independent of private labels; data-quality evidence alone does not establish learning, retention, native readiness, full difficulty equivalence, or isolated diversity causality.",
        "resource_measurement": {"per_selected_input_byte_bound": MAX_INPUT, "actual_selected_input_bytes": sum(ref["bytes"] for ref in refs.values()),
            "CPU_seconds": time.process_time() - CPU_START, "wall_seconds": time.monotonic() - WALL_START, "CPU_and_wall_budget_seconds": 300,
            "tracemalloc_peak_bytes": peak, "tracemalloc_scope": "Python allocator traces only, not RSS/process-tree peak or hard containment",
            "existing_SYNTH_512MiB_policy_unchanged": True, "native_resource_admission_licensed": False},
        "native_model_loads": 0, "native_forward_calls": 0, "native_or_HF_tokenizer_instantiated": False, "optimizer_updates": 0,
        "protected_or_DEV_or_confirmation_oracle_reads": 0, "training_data_written_or_derivatives_admitted": False,
        "native_quality_or_retention_established": False, "training_or_promotion_permission": False}
    report["sha256"] = digest(report)
    path = OUT / "SYNTH_ACTUAL640_PUBLIC_GOLD_AUDIT_001.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "status": report["status"],
                      "counts": report["observed_counts"], "resources": report["resource_measurement"], "failures": failures[:5]}, indent=2))


if __name__ == "__main__":
    main()
