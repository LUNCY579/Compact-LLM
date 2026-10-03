"""Fresh-family R/S/RS CPU DEV preparation, never model or launch authority.

Source selection is frozen before student outputs. A distinct ledger parser and
fixed-point/combination solver check both renderings against direct-walk/DP.
Policy exports contain routing IDs and messages only. All oracle information
stays in a separate, already exposed CPU DEV sidecar, not confirmation labels.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import io
from itertools import combinations
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import unittest

_HERE = Path(__file__).resolve().parent
PACKAGE = _HERE.parents[1] if _HERE.name == "eval_cpu_work" and _HERE.parent.name == "artifacts" else _HERE
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))
import rs_contract as contract
import rs_solver as primary
import rs_verifier as old_verifier
import balanced_corpus as balanced
import task_compiler as compiler

VERSION = "synth-fresh-dev-panel-2"
POLICY_PATH = _HERE / "EVALUATION_PANEL_POLICY.json"
POLICY_002_PATH = _HERE / "EVALUATION_PANEL_POLICY_002.json"
MAX_FILE_BYTES = 16 * 1024 * 1024
FIXTURE_MODULES = ("test_balanced_corpus", "test_cpu_tokenizer", "test_matching_schedule",
                   "test_prepare_corpus", "test_registration", "test_rs_semantics",
                   "test_source_policy2", "test_task_compiler")
FRAMEWORKS = {"torch", "transformers", "tensorflow", "jax", "flax"}
LEDGER_INTRO = "Interpret this instance ledger using the stated finite rules."
FINAL_RULE = "Order does not matter. There are no updates, filtering, or precedence rules."


def _no_frameworks():
    contract.require(not any(name.split(".", 1)[0] in FRAMEWORKS for name in sys.modules),
                     "CPU evaluation preparation must not contain model frameworks")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
    contract.require(len(raw) <= MAX_FILE_BYTES, "bounded CPU JSON file exceeded")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=contract._pairs,
                      parse_constant=contract._bad_constant)


def policy(source_policy=2):
    contract.require(type(source_policy) is int and source_policy in (1, 2), "registered evaluation source policy")
    result = _json(POLICY_PATH)
    contract.require(type(result.get("source_pool_seed")) is int and result["source_pool_seed"] == 2026100307 and
                     type(result.get("source_candidates_per_skill")) is int and result["source_candidates_per_skill"] == 256 and
                     type(result.get("component_quota_per_skill")) is int and result["component_quota_per_skill"] == 32 and
                     result["skills"] == ["R", "S", "RS"] and result["population"] == "EXPOSED_CPU_DEV_ONLY_NOT_CONFIRMATION" and
                     result["native_launch_authorized"] is False and result["training_authorized"] is False and
                     result["protected_bank_access"] is False and result["confirmation_eligible"] is False and
                     result["baseline_membership_enrolled"] is False,
                     "unregistered evaluation policy")
    controls = {"item_count_range": [13, 15], "reference_request_counts": [4, 5],
                "reference_standard_depth_range": [1, 5], "reference_deeper_first_depth_range": [6, 8],
                "selection_cardinality_range": [1, 3], "conditions_per_parent": 2,
                "renderings_per_condition": ["familiar", "heldout_ledger"],
                "stratum_component_quotas": {"R": {"wider_request": 16, "deeper_reference": 8, "solved_to_UNKNOWN": 8},
                    "S": {"valid_target_change": 24, "solved_to_UNKNOWN": 8},
                    "RS": {"valid_target_change": 24, "solved_to_UNKNOWN": 8}}}
    contract.require(all(contract.canonical(result.get(key)) == contract.canonical(expected)
                         for key, expected in controls.items()), "registered source strata or cell quota drift")
    if source_policy == 2:
        amendment = _json(POLICY_002_PATH)
        expected = {**result, "id": "SYNTH-FRESH-DEV-PANEL-002", "source_policy": 2,
                    "amendment": "UNIFORM_COMPACT_ALIAS_BIJECTION_BEFORE_ANY_STUDENT_DEV_OUTPUTS",
                    "predecessor_policy": "EVALUATION_PANEL_POLICY.json",
                    "predecessor_policy_sha256": _sha(POLICY_PATH),
                    "alias_renaming": "Sort the union of all alias keys, string link targets and requests across both parent cells; map bijectively to a..z,aa..az,ba.. with no RNG draws; apply to every R/RS candidate before footprints and selection.",
                    "source_identity": "Keep pre-lexical source hashes/parent IDs and separately bind compact display cells. Renaming creates no fresh families.",
                    "native_context_cap_tokens": 512, "evaluation_prompt_cap_tokens": 480,
                    "evaluation_generation_reserve_tokens": 32, "training_corpus_modified": False}
        contract.require(contract.canonical(amendment) == contract.canonical(expected), "lexical amendment or inherited policy drift")
        return amendment
    return result


def render_ledger(spec):
    sig = contract.validate_spec(spec)
    rules = [contract.R_RULE if "R" in sig else "All items are eligible; no reference lookup is needed.",
             contract.S_RULE if "S" in sig else "Return ALL resolved items, each exactly once.", FINAL_RULE]
    lines = [LEDGER_INTRO, *rules, "BEGIN INSTANCE LEDGER"]
    lines += [f"ITEM {i}: {contract.canonical(value)}" for i, value in enumerate(spec["initial"])]
    if "R" in sig:
        lines += ["LINK " + alias + ": " + contract.canonical(target)
                  for alias, target in sorted(spec["lookup"]["links"].items())]
        lines.append("REQUESTS: " + contract.canonical(spec["lookup"]["requested"]))
    if "S" in sig:
        lines.append("EXACT CHOICE: " + contract.canonical({"count": spec["selection"]["k"],
                                                           "sum": spec["selection"]["target"]}))
    lines.append("END INSTANCE LEDGER")
    text = "\n".join(lines)
    contract.require(contract.utf8_size(text) <= contract.MAX_BYTES, "overlength ledger")
    return [{"role": "system", "content": contract.SYSTEM}, {"role": "user", "content": text}]


def parse_ledger(messages):
    """Independently reconstruct the public spec from literal typed ledger lines."""
    contract.require(type(messages) is list and len(messages) == 2 and
                     messages[0] == {"role": "system", "content": contract.SYSTEM} and
                     type(messages[1]) is dict and set(messages[1]) == {"role", "content"} and
                     messages[1]["role"] == "user" and type(messages[1]["content"]) is str,
                     "ledger policy boundary")
    text = messages[1]["content"]
    contract.require(contract.utf8_size(text) <= contract.MAX_BYTES, "overlength ledger")
    lines = text.splitlines()
    contract.require(len(lines) >= 12 and lines[0] == LEDGER_INTRO and
                     lines[4] == "BEGIN INSTANCE LEDGER" and lines[-1] == "END INSTANCE LEDGER", "ledger frame")
    values, links, requested, selection = [], {}, None, None
    phase = "items"
    for line in lines[5:-1]:
        if line.startswith("ITEM "):
            contract.require(phase == "items", "item order")
            left, raw = line[5:].split(": ", 1)
            contract.require(left == str(len(values)), "duplicate/noncanonical item index")
            values.append(contract.strict_json(raw))
        elif line.startswith("LINK "):
            contract.require(phase in ("items", "links"), "link order")
            phase = "links"
            alias, raw = line[5:].split(": ", 1)
            contract.require(alias not in links, "duplicate ledger link")
            links[alias] = contract.strict_json(raw)
        elif line.startswith("REQUESTS: "):
            contract.require(phase == "links" and requested is None, "requested boundary")
            phase, requested = "requested", contract.strict_json(line[len("REQUESTS: "):])
        elif line.startswith("EXACT CHOICE: "):
            contract.require(phase in ("items", "requested") and selection is None, "choice order")
            phase = "choice"
            choice = contract.strict_json(line[len("EXACT CHOICE: "):])
            contract.require(type(choice) is dict and set(choice) == {"count", "sum"}, "choice fields")
            selection = {"k": choice["count"], "target": choice["sum"]}
        else:
            raise ValueError("unrecognized ledger line")
    spec = {"initial": values}
    if links:
        contract.require(requested is not None, "links without requests")
        spec["lookup"] = {"links": links, "requested": requested}
    if selection is not None:
        spec["selection"] = selection
    contract.validate_spec(spec)
    expected_rules = [contract.R_RULE if "lookup" in spec else "All items are eligible; no reference lookup is needed.",
                      contract.S_RULE if "selection" in spec else "Return ALL resolved items, each exactly once.",
                      "Order does not matter. There are no updates, filtering, or precedence rules."]
    contract.require(lines[1:4] == expected_rules, "ledger rule drift")
    contract.require(list(links) == sorted(links), "noncanonical ledger link order")
    return spec


def parse_visible(messages):
    if type(messages) is list and len(messages) == 2 and type(messages[1]) is dict and \
            type(messages[1].get("content")) is str and messages[1]["content"].startswith(LEDGER_INTRO):
        return parse_ledger(messages)
    return contract.parse_visible(messages)


def solve_visible(messages):
    """Fixed-point alias propagation and combination enumeration, no DP/walk call."""
    spec = parse_visible(messages)
    eligible = set(range(len(spec["initial"])))
    if "lookup" in spec:
        links = spec["lookup"]["links"]
        terminals = {alias: target for alias, target in links.items() if type(target) is int}
        while True:
            additions = {alias: terminals[target] for alias, target in links.items()
                         if alias not in terminals and type(target) is str and target in terminals}
            if not additions: break
            terminals.update(additions)
        if any(alias not in terminals for alias in spec["lookup"]["requested"]):
            return {"status": "insufficient_evidence", "solutions": []}
        eligible = {terminals[alias] for alias in spec["lookup"]["requested"]}
    if "selection" not in spec:
        return {"status": "solved", "solutions": [sorted(eligible)]}
    selection = spec["selection"]
    solutions = [list(indices) for indices in combinations(sorted(eligible), selection["k"])
                 if sum(spec["initial"][i] for i in indices) == selection["target"]]
    return {"status": "solved" if solutions else "impossible", "solutions": solutions}


def verify_response(messages, response):
    try:
        spec = parse_visible(messages)
        outcome = solve_visible(messages)
        answer = contract.parse_answer(response, len(spec["initial"]))
    except (ValueError, TypeError, KeyError, IndexError, RecursionError):
        return {"accepted": False, "reason": "invalid_prompt_or_answer"}
    accepted = not outcome["solutions"] if answer == "UNKNOWN" else answer in outcome["solutions"]
    return {"accepted": accepted, "reason": "verified" if accepted else "incorrect_answer"}


def footprints(spec):
    """Conservative existing family screens, plus cross-skill atomic projections."""
    contract.validate_spec(spec)
    result = {"structural:" + compiler.structural_key(spec)}
    try:
        result.add("balanced:" + balanced.family_key(spec))
    except ValueError:
        # Public structural_key covers missing/cyclic requested chains. These
        # are never allowed to erase the solved sibling's balanced footprint.
        pass
    if "lookup" in spec and "selection" in spec:
        result |= footprints({"initial": spec["initial"], "lookup": spec["lookup"]})
        result |= footprints({"initial": spec["initial"], "selection": spec["selection"]})
    return result


def _walk_specs(value):
    if type(value) is dict:
        if "initial" in value:
            try:
                contract.validate_spec(value)
                yield value
            except ValueError:
                pass
        if "messages" in value:
            # A student-only export may lack a raw spec. Recover its public
            # fields from the visible parser rather than ignore that exposure.
            yield parse_visible(value["messages"])
        for part in value.values(): yield from _walk_specs(part)
    elif type(value) is list:
        for part in value: yield from _walk_specs(part)


def _fixture_census():
    """Private isolated subprocess entry: observe every valid public CPU fixture."""
    _no_frameworks()
    original = contract.validate_spec
    observed = set()
    def watch(spec):
        sig = original(spec)
        observed.add(contract.canonical(spec))
        return sig
    for module in list(sys.modules.values()):
        if getattr(module, "validate_spec", None) is original:
            module.validate_spec = watch
    sys.path.insert(0, str(PACKAGE / "tests"))
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in FIXTURE_MODULES)
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=0).run(suite)
    contract.validate_spec = original
    # All imported aliases must be restored before computing fingerprints.
    for module in list(sys.modules.values()):
        if getattr(module, "validate_spec", None) is watch: module.validate_spec = original
    _no_frameworks()
    contract.require(result.wasSuccessful() and not result.skipped, "CPU fixture census failed/skipped: " + output.getvalue()[-5000:])
    keys = set()
    for raw in observed: keys |= footprints(contract.strict_json(raw))
    return {"fixture_tests": result.testsRun, "fixture_specs": len(observed), "footprints": sorted(keys)}


def _materialized_files():
    literal_root = PACKAGE / "data"
    root = literal_root.resolve()
    contract.require(not literal_root.is_symlink() and root == PACKAGE.resolve() / "data", "owned CPU data root")
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl"))
    contract.require(0 < len(paths) <= 256, "bounded materialized CPU inventory required")
    contract.require(all(not p.is_symlink() and root in p.resolve().parents for p in paths),
                     "redirected CPU data file")
    return paths


def _fixture_sources():
    paths = [PACKAGE / name for name in ("rs_contract.py", "rs_solver.py", "rs_verifier.py",
             "task_compiler.py", "balanced_corpus.py", "cpu_tokenizer.py", "matching_schedule.py", "prepare_corpus.py")]
    paths += [PACKAGE / "tests" / (name + ".py") for name in FIXTURE_MODULES]
    paths += [PACKAGE.parent / "post_training_mimo_centralized_v1/receipts/titan_static_v1/PRIMARY_STUDENT_RECEIPT.json",
              Path(__file__), POLICY_PATH]
    return paths


def collect_repository_exclusions():
    """Read only package CPU data and source-bound CPU tests; no protected banks."""
    _no_frameworks()
    root = (PACKAGE / "data").resolve()
    pins, keys, specs = {}, set(), 0
    paths = _materialized_files()
    for path in paths:
        contract.require(not path.is_symlink() and root in path.resolve().parents, "redirected CPU data file")
        pins[str(path.resolve())] = _sha(path)
        if path.suffix == ".json":
            documents = [_json(path)]
        else:
            with path.open("rb") as stream:
                raw = stream.read(MAX_FILE_BYTES + 1)
            contract.require(len(raw) <= MAX_FILE_BYTES, "CPU JSONL bound")
            documents = [contract.strict_json(line.decode()) for line in raw.splitlines() if line]
        for document in documents:
            for spec in _walk_specs(document):
                keys |= footprints(spec)
                specs += 1
    source_paths = _fixture_sources()
    source_pins = {str(path.resolve()): _sha(path) for path in source_paths}
    code = "import sys;sys.path.insert(0,sys.argv[1]);import evaluation_panel as e;print(e.contract.canonical(e._fixture_census()))"
    static_path = PACKAGE.parent / "post_training_mimo_centralized_v1/receipts/titan_static_v1/PRIMARY_STUDENT_RECEIPT.json"
    static = _json(static_path)
    # These two opt-in tests use the already pinned local CPU tokenizer only;
    # no framework/model or GPU can be imported by this census.
    child_env = dict(os.environ, SYNTH_CRS_LOCAL_TOKENIZER_DIR=static["checkpoint_path"],
                     CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    source_pins[str(static_path.resolve())] = _sha(static_path)
    child = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(_HERE)],
                           text=True, capture_output=True, timeout=120, env=child_env)
    contract.require(child.returncode == 0 and len(child.stdout.encode()) <= MAX_FILE_BYTES,
                     "CPU fixture census did not complete: " + child.stderr[-2000:])
    census = json.loads(child.stdout, object_pairs_hook=contract._pairs, parse_constant=contract._bad_constant)
    keys.update(census["footprints"])
    contract.require({str(p.resolve()) for p in _materialized_files()} == set(pins),
                     "CPU materialized inventory changed during census")
    for path, expected in {**pins, **source_pins}.items():
        contract.require(_sha(path) == expected, "CPU exclusion source changed during census")
    result = {"scope": "all_package_materialized_CPU_data_and_source_bound_fixture_census",
              "file_pins": pins, "source_pins": source_pins, "materialized_spec_occurrences": specs,
              "fixture_tests": census["fixture_tests"], "fixture_specs": census["fixture_specs"],
              "footprints": sorted(keys), "protected_banks_read": False, "model_outputs_used": False}
    result["sha256"] = contract.digest(result)
    return result


def _validate_exclusions(value):
    contract.require(type(value) is dict and value.get("sha256") == contract.digest({k: v for k, v in value.items() if k != "sha256"}), "exclusion binding")
    contract.require(value.get("scope") in ("all_package_materialized_CPU_data_and_source_bound_fixture_census", "TEST_ONLY_NOT_REPOSITORY_ADMISSION") and
                     type(value.get("footprints")) is list and
                     all(type(key) is str and re.fullmatch(r"(structural|balanced):[0-9a-f]{64}", key)
                         for key in value["footprints"]) and value["footprints"] == sorted(set(value["footprints"])),
                     "exclusion footprint inventory")
    if value["scope"] != "TEST_ONLY_NOT_REPOSITORY_ADMISSION":
        contract.require(type(value.get("file_pins")) is dict and bool(value["file_pins"]) and
                         type(value.get("source_pins")) is dict and bool(value["source_pins"]) and
                         type(value.get("fixture_tests")) is int and value["fixture_tests"] > 0 and
                         type(value.get("fixture_specs")) is int and value["fixture_specs"] > 0 and
                         value.get("protected_banks_read") is False and value.get("model_outputs_used") is False,
                         "repository fixture/data evidence missing")
        contract.require(set(value["file_pins"]) == {str(p.resolve()) for p in _materialized_files()},
                         "CPU materialized inventory closure changed")
        contract.require(set(value["source_pins"]) == {str(p.resolve()) for p in _fixture_sources()},
                         "CPU fixture source closure changed")
        for path, expected in {**value["file_pins"], **value["source_pins"]}.items():
            contract.require(_sha(path) == expected, "exclusion-bound source changed")


def _candidate(rng, skill, index):
    n = rng.randint(13, 15)
    values = [rng.randint(-100, 100) for _ in range(n)]
    values[0], values[1] = rng.randint(61, 100), rng.randint(-100, -61)
    spec = {"initial": values}
    stratum = "valid_target_change" if index % 4 != 3 else "solved_to_UNKNOWN"
    if "R" in skill:
        if skill == "R" and index % 4 in (0, 1): stratum = "wider_request"
        if skill == "R" and index % 4 == 2: stratum = "deeper_reference"
        count = rng.choice((4, 5))
        endpoints = rng.sample(range(n), count + 1)
        links = {}
        for branch in range(count):
            depth = rng.randint(6, 8) if branch == 0 and stratum == "deeper_reference" else rng.randint(1, 5)
            for hop in range(depth):
                links[f"r{branch}_{hop}"] = endpoints[branch] if hop == depth - 1 else f"r{branch}_{hop + 1}"
        spec["lookup"] = {"links": links, "requested": [f"r{i}_0" for i in range(count)]}
    if "S" in skill:
        eligible = primary.solve({"initial": values, "lookup": spec["lookup"]})["solutions"][0] if skill == "RS" else list(range(n))
        k = rng.randint(1, min(3, len(eligible)))
        if skill == "RS":
            outside = [i for i in range(n) if i not in eligible]
            for source, destination in zip(eligible, outside): values[destination] = values[source]
        attainable = sorted({sum(values[i] for i in chosen) for chosen in combinations(eligible, k)})
        contract.require(len(attainable) >= 2, "no answer-changing target pair")
        spec["selection"] = {"k": k, "target": attainable[len(attainable) // 3]}
        altered = copy.deepcopy(spec)
        if stratum == "solved_to_UNKNOWN":
            if skill == "RS":
                unfiltered = sorted({sum(values[i] for i in chosen) for chosen in combinations(range(n), k)} - set(attainable))
                contract.require(bool(unfiltered), "RS lacks unfiltered/eligible counterexample")
                changed_target = unfiltered[len(unfiltered) // 2]
            else:
                changed_target = attainable[-1] + 1
        else:
            changed_target = next(value for value in reversed(attainable) if value != spec["selection"]["target"])
        altered["selection"]["target"] = changed_target
    else:
        altered = copy.deepcopy(spec)
        node = spec["lookup"]["requested"][0]
        while type(altered["lookup"]["links"][node]) is str: node = altered["lookup"]["links"][node]
        if stratum == "solved_to_UNKNOWN":
            if index % 8 == 3: del altered["lookup"]["links"][node]
            else: altered["lookup"]["links"][node] = node
        else:
            altered["lookup"]["links"][node] = endpoints[-1]
    if skill == "RS":
        for world in (spec, altered):
            restricted = primary.solve(world)["solutions"]
            unrestricted = primary.solve({"initial": values, "selection": world["selection"]})["solutions"]
            contract.require(any(answer not in restricted for answer in unrestricted), "RS reference step is vacuous")
    return {"skill": skill, "stratum": stratum, "candidate_index": index, "cells": [spec, altered]}


def _compact_alias_name(index):
    contract.require(type(index) is int and index >= 0, "alias ordinal")
    letters = []
    while index >= 0:
        letters.append(chr(ord("a") + index % 26))
        index = index // 26 - 1
    return "".join(reversed(letters))


def compact_alias_cells(cells):
    """One parent-wide bijection includes missing links, requests and cycles.

    Both conditions receive the same names. This pure representation change
    draws no random numbers and preserves indices, initial values and selection.
    """
    contract.require(type(cells) is list and len(cells) == 2, "two parent cells required")
    aliases = set()
    for cell in cells:
        contract.validate_spec(cell)
        if "lookup" in cell:
            lookup = cell["lookup"]
            aliases.update(lookup["links"])
            aliases.update(target for target in lookup["links"].values() if type(target) is str)
            aliases.update(lookup["requested"])
    mapping = {name: _compact_alias_name(index) for index, name in enumerate(sorted(aliases))}
    contract.require(len(mapping) == len(set(mapping.values())) and all(contract.alias(name) for name in mapping.values()), "compact alias bijection")
    result = copy.deepcopy(cells)
    for cell in result:
        if "lookup" in cell:
            lookup = cell["lookup"]
            lookup["links"] = {mapping[name]: mapping[target] if type(target) is str else target
                               for name, target in lookup["links"].items()}
            lookup["requested"] = [mapping[name] for name in lookup["requested"]]
        contract.validate_spec(cell)
    return result, mapping


def _pool(frozen):
    rng = random.Random(frozen["source_pool_seed"])
    candidates, attempts = [], []
    for skill in frozen["skills"]:
        for index in range(frozen["source_candidates_per_skill"]):
            try:
                candidate = _candidate(rng, skill, index)
            except ValueError as error:
                attempts.append({"skill": skill, "index": index, "accepted": False, "reason": str(error)})
                continue
            # The source family was declared by policy001; compact strings are
            # a view of that same source, with a separate display digest.
            original_hash = contract.digest(candidate)
            if frozen.get("source_policy", 1) == 2:
                original_cells_digest = contract.digest(candidate["cells"])
                candidate["cells"], mapping = compact_alias_cells(candidate["cells"])
                candidate["alias_bijection"] = mapping
                candidate["prelexical_cells_sha256"] = original_cells_digest
                candidate["display_cells_sha256"] = contract.digest(candidate["cells"])
            candidate["source_hash"] = original_hash
            candidate["parent_id"] = "dev_" + candidate["source_hash"][:24]
            candidate["footprints"] = sorted(set().union(*(footprints(cell) for cell in candidate["cells"])))
            candidates.append(candidate)
            attempts.append({"skill": skill, "index": index, "accepted": True, "source_hash": candidate["source_hash"]})
    return candidates, attempts


def compile_panel(exclusions, source_policy=2):
    _no_frameworks()
    _validate_exclusions(exclusions)
    frozen = policy(source_policy)
    candidates, attempts = _pool(frozen)
    leaders = {candidate["parent_id"]: candidate["parent_id"] for candidate in candidates}
    owners = {}
    def leader(pid):
        while leaders[pid] != pid:
            leaders[pid] = leaders[leaders[pid]]
            pid = leaders[pid]
        return pid
    for candidate in candidates:
        for token in candidate["footprints"] + ["source:" + candidate["source_hash"]]:
            if token in owners:
                a, b = sorted((leader(candidate["parent_id"]), leader(owners[token])))
                leaders[b] = a
            else: owners[token] = candidate["parent_id"]
    groups = defaultdict(list)
    for candidate in candidates: groups[leader(candidate["parent_id"])].append(candidate)
    excluded = set(exclusions["footprints"])
    components = {}
    for members in groups.values():
        footprint = sorted(set().union(*(set(member["footprints"]) for member in members)))
        component_id = contract.digest({"sources": sorted(member["source_hash"] for member in members), "footprints": footprint})
        components[component_id] = {"members": [member["parent_id"] for member in members],
                                    "footprints": footprint, "excluded": bool(excluded.intersection(footprint))}
        for member in members: member["component_id"] = component_id
    selected, used, shortages = [], set(), []
    for skill in frozen["skills"]:
        for stratum, quota in frozen["stratum_component_quotas"][skill].items():
            count = 0
            for candidate in candidates:
                component_id = candidate["component_id"]
                if candidate["skill"] == skill and candidate["stratum"] == stratum and component_id not in used and not components[component_id]["excluded"]:
                    selected.append(candidate)
                    used.add(component_id)
                    count += 1
                    if count == quota: break
            if count != quota: shortages.append({"skill": skill, "stratum": stratum, "required": quota, "found": count})
    panel = {"version": VERSION, "population": frozen["population"], "policy": frozen,
             "policy_sha256": contract.digest(frozen), "exclusions": copy.deepcopy(exclusions),
             "generator_and_solver_source_pins": {str(path.resolve()): _sha(path) for path in
                 ((Path(__file__), POLICY_PATH) + ((POLICY_002_PATH,) if source_policy == 2 else ()) + (Path(contract.__file__), Path(primary.__file__),
                  Path(old_verifier.__file__), Path(compiler.__file__), Path(balanced.__file__)))},
             "pool_sha256": contract.digest(candidates), "attempts": attempts, "components": components,
             "selected_parents": selected if not shortages else [], "shortages": shortages,
             "status": "BLOCKED_FRESH_FAMILY_QUOTAS" if shortages else "CPU_DEV_PANEL_COMPILED_NATIVE_UNADMITTED",
             "native_launch_authorized": False, "training_authorized": False, "confirmation_eligible": False,
             "baseline_membership_enrolled": False}
    panel["sha256"] = contract.digest(panel)
    return panel


def validate_panel(panel):
    contract.require(type(panel) is dict and "exclusions" in panel, "panel object")
    expected = compile_panel(panel["exclusions"], panel.get("policy", {}).get("source_policy", 1))
    contract.require(contract.canonical(panel) == contract.canonical(expected), "panel/selection/source/footprint drift")
    contract.require(panel["status"] == "CPU_DEV_PANEL_COMPILED_NATIVE_UNADMITTED", "fresh-family quotas blocked")


def audit_alias_amendment(exclusions):
    """Check all768 sources, not only the selected96, before CPU token fitting."""
    original, old_attempts = _pool(policy(1))
    compact, new_attempts = _pool(policy(2))
    contract.require(len(original) == len(compact) == 768 and old_attempts == new_attempts,
                     "lexical amendment changed source-pool attempts or identities")
    checked_cells = 0
    for before, after in zip(original, compact):
        contract.require((before["skill"], before["stratum"], before["candidate_index"], before["source_hash"], before["parent_id"])
                         == (after["skill"], after["stratum"], after["candidate_index"], after["source_hash"], after["parent_id"]),
                         "lexical amendment changed candidate identity or stratum")
        expected, mapping = compact_alias_cells(before["cells"])
        contract.require(after["cells"] == expected and after["alias_bijection"] == mapping and
                         before["footprints"] == after["footprints"], "source bijection/family footprint changed")
        for old_cell, new_cell in zip(before["cells"], after["cells"]):
            contract.require({key: value for key, value in old_cell.items() if key != "lookup"} ==
                             {key: value for key, value in new_cell.items() if key != "lookup"},
                             "lexical amendment changed non-reference task data")
            if "lookup" in old_cell:
                inverse = {compact_name: original_name for original_name, compact_name in mapping.items()}
                restored_lookup = {"links": {inverse[name]: inverse[target] if type(target) is str else target
                                             for name, target in new_cell["lookup"]["links"].items()},
                                   "requested": [inverse[name] for name in new_cell["lookup"]["requested"]]}
                contract.require(restored_lookup == old_cell["lookup"], "compact alias graph is not the original bijective graph")
            old_solution, new_solution = primary.solve(old_cell), primary.solve(new_cell)
            contract.require(old_solution == new_solution and footprints(old_cell) == footprints(new_cell),
                             "lexical amendment changed cell answers or projection family")
            for renderer in (lambda spec: contract.render(spec, 0), render_ledger):
                old_visible, new_visible = solve_visible(renderer(old_cell)), solve_visible(renderer(new_cell))
                contract.require(old_visible == new_visible and new_visible["status"] == new_solution["status"] and
                                 new_visible["solutions"] == new_solution["solutions"], "independent visible solver changed under renaming")
            checked_cells += 1
    old_panel, new_panel = compile_panel(exclusions, 1), compile_panel(exclusions, 2)
    contract.require(old_panel["components"] == new_panel["components"] and old_panel["shortages"] == new_panel["shortages"],
                     "lexical amendment changed joined-component membership")
    old_selection = [(item["skill"], item["stratum"], item["candidate_index"], item["parent_id"], item["component_id"])
                     for item in old_panel["selected_parents"]]
    new_selection = [(item["skill"], item["stratum"], item["candidate_index"], item["parent_id"], item["component_id"])
                     for item in new_panel["selected_parents"]]
    contract.require(old_selection == new_selection, "lexical amendment changed selection")
    return {"status": "CPU_LEXICAL_AMENDMENT_INVARIANCE_PASSED_NOT_NATIVE_ADMISSION",
            "source_candidates": 768, "checked_cells": checked_cells,
            "independent_visible_comparisons": checked_cells * 2,
            "solution_sets_and_statuses_unchanged": True, "full_and_atomic_projection_footprints_unchanged": True,
            "source_hashes_and_parent_ids_unchanged": True, "joined_component_members_unchanged": True,
            "selected_candidate_indices_and_strata_unchanged": True, "selected_components": len(new_selection),
            "old_policy_sha256": contract.digest(policy(1)), "new_policy_sha256": contract.digest(policy(2)),
            "new_panel_sha256": new_panel["sha256"], "native_launch_authorized": False,
            "student_DEV_outputs_used": False, "training_corpus_modified": False}


def export_panel(panel):
    """Return separate policy and CPU-exposed oracle documents; no file writes."""
    validate_panel(panel)
    inputs, oracle = [], []
    for parent in panel["selected_parents"]:
        answers = []
        for condition, spec in enumerate(parent["cells"]):
            first = primary.solve(spec)
            answers.append({tuple(solution) for solution in first["solutions"]})
            for rendering in panel["policy"]["renderings_per_condition"]:
                messages = contract.render(spec, 0) if rendering == "familiar" else render_ledger(spec)
                visible = solve_visible(messages)
                contract.require(parse_visible(messages) == spec and first["status"] == visible["status"] and
                                 first["solutions"] == visible["solutions"], "independent visible solution disagreement")
                row_id = contract.digest({"panel": panel["sha256"], "parent": parent["parent_id"],
                                           "condition": condition, "rendering": rendering})
                row = {"row_id": row_id, "messages": messages, "prompt_sha256": contract.digest(messages)}
                inputs.append(row)
                oracle.append({"row_id": row_id, "parent_id": parent["parent_id"], "component_id": parent["component_id"],
                               "skill": parent["skill"], "stratum": parent["stratum"], "condition": condition,
                               "rendering": rendering, "spec": copy.deepcopy(spec), "outcome": visible,
                               "prompt_sha256": row["prompt_sha256"], "source_hash": parent["source_hash"]})
        contract.require(bool(answers[0]) and answers[0].isdisjoint(answers[1]), "condition cells lack answer change")
    header = {"panel_sha256": panel["sha256"], "population": panel["population"],
              "native_launch_authorized": False, "confirmation_eligible": False}
    return {**header, "rows": inputs}, {**header, "oracle_labels_exposed_CPU_DEV": True, "rows": oracle}


def audit_exports(panel, inputs, oracle):
    expected_inputs, expected_oracle = export_panel(panel)
    contract.require(contract.canonical(inputs) == contract.canonical(expected_inputs) and
                     contract.canonical(oracle) == contract.canonical(expected_oracle), "policy/oracle export drift or leakage")
    for row in inputs["rows"]:
        contract.require(set(row) == {"row_id", "messages", "prompt_sha256"}, "oracle metadata in policy row")
    counts = Counter(row["skill"] for row in oracle["rows"])
    components = {row["component_id"] for row in oracle["rows"]}
    contract.require(counts == {"R": 128, "S": 128, "RS": 128} and len(components) == 96, "panel quotas/cells/components")
    negative_count = 0
    for policy_row, oracle_row in zip(inputs["rows"], oracle["rows"]):
        n = len(oracle_row["spec"]["initial"])
        correct = oracle_row["outcome"]["solutions"]
        target = contract.format_answer(correct[0] if correct else "UNKNOWN")
        contract.require(verify_response(policy_row["messages"], target)["accepted"], "oracle certificate rejected")
        wrong = ['{"answer":[]}', '{"answer":[0,0]}', '{"answer":[0],"oracle":true}']
        wrong.append('{"answer":"UNKNOWN"}' if correct else '{"answer":[0]}')
        for response in wrong:
            contract.require(not verify_response(policy_row["messages"], response)["accepted"], "deliberately wrong certificate accepted")
            negative_count += 1
    return {"status": "CPU_DEV_PANEL_AUDITED_NOT_NATIVE_ADMISSION", "rows": len(inputs["rows"]),
            "components_per_skill": {skill: 32 for skill in counts}, "independent_solution_agreements": len(inputs["rows"]),
            "negative_certificates_rejected": negative_count, "condition_change_parent_pairs": 96,
            "oracle_fields_in_student_rows": False, "heldout_rendering": "heldout_ledger",
            "prior_correct_retention_measured": False, "transfer_measured": False,
            "exclusion_scope": panel["exclusions"]["scope"], "tokenizer_fit_qualified": False,
            "native_launch_authorized": False, "confirmation_eligible": False}


def write_bundle(panel, directory):
    """Exclusively save separate policy/oracle packets under owned CPU artifacts.

    The oracle directory is a workflow separation, not a cryptographic seal.
    Labels are explicitly exposed DEV; no confirmation population is created.
    """
    _no_frameworks()
    target = Path(directory).absolute()
    root = PACKAGE / "artifacts" / "evaluation_dev"
    contract.require(not (PACKAGE / "artifacts").is_symlink() and not root.is_symlink() and
                     PACKAGE.resolve() in root.resolve().parents, "redirected owned artifact root")
    contract.require(target.parent.resolve() == root.resolve() and not target.is_symlink() and
                     not target.exists(), "new direct owned evaluation_dev artifact leaf required")
    inputs, oracle = export_panel(panel)
    audit = audit_exports(panel, inputs, oracle)
    root.mkdir(parents=True, exist_ok=True)
    contract.require(root.resolve().parent == (PACKAGE / "artifacts").resolve(), "redirected CPU evaluation root")
    target.mkdir()
    (target / "policy").mkdir()
    (target / "oracle_CPU_DEV_exposed").mkdir()
    files = {"panel.json": panel, "audit.json": audit, "policy/prompts.json": inputs,
             "oracle_CPU_DEV_exposed/labels.json": oracle}
    manifest = {"population": panel["population"], "native_launch_authorized": False,
                "confirmation_eligible": False, "oracle_labels_exposed_CPU_DEV": True, "files": {}}
    for name, value in files.items():
        raw = (contract.canonical(value) + "\n").encode("ascii")
        contract.require(len(raw) <= MAX_FILE_BYTES, "CPU export packet size bound")
        with (target / name).open("xb", buffering=0) as stream:
            contract.require(stream.write(raw) == len(raw), "short CPU export write")
        manifest["files"][name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    with (target / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(contract.canonical(manifest) + "\n")
    return manifest
