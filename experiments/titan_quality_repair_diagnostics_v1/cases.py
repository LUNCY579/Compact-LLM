"""Training/development input diagnostics only; no workers, models, or bank I/O.

This sibling leaves the reviewed quality package and its prospective native run
unchanged. Findings concern deterministic input construction, never model effects.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

REPAIR_FAMILIES = ("ledger", "register", "answerability", "grounded_qa")
STYLES = ("canonical", "native_a", "native_b", "natural")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _validate(row):
    required = {"row_id", "world_id", "family", "variant_id", "split", "operative_key",
                "payload", "strata", "answerable", "expected"}
    if not isinstance(row, dict) or not required <= row.keys():
        raise ValueError("STOP_CASE_AUDIT_INCOMPLETE_ROW")
    if row["split"] not in {"train", "replay", "dev", "fixture"}:
        raise ValueError("STOP_CASE_AUDIT_TRAINING_DEVELOPMENT_ONLY")
    if row["variant_id"] not in STYLES or type(row["answerable"]) is not bool:
        raise ValueError("STOP_CASE_AUDIT_STYLE_OR_CLASS")
    if row["payload"].get("expected_kind") == "integer" and row["answerable"] and type(row["expected"]) is not int:
        raise ValueError("STOP_CASE_AUDIT_TYPED_INTEGER_GOLD")
    return row


def _sign(value):
    if type(value) is not int:
        return "not_integer"
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def _scope(row):
    if row.get("is_replay") or row.get("replay") or row["split"] == "replay":
        return "historical_replay"
    if row["split"] == "dev":
        return "development"
    return "fresh_repair" if row["family"] in REPAIR_FAMILIES else "other_training"


def _case(row):
    _validate(row)
    payload, family = row["payload"], row["family"]
    event = payload.get("event", {})
    sign = row["strata"].get("operand_sign", "not_declared")
    branch, margin = "not_conditional", None
    if family in {"ledger", "register"}:
        value = event.get("amount") if family == "ledger" else event.get("value")
        sign = _sign(value)
        if family == "register" and event.get("kind") == "conditional":
            margin = payload["before"][event["other"]] - event["threshold"]
            branch = "true" if margin > 0 else "false"
            declared = row["strata"].get("conditional_branch")
            if declared is not None and declared is not (margin > 0):
                raise ValueError("STOP_CASE_AUDIT_BRANCH_METADATA_MISMATCH")
    return {"scope": _scope(row), "family": family, "style": row["variant_id"],
            "event_kind": event.get("kind", "not_transition"), "operand_sign": sign,
            "branch": branch, "branch_margin": margin, "answerable": row["answerable"],
            "scenario": row["strata"].get("scenario", "not_declared"),
            "supported_numeric_gold_sign": _sign(row["expected"]) if row["answerable"] else "unanswerable",
            "support_expansion": row.get("support_expansion"), "pair_id": row.get("pair_id"),
            "world_id": row["world_id"], "operative_key": row["operative_key"], "row_id": row["row_id"]}


def _table(records, columns):
    counts = Counter(canonical({key: record[key] for key in columns}) for record in records)
    return [{**json.loads(key), "count": counts[key]} for key in sorted(counts)]


def _zero_cells(records, expected_styles):
    """Explicit absent sign/branch/gold cells, including intentional canonical arms."""
    fresh = [record for record in records if record["scope"] == "fresh_repair"]
    absent = []
    for family in REPAIR_FAMILIES:
        group = [record for record in fresh if record["family"] == family]
        if not group:
            continue
        for style in expected_styles:
            for sign in ("positive", "negative"):
                if not any(r["style"] == style and r["operand_sign"] == sign for r in group):
                    absent.append({"family": family, "style": style, "dimension": "operand_sign", "missing": sign})
            if family == "register":
                for branch in ("true", "false"):
                    conditional = [r for r in group if r["event_kind"] == "conditional"]
                    if conditional and not any(r["style"] == style and r["branch"] == branch for r in conditional):
                        absent.append({"family": family, "style": style, "dimension": "conditional_branch", "missing": branch})
                for sign in ("positive", "negative"):
                    for branch in ("true", "false"):
                        conditional = [r for r in group if r["event_kind"] == "conditional"]
                        if conditional and not any(r["style"] == style and r["operand_sign"] == sign and r["branch"] == branch for r in conditional):
                            absent.append({"family": family, "style": style, "dimension": "conditional_sign_branch", "missing": [sign, branch]})
            if family == "answerability":
                for sign in ("positive", "negative"):
                    if not any(r["style"] == style and r["supported_numeric_gold_sign"] == sign for r in group):
                        absent.append({"family": family, "style": style, "dimension": "supported_numeric_gold_sign", "missing": sign})
    return absent


def summarize_rows(rows, recipe=None):
    records = [_case(row) for row in rows]
    if not records:
        raise ValueError("STOP_CASE_AUDIT_EMPTY_INPUT")
    expected_styles = STYLES if recipe and recipe.get("prompt_coverage") == "diverse" else (
        ("canonical",) if recipe else tuple(style for style in STYLES if any(r["style"] == style for r in records)))
    worlds, pairs = defaultdict(list), defaultdict(list)
    for record in records:
        worlds[(record["scope"], record["family"], record["world_id"])].append(record)
        if record["pair_id"]:
            pairs[(record["scope"], record["family"], record["pair_id"])].append(record)
    world_records = [{"scope": scope, "family": family, "rows": len(group),
                      "distinct_styles": len({r["style"] for r in group}),
                      "distinct_operative_keys": len({r["operative_key"] for r in group})}
                     for (scope, family, _), group in worlds.items()]
    pair_records = [{"scope": scope, "family": family, "rows": len(group),
                     "distinct_world_ids": len({r["world_id"] for r in group}),
                     "distinct_styles": len({r["style"] for r in group}),
                     "distinct_operative_keys": len({r["operative_key"] for r in group}),
                     "answerable_rows": sum(r["answerable"] for r in group)}
                    for (scope, family, _), group in pairs.items()]
    return {"input_rows": len(rows), "model_outcomes_read": False, "native_effects": "UNMEASURED",
            "by_family_style": _table(records, ("scope", "family", "style")),
            "transition_joint_cases": _table([r for r in records if r["family"] in {"ledger", "register"}],
                ("scope", "family", "style", "event_kind", "operand_sign", "branch", "support_expansion")),
            "conditional_margins": _table([r for r in records if r["branch_margin"] is not None],
                ("scope", "style", "branch", "branch_margin")),
            "facts_style_scenario_support": _table([r for r in records if r["family"] in {"answerability", "grounded_qa"}],
                ("scope", "family", "style", "scenario", "answerable", "operand_sign", "supported_numeric_gold_sign")),
            "supported_numeric_gold": _table([r for r in records if r["family"] == "answerability"],
                ("scope", "style", "answerable", "supported_numeric_gold_sign")),
            "explicit_zero_cells": _zero_cells(records, expected_styles),
            "parent_world_structure": _table(world_records, ("scope", "family", "rows", "distinct_styles", "distinct_operative_keys")),
            "hard_pair_structure": _table(pair_records, ("scope", "family", "rows", "distinct_world_ids", "distinct_styles", "distinct_operative_keys", "answerable_rows")),
            "metadata_sign_caveat": "Facts operand_sign is latent metadata; supported_numeric_gold_sign is the actual supplied target. Grounded-QA dock labels use abs(value).",
            "style_caveat": "Historical replay retains original system prompts; a canonical variant label does not prove identical canonical repair wording."}


def _semantic(row):
    _validate(row)
    # Exclude style/system/output contract; retain facts, gold and group identity.
    return {key: row.get(key) for key in ("family", "world_id", "pair_id", "operative_key", "payload", "user", "expected", "answerable")}


def _supervised_target(row):
    if _scope(row) == "historical_replay":
        return row.get("replay_target", row.get("reference"))
    if not row["answerable"]:
        return row.get("contract", {}).get("abstention")
    return canonical(row["expected"]) if isinstance(row["expected"], dict) else str(row["expected"])


def compare_membership(left, right):
    """Exact held-factor and seed comparisons; no claims of isolated treatment."""
    by_left = Counter(canonical(_semantic(row)) for row in left)
    by_right = Counter(canonical(_semantic(row)) for row in right)
    keys_left, keys_right = {r["operative_key"] for r in left}, {r["operative_key"] for r in right}
    worlds_left, worlds_right = {r["world_id"] for r in left}, {r["world_id"] for r in right}
    targets_left, targets_right = [_supervised_target(row) for row in left], [_supervised_target(row) for row in right]
    return {"left_rows": len(left), "right_rows": len(right),
            "semantic_multiset_equal": by_left == by_right,
            "identical_semantic_occurrences": sum((by_left & by_right).values()),
            "operative_key_intersection": len(keys_left & keys_right),
            "left_unique_operative_keys": len(keys_left), "right_unique_operative_keys": len(keys_right),
            "world_intersection": len(worlds_left & worlds_right),
            "left_worlds": len(worlds_left), "right_worlds": len(worlds_right),
            "row_identity_multiset_equal": Counter(r["row_id"] for r in left) == Counter(r["row_id"] for r in right),
            "row_order_equal": [r["row_id"] for r in left] == [r["row_id"] for r in right],
            "supervised_targets_available": all(isinstance(value, str) for value in targets_left + targets_right),
            "supervised_target_multiset_equal": Counter(targets_left) == Counter(targets_right),
            "order_sha256": {"left": digest([r["row_id"] for r in left]), "right": digest([r["row_id"] for r in right])},
            "interpretation": "Membership equality is prerequisite, not sufficient proof, for an isolated factor contrast; difficulty also changes paired dependence and target cases.",
            "contract_caveat": "The natural answerability contract changes unsupported target strings from UNKNOWN to the stated abstention phrase, even where semantic gold/worlds match.",
            "native_effects": "UNMEASURED"}


def _validate_prepared_metadata(validation, frozen, fixed_registry, validation_sha256):
    expected_trials = {trial["trial_id"] for trial in fixed_registry["trials"]}
    if (validation.get("offline_validation_passed") is not True
            or validation.get("native_execution_authorized") is not False
            or frozen.get("validation_sha256") != validation_sha256
            or validation.get("registry_sha256") != fixed_registry["registry_sha256"]
            or frozen.get("registry") != fixed_registry
            or set(validation.get("datasets", {})) != expected_trials
            or len(expected_trials) != 16):
        raise ValueError("STOP_CASE_AUDIT_PREPARED_PACKET_BINDING")
    return validation["datasets"]


def prospective_repair_decisions():
    return {"status": "UNADOPTED_REQUIRES_PROSPECTIVE_DATA_DECISION_BEFORE_MODEL_OUTCOMES",
            "current_recipes_and_endpoints_changed": False, "native_effects": "UNMEASURED",
            "alternatives": [
                {"problem": "regular supported scalar targets are positive-only",
                 "prospective_alternative": "Cross actual supported numeric sign with support/missing/conflict and prompt style; distinguish latent metadata from visible evidence and gold.",
                 "current_data_reusable_without_regeneration": False},
                {"problem": "prompt styles predict transition sign/branch or fact scenario",
                 "prospective_alternative": "Use the unadopted semantic-cell style assignment, preserve all world/evidence/semantic-gold identities, then re-render contract targets and repeat tokenizer audits.",
                 "current_data_reusable_without_regeneration": True},
                {"problem": "all fresh conditional margins are zero or one",
                 "prospective_alternative": "Declare wider positive/negative comparison margins and held-out strata before generation, including an explicit tie stratum; keep modulo-97 support limitations separate.",
                 "current_data_reusable_without_regeneration": False},
                {"problem": "difficulty changes parent worlds, gold signs and paired dependence",
                 "prospective_alternative": "Either report the current arms as bundled curricula, or prospectively construct common latent parent populations with matched case margins and paired units in both controls and treatments.",
                 "current_data_reusable_without_regeneration": False}],
            "required_next_decision": "Retain the reviewed bundled-recipe experiment with these disclosed input patterns, or explicitly version amended inputs and obtain new source/data review. No automatic replacement."}


def observed_input_patterns(rows, recipe):
    fresh = [_case(row) for row in rows if _scope(row) == "fresh_repair"]
    supported = [record for record in fresh if record["family"] == "answerability" and record["answerable"]]
    gold_signs = Counter(record["supported_numeric_gold_sign"] for record in supported)
    ledger = [record for record in fresh if record["family"] == "ledger"]
    conditional = [record for record in fresh if record["family"] == "register" and record["branch"] != "not_conditional"]
    ledger_style_signs = {style: sorted({record["operand_sign"] for record in ledger if record["style"] == style}) for style in STYLES}
    conditional_style_branches = {style: sorted({record["branch"] for record in conditional if record["style"] == style}) for style in STYLES}
    margin_counts = Counter(record["branch_margin"] for record in conditional)
    diverse, hard = recipe["prompt_coverage"] == "diverse", recipe["difficulty"] == "counterfactual"
    return {"supported_numeric_gold_sign_counts": dict(gold_signs),
            "regular_supported_scalar_positive_only": bool(supported) and set(gold_signs) == {"positive"} if not hard else None,
            "ledger_style_operand_signs": ledger_style_signs,
            "hard_diverse_ledger_style_sign_mapping_observed": (ledger_style_signs == {
                "canonical": ["positive"], "native_a": ["negative"], "native_b": ["positive"], "natural": ["negative"]}) if diverse and hard else None,
            "conditional_style_branches": conditional_style_branches,
            "diverse_conditional_branch_style_mapping_observed": (conditional_style_branches == {
                "canonical": ["true"], "native_a": ["false"], "native_b": ["true"], "natural": ["false"]}) if diverse else None,
            "conditional_margin_counts": [{"margin": value, "count": margin_counts[value]} for value in sorted(margin_counts)],
            "fresh_conditional_margins_exactly_zero_one": set(margin_counts) == {0, 1},
            "native_effects": "UNMEASURED", "model_quality_gain": False}


def audit_prepared_inputs(packet_dir, output_dir):
    """Run ONLY inside the sibling's owned CPU supervisor; never launches a worker.

    Reads prepared pinned training datasets and DEV. It never calls verify_packet
    (which would read a final archive), opens final membership, regenerates rows,
    loads a model, or changes any reviewed source/proposal. At most two full
    training datasets are resident; comparisons reread inexpensive gzip inputs.
    """
    from ..titan_quality_repair_v1 import custody, packet, registry
    packet_root, output_root = custody.external(packet_dir), custody.external(output_dir)
    if output_root == packet_root or output_root.is_relative_to(packet_root):
        raise ValueError("STOP_CASE_AUDIT_OUTPUT_MUST_BE_FRESH_SIBLING_ARTIFACT")
    reviewed_before = custody.source_manifest()
    frozen_path = packet_root / "FROZEN_RECIPE_REGISTRY.json"
    frozen_bytes = frozen_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    fixed = registry.build_registry()
    registry.validate_registry(frozen["registry"])
    validation_path = packet_root / "CPU_VALIDATION.json"
    validation_bytes = custody.read_pinned(validation_path, frozen["validation_sha256"])
    validation = json.loads(validation_bytes)
    descriptors = _validate_prepared_metadata(validation, frozen, fixed, custody.sha_bytes(validation_bytes))
    supervisor_path = packet_root / "OFFLINE_SUPERVISOR_RECEIPT.json"
    supervisor_bytes = supervisor_path.read_bytes()
    supervisor = json.loads(supervisor_bytes)
    if (supervisor.get("status") != "PASS" or supervisor.get("stage") != "prepare"
            or supervisor.get("source_unchanged_during_worker") is not True):
        raise ValueError("STOP_CASE_AUDIT_NO_COMPLETE_PREPARE_RECEIPT")
    # Generated inputs are not regenerated. Pin their construction identities
    # separately from later added native worker/supervision source files.
    input_modules = ("generator.py", "schema.py", "registry.py", "oracles.py", "replay.py", "exclusion.py", "token_audit.py")
    origin = validation.get("source_manifest", {})
    if any(origin.get(name) != reviewed_before.get(name) for name in input_modules):
        raise ValueError("STOP_CASE_AUDIT_INPUT_CONSTRUCTION_SOURCE_DRIFT")
    recipes = {recipe["recipe_id"]: recipe for recipe in fixed["recipes"]}
    trial_lookup = {(trial["recipe_id"], trial["seed"]): trial["trial_id"] for trial in fixed["trials"]}
    dataset_pins, summaries, alternatives = {}, {}, {}
    def read_trial(trial_id):
        descriptor = descriptors[trial_id]
        expected_path = packet_root / "training" / (trial_id + ".jsonl.gz")
        if Path(descriptor["path"]).resolve() != expected_path.resolve():
            raise ValueError("STOP_CASE_AUDIT_TRAINING_PATH_BINDING")
        return packet.read_training_dataset(descriptor)
    for trial in fixed["trials"]:
        trial_id, recipe_id = trial["trial_id"], trial["recipe_id"]
        rows = read_trial(trial_id)
        summary = summarize_rows(rows, recipes[recipe_id])
        summary["observed_input_patterns"] = observed_input_patterns(rows, recipes[recipe_id])
        descriptor = descriptors[trial_id]
        dataset_pins[trial_id] = {key: descriptor[key] for key in ("path", "sha256", "payload_sha256", "payload_bytes", "recipe_id", "seed")}
        summary_path = output_root / (trial_id + "-CASE_SUMMARY.json")
        custody.write_json(summary_path, summary)
        summaries[trial_id] = {"path": str(summary_path), "sha256": custody.sha_bytes(summary_path.read_bytes())}
        if recipes[recipe_id]["prompt_coverage"] == "diverse" and recipes[recipe_id]["difficulty"] == "counterfactual":
            unadopted = prospective_balanced_style_assignment(rows)
            assignment = unadopted["assignments"]
            # Shallow views change only style for the counterfactual INPUT audit.
            # Neither originals nor contract targets/tokenizer artifacts change.
            views = [dict(row, variant_id=assignment[row["row_id"]]) if row["row_id"] in assignment else row for row in rows]
            assignment_summary = summarize_rows(views, recipes[recipe_id])
            before_families = Counter(row["family"] for row in rows if _scope(row) == "fresh_repair")
            after_styles = Counter((row["family"], row["variant_id"]) for row in views if _scope(row) == "fresh_repair")
            totals_ok = all(after_styles[(family, style)] == count // 4 for family, count in before_families.items() for style in STYLES)
            strata_counts = defaultdict(Counter)
            for row in views:
                if _scope(row) == "fresh_repair":
                    strata_counts[(row["family"], canonical(_assignment_stratum(row)))][row["variant_id"]] += 1
            floors_ok = all(max(counts.get(style, 0) for style in STYLES) - min(counts.get(style, 0) for style in STYLES) <= 1 for counts in strata_counts.values())
            if not totals_ok or not floors_ok or not compare_membership(rows, views)["semantic_multiset_equal"]:
                raise ValueError("STOP_UNADOPTED_ASSIGNMENT_INVARIANT")
            unadopted.update(exact_family_style_totals_verified=totals_ok, observed_cell_floor_ceil_verified=floors_ok,
                            prospective_case_summary=assignment_summary, semantic_membership_preserved=True,
                            targets_rerendered=False, tokenizer_audit_performed=False)
            path = output_root / (trial_id + "-UNADOPTED_STYLE_MAP.json")
            custody.write_json(path, unadopted)
            alternatives[trial_id] = {"path": str(path), "sha256": custody.sha_bytes(path.read_bytes()), "adopted": False}
            del views, unadopted, assignment
        del rows
    comparisons = []
    for seed in fixed["seeds"]:
        values = list(recipes.values())
        for left_index, left_recipe in enumerate(values):
            for right_recipe in values[left_index + 1:]:
                factors = ("prompt_coverage", "difficulty", "replay_fraction")
                changed = [key for key in factors if left_recipe[key] != right_recipe[key]]
                if len(changed) != 1:
                    continue
                left_id = trial_lookup[(left_recipe["recipe_id"], seed)]
                right_id = trial_lookup[(right_recipe["recipe_id"], seed)]
                left, right = read_trial(left_id), read_trial(right_id)
                by_family = {family: compare_membership([r for r in left if r["family"] == family and _scope(r) == "fresh_repair"],
                    [r for r in right if r["family"] == family and _scope(r) == "fresh_repair"]) for family in REPAIR_FAMILIES}
                comparisons.append({"kind": "held_factor", "changed_factor": changed[0], "seed": seed,
                                    "left_trial": left_id, "right_trial": right_id,
                                    "all_rows": compare_membership(left, right), "fresh_repair_by_family": by_family})
                del left, right
    for recipe_id in recipes:
        left_id, right_id = (trial_lookup[(recipe_id, seed)] for seed in fixed["seeds"])
        left, right = read_trial(left_id), read_trial(right_id)
        comparisons.append({"kind": "seed_replication", "left_trial": left_id, "right_trial": right_id,
                            "comparison": compare_membership(left, right)})
        del left, right
    development = validation["development"]
    dev_path = custody.external(development["path"])
    if dev_path.resolve() != (packet_root / "DEVELOPMENT.jsonl").resolve():
        raise ValueError("STOP_CASE_AUDIT_DEVELOPMENT_PATH_BINDING")
    dev_bytes = custody.read_pinned(dev_path, development["sha256"])
    dev_rows = [json.loads(line) for line in dev_bytes.decode("utf-8").splitlines() if line.strip()]
    if len(dev_rows) != 512 or any(row["split"] != "dev" for row in dev_rows):
        raise ValueError("STOP_CASE_AUDIT_DEVELOPMENT_MEMBERSHIP")
    dev_summary = summarize_rows(dev_rows)
    del dev_rows, dev_bytes
    reviewed_after = custody.source_manifest()
    if reviewed_before != reviewed_after:
        raise ValueError("STOP_CASE_AUDIT_REVIEWED_SOURCE_CHANGED")
    report = {"status": "VERIFIED_INPUT_CASE_PATTERN_NO_MODEL_GAIN", "packet_dir": str(packet_root),
              "provenance": {"frozen_registry_path": str(frozen_path), "frozen_registry_sha256": custody.sha_bytes(frozen_bytes),
                  "cpu_validation_path": str(validation_path), "cpu_validation_sha256": custody.sha_bytes(validation_bytes),
                  "prepare_supervisor_sha256": custody.sha_bytes(supervisor_bytes), "registry_sha256": fixed["registry_sha256"],
                  "training_datasets": dataset_pins, "development_sha256": development["sha256"],
                  "reviewed_package_source_sha256": reviewed_before,
                  "diagnostic_source_sha256": {path.name: custody.sha_bytes(path.read_bytes()) for path in Path(__file__).parent.glob("*.py")}},
              "trial_case_summaries": summaries, "development_case_summary": dev_summary,
              "factor_and_seed_membership_comparisons": comparisons, "unadopted_style_maps": alternatives,
              "prospective_repair_decisions": prospective_repair_decisions(),
              "max_resident_training_datasets": 2, "existing_source_and_proposal_changed": False,
              "final_payload_or_archive_opened": False, "old_banks_opened": False, "weights_opened": False,
              "native_effects": "UNMEASURED", "model_quality_gain": False,
              "native_model_loads": 0, "native_model_forwards": 0, "recipes_changed": False,
              "endpoints_changed": False, "mechanical_input_audit_only": True}
    custody.write_json(output_root / "INPUT_CASE_AUDIT.json", report)
    return report


def _assignment_stratum(row):
    c = _case(row)
    if row["family"] in {"ledger", "register"}:
        return (c["event_kind"], c["operand_sign"], c["branch"])
    return (c["scenario"], c["answerable"], c["supported_numeric_gold_sign"])


def prospective_balanced_style_assignment(rows):
    """Return an UNADOPTED map; never mutate/relabel a frozen training row.

    Fixed family/style totals and each semantic cell's floor/ceil quotas are
    preserved. Pair members may have different styles, as in the original
    corpus. Assignment alone cannot create absent supported negative targets,
    widen threshold margins, or make different difficulty worlds identical.
    Applying it needs a separately reviewed data/version decision and re-rendered
    format targets/token audits. No current recipe or permit consumes this map.
    """
    groups, assignments = defaultdict(list), {}
    for row in rows:
        _validate(row)
        if _scope(row) != "fresh_repair":
            continue  # historical replay is never regenerated
        groups[row["family"]].append(row)
    if not groups:
        raise ValueError("STOP_PROSPECTIVE_ASSIGNMENT_NO_FRESH_REPAIR")
    for family, members in groups.items():
        if len(members) % 4:
            raise ValueError("STOP_PROSPECTIVE_ASSIGNMENT_FOUR_STYLE_TOTAL")
        if len({r["row_id"] for r in members}) != len(members):
            raise ValueError("STOP_PROSPECTIVE_ASSIGNMENT_DUPLICATE_ROW_ID")
        strata = defaultdict(list)
        for row in members:
            strata[canonical(_assignment_stratum(row))].append(row)
        base_total = sum(len(group) // 4 for group in strata.values())
        capacities = [len(members) // 4 - base_total] * 4
        for group_index, key in enumerate(sorted(strata)):
            group = sorted(strata[key], key=lambda row: digest({"world_id": row["world_id"], "operative_key": row["operative_key"]}))
            base, remainder = divmod(len(group), 4)
            quotas = [base] * 4
            priority = sorted(range(4), key=lambda index: (-capacities[index], (index - group_index) % 4))
            for index in priority[:remainder]:
                if capacities[index] <= 0:
                    raise ValueError("STOP_PROSPECTIVE_ASSIGNMENT_CAPACITY")
                capacities[index] -= 1
                quotas[index] += 1
            offset = 0
            for index, quota in enumerate(quotas):
                for row in group[offset:offset + quota]:
                    assignments[row["row_id"]] = STYLES[index]
                offset += quota
        if any(capacities):
            raise ValueError("STOP_PROSPECTIVE_ASSIGNMENT_UNREALIZED_TOTAL")
    return {"status": "PROSPECTIVE_UNADOPTED_NO_CURRENT_RECIPE_OR_PERMIT_CHANGE",
            "assignments": assignments, "assignment_sha256": digest(assignments),
            "native_effects": "UNMEASURED", "source_rows_mutated": False,
            "fixes": "Observed semantic sign/branch/scenario cells are spread across all four styles to floor/ceil quotas, with exact existing family doses.",
            "does_not_fix": ["regular supported scalar golds have no negative values",
                              "conditional margins remain restricted to 0 and 1",
                              "difficulty populations and paired dependence differ",
                              "replay displaces repair examples and alters the register sampling stream"],
            "required_decision": "Keep the current bundled-recipe study with disclosed confounds, or explicitly version a new prospectively matched/balanced dataset before native outcomes."}
