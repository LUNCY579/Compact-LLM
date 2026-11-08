"""Freeze before observing confirmation, consume once, fail closed on missing gates."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from .io import (assert_disjoint, digest, file_hash, load, require, seal, validate_bank,
                 verify, write_new)
from .metrics import compare, paired_outcomes


def freeze(inputs: dict[str, str], settings: dict, ledger: str) -> dict:
    needed = {"confirmation_bank", "training_bank", "development_bank", "reference_spec", "candidate_spec"}
    require(needed <= inputs.keys(), "Freeze needs training, development, confirmation and both model specs")
    require(settings.get("approved") is True, "The protocol must be explicitly approved before freezing")
    require(isinstance(settings.get("claim"), str) and bool(settings["claim"]), "Register bounded claim wording")
    require(type(settings.get("bootstrap_repeats")) is int and 2000 <= settings["bootstrap_repeats"] <= 10000,
            "Confirmation bootstrap must use 2000..10000 repeats")
    require(type(settings.get("seed")) is int, "Register the analysis seed")
    rules = settings.get("behavior_gates", [])
    require(rules and len({x["endpoint"] for x in rules}) == len(rules), "Distinct behavioral gates required")
    require(any(x["mode"] == "noninferiority" for x in rules), "A separate protected-behavior gate is required")
    for rule in rules:
        require(rule["mode"] in {"superiority", "noninferiority"} and rule["direction"] in {"higher", "lower"}, "Invalid gate")
        require(isinstance(rule["margin"], (int, float)) and 0 <= rule["margin"] < float("inf"), "Invalid gate margin")
        require(type(rule.get("expected_items")) is int and rule["expected_items"] >= 8, "Register endpoint coverage")
    banks = [load(inputs[name]) for name in ("training_bank", "development_bank", "confirmation_bank")]
    for bank, split in zip(banks, ("train", "development", "confirmation")):
        validate_bank(bank)
        require(all(i["split"] == split for i in bank["items"]), "Mislabelled freeze bank")
    assert_disjoint(*banks)
    require(len({b["evidence_class"] for b in banks}) == 1, "Evidence class mismatch in freeze")
    require(len({b["tokenizer_sha256"] for b in banks}) == 1, "Tokenizer mismatch in freeze")
    for name in ("reference_spec", "candidate_spec"):
        spec = load(inputs[name])
        require(spec["evidence_class"] == banks[0]["evidence_class"], "Spec evidence class mismatch")
        if banks[0]["evidence_class"] == "native_model":
            from .native import inspect_spec
            inspect_spec(spec)
    if "behavior_bank" in inputs:
        behavior_bank = load(inputs["behavior_bank"])
        validate_bank(behavior_bank)
        require(all(i["split"] == "confirmation" for i in behavior_bank["items"]), "Behavior bank must be confirmation")
        assert_disjoint(banks[0], banks[1], behavior_bank)
    ledger_path = Path(ledger).resolve()
    require(not ledger_path.exists(), "Confirmation ledger already exists")
    return seal({"kind": "confirmation_freeze", "schema_version": 1, "settings": settings,
                 "created_utc": datetime.now(timezone.utc).isoformat(), "ledger": str(ledger_path),
                 "inputs": {name: {"path": str(Path(path).resolve()), "sha256": file_hash(path)}
                            for name, path in sorted(inputs.items())},
                 "evidence_class": banks[0]["evidence_class"], "release_permitted": False})


def consume(manifest: dict) -> Path:
    verify(manifest)
    require(manifest.get("kind") == "confirmation_freeze", "Not a confirmation freeze")
    for item in manifest["inputs"].values():
        require(file_hash(item["path"]) == item["sha256"], "Frozen input changed")
    ledger = Path(manifest["ledger"])
    ledger.mkdir(parents=True, exist_ok=False)  # atomic one-attempt claim before reading outputs
    write_new(ledger / "CLAIM.json", {"freeze_sha256": manifest["content_sha256"],
              "claimed_utc": datetime.now(timezone.utc).isoformat(), "state": "CONSUMED_BEFORE_ANALYSIS"})
    return ledger


def confirm(manifest: dict, reference_path: str, candidate_path: str,
            reference_outcomes_path: str | None, candidate_outcomes_path: str | None) -> dict:
    ledger = consume(manifest)
    try:
        reference, candidate = load(reference_path), load(candidate_path)
        inputs, settings = manifest["inputs"], manifest["settings"]
        bank = load(inputs["confirmation_bank"]["path"])
        for artifact, key in ((reference, "reference_spec"), (candidate, "candidate_spec")):
            spec = load(inputs[key]["path"])
            require(artifact["bank_sha256"] == bank["content_sha256"], "Wrong confirmation bank")
            require(artifact["model_id"] == digest(spec), "Artifact model differs from frozen spec")
            require(artifact["evidence_class"] == manifest["evidence_class"], "Evidence class escalation")
        diagnostic = compare(reference, candidate, settings["bootstrap_repeats"], settings["seed"])
        gates = []
        if reference_outcomes_path and candidate_outcomes_path:
            ro, co = load(reference_outcomes_path), load(candidate_outcomes_path)
            for outcomes, artifact in ((ro, reference), (co, candidate)):
                require(outcomes.get("freeze_sha256") == manifest["content_sha256"], "Behavioral job used a different freeze")
                require(outcomes["model_id"] == artifact["model_id"], "Behavioral model identity mismatch")
                require(outcomes["evidence_class"] == manifest["evidence_class"], "Behavioral evidence class escalation")
                # A separate rollout bank must be frozen when different from the NLL bank.
                expected = load(inputs.get("behavior_bank", inputs["confirmation_bank"])["path"])
                require(outcomes["bank_sha256"] == expected["content_sha256"], "Unregistered behavioral bank")
                known_items = {r["item_id"]: r for r in expected["items"]}
                for outcome in outcomes["rows"]:
                    require(outcome["item_id"] in known_items, "Unregistered behavioral item")
                    original = known_items[outcome["item_id"]]
                    require(all(outcome[k] == original[k] for k in ("parent_id", "family", "payload_sha256", "split")),
                            "Behavioral item provenance mismatch")
                for rule in settings["behavior_gates"]:
                    subset = [r for r in outcomes["rows"] if r["endpoint"] == rule["endpoint"]]
                    require(len(subset) == rule["expected_items"], "Incomplete endpoint coverage")
                    require(all(r["split"] == "confirmation" for r in subset), "Development behavior cannot confirm")
            alpha = 0.05 / len(settings["behavior_gates"])
            for rule in settings["behavior_gates"]:
                gates.append(paired_outcomes(ro, co, rule["endpoint"], rule["margin"], rule["direction"],
                    settings["bootstrap_repeats"], settings["seed"], rule["mode"], alpha))
        verdict = "BLOCKED_MISSING_BEHAVIOR"
        if gates:
            verdict = "CONFIRMED_FIXED_ARTIFACT" if all(g["verdict"] == "PASS" for g in gates) else "NOT_CONFIRMED"
        if manifest["evidence_class"] != "native_model":
            verdict = "SYNTHETIC_OR_TINY_VALIDATION_ONLY"
        result = seal({"kind": "confirmation_result", "schema_version": 1, "verdict": verdict,
            "freeze_sha256": manifest["content_sha256"], "registered_claim": settings["claim"],
            "diagnostic": diagnostic, "behavior_gates": gates,
            "source_file_sha256": {"reference": file_hash(reference_path), "candidate": file_hash(candidate_path),
                "reference_outcomes": file_hash(reference_outcomes_path) if reference_outcomes_path else None,
                "candidate_outcomes": file_hash(candidate_outcomes_path) if candidate_outcomes_path else None}, "multiplicity": "Bonferroni two-sided parent-bootstrap CIs",
            "quality_improvement_established": False, "release_permitted": False})
        write_new(ledger / "RESULT.json", result)
        return result
    except Exception as exc:
        write_new(ledger / "FAILED.json", {"status": "FAILED_READ_CONSUMED", "error_type": type(exc).__name__})
        raise


def render_report(result: dict) -> str:
    """Human-readable companion; JSON remains the machine source of truth."""
    body = result.get("diagnostic", result)
    profile = body.get("token_profile")
    lines = ["# RGI-001 diagnostic report", "", "Status: " + result.get("verdict", "DIAGNOSTIC_ONLY"), "",
             "No automatic intelligence, causal, retention or release claim follows.", ""]
    if profile:
        lines.extend(["| Measure | Value |", "|---|---:|"])
        for key, value in profile.items():
            lines.append(f"| {key} | {value} |")
        lines += ["", "## Interpretation", "",
                  "Negative offset means lower mean NLL. Centered CCC tests unit-slope agreement; Pearson does not.",
                  "Slice residuals retain the global offset. Content, format and mixed tokens are reported separately.",
                  "Bootstrap intervals describe held-out parent sampling, not uncertainty over training seeds."]
    return "\n".join(lines) + "\n"


def collect_frozen(manifest: dict, out: str, execute: bool = False) -> dict:
    """One frozen collection attempt, separate from the single analysis read."""
    require(execute, "Frozen collection requires --execute")
    verify(manifest)
    require(manifest.get("kind") == "confirmation_freeze", "Wrong freeze kind")
    for item in manifest["inputs"].values():
        require(file_hash(item["path"]) == item["sha256"], "Frozen input changed")
    ledger = Path(manifest["ledger"] + ".collection")
    ledger.mkdir(parents=True, exist_ok=False)
    write_new(ledger / "CLAIM.json", {"freeze_sha256": manifest["content_sha256"], "state": "COLLECTION_STARTED"})
    try:
        root = Path(out)
        root.mkdir(parents=True, exist_ok=False)
        bank = load(manifest["inputs"]["confirmation_bank"]["path"])
        from .native import collect
        paths = {}
        for name in ("reference", "candidate"):
            spec = load(manifest["inputs"][name + "_spec"]["path"])
            value = collect(bank, spec)
            path = root / (name + ".json")
            write_new(path, value)
            paths[name] = {"path": str(path.resolve()), "sha256": file_hash(path)}
        result = {"status": "COLLECTED_NOT_ANALYZED", "freeze_sha256": manifest["content_sha256"], "outputs": paths}
        write_new(ledger / "RESULT.json", result)
        return result
    except Exception as exc:
        write_new(ledger / "FAILED.json", {"status": "FAILED_COLLECTION_CONSUMED", "error_type": type(exc).__name__})
        raise
