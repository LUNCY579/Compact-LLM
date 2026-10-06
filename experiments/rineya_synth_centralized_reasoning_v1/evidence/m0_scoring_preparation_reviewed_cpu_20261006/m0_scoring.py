"""Staged CPU scoring preparation. Native capture access remains closed here."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1] if HERE.name == "m0_scoring_preparation_work" else HERE.parent
SNAPSHOT = PACKAGE / "evidence/fresh_dev_compact002_20261003/code_snapshot"
SCOPE = "M0_FIXED384_DEV_WHOLE_RESPONSE_SCORING"
FIXTURE = "CPU_AUTHORED_M0_SCORING_FIXTURE"
SCORER_PINS = {
    "EVALUATION_PANEL_POLICY.json": "1b30b9041b8f009702dc75f284f8637d3cffef6f445cd8deca632ddee0f7c242",
    "EVALUATION_PANEL_POLICY_002.json": "7e80d5d52ba6c0a14654151a3be40cfa177faf3a120953d5d177792d9a33ee92",
    "balanced_corpus.py": "0577bb4a15e08ae38b97289171bcd28cb4dc45544393f9e1f3075fbab270ca34",
    "evaluation_panel.py": "e8ed801cf258cd42edc2a8b08a560cd80914822bd000e833e08208c1bde46598",
    "rs_contract.py": "cef135a06d4449917798e7e40685ac3589615663bfe47662e0a49207b67ccc6a",
    "rs_solver.py": "6acf2fa26ef3f5a7b103818f4bb0eef972d5d975e9f3b39614ad6b253131acaf",
    "rs_verifier.py": "36d2322ac077f4731b621defbc77ae39d1bfde2f1d0bcc15c73082b5f897ec98",
    "task_compiler.py": "1ae26b485d7a6ee48fe9667df51886c2c64af09d89dd0edcb622794d33642f60",
}
SCORER_BYTES = {"EVALUATION_PANEL_POLICY.json": 2163, "EVALUATION_PANEL_POLICY_002.json": 3106,
    "balanced_corpus.py": 18956, "evaluation_panel.py": 41113, "rs_contract.py": 7033,
    "rs_solver.py": 2037, "rs_verifier.py": 3704, "task_compiler.py": 17220}
FINISH_FILE = "db808f1b53b1ebd8c716a246e80a4eb13a8275aad643b13a05017c9162990e7d"
IMPORTER_PIN = "f859973eabc5ff0d04cd9befe0b41f97c7474abf8b1d46e768ebf4abc6ff551a"
_TOKEN = object()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def seal(value):
    result = deepcopy(value)
    result["sha256"] = digest(result)
    return result


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def read_pinned(path, pin):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "selected regular file")
    before = path.stat()
    require(0 < before.st_size <= 16 * 1024**2, "bounded selected file")
    with path.open("rb") as stream:
        raw = stream.read(before.st_size + 1)
    after = path.stat()
    require(before.st_size == after.st_size == len(raw) and before.st_mtime_ns == after.st_mtime_ns,
            "selected file changed")
    require(hashlib.sha256(raw).hexdigest() == pin, "selected byte identity")
    return raw


def read_anchor(anchor):
    require(type(anchor) is dict and set(anchor) == {"path", "file_sha256", "body_sha256"},
            "externally selected file/body anchor")
    value = json.loads(read_pinned(anchor["path"], anchor["file_sha256"]), object_pairs_hook=pairs)
    require(value.get("sha256") == anchor["body_sha256"] == digest(
        {k: v for k, v in value.items() if k != "sha256"}), "selected body identity")
    return value


@contextmanager
def frozen_verifier():
    """Execute the exact frozen eight-file stdlib source closure, not an injected grader."""
    raw = {name: read_pinned(SNAPSHOT / name, pin) for name, pin in SCORER_PINS.items()}
    require(all(len(raw[name]) == SCORER_BYTES[name] for name in raw), "frozen source descriptor sizes")
    names = ["rs_contract", "rs_solver", "rs_verifier", "balanced_corpus", "task_compiler", "evaluation_panel"]
    saved = {name: sys.modules.get(name) for name in names}
    old_path = list(sys.path)
    try:
        for name in names:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(SNAPSHOT))
        for name in names:
            module = ModuleType(name)
            module.__file__ = str(SNAPSHOT / (name + ".py"))
            module.__package__ = ""
            module.__spec__ = importlib.util.spec_from_file_location(name, module.__file__)
            sys.modules[name] = module
            # Compile the selected bytes directly, avoiding an unselected cache.
            exec(compile(raw[name + ".py"], module.__file__, "exec"), module.__dict__)
            require(Path(module.__file__).resolve() == (SNAPSHOT / (name + ".py")).resolve(),
                    "selected scorer module origin")
        yield sys.modules["evaluation_panel"]
        require(raw == {name: read_pinned(SNAPSHOT / name, pin) for name, pin in SCORER_PINS.items()},
                "frozen scorer changed during execution")
    finally:
        sys.path[:] = old_path
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def authored_specs():
    """Independent explicit tasks; no DEV population or oracle file is consulted."""
    return [
        ("R", {"initial": [2, 3, 5, 7, 11, 13], "lookup": {"links": {"a": "b", "b": 1, "c": 3}, "requested": ["a", "c"]}}),
        ("S", {"initial": [1, 2, 3, 4, 8, 9], "selection": {"k": 2, "target": 5}}),
        ("RS", {"initial": [1, 2, 3, 4, 8, 9], "lookup": {"links": {"a": 0, "b": 3}, "requested": ["a", "b"]}, "selection": {"k": 2, "target": 5}}),
    ]


def qualify_frozen_verifier():
    """Execute authored positive/negative/counterfactual vectors; adds no authority."""
    checks = []
    with frozen_verifier() as verifier:
        for skill, spec in authored_specs():
            altered = deepcopy(spec)
            if skill == "R":
                altered["lookup"]["links"]["b"] = 2
            else:
                altered["selection"]["target"] = 99
            # Direct-walk/cardinality-sum DP reference is algorithmically independent
            # of the fixed-point/combination verifier; the public parser is shared.
            for rendering in ("familiar", "ledger"):
                render = verifier.render_ledger if rendering == "ledger" else verifier.contract.render
                messages, changed = render(spec), render(altered)
                expected = verifier.primary.solve(spec)["solutions"]
                require(expected == verifier.solve_visible(messages)["solutions"], "independent solution disagreement")
                answer = verifier.contract.format_answer(expected[0])
                candidates = [(answer, True), ('{"answer":"UNKNOWN"}', False),
                    ('{"answer":[true]}', False), ('{"answer":[0],"answer":[1]}', False),
                    ('prefix ' + answer, False), (answer + '\n{"answer":[0]}', False),
                    ('{"answer":[]}', False), ('{"answer":[0,0]}', False)]
                for solution in expected:
                    candidates.append((verifier.contract.format_answer(list(reversed(solution))), True))
                for candidate, correct in candidates:
                    result = verifier.verify_response(messages, candidate)
                    require(type(result.get("accepted")) is bool and result["accepted"] is correct,
                            "authored semantic qualification failed")
                    checks.append({"skill": skill, "rendering": rendering,
                        "messages_sha256": digest(messages), "candidate_sha256": digest(candidate), "expected": correct})
                result = verifier.verify_response(changed, answer)
                require(result["accepted"] is False, "relevant counterfactual accepted stale answer")
                require(verifier.primary.solve(altered)["solutions"] == verifier.solve_visible(changed)["solutions"],
                        "counterfactual independent solution disagreement")
                if skill != "R":
                    require(verifier.verify_response(changed, '{"answer":"UNKNOWN"}')["accepted"] is True,
                            "impossible counterfactual rejected UNKNOWN")
                checks.append({"skill": skill, "rendering": rendering,
                    "counterfactual_messages_sha256": digest(changed), "stale_answer_rejected": True})
    return seal({"kind": "CPU_AUTHORED_FROZEN_VERIFIER_QUALIFICATION", "scorer_sources": SCORER_PINS,
        "checks": checks, "checks_count": len(checks), "actual_responses_read": 0,
        "independent_review_required": True, "native_scoring_authorized": False,
        "native_membership_admitted": False, "quality_gain_established": False})


class _Fixture:
    def __init__(self, rows, token):
        require(token is _TOKEN, "authored fixture factory required")
        self.rows = deepcopy(rows)
        self.sha256 = digest(rows)
        self._factory_sha256 = self.sha256


def authored_fixture():
    """Construct exactly384 synthetic records; accepts no native paths or data."""
    rows = []
    with frozen_verifier() as verifier:
        for skill, spec in authored_specs():
            for parent in range(32):
                for condition in (0, 1):
                    for rendering in ("familiar", "ledger"):
                        messages = verifier.render_ledger(spec) if rendering == "ledger" else verifier.contract.render(spec)
                        answer = verifier.contract.format_answer(verifier.primary.solve(spec)["solutions"][0])
                        ordinal = len(rows)
                        response = answer if ordinal % 3 else "extra " + answer
                        rows.append({"row_id": "fixture_%s_%02d_%d_%s" % (skill, parent, condition, rendering),
                            "component_id": "fixture_%s_%02d" % (skill, parent), "skill": skill,
                            "condition": condition, "rendering": rendering, "messages": messages,
                            "raw": {"scope": FIXTURE, "decoded_response": response,
                                "finish_reason": "eos" if condition == 0 else "generation_cap",
                                "raw_generated_ids": [7, 151645] if condition == 0 else [7] * 32}})
    return _Fixture(rows, _TOKEN)


def _score_rows(rows, scope, lineage):
    require(len(rows) == 384 and len({r["row_id"] for r in rows}) == 384
            and len({r["component_id"] for r in rows}) == 96, "complete384/96 score population")
    results = []
    with frozen_verifier() as verifier:
        for row in rows:
            result = verifier.verify_response(row["messages"], row["raw"]["decoded_response"])
            require(type(result.get("accepted")) is bool, "accepted field must be exact bool")
            results.append({"row_id": row["row_id"], "component_id": row["component_id"],
                "skill": row["skill"], "condition": row["condition"], "rendering": row["rendering"],
                "messages_sha256": digest(row["messages"]), "raw_sha256": digest(row["raw"]),
                "finish_reason": row["raw"]["finish_reason"], "raw_generated_ids": deepcopy(row["raw"]["raw_generated_ids"]),
                "accepted": result["accepted"] is True, "reason": result["reason"]})
    return seal({"kind": "COMPLETE_M0_WHOLE_RESPONSE_SCORE_PACKET", "scope": scope, "arm": "M0",
        "rows": results, "row_order_sha256": digest([r["row_id"] for r in rows]),
        "input_rows_sha256": digest(rows), "lineage": deepcopy(lineage), "scorer_sources": SCORER_PINS,
        "verifier_calls": 384, "comparison_authorized": False, "retention_certified": False,
        "training_authorized": False, "promotion_authorized": False, "quality_gain_established": False})


def score_authored_fixture(fixture):
    require(type(fixture) is _Fixture and fixture.sha256 == fixture._factory_sha256 == digest(fixture.rows),
            "unmodified authored fixture")
    require(all(r["row_id"].startswith("fixture_") and r["raw"]["scope"] == FIXTURE for r in fixture.rows),
            "authored fixture scope")
    return _score_rows(fixture.rows, FIXTURE, {"fixture_factory_sha256": fixture.sha256})


def _public_location_gate():
    require(HERE.name == "m0_scoring_v1" and HERE.parent == PACKAGE,
            "staging cannot score native responses; reviewed promotion required")


def validate_m0_scoring_registration(registration_anchor, *, approval_anchor, review_anchor):
    """Call location gate before reading ANY registration, response or approval file."""
    _public_location_gate()
    approval = read_anchor(approval_anchor)
    require(approval.get("kind") == "OWNER_M0_ONLY_SCORING_AUTHORIZATION"
            and approval.get("scope") == SCOPE and approval.get("approved") is True
            and type(approval.get("human_evidence_reference")) is str
            and approval["human_evidence_reference"].strip(), "explicit M0-only scoring permission")
    review = read_anchor(review_anchor)
    require(review.get("kind") == "INDEPENDENT_M0_SCORER_QUALIFICATION_REVIEW"
            and review.get("scope") == SCOPE and review.get("qualified") is True,
            "trusted independently selected scorer qualification required")
    registration = read_anchor(registration_anchor)
    require(registration.get("kind") == "REGISTERED_M0_ONLY_SCORING_V1" and registration.get("scope") == SCOPE
            and registration.get("native_scoring_authorized") is True, "registered M0-only scoring scope")
    require(approval.get("registration_body_sha256") == registration["sha256"]
            and review.get("registration_body_sha256") == registration["sha256"], "external registration joins")
    require(registration.get("scorer_sources") == SCORER_PINS
            and registration.get("finish_policy_file_sha256") == FINISH_FILE
            and registration.get("importer_file_sha256") == IMPORTER_PIN, "unchanged scorer/finish/importer closure")
    require(type(registration.get("implementation_files")) is dict
            and set(registration["implementation_files"]) == {"m0_scoring.py", "m0_score_audit.py"},
            "reviewed scorer implementation closure")
    for name, pin in registration["implementation_files"].items():
        read_pinned(HERE / name, pin)
    amendment = read_anchor(registration["baseline_first_amendment_anchor"])
    require(amendment.get("id") == "SYNTH-BASELINE-FIRST-MEASUREMENT-AMENDMENT-001"
            and amendment["predecessor"]["finish_policy"]["sha256"] == FINISH_FILE,
            "baseline-first frozen finish-policy amendment")
    read_pinned(PACKAGE / "DEV_FINISH_POLICY_001.json", FINISH_FILE)
    read_pinned(PACKAGE / "score_import.py", IMPORTER_PIN)
    qualification = read_anchor(registration["qualification_anchor"])
    require(qualification.get("kind") == "CPU_AUTHORED_FROZEN_VERIFIER_QUALIFICATION"
            and qualification.get("scorer_sources") == SCORER_PINS
            and review.get("qualification_body_sha256") == qualification["sha256"], "independently reviewed qualification join")
    require(qualification == qualify_frozen_verifier(), "qualification must reproduce authored execution")
    require(type(registration.get("batches")) is list and len(registration["batches"]) == 6,
            "six complete M0 audit anchors")
    require(registration.get("requires_treatment_checkpoints") is False, "M0-only excludes future endpoint prerequisite")
    return registration


def _load_importer():
    raw = read_pinned(PACKAGE / "score_import.py", IMPORTER_PIN)
    module = ModuleType("_selected_m0_score_import")
    module.__file__ = str(PACKAGE / "score_import.py")
    module.__package__ = ""
    module.__spec__ = importlib.util.spec_from_file_location(module.__name__, module.__file__)
    exec(compile(raw, module.__file__, "exec"), module.__dict__)
    return module


def prepare_complete_m0(registration_anchor, *, approval_anchor, review_anchor):
    registration = validate_m0_scoring_registration(registration_anchor,
        approval_anchor=approval_anchor, review_anchor=review_anchor)
    importer = _load_importer()
    panel = importer.load_frozen_panel(PACKAGE)
    require(panel["sha256"] == registration.get("panel_registration_sha256")
            and panel["row_order_sha256"] == registration.get("row_order_sha256"), "registered complete panel/order")
    batches = [importer.load_m0_batch(b["preserved_directory"], b["audit_path"],
        expected_audit_sha256=b["expected_audit_sha256"], expected_config_sha256=b["expected_config_sha256"])
        for b in registration["batches"]]
    packet = importer.prepare_import(panel, batches, arm="M0")
    require(packet["batch_references"][0]["model_pins"] == registration.get("model_pins"), "unchanged exact M0 model pins")
    rows = [{**r, "messages": r["prompt"]["messages"]} for r in packet["rows"]]
    lineage = {"pre_score_packet_sha256": packet["sha256"],
        "registration_anchor": registration_anchor, "approval_anchor": approval_anchor,
        "review_anchor": review_anchor, "batch_references": packet["batch_references"],
        "qualification_anchor": registration["qualification_anchor"]}
    return rows, lineage


def score_complete_m0(registration_anchor, *, approval_anchor, review_anchor):
    rows, lineage = prepare_complete_m0(registration_anchor,
        approval_anchor=approval_anchor, review_anchor=review_anchor)
    return _score_rows(rows, SCOPE, lineage)
