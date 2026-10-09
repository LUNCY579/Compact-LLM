"""Arithmetic64 adaptation of the preserved complete RAM2 raw auditor.

No framework, tokenizer or native API is imported at module load. The inherited
full audit remains conditional on selected terminal/config/preservation/source
anchors and the original current qualification checks. CPU controls cannot
publish its native-compatible receipt.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
import ast
import collections
import hashlib
import importlib.metadata
import json
import math
import os
import sys

HERE = Path(__file__).resolve().parent
PACKAGE = next(p for p in HERE.parents if p.name == "rineya_synth_centralized_reasoning_v1")
EXPECTED = PACKAGE / "arithmetic_raw_audit_v1"
CAPTURE = PACKAGE / "arithmetic_capture_v1"
ORIGINAL = PACKAGE / "evidence/short_ram_successor_reviewed_cpu_20261005/audit_work/auditor.py"
ORIGINAL_SHA256 = "d69810dccd11a32595dace73a788c8868efabb2a09f593fd33316e92dcb47442"
SIGNAL = "ROOT_TERMINAL_AND_PRESERVED"
KIND = "arithmetic_guard_DEV"
REGISTRATION_ID = "SYNTH_ARITHMETIC_RAW_AUDIT_INVENTORY001"
SECTIONS = {"source": "source_pins", "runtime": "runtime_pins", "decoder": "decoder_runtime_pins",
            "data": "data_pins", "model": "model_pins"}
REF_FIELDS = {"path", "bytes", "sha256"}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("ascii")).hexdigest()


def hash_pin(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def read_ref(ref):
    require(type(ref) is dict and set(ref) == REF_FIELDS and type(ref["bytes"]) is int
            and 0 < ref["bytes"] <= 32 * 1024**2 and hash_pin(ref["sha256"]), "exact bounded selected JSON ref")
    path = Path(ref["path"])
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), "selected regular JSON file")
    with path.open("rb") as stream:
        require(os.fstat(stream.fileno()).st_size == ref["bytes"], "selected JSON size before read")
        raw = stream.read(ref["bytes"] + 1)
        require(os.fstat(stream.fileno()).st_size == ref["bytes"], "selected JSON size after read")
    require(len(raw) == ref["bytes"] and hashlib.sha256(raw).hexdigest() == ref["sha256"], "selected JSON bytes")
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def finite(text):
        value = float(text)
        require(math.isfinite(value), "nonfinite JSON float")
        return value
    return json.loads(raw, object_pairs_hook=pairs, parse_float=finite,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def selector(value):
    require(type(value) is tuple and len(value) == 2 and value[0] == KIND
            and type(value[1]) is int and 1 <= value[1] <= 8, "exact arithmetic batch1..8")
    return value


def prepared_source():
    raw = ORIGINAL.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ORIGINAL_SHA256, "preserved RAM2 full auditor source")
    text = raw.decode("utf-8")
    changes = (
        ('require(HERE.name=="audit_work" and HERE.parent.name=="short_ram_successor_work" and HERE.parent.parent.name=="artifacts","stage-only new-profile auditor")',
         'require(HERE == EXPECTED,"installed arithmetic raw auditor required")'),
        ('spec=importlib.util.spec_from_file_location("short_RAM2_audit_integrated_contract",package/\'short_collection_contract.py\')\n contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)',
         'contract=_capture_view()'),
        ('require(contract.VERSION==CONTRACT_VERSION and contract.HERE.resolve()==package.resolve() and not contract.STAGED,',
         'require(contract.VERSION==CONTRACT_VERSION and contract.HERE.resolve()==CAPTURE.resolve() and not contract.STAGED,'),
        ('require(type(expected_selector) is tuple and len(expected_selector)==2\n         and expected_selector[0] in ("atomic_RS_DEV","task_guard_DEV")\n         and type(expected_selector[1]) is int and 1<=expected_selector[1]<=(6 if expected_selector[0]=="atomic_RS_DEV" else 12),"one preregistered frozen batch selector")',
         'selector(expected_selector)'),
        ('str((package/name).resolve()):contract.native.filehash(contract.HERE/name)',
         'str((contract.HERE/name).resolve()):contract.native.filehash(contract.HERE/name)'),
        ("str(package/'short_collection_worker.py')", "str(CAPTURE/'arithmetic_capture_owned.py')"),
        ('counts.get("data")==5', 'counts.get("data")==4'),
        ('independently_registered_source_runtime65_decoder17_data5_model7_and_all_raw_files_verified',
         'independently_registered_source_runtime65_decoder17_data4_model7_and_all_raw_files_verified'),
    )
    applied = []
    for before, after in changes:
        require(text.count(before) == 1, "single bounded arithmetic adaptation: " + before)
        text = text.replace(before, after, 1)
        applied.append({"original": before, "replacement": after})
    # Module-level imports/CLI are never executed. Preserve every audit check.
    tree = ast.parse(text)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
             ("_inventory_path", "_registered_admission", "_hash", "_early_terminal_checks", "audit_terminal")]
    require(len(nodes) == 5, "exact original complete audit functions")
    return ast.Module(body=nodes, type_ignores=[]), applied, text


def _capture_module(expected_source_sha256):
    require(HERE == EXPECTED, "staged arithmetic raw audit has no native receipt authority")
    require(hash_pin(expected_source_sha256), "independently selected capture source before execution")
    path = CAPTURE / "arithmetic_capture_contract.py"
    require(path.is_file() and not path.is_symlink(), "reviewed arithmetic capture integration absent")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_source_sha256, "selected capture source bytes before execution")
    module = ModuleType("selected_arithmetic_capture_audit_contract")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    module.integrated()
    return module


def admit(scope, plan, registration, expected_scope, *, selected, source_commit, profile_sha256):
    """Pure source/order admission; neither Boolean approval nor native PASS."""
    selector(selected)
    require(type(source_commit) is str and len(source_commit) == 40
            and all(c in "0123456789abcdef" for c in source_commit), "independently selected launch commit")
    require(scope == expected_scope, "current original qualified arithmetic scope differs")
    require(registration.get("id") == REGISTRATION_ID and type(registration.get("schema_version")) is int
            and registration["schema_version"] == 1 and registration.get("source_commit") == source_commit,
            "exact selected arithmetic inventory/commit")
    require(registration.get("sha256") == digest({k: v for k, v in registration.items() if k != "sha256"}),
            "selected inventory body seal")
    require(registration.get("profile_body_sha256") == profile_sha256
            == digest(scope["short_RAM_monitor_profile"]), "selected original RAM profile")
    require(plan.get("fixture_only") is False and plan.get("rows") == 512 and plan.get("parent_components") == 128
            and plan.get("sha256") == digest({k: v for k, v in plan.items() if k != "sha256"})
            and len(plan["batches"][KIND]) == 8, "whole selected arithmetic512/128 plan")
    batch = plan["batches"][KIND][selected[1] - 1]
    require(scope["batch_selector"] == {"source_kind": KIND, "ordinal": selected[1]}
            and scope["rows"] == batch["rows"] and len(scope["rows"]) == 64
            and scope["source_packet_sha256"] == digest(scope["rows"])
            and batch["source_packet_sha256"] == scope["source_packet_sha256"], "exact selected plan64 row/order/hash")
    for section, key in SECTIONS.items():
        require(registration["inventories"][section] == scope[key]
                and type(registration["counts"][section]) is int
                and registration["counts"][section] == len(scope[key]), "complete selected inventory " + section)
    require({k: registration["counts"][k] for k in ("runtime", "decoder", "data", "model")}
            == {"runtime": 65, "decoder": 17, "data": 4, "model": 7}, "unchanged original runtime/decoder/model plus arithmetic data4")
    require(scope["model_role"] == "fresh_dense_M0_zero_B_only_no_STATE1_or_study_checkpoint"
            and scope["dose"]["rows"] == 64 and scope["dose"]["maximum_forward_calls"] == 2050
            and scope["dose"]["context_cap"] == 512 and scope["dose"]["maximum_generated_tokens"] == 32,
            "unchanged M0/model/dose profile")
    return {"fixture_only": True, "native_PASS": False, "source_kind": KIND, "ordinal": selected[1]}


def selected_sources(registration, source_commit):
    """Execute nothing until the externally selected source closure agrees."""
    require(registration.get("source_commit") == source_commit, "selected source commit before imports")
    sources = registration["inventories"]["source"]
    required = {str((CAPTURE / name).resolve()) for name in
                ("arithmetic_capture_contract.py", "arithmetic_capture_owned.py", "arithmetic_capture_pipeline.py")}
    require(type(sources) is dict and required <= set(sources), "complete executing capture/auditor source closure")
    dependencies = registration.get("auditor_dependency_pins")
    required_dependencies = {str(Path(__file__).resolve()), str(ORIGINAL.resolve())}
    require(type(dependencies) is dict and set(dependencies) == required_dependencies,
            "exact independently selected auditor dependency closure")
    for text, pin in {**sources, **dependencies}.items():
        path = Path(text)
        require(path.is_absolute() and path.is_file() and not path.is_symlink() and hash_pin(pin), "selected source descriptor")
        require(path.suffix.lower() in (".py", ".md", ".json"), "source metadata excludes model/native binaries")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == pin, "selected source before imports")
    require(dependencies[str(ORIGINAL.resolve())] == ORIGINAL_SHA256, "original full auditor source in selected closure")
    return sources[str((CAPTURE / "arithmetic_capture_contract.py").resolve())]


def arithmetic_references(plan, plan_ref, expected_plan_file_sha256):
    refs = {"plan": plan_ref, "projection": plan["measurement_reference"], "prompt": plan["prompt_reference"],
            "policy": plan["policy_reference"], "ledger": plan["ledger_reference"]}
    require(plan_ref["sha256"] == expected_plan_file_sha256, "selected original arithmetic plan bytes")
    for ref in refs.values():
        require(Path(ref["path"]).suffix.lower() == ".json", "arithmetic metadata files only")
        read_ref(ref)  # Every independently selected data reference, before assets.
    return deepcopy(refs)


def audit_terminal(config_path, raw_directory, preserved_directory, *, expected_selector,
                   expected_config_file_sha256, expected_preservation_body_sha256,
                   inventory_registration_path, expected_inventory_registration_file_sha256,
                   expected_profile_body_sha256, expected_plan_file_sha256, expected_source_commit,
                   terminal_preservation_signal=None, output_path=None):
    require(terminal_preservation_signal == SIGNAL, "selected terminal and preservation signal before reads")
    selector(expected_selector)
    for value in (expected_config_file_sha256, expected_preservation_body_sha256,
                  expected_inventory_registration_file_sha256, expected_profile_body_sha256, expected_plan_file_sha256):
        require(hash_pin(value), "independently selected actual evidence anchors required")
    require(HERE == EXPECTED, "staged arithmetic raw audit has no native receipt authority")
    inventory = read_ref({"path": str(Path(inventory_registration_path).absolute()),
                          "bytes": Path(inventory_registration_path).stat().st_size,
                          "sha256": expected_inventory_registration_file_sha256})
    require(inventory.get("id") == REGISTRATION_ID and type(inventory.get("schema_version")) is int
            and inventory["schema_version"] == 1
            and inventory.get("sha256") == digest({k: v for k, v in inventory.items() if k != "sha256"})
            and inventory.get("config_file_sha256") == expected_config_file_sha256
            and inventory.get("profile_body_sha256") == expected_profile_body_sha256
            and inventory.get("arithmetic_plan_file_sha256") == expected_plan_file_sha256,
            "independently selected scope/plan/inventory header before imports")
    capture_pin = selected_sources(inventory, expected_source_commit)
    capture = _capture_module(capture_pin)
    # Original qualification/source/runtime checks happen before inherited
    # decoder/model-byte closure. Missing current qualification is a real refusal.
    plan_ref = capture.PLAN_REF
    require(type(plan_ref) is dict and plan_ref["sha256"] == expected_plan_file_sha256, "independently selected arithmetic plan")
    plan = capture.plan()
    extra_data = arithmetic_references(plan, plan_ref, expected_plan_file_sha256)
    expected = capture.build_scope(*expected_selector)
    scope = read_ref({"path": str(Path(config_path).absolute()), "bytes": Path(config_path).stat().st_size,
                      "sha256": expected_config_file_sha256})
    view = capture._original()
    view.build_scope = capture.build_scope
    view.preflight = capture.preflight
    view.integrated = capture.integrated
    admit(scope, plan, inventory, {**expected, "sha256": digest(expected)}, selected=expected_selector,
          source_commit=expected_source_commit, profile_sha256=expected_profile_body_sha256)
    if str(PACKAGE) not in sys.path:
        sys.path.insert(0, str(PACKAGE))
    from inference_component import validate_row, ROW_FIELDS, EOS_IDS, VERSION
    tree, changes, text = prepared_source()
    namespace = {"Path": Path, "datetime": datetime, "timezone": timezone, "collections": collections,
                 "json": json, "math": math, "sys": sys, "os": os, "importlib": importlib,
                 "require": require, "digest": digest, "contract": view, "HERE": HERE, "EXPECTED": EXPECTED,
                 "CAPTURE": CAPTURE, "package": PACKAGE, "root": PACKAGE.parents[1], "SIGNAL": SIGNAL,
                 "CONTRACT_VERSION": view.VERSION, "PROFILE_ID": "SHORT_M0_PHYSICAL_RAM_COHERENT_EXTERNAL_CHECK_V2",
                 "REGISTRATION_ID": REGISTRATION_ID, "SECTIONS": SECTIONS, "selector": selector,
                 "_capture_view": lambda: view, "validate_row": validate_row, "ROW_FIELDS": ROW_FIELDS,
                 "EOS_IDS": EOS_IDS, "VERSION": VERSION}
    original_tree = ast.parse(ORIGINAL.read_bytes())
    operation = next(n for n in original_tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "ALLOWED_OPERATIONS" for t in n.targets))
    namespace["ALLOWED_OPERATIONS"] = ast.literal_eval(operation.value)
    exec(compile(tree, str(ORIGINAL), "exec"), namespace)
    result = namespace["audit_terminal"](config_path, raw_directory, preserved_directory,
        expected_selector=expected_selector, expected_config_file_sha256=expected_config_file_sha256,
        expected_preservation_body_sha256=expected_preservation_body_sha256,
        inventory_registration_path=inventory_registration_path,
        expected_inventory_registration_file_sha256=expected_inventory_registration_file_sha256,
        expected_profile_body_sha256=expected_profile_body_sha256,
        terminal_preservation_signal=SIGNAL, output_path=None)
    arithmetic_references(plan, plan_ref, expected_plan_file_sha256)
    selected_sources(inventory, expected_source_commit)
    result.update(arithmetic_data_reference_closure=extra_data,
                  arithmetic_original_auditor_sha256=ORIGINAL_SHA256,
                  arithmetic_exact_source_adaptations=changes,
                  arithmetic_current_qualified_scope_verified=True)
    result["sha256"] = digest({k: v for k, v in result.items() if k != "sha256"})
    if output_path is not None:
        path = Path(output_path).absolute()
        require(path.parent.resolve() == (PACKAGE / "evidence").resolve() and path.suffix == ".json"
                and path.name.startswith(Path(raw_directory).name + "_independent_audit_")
                and not path.exists() and not path.is_symlink(), "new owned arithmetic audit receipt")
        raw = (canonical(result) + "\n").encode("ascii")
        with path.open("xb") as stream:
            require(stream.write(raw) == len(raw), "complete audit publication")
            stream.flush()
            os.fsync(stream.fileno())
        require(path.read_bytes() == raw, "published audit bytes")
    return result
