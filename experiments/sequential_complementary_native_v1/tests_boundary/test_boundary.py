"""Fresh stdlib boundary tests, not inherited/native/full-checkout qualification."""
import ast
import copy
import inspect
import json
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest
from experiments.sequential_complementary_native_v1 import boundary as b
from experiments.sequential_complementary_native_v1.receipt_review import (
    LEGACY_CLOSED, REQUIRED_CHECKS, SUBJECT, screen_student_result,
)

PACKAGE = Path(b.__file__).parent


def host():
    return {**{key: "fixture-" + key for key in b.HOST_TEXT},
            "python_executable_sha256": "1" * 64, "gpu_uuid": "GPU-fixture-only",
            "gpu_total_bytes": 8 * 2**30, "runtime": copy.deepcopy(b.RUNTIME),
            "device": "cuda:0", "compute_dtype": "float32", "attention": "eager"}


def checkout():
    return {"commit": "2" * 40, "tree": "3" * 40, "branch": b.BRANCH, "parent": b.PARENT}


def request():
    return b.request_for(host(), checkout())


def reseal(body):
    return {**body, "sha256": b.digest(body)}


def receipt_fixture():
    binding = {"schema": "SCP_NATIVE_BINDING_V1", "experiment": "SCP-NQ-001",
               **{key: False for key in LEGACY_CLOSED}}
    runtime = {"fixture_only": True, "host": "never-native"}
    state = {"parameters": "4" * 64, "buffers": "5" * 64}
    body = {"schema": "SCP_NATIVE_RESULT_V1", "experiment": "SCP-NQ-001", "phase": "student",
            "status": "PASS_QUALIFICATION_ONLY", "evidence_class": "NATIVE_QUALIFICATION_ONLY",
            "binding_sha256": b.digest(binding), "runtime": runtime, "launch_sha256": "6" * 64,
            "worker_pid": 123, "student_run": None,
            "loading": {"subject_id": SUBJECT, "source_weight_sha256": b.WEIGHTS["source"],
                        "trained_partial_sha256": b.WEIGHTS["trained_partial"],
                        "subject_weight_sha256": b.WEIGHTS["subject"],
                        "full_tensors_verified": 310, "trained_tensors_reconstructed": 220,
                        "subject_state": state, "storage_tying": True, "optimizer_steps": 0},
            "checks": {"state_before": state, "state_after": copy.deepcopy(state), "zero_update": True,
                       "backward_calls": 0, "optimizer_steps": 0, "currently_trainable_tensors": 0,
                       "answers_scored": 0, "training_shape_qualified": False,
                       "qualified_sequence_tokens": 2048, "checks": sorted(REQUIRED_CHECKS)},
            "generation_requests": 0, "generated_tokens_actual": 0,
            "forward_calls_actual_including_failures": 28,
            **{key: False for key in LEGACY_CLOSED}}
    return binding, runtime, body


def test_lock_has_no_bypass_parameter():
    assert not inspect.signature(b.require_native_review).parameters
    with pytest.raises(b.NativeReviewLocked, match="NATIVE_REVIEW_LOCKED"):
        b.require_native_review()


@pytest.mark.parametrize("env", ["SCP_NATIVE_APPROVED", "SCP_NATIVE_ENABLED", "SCP_REVIEW_PASSED", "TRAINING_PERMISSION"])
def test_environment_never_opens_lock(monkeypatch, env):
    monkeypatch.setenv(env, "true")
    with pytest.raises(b.NativeReviewLocked):
        b.require_native_review()


def test_pending_request_is_not_grant_and_is_copied():
    h, c = host(), checkout()
    value = b.request_for(h, c)
    h["gpu_uuid"] = "changed"; c["commit"] = "changed"
    b.validate_request(value)
    assert value["limits"]["model_worker_launches"] == 1
    assert value["limits"]["generation_requests"] == 0
    assert value["approval_reference"] is None
    assert not any(value["permissions"].values())
    assert value["native_review_lock"] is True


@pytest.mark.parametrize("key", b.CLOSED)
def test_each_permission_stays_false(key):
    value = request(); value["permissions"][key] = True
    with pytest.raises(ValueError): b.validate_request(value)


@pytest.mark.parametrize("key", b.LIMITS)
def test_every_numeric_bound_is_frozen(key):
    value = request(); value["limits"][key] += 1
    with pytest.raises(ValueError): b.validate_request(value)


@pytest.mark.parametrize("invalid", [False, True, 1.0, "1", None])
def test_numeric_type_confusion_refused(invalid):
    value = request(); value["limits"]["model_worker_launches"] = invalid
    with pytest.raises(ValueError): b.validate_request(value)


@pytest.mark.parametrize("key", b.HOST_TEXT)
def test_each_exact_host_text_field_required(key):
    h = host(); h[key] = None
    with pytest.raises(ValueError): b.request_for(h, checkout())


@pytest.mark.parametrize("key,value", [
    ("gpu_uuid", "cuda:0"), ("gpu_total_bytes", True), ("python_executable_sha256", "missing"),
    ("runtime", {"torch": "2.11"}), ("compute_dtype", "bfloat16"),
    ("attention", "flash_attention_2"), ("device", "cpu"),
])
def test_wrong_host_or_runtime_refused(key, value):
    h = host(); h[key] = value
    with pytest.raises(ValueError): b.request_for(h, checkout())


@pytest.mark.parametrize("key,value", [
    ("status", "APPROVED"), ("approval_reference", "old RRP permit"),
    ("assets_verified", True), ("native_review_lock", False),
    ("host_observation_independently_verified", True), ("phase", "mr"),
])
def test_draft_cannot_self_approve(key, value):
    item = request(); item[key] = value
    with pytest.raises(ValueError): b.validate_request(item)


def test_unknown_grant_fields_and_changed_host_hash_refused():
    item = request(); item["approved"] = True
    with pytest.raises(ValueError): b.validate_request(item)
    item = request(); item["host"]["driver_version"] = "other"
    with pytest.raises(ValueError): b.validate_request(item)


def test_three_assets_cannot_be_conflated():
    item = request(); item["weight_sha256"]["subject"] = b.WEIGHTS["source"]
    with pytest.raises(ValueError): b.validate_request(item)


@pytest.mark.parametrize("value", [None, [], (), True, "7", [True], [7, 7], [-1], [151936], [8]])
def test_bad_generation_eos_refused_without_fallback(value):
    with pytest.raises(ValueError): b.generation_eos_ids(value, vocab_size=151936, tokenizer_eos=7)


def test_full_generation_eos_preserved_and_used():
    ids = b.generation_eos_ids([9, 7], vocab_size=151936, tokenizer_eos=7)
    assert ids == (7, 9)
    assert b.terminal_token(9, ids)  # Regression: non-tokenizer EOS also stops.
    assert b.terminal_token(7, ids)
    assert not b.terminal_token(8, ids)
    assert b.generation_eos_ids(7, vocab_size=151936, tokenizer_eos=7) == (7,)


@pytest.mark.parametrize("token", [True, -1, "7", None])
def test_bad_emitted_token_refused(token):
    with pytest.raises(ValueError): b.terminal_token(token, (7, 9))


@pytest.mark.parametrize("command", ["qualify", "_worker", "propose"])
def test_cli_refuses_before_runtime_import_or_argument_access(command, tmp_path):
    # A partial source checkout with NO MQ/RRP/transformers dependencies is enough
    # to test this refusal. Any attempted native import would fail this test.
    result = subprocess.run([sys.executable, "-B", "-m", "experiments.sequential_complementary_native_v1",
                             command, "--grant", str(tmp_path / "nonexistent.json")],
                            cwd=PACKAGE.parents[1], text=True, capture_output=True, timeout=10)
    assert result.returncode == 2
    assert "NATIVE_REVIEW_LOCKED" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert list(tmp_path.iterdir()) == []


def function_node(path, name):
    tree = ast.parse(path.read_text())
    found = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(found) == 1
    return copy.deepcopy(found[0])


@pytest.mark.parametrize("file,name,args", [
    ("run.py", "launch", [None]), ("run.py", "worker", [None]),
    ("native.py", "_load_subject", [None, None]),
    ("native.py", "open_backend", [None] * 5),
    ("native.py", "require_native_backend", [None] * 5),
    ("native.py", "authorize", [None] * 5), ("native.py", "generate", [None] * 3),
])
def test_real_function_first_statement_refuses_before_dependencies(file, name, args):
    # Execute the source function in isolation with NO model/dependency globals.
    # This proves ordering of the actual guard, not full-module import qualification.
    node = function_node(PACKAGE / file, name)
    assert isinstance(node.body[0], ast.Expr)
    assert ast.unparse(node.body[0]) == "require_native_review()"
    node.decorator_list = []
    namespace = {"require_native_review": b.require_native_review}
    exec(compile(ast.Module(body=[node], type_ignores=[]), file, "exec"), namespace)
    with pytest.raises(b.NativeReviewLocked): namespace[name](*args)


def test_live_forward_guard_is_before_model_and_charge():
    node = function_node(PACKAGE / "native.py", "forward")
    assert ast.unparse(node.body[0].test) == "self._key is _LIVE"
    assert ast.unparse(node.body[0].body[0]) == "require_native_review()"


def test_eos_fix_is_wired_to_real_loader_and_decoder():
    source = (PACKAGE / "native.py").read_text()
    loader = ast.unparse(function_node(PACKAGE / "native.py", "_load_subject"))
    decoder = ast.unparse(function_node(PACKAGE / "native.py", "generate"))
    assert 'generation_config.json' in loader and 'loaded_eos == declared_eos' in loader
    assert 'generation_eos_ids(self.model.generation_config.eos_token_id' in decoder
    assert 'terminal_token(token, eos_ids)' in decoder
    assert 'token == self.tokenizer.eos_token_id' not in decoder
    assert 'generation_eos_ids' in source


def test_records_create_only_and_external(tmp_path):
    root = tmp_path / "checkout"; root.mkdir()
    out = tmp_path / "request.json"
    b.write_new(out, request(), root)
    assert b.read_object(out) == request()
    with pytest.raises(FileExistsError): b.write_new(out, request(), root)
    with pytest.raises(ValueError): b.write_new(root / "inside.json", request(), root)


def test_record_symlinks_and_traversal_refused(tmp_path):
    root = tmp_path / "checkout"; root.mkdir()
    target = tmp_path / "target"; target.mkdir()
    link = tmp_path / "link"; link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError): b.write_new(link / "x.json", request(), root)
    with pytest.raises(ValueError): b.write_new(target / ".." / "x.json", request(), root)
    (target / "x.json").write_text('{}')
    with pytest.raises(ValueError): b.read_object(link / "x.json")


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '[]'])
def test_bad_json_rejected(text, tmp_path):
    path = tmp_path / "bad.json"; path.write_text(text)
    with pytest.raises(ValueError): b.read_object(path)


def test_oversized_json_refused(tmp_path):
    path = tmp_path / "big.json"; path.write_bytes(b' ' * (2**20 + 1))
    with pytest.raises(ValueError): b.read_object(path)


def test_result_candidate_never_opens_any_lane():
    binding, runtime, body = receipt_fixture()
    result = screen_student_result(reseal(body), binding=binding, runtime=runtime)
    assert result["status"] == "CANDIDATE_REQUIRES_TERMINAL_AND_JOURNAL_VERIFICATION"
    assert result["native_pass"] is False and not any(result["permissions"].values())
    assert len(result["unverified"]) == 4


@pytest.mark.parametrize("key,value", [
    ("evidence_class", "SOFTWARE_FIXTURES_ONLY"), ("phase", "mr"),
    ("status", "EXITED"), ("generation_requests", 1), ("generated_tokens_actual", 1),
    ("generation_requests", False), ("forward_calls_actual_including_failures", 65),
    ("forward_calls_actual_including_failures", 0), ("forward_calls_actual_including_failures", True),
    ("worker_pid", True), ("student_run", "old-run"), ("binding_sha256", "0" * 64),
])
def test_wrong_result_scope_refused(key, value):
    binding, runtime, body = receipt_fixture(); body[key] = value
    with pytest.raises(ValueError): screen_student_result(reseal(body), binding=binding, runtime=runtime)


@pytest.mark.parametrize("key", ["backward_calls", "optimizer_steps", "currently_trainable_tensors", "answers_scored"])
def test_closed_operation_counts_refused(key):
    binding, runtime, body = receipt_fixture(); body["checks"][key] = 1
    with pytest.raises(ValueError): screen_student_result(reseal(body), binding=binding, runtime=runtime)


@pytest.mark.parametrize("key", LEGACY_CLOSED)
def test_result_cannot_open_closed_permissions(key):
    binding, runtime, body = receipt_fixture(); body[key] = True
    with pytest.raises(ValueError): screen_student_result(reseal(body), binding=binding, runtime=runtime)


def test_result_state_subject_runtime_and_seal_refused():
    binding, runtime, body = receipt_fixture()
    altered = copy.deepcopy(body); altered["checks"]["state_after"]["parameters"] = "0" * 64
    with pytest.raises(ValueError): screen_student_result(reseal(altered), binding=binding, runtime=runtime)
    altered = copy.deepcopy(body); altered["loading"]["subject_weight_sha256"] = b.WEIGHTS["source"]
    with pytest.raises(ValueError): screen_student_result(reseal(altered), binding=binding, runtime=runtime)
    with pytest.raises(ValueError): screen_student_result(reseal(body), binding=binding, runtime={"other": True})
    altered = reseal(body); altered["worker_pid"] = 456
    with pytest.raises(ValueError): screen_student_result(altered, binding=binding, runtime=runtime)


def test_missing_mechanical_check_refused():
    binding, runtime, body = receipt_fixture(); body["checks"]["checks"].pop()
    with pytest.raises(ValueError): screen_student_result(reseal(body), binding=binding, runtime=runtime)


def test_status_and_template_do_not_import_model_stack():
    result = subprocess.run([sys.executable, "-B", "-c",
        "import sys; from experiments.sequential_complementary_native_v1.boundary import status,host_template; "
        "assert not status()['native_launch_available']; assert host_template()['host_id'] is None; "
        "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],
        cwd=PACKAGE.parents[1], text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_checkout_snapshot_rejects_wrong_branch_and_dirty_tree(tmp_path, monkeypatch):
    def output(command, **kwargs):
        args = command[3:]
        if args == ['rev-parse', '--show-toplevel']: return str(tmp_path)
        if args == ['branch', '--show-current']: return 'main'
        raise AssertionError('later Git operation reached')
    monkeypatch.setattr(b.subprocess, 'check_output', output)
    with pytest.raises(ValueError, match='wrong successor branch'): b.checkout_snapshot(tmp_path)
    def dirty(command, **kwargs):
        args = command[3:]
        if args == ['rev-parse', '--show-toplevel']: return str(tmp_path)
        if args == ['branch', '--show-current']: return b.BRANCH
        if args == ['status', '--porcelain', '--untracked-files=all']: return '?? dirty.py'
        raise AssertionError('later Git operation reached')
    monkeypatch.setattr(b.subprocess, 'check_output', dirty)
    with pytest.raises(ValueError, match='dirty checkout'): b.checkout_snapshot(tmp_path)
