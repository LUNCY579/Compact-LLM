"""Containment marker/clock checks before any model import or child launch."""
from types import SimpleNamespace
from pathlib import Path
import pytest
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3.fresh_entry import (
    validate_parent_binding, validate_clock, validate_release, validate_process_chain,
)


def fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    args = SimpleNamespace(source=source, out=run / "result", release=run / "release.json")
    admission = {"source": str(source), "lane": "lane", "study_id": "study"}
    lock = {"out": str(run), "lane": "lane", "parent_pid": 71, "token": "one-owned-token"}
    return args, admission, lock


def test_exact_parent_paths_and_owner(tmp_path):
    args, admission, lock = fixture(tmp_path)
    validate_parent_binding(args, admission, lock)
    args.release = args.out / "release.json"
    with pytest.raises(PermissionError): validate_parent_binding(args, admission, lock)


@pytest.mark.parametrize("field,value", [("token", ""), ("parent_pid", 0), ("lane", "other")])
def test_unowned_marker_cannot_import_worker(tmp_path, field, value):
    args, admission, lock = fixture(tmp_path)
    lock[field] = value
    with pytest.raises(PermissionError): validate_parent_binding(args, admission, lock)


def test_source_and_output_arguments_cannot_redirect(tmp_path):
    args, admission, lock = fixture(tmp_path)
    args.source = tmp_path / "other_source"
    with pytest.raises(PermissionError): validate_parent_binding(args, admission, lock)
    args.source = Path(admission["source"])
    args.out = tmp_path / "other_output"
    with pytest.raises(PermissionError): validate_parent_binding(args, admission, lock)


@pytest.mark.parametrize("started,now", [(float("nan"), 1.), (float("inf"), 1.),
    (2., 1.), (1., 15601.), (True, 2.), (1., float("nan"))])
def test_invalid_parent_clock_stops_before_import(started, now):
    with pytest.raises(PermissionError): validate_clock(started, now, 15600)


def test_live_clock_and_token_release():
    validate_clock(10., 11., 15600)
    admission = {"study_id": "study"}; lock = {"token": "owned"}
    release = {"contained": True, "admission_sha256": "sha", "study_id": "study", "token": "owned"}
    validate_release(release, admission, lock, "sha")
    release["token"] = "forged"
    with pytest.raises(PermissionError): validate_release(release, admission, lock, "sha")


def chain_fixture():
    lock = {"parent_pid": 10, "parent_created": 1., "runtime_exe": "runtime.exe",
            "launcher_pid": 20, "launcher_created": 2., "launcher_exe": "venv.exe"}
    supervisor = {"pid": 10, "parent_pid": 1, "created": 1., "exe": "runtime.exe"}
    launcher = {"pid": 20, "parent_pid": 10, "created": 2., "exe": "venv.exe"}
    child = {"pid": 30, "parent_pid": 20, "created": 3., "exe": "runtime.exe"}
    return lock, supervisor, launcher, child


def test_owned_direct_and_windows_redirector_processes():
    lock, supervisor, launcher, child = chain_fixture()
    validate_process_chain(lock, child, launcher, supervisor)
    direct = {**launcher, "exe": "runtime.exe"}
    validate_process_chain(lock, direct, supervisor, supervisor)


@pytest.mark.parametrize("field,value", [("pid", 21), ("created", 8.), ("parent_pid", 11), ("exe", "foreign.exe")])
def test_foreign_or_reused_redirector_rejected(field, value):
    lock, supervisor, launcher, child = chain_fixture()
    launcher[field] = value
    with pytest.raises(PermissionError): validate_process_chain(lock, child, launcher, supervisor)
