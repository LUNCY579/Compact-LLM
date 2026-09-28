"""Amendment 1 sections 1 and 11: data gates, scope receipts and the sealed-bank guard.

Synthetic fixtures run everywhere. Data-backed tests read only TRAIN, DEV and CALIBRATION files and
skip when the dataset is absent (COMP_RS_DATA overrides the default root).
"""
from __future__ import annotations

import _io
import copy
import io
import json
import os
from pathlib import Path
import shutil
import tokenize

import pytest

from experiments.maxrl_comp002_pretraining_v1 import data_gates as dg
from experiments.maxrl_comp002_pretraining_v1 import scope_receipt as sr
from experiments.maxrl_comp002_pretraining_v1.preflight import digest
from experiments.maxrl_comp002_pretraining_v1.q1_dev_scope import sha256 as q1_sha256

PACKAGE = Path(__file__).resolve().parents[1]
DIAGNOSTICS = PACKAGE.parent / "maxrl_composable_skills_v2/evidence/quality_diagnostics.json"
SEALED_MANIFESTS = ("test_core.json", "test_boundary.json", "challenge_core.json", "challenge_boundary.json",
                    "guard_test.json", "guard_challenge.json")


def norm(path) -> str:
    """The form conftest's Guard.record_opens logs."""
    return os.path.normcase(os.path.realpath(path))


def opened_under(opened: list[str], root: Path) -> list[str]:
    base = norm(root)
    return [p for p in opened if p == base or p.startswith(base + os.sep)]


# ---------------------------------------------------------------- sealed-bank guard

def test_guard_fires_on_synthetic_sealed_file(tmp_path, forbidden_data_guard, monkeypatch):
    canonical = tmp_path / "data" / "canonical"
    canonical.mkdir(parents=True)
    target = canonical / "test.jsonl"
    with forbidden_data_guard.real_open(target, "xb") as stream:
        stream.write(b"{}\n")
    monkeypatch.chdir(canonical)
    attempts = (lambda: open(target), lambda: open(str(target), "rb"), lambda: io.open(target),
                lambda: target.open(), target.read_text, target.read_bytes, lambda: q1_sha256(target),
                lambda: open("test.jsonl"), lambda: Path("test.jsonl").read_bytes(),
                lambda: open(target, "a"))
    for attempt in attempts:
        with pytest.raises(forbidden_data_guard.error):
            attempt()
    recorded = len(forbidden_data_guard.violations)
    forbidden_data_guard.violations.clear()
    assert recorded >= len(attempts)
    assert target.stat().st_size == 3  # nothing was appended


def test_guard_records_even_when_the_caller_swallows(tmp_path, forbidden_data_guard):
    sealed = tmp_path / "data" / "oracles" / "guard_challenge.jsonl"
    sealed.parent.mkdir(parents=True)
    try:
        open(sealed, "w")
    except Exception:
        pass
    recorded = list(forbidden_data_guard.violations)
    forbidden_data_guard.violations.clear()
    assert recorded == [str(sealed)]
    assert not sealed.exists()


def test_guard_name_rule(tmp_path, forbidden_data_guard, monkeypatch):
    check = forbidden_data_guard.is_forbidden
    sealed = ("test.jsonl", "challenge.jsonl", "guard_test.jsonl", "guard_challenge.jsonl", "TEST.JSONL",
              "test.jsonl.", "test.jsonl ", "test.jsonl::$DATA") + SEALED_MANIFESTS
    for name in sealed:
        assert check(tmp_path / "data" / "canonical" / name)
        assert check(tmp_path / "Data" / "x" / "y" / name)
    relocated = tmp_path / "rineya_copy"
    assert not any(part.casefold() == "data" for part in relocated.parts)
    monkeypatch.delenv("COMP_RS_DATA", raising=False)
    for path in (tmp_path / "data" / "canonical" / "dev.jsonl", tmp_path / "data" / "test.jsonl.bak",
                 tmp_path / "data" / "manifests" / "dev_core.json", relocated / "canonical" / "test.jsonl"):
        assert not check(path)
    assert not check(3)
    # DS-2: a data root named by COMP_RS_DATA is guarded whatever its folder name; so is the default root.
    assert os.path.normcase(os.path.abspath(forbidden_data_guard.default_data)) in forbidden_data_guard.guarded_roots()
    monkeypatch.setenv("COMP_RS_DATA", str(relocated))
    assert os.path.normcase(os.path.abspath(relocated)) in forbidden_data_guard.guarded_roots()
    for name in sealed:
        assert check(relocated / "canonical" / name) and check(relocated / name)
    assert not check(relocated / "canonical" / "dev.jsonl")
    assert not check(tmp_path / "elsewhere" / "test.jsonl")  # synthetic sealed-role fixtures live outside every root
    monkeypatch.setenv("COMP_RS_DATA", str(Path("/srv/other")))
    assert check(Path("/srv/other/test.jsonl")) and check(Path("/srv/other/manifests/guard_test.json"))


def _pair(guard, root: Path, route: str, content: bytes) -> tuple[Path, Path]:
    """An allowed file and a sealed-named file with the same bytes, in their own folder under `root`."""
    folder = root / route.replace(".", "_")
    folder.mkdir(parents=True)
    allowed, sealed = folder / "dev.jsonl", folder / "test.jsonl"
    for path in (allowed, sealed):
        with guard.real_open(path, "xb") as stream:
            stream.write(content)
    return allowed, sealed


def _assert_every_route_blocked(guard, root: Path, routes: dict) -> None:
    """Each route reads the allowed file (positive control) and raises the guard's error on the sealed one."""
    for label, (content, read, expected) in routes.items():
        allowed, sealed = _pair(guard, root, label, content)
        assert read(allowed) == expected, label
        before = len(guard.violations)
        with pytest.raises(guard.error):
            read(sealed)
        assert len(guard.violations) > before, label
    guard.violations.clear()


def _read(stream):
    with stream as handle:
        return handle.read()


def _os_read(path) -> bytes:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        return os.read(fd, 1 << 16)
    finally:
        os.close(fd)


def _copy_then_read(path: Path) -> bytes:
    copy_path = path.with_name(path.name + ".copy")
    shutil.copyfile(path, copy_path)
    return copy_path.read_bytes()


def _link_then_read(path: Path) -> bytes:
    link = path.with_name("linked.bin")
    os.link(path, link)
    return link.read_bytes()


def test_guard_blocks_every_open_route_under_a_relocated_data_root(tmp_path, forbidden_data_guard, monkeypatch):
    """DS-2: a synthetic root not named 'data', guarded through COMP_RS_DATA; stdlib and numpy routes."""
    np = pytest.importorskip("numpy")
    root = tmp_path / "rineya_copy"
    assert not any(part.casefold() == "data" for part in root.parts)
    monkeypatch.setenv("COMP_RS_DATA", str(root))
    raw = b'{"id":"synthetic"}\n'
    npy = io.BytesIO()
    np.save(npy, np.arange(4, dtype=np.int64))
    routes = {
        "builtins.open": (raw, lambda p: _read(open(p, "rb")), raw),
        "io.open": (raw, lambda p: _read(io.open(p, "rb")), raw),
        "_io.open": (raw, lambda p: _read(_io.open(p, "rb")), raw),
        "io.FileIO": (raw, lambda p: _read(io.FileIO(p, "r")), raw),
        "io.open_code": (raw, lambda p: _read(io.open_code(str(p))), raw),
        "os.open": (raw, _os_read, raw),
        "tokenize.open": (raw, lambda p: _read(tokenize.open(p)), raw.decode("utf-8")),
        "Path.open": (raw, lambda p: _read(p.open("rb")), raw),
        "Path.read_bytes": (raw, lambda p: p.read_bytes(), raw),
        "Path.read_text": (raw, lambda p: p.read_text(encoding="utf-8"), raw.decode("utf-8")),
        "shutil.copyfile": (raw, _copy_then_read, raw),
        "os.link": (raw, _link_then_read, raw),
        "np.fromfile": (raw, lambda p: np.fromfile(p, dtype=np.uint8).tobytes(), raw),
        "np.memmap": (raw, lambda p: np.memmap(p, dtype=np.uint8, mode="r").tobytes(), raw),
        "np.load": (npy.getvalue(), lambda p: np.load(p).tolist(), [0, 1, 2, 3]),
    }
    if os.name == "nt":  # spellings Windows opens as the same file
        for suffix, label in ((".", "dot"), (" ", "space"), ("::$DATA", "stream")):
            routes["os.open+" + label] = (raw, lambda p, s=suffix: _os_read(str(p) + s), raw)
            routes["open+" + label] = (raw, lambda p, s=suffix: _read(open(str(p) + s, "rb")), raw)
        routes["os.open+upper"] = (raw, lambda p: _os_read(p.with_name(p.name.upper())), raw)
    _assert_every_route_blocked(forbidden_data_guard, root, routes)
    # A symlink with an innocent name is resolved (skipped where symlinks need a privilege).
    target = root / "builtins_open" / "test.jsonl"
    link = root / "innocent.txt"
    try:
        os.symlink(target, link)
    except OSError:
        return
    with pytest.raises(forbidden_data_guard.error):
        link.read_bytes()
    forbidden_data_guard.violations.clear()


@pytest.mark.filterwarnings("ignore:TypedStorage is deprecated")
def test_guard_blocks_torch_and_safetensors_routes_under_a_relocated_data_root(tmp_path, forbidden_data_guard,
                                                                              monkeypatch):
    """DS-2: the C++/Rust opens the audit hook cannot see, including a `from safetensors import safe_open` binding."""
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors")
    import safetensors.numpy
    import safetensors.torch
    from experiments.maxrl_comp002_pretraining_v1 import arm_build
    root = tmp_path / "rineya_copy"
    monkeypatch.setenv("COMP_RS_DATA", str(root))
    raw = bytes(range(16))
    tensors = safetensors.torch.save({"w": torch.arange(3, dtype=torch.float32)})
    pickled = io.BytesIO()
    torch.save(torch.arange(3), pickled)

    def keys_and_values(handle_factory):
        with handle_factory() as handle:
            return {k: handle.get_tensor(k).tolist() for k in handle.keys()}

    routes = {
        "torch.from_file": (raw, lambda p: torch.from_file(str(p), size=len(raw), dtype=torch.uint8).numpy().tobytes(),
                            raw),
        "UntypedStorage.from_file": (raw, lambda p: bytes(torch.UntypedStorage.from_file(str(p), False, len(raw)).tolist()),
                                     raw),
        "ByteStorage.from_file": (raw, lambda p: bytes(torch.ByteStorage.from_file(str(p), False, len(raw)).tolist()),
                                  raw),
        "torch.load": (pickled.getvalue(), lambda p: torch.load(p, weights_only=True).tolist(), [0, 1, 2]),
        "safetensors.safe_open": (tensors, lambda p: keys_and_values(lambda: safetensors.safe_open(str(p), "pt")),
                                  {"w": [0.0, 1.0, 2.0]}),
        "arm_build.safe_open": (tensors, lambda p: keys_and_values(lambda: arm_build.safe_open(str(p), framework="pt")),
                                {"w": [0.0, 1.0, 2.0]}),
        "safetensors.torch.load_file": (tensors, lambda p: {k: v.tolist() for k, v in
                                                            safetensors.torch.load_file(str(p)).items()},
                                        {"w": [0.0, 1.0, 2.0]}),
        "safetensors.numpy.load_file": (tensors, lambda p: {k: v.tolist() for k, v in
                                                            safetensors.numpy.load_file(str(p)).items()},
                                        {"w": [0.0, 1.0, 2.0]}),
    }
    _assert_every_route_blocked(forbidden_data_guard, root, routes)


def test_guard_fails_the_test_for_attempts_inside_it_and_outside_any_test(forbidden_data_guard):
    """An attempt recorded outside any test (collection, a wider-scoped fixture) fails the next test's setup."""
    fresh = type(forbidden_data_guard)()
    fresh.violations.append("attempt during collection")
    with pytest.raises(pytest.fail.Exception, match="outside any test"):
        next(fresh.per_test())
    assert fresh.violations == []
    body = fresh.per_test()
    assert next(body) is fresh
    fresh.violations.append("attempt in the test")
    with pytest.raises(pytest.fail.Exception, match="opened: attempt in the test"):
        next(body)
    assert fresh.violations == []
    with pytest.raises(ValueError):
        forbidden_data_guard.real_open(Path("unused.bin"), "rb")  # the fixture writer never reads


def test_guard_allows_other_files(tmp_path, forbidden_data_guard):
    if any(part.lower() == "data" for part in tmp_path.parts):
        pytest.skip("tmp_path already lies under a 'data' directory")
    allowed = [tmp_path / "data" / "canonical" / "dev.jsonl", tmp_path / "elsewhere" / "test.jsonl"]
    for path in allowed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}\n")
        assert path.read_text(encoding="utf-8") == "{}\n"
        with open(path, "rb") as stream:
            assert stream.read() == b"{}\n"
    assert forbidden_data_guard.violations == []


# ---------------------------------------------------------------- synthetic fixtures

def links_for(resolved: list[int], n: int) -> dict:
    links = {"r" + str(i): i for i in range(n)}
    links.update({"a" + str(i): "r" + str(i) for i in resolved})
    return {"links": links, "requested": ["a" + str(i) for i in reversed(resolved)]}


INITIAL = [1, 1, 1, 2, 2, 2, 4, 4, 4]


def make_row(skills, spec, *, parent="world-1", role="dev", tag="core", rid=None):
    return {"id": rid or ("MC2-" + "".join(skills) + "-" + parent), "parent_id": parent, "role": role,
            "kind": "composition", "tag": tag, "skills": list(skills), "data_source": "rineya_maxrl_comp_v2",
            "verifier": {"version": "2.0.0", "kind": "composition", "spec": spec}}


def rs_spec(resolved=(0, 1, 3, 4, 6, 7), k=2, target=3):
    return {"initial": INITIAL[:], "lookup": links_for(list(resolved), 9), "selection": {"k": k, "target": target}}


def test_rs_floor_by_hand():
    rows = [make_row("RS", rs_spec(), parent="world-a"),
            make_row("RS", rs_spec(resolved=(1, 2, 4, 5, 7, 8)), parent="world-b"),
            make_row("RS", rs_spec(k=3, target=7), parent="world-c"),
            make_row("RS", rs_spec(), parent="world-d", tag="boundary"),
            make_row("S", {"initial": INITIAL[:], "selection": {"k": 2, "target": 3}}, parent="world-a")]
    result = dg.rs_floor(rows)
    assert result["rows"] == 3
    # k=2 target 3: 9 blind pairs (one 1 and one 2), 2x2 inside. k=3 target 7: 27 triples, 2x2x2 inside.
    p = {r["parent_id"]: r["p_blind"] for r in result["per_row"]}
    assert p == {"world-a": 4 / 9, "world-b": 4 / 9, "world-c": 8 / 27}
    assert result["by_k"]["2"] == {"rows": 2, "mean_p_blind": 4 / 9, "first_lex_pass": 1}
    assert result["by_k"]["3"]["first_lex_pass"] == 1
    assert result["first_lex_pass"] == 2  # (0,3) is inside for world-a and world-c, outside for world-b
    empty = dg.rs_floor([])
    assert empty["mean_p_blind"] is None and empty["first_lex_pass"] is None and empty["status"] == "NO_ROWS"
    assert result["status"] == "COMPUTED"


def test_rs_floor_refuses_when_the_verifier_disagrees_with_resolved_set_inclusion(monkeypatch):
    """TA-12: the floor/verifier cross-check raises instead of reporting a floor."""
    rows = [make_row("RS", rs_spec(), parent="world-a")]
    real_score = dg.score
    monkeypatch.setattr(dg, "score", lambda spec, response: 1 - real_score(spec, response))
    with pytest.raises(ValueError, match="verifier disagrees"):
        dg.rs_floor(rows)


def test_triple_structure_pass_and_fail():
    good = make_row("RS", rs_spec(), parent="world-a")
    r_row = make_row("R", {"initial": INITIAL[:], "lookup": links_for([0, 2, 3, 5, 6, 8], 9)}, parent="world-a")
    three = make_row("R", {"initial": INITIAL[:], "lookup": links_for([0, 1, 2, 3, 4, 6, 7], 9)}, parent="world-b")
    quad = make_row("RS", {"initial": [1, 1, 1, 1, 2, 2, 4, 4, 4], "lookup": links_for([0, 1, 4, 6, 7], 9),
                           "selection": {"k": 2, "target": 3}}, parent="world-c")
    assert dg.triple_structure([good, r_row])["status"] == "PASS"
    result = dg.triple_structure([good, r_row, three, quad])
    assert result["status"] == "FAIL"
    assert result["failures"] == sorted([three["id"], quad["id"]])
    assert result["by_signature"] == {"R": {"rows": 2, "pass": 1}, "RS": {"rows": 2, "pass": 1}}
    # F13 / C9: descriptive only. good excludes the highest member of each triple, r_row the middle one.
    descriptive = result["descriptive"]
    assert descriptive["excluded_member_index_rank"] == {"lowest": 0, "middle": 3, "highest": 3}
    chi = descriptive["excluded_member_rank_chi_square"]
    assert chi["statistic"] == 3.0 and chi["df"] == 2  # expected 2 per rank: (4 + 1 + 1) / 2
    assert "descriptive" in chi["note"]
    assert dg.triple_structure([good, r_row])["status"] == "PASS"  # a skewed rank never fails the gate
    assert dg.triple_structure([])["status"] == "NO_ROWS"
    assert dg.triple_structure([])["descriptive"]["excluded_member_rank_chi_square"]["statistic"] is None
    assert dg.chi_square_uniform([630, 702, 666]) == pytest.approx(3.8918918918918)


def test_s_rs_leak_counts():
    rows, oracles = [], {}
    for parent, (s_target, reference) in {"world-a": (3, [0, 3]), "world-b": (3, [2, 5]),
                                          "world-c": (5, [0, 6])}.items():
        s_row = make_row("S", {"initial": INITIAL[:], "selection": {"k": 2, "target": s_target}}, parent=parent)
        rows += [s_row, make_row("RS", rs_spec(), parent=parent)]
        oracles[s_row["id"]] = {"id": s_row["id"], "parent_id": parent,
                                "references": [json.dumps({"answer": reference}, separators=(",", ":"))]}
    result = dg.s_rs_leak(rows, oracles)
    assert result["rs_rows"] == 3 and result["same_k_target"] == 2
    assert result["s_first_reference_passes_rs"] == 1  # [2,5] uses excluded items; [0,6] sums to 5
    assert result["s_first_reference_passes_rs_ids"] == [rows[1]["id"]]
    bad = copy.deepcopy(oracles)
    bad[rows[0]["id"]]["references"] = ['{"answer":[0,1]}']
    with pytest.raises(ValueError):
        dg.s_rs_leak(rows, bad)
    with pytest.raises(ValueError):
        dg.s_rs_leak(rows[1:], oracles)


def test_shortcut_definition_and_scoring():
    assert json.loads(DIAGNOSTICS.read_text(encoding="utf-8"))["shortcut_definition"] == dg.SHORTCUT_DEFINITION
    hit = make_row("S", {"initial": INITIAL[:], "selection": {"k": 2, "target": 2}}, parent="world-a")
    miss = make_row("S", {"initial": INITIAL[:], "selection": {"k": 2, "target": 3}}, parent="world-b")
    rset = make_row("R", {"initial": INITIAL[:], "lookup": links_for(list(range(9)), 9)}, parent="world-c")
    result = dg.shortcut([hit, miss, rset])["by_signature"]
    assert result["S"] == {"total": 2, "first_indices_correct": 1, "unknown_correct": 0,
                           "first_indices_correct_ids": [hit["id"]]}
    assert result["R"]["first_indices_correct"] == 1  # "all indices without selection" when all resolve


def test_overlap_and_controls():
    train = [make_row("S", {"initial": [3, 3, 3, 5, 5, 5, 8, 8, 8], "selection": {"k": 2, "target": 8}},
                      parent="world-t1", role="train"),
             make_row("S", {"initial": [2, 2, 2, 6, 6, 6, 9, 9, 9], "selection": {"k": 3, "target": 17}},
                      parent="world-t2", role="train")]
    exposed = make_row("S", {"initial": [8, 5, 3, 8, 5, 3, 8, 5, 3], "selection": {"k": 2, "target": 8}},
                       parent="world-h1")
    shifted = make_row("S", {"initial": [3, 3, 3, 5, 5, 5, 8, 8, 8], "selection": {"k": 2, "target": 9}},
                       parent="world-h2")
    rs = make_row("RS", {"initial": [2, 2, 2, 6, 6, 6, 9, 9, 9], "lookup": links_for([0, 1, 3, 4, 6, 7], 9),
                         "selection": {"k": 3, "target": 17}}, parent="world-h1")
    result = dg.overlap(train, [exposed, shifted, rs])
    assert result["by_signature"]["S"]["rows"] == 2 and result["by_signature"]["S"]["exposed_ids"] == [exposed["id"]]
    assert result["by_signature"]["RS"]["matches"] == [{"id": rs["id"], "train_ids": [train[1]["id"]]}]
    controls = result["controls"]
    assert controls["status"] == "PASS" and controls["permuted_train_spec_recovers_its_row"] == 2
    assert controls["target_plus_one_recovers_its_row"] == 0
    with pytest.raises(ValueError):
        dg.overlap([rs], [exposed])


def overlap_pool():
    return [make_row("S", {"initial": [3, 3, 3, 5, 5, 5, 8, 8, 8], "selection": {"k": 2, "target": 8}},
                     parent="world-t1", role="train"),
            make_row("S", {"initial": [2, 2, 2, 6, 6, 6, 9, 9, 9], "selection": {"k": 3, "target": 17}},
                     parent="world-t2", role="train")]


def test_overlap_controls_fail_when_the_key_is_broken(monkeypatch):
    """TA-12: each control can FAIL. A key blind to target recovers target+1; an order-sensitive key misses permutations."""
    pool = overlap_pool()
    assert dg.overlap(pool, [])["controls"]["status"] == "PASS"
    monkeypatch.setattr(dg, "content_key", lambda spec: (tuple(sorted(spec["initial"])), spec["selection"]["k"]))
    controls = dg.overlap(pool, [])["controls"]
    assert controls["status"] == "FAIL"
    assert controls["target_plus_one_recovers_its_row"] == 2 and controls["permuted_train_spec_recovers_its_row"] == 2
    monkeypatch.setattr(dg, "content_key", lambda spec: (tuple(spec["initial"]), spec["selection"]["k"],
                                                         spec["selection"]["target"]))
    controls = dg.overlap(pool, [])["controls"]
    assert controls["status"] == "FAIL"
    assert controls["permuted_train_spec_recovers_its_row"] == 0 and controls["target_plus_one_recovers_its_row"] == 0
    monkeypatch.undo()
    assert dg.overlap([], [])["controls"]["status"] == "FAIL"  # an empty pool certifies nothing


def test_atomic_pool_rejects_contamination():
    good = make_row("S", {"initial": INITIAL[:], "selection": {"k": 2, "target": 3}}, role="train")
    assert dg.atomic_pool([good], [good["id"]]) == [good]
    for ids in ([good["id"], good["id"]], [good["id"], "MC2-missing"]):
        with pytest.raises(ValueError):
            dg.atomic_pool([good], ids)
    boundary = dict(good, tag="boundary")
    with pytest.raises(ValueError):
        dg.atomic_pool([boundary], [good["id"]])


def test_data_gates_cli_refuses_test_without_freeze_before_any_read(tmp_path):
    with pytest.raises(SystemExit, match="at-freeze"):
        dg.main(["--data", str(tmp_path / "absent"), "--roles", "dev", "test", "--out", str(tmp_path / "o.json")])
    with pytest.raises(SystemExit):
        dg.main(["--data", str(tmp_path), "--roles", "challenge", "--out", str(tmp_path / "o.json")])
    data = tmp_path / "data"
    data.mkdir()
    for relative in ("canonical/challenge.jsonl", "oracles/guard_test.jsonl", "canonical/test.jsonl"):
        with pytest.raises(SystemExit, match="sealed"):
            dg.verified_bytes(data, relative, {"files": {}}, {})
    with pytest.raises(SystemExit, match="at-freeze"):
        dg.role_file("canonical", "test", False)
    assert not (tmp_path / "o.json").exists()


def test_verified_bytes_admits_only_exact_manifest_keys_before_any_read(tmp_path, forbidden_data_guard):
    """DS-3. The root is synthetic and outside every guarded root, so only data_gates' own refusal is tested."""
    root = tmp_path / "synth"
    raw = b'{"id":"synthetic"}\n'
    for relative in ("canonical/test.jsonl", "canonical/dev.jsonl", "oracles/test.jsonl", "canonical/guard_dev.jsonl"):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(raw)
    manifest = {"files": {k: {"sha256": dg.sha256_bytes(raw)} for k in
                          ("canonical/test.jsonl", "canonical/dev.jsonl", "oracles/test.jsonl", "canonical/guard_dev.jsonl")}}
    spellings = ["canonical/TEST.jsonl", "canonical/Test.JSONL", "canonical/test.jsonl ", "canonical/test.jsonl.",
                 "canonical/test.jsonl::$DATA", "canonical\\test.jsonl", "./canonical/test.jsonl",
                 "canonical//test.jsonl", "canonical/../canonical/test.jsonl", str(root / "canonical" / "test.jsonl")]
    with forbidden_data_guard.record_opens() as opened:
        for relative in spellings:
            for flag in (False, True):
                with pytest.raises(SystemExit):
                    dg.verified_bytes(root, relative, manifest, {}, at_freeze=flag)
        for flag in (1, "yes", [True], None, False):  # at_freeze unseals only as the literal True
            with pytest.raises(SystemExit, match="sealed"):
                dg.verified_bytes(root, "canonical/test.jsonl", manifest, {}, at_freeze=flag)
        with pytest.raises(SystemExit, match="sealed"):
            dg.verified_bytes(root, "oracles/test.jsonl", manifest, {}, at_freeze=True)  # TEST oracles, never
        with pytest.raises(SystemExit, match="not admitted"):
            dg.verified_bytes(root, "canonical/guard_dev.jsonl", manifest, {})  # a key this module never builds
        with pytest.raises(SystemExit, match="not in the manifest"):
            dg.verified_bytes(root, "canonical/dev.jsonl", {"files": {}}, {})  # membership before the read
        with pytest.raises(SystemExit):
            dg.verified_bytes(root, Path("canonical/dev.jsonl"), manifest, {})
        with pytest.raises(SystemExit, match="at-freeze"):
            dg.role_file("canonical", "test", 1)
        with pytest.raises(SystemExit, match="only TEST specs"):
            dg.role_file("oracles", "test", True)
    assert opened_under(opened, root) == []
    record = {}
    with forbidden_data_guard.record_opens() as opened:  # positive control: the recorder sees these reads
        assert dg.verified_bytes(root, "canonical/dev.jsonl", manifest, record) == raw
        assert dg.verified_bytes(root, "canonical/test.jsonl", manifest, record, at_freeze=True) == raw
    assert set(opened_under(opened, root)) == {norm(root / "canonical/dev.jsonl"), norm(root / "canonical/test.jsonl")}
    assert record == {"canonical/dev.jsonl": dg.sha256_bytes(raw), "canonical/test.jsonl": dg.sha256_bytes(raw)}
    assert dg.role_file("canonical", "test", True) == "canonical/test.jsonl"


def write_synthetic_bank(root: Path, monkeypatch, roles=("dev", "test")) -> dict:
    """A synthetic dataset outside every guarded root: its test.jsonl files are synthetic, not sealed data."""
    files = {}

    def put(relative, raw):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(raw)
        files[relative] = {"sha256": dg.sha256_bytes(raw)}

    def jsonl(rows):
        return b"".join(json.dumps(row, sort_keys=True).encode("utf-8") + b"\n" for row in rows)

    pool = overlap_pool()
    put(dg.ATOMIC_S_MANIFEST, json.dumps({"ids": [row["id"] for row in pool]}).encode("utf-8"))
    put("canonical/train.jsonl", jsonl(pool))
    for role in roles:
        rows, oracles = [], []
        for parent in ("world-a", "world-b"):
            s_row = make_row("S", {"initial": INITIAL[:], "selection": {"k": 2, "target": 3}}, parent=parent, role=role)
            rows += [s_row, make_row("RS", rs_spec(), parent=parent, role=role),
                     make_row("R", {"initial": INITIAL[:], "lookup": links_for([0, 2, 3, 5, 6, 8], 9)},
                              parent=parent, role=role)]
            oracles.append({"id": s_row["id"], "parent_id": parent, "references": ['{"answer":[0,3]}']})
        put(f"canonical/{role}.jsonl", jsonl(rows))
        put(f"oracles/{role}.jsonl", jsonl(oracles))
    manifest = json.dumps({"files": files}, sort_keys=True).encode("utf-8")
    (root / "manifest.json").write_bytes(manifest)
    monkeypatch.setattr(dg, "MANIFEST", dg.sha256_bytes(manifest))
    return files


def test_freeze_run_computes_only_the_section_11_test_gates_and_binds_its_code(tmp_path, monkeypatch,
                                                                                forbidden_data_guard, capsys):
    """DS-4: TEST at freeze gets the floor and the triple structure only; its oracles are never opened.
    DS-6: the receipt binds the import-time hashes of every gate dependency, re-checked before writing."""
    root = tmp_path / "synth"
    files = write_synthetic_bank(root, monkeypatch)
    out = tmp_path / "out" / "gates.json"
    argv = ["--data", str(root), "--roles", "dev", "test", "--at-freeze"]
    with forbidden_data_guard.record_opens() as opened:
        assert dg.main(argv + ["--out", str(out)]) == 0
    seen = set(opened_under(opened, root))
    assert norm(root / "canonical/test.jsonl") in seen and norm(root / "oracles/dev.jsonl") in seen  # control
    assert norm(root / "oracles/test.jsonl") not in seen
    receipt = json.loads(out.read_text(encoding="utf-8"))
    assert set(receipt["files_sha256"]) == set(files) - {"oracles/test.jsonl"}
    assert set(receipt["gates"]["test"]) == set(dg.FREEZE_GATES) == {"rs_floor", "triple_structure"}
    assert receipt["freeze_role_gates"] == ["rs_floor", "triple_structure"]
    assert set(receipt["gates"]["dev"]) == {"rs_floor", "triple_structure", "s_rs_leak", "shortcut", "overlap"}
    test = receipt["gates"]["test"]
    assert test["rs_floor"]["rows"] == 2 and test["rs_floor"]["mean_p_blind"] == 4 / 9
    assert test["triple_structure"]["status"] == "PASS"
    assert receipt["gates"]["dev"]["s_rs_leak"]["s_first_reference_passes_rs"] == 2
    assert receipt["at_freeze"] is True and receipt["test_challenge_read"] is True
    assert "leak" not in json.loads(capsys.readouterr().out)["test"]
    # DS-6
    assert receipt["code_sha256"] == dg._IMPORT_CODE and set(receipt["code_sha256"]) == set(dg.CODE_FILES)
    assert receipt["code_bundle_sha256"] == digest(dg._IMPORT_CODE)
    assert b"\r\n" not in out.read_bytes()
    monkeypatch.setattr(dg, "_IMPORT_CODE", {**dg._IMPORT_CODE, "maxrl_composable_skills_v2/verifier.py": "0" * 64})
    edited = tmp_path / "edited" / "gates.json"
    with pytest.raises(SystemExit, match="changed since import"):
        dg.main(argv + ["--out", str(edited)])
    assert not edited.exists()


def test_code_binding_covers_every_gate_dependency_and_refuses_an_edit_before_writing(tmp_path, monkeypatch):
    """DS-6: data_gates, preflight, and the verifier, corpus and contracts the gate values come from."""
    from experiments.maxrl_comp002_pretraining_v1 import preflight
    from experiments.maxrl_composable_skills_v2 import contracts, corpus, verifier
    assert dg.CODE_FILES == {"data_gates.py": Path(dg.__file__).resolve(),
                             "preflight.py": Path(preflight.__file__).resolve(),
                             "maxrl_composable_skills_v2/verifier.py": Path(verifier.__file__).resolve(),
                             "maxrl_composable_skills_v2/corpus.py": Path(corpus.__file__).resolve(),
                             "maxrl_composable_skills_v2/contracts.py": Path(contracts.__file__).resolve()}
    assert dg._IMPORT_CODE == dg.code_sha256()
    assert dg._IMPORT_CODE["maxrl_composable_skills_v2/verifier.py"] == q1_sha256(Path(verifier.__file__))
    probe = tmp_path / "probe.py"
    probe.write_bytes(b"x = 1\n")
    monkeypatch.setattr(dg, "CODE_FILES", {"probe.py": probe})
    monkeypatch.setattr(dg, "_IMPORT_CODE", dg.code_sha256())
    written = tmp_path / "unchanged" / "receipt.json"
    dg.write_receipt({"a": 1}, written)
    assert json.loads(written.read_text(encoding="utf-8")) == {"a": 1}
    with pytest.raises(FileExistsError):
        dg.write_receipt({"a": 2}, written)
    probe.write_bytes(b"x = 2\n")
    refused = tmp_path / "edited" / "receipt.json"
    with pytest.raises(SystemExit, match="changed since import"):
        dg.write_receipt({"a": 1}, refused)
    assert not refused.parent.exists()


def test_scope_cli_refuses_test_without_ack_before_any_read(tmp_path):
    with pytest.raises(SystemExit, match="spec-only-ack"):
        sr.main(["--data", str(tmp_path / "absent"), "--role", "test", "--out", str(tmp_path / "o.json")])
    with pytest.raises(SystemExit):
        sr.main(["--data", str(tmp_path), "--role", "dev", "--out", str(tmp_path / "o.json")])


def test_signature_scope_synthetic():
    rows = [{"id": "a", "parent_id": "p1", "tag": "core", "skills": ["R"]},
            {"id": "b", "parent_id": "p1", "tag": "core", "skills": ["S"]},
            {"id": "c", "parent_id": "p2", "tag": "core", "skills": ["R"]},
            {"id": "d", "parent_id": "p2", "tag": "core", "skills": ["S"]},
            {"id": "e", "parent_id": "p3", "tag": "boundary", "skills": ["S"]},
            {"id": "f", "parent_id": "p1", "tag": "core", "skills": ["U"]}]
    scope = sr.signature_scope(rows)
    assert scope["rows"] == 6 and scope["absent_signatures"] == ["P", "RS"]
    assert scope["shared_parent_worlds"] == 2
    assert scope["signatures"]["R"]["parents_sha256"] == scope["signatures"]["S"]["parents_sha256"]
    assert sr.signature_scope(rows[:3])["shared_parent_worlds"] is None
    with pytest.raises(ValueError):
        sr.signature_scope(rows + [{"id": "g", "parent_id": "p1", "tag": "core", "skills": ["R"]}])
    with pytest.raises(ValueError):
        sr.signature_scope(rows + [rows[0]])
    assert sr.project('{"id":"x","parent_id":"p","tag":"core","skills":["R"],"prompt":"hidden"}') == {
        "id": "x", "parent_id": "p", "tag": "core", "skills": ["R"]}


# ---------------------------------------------------------------- real data (TRAIN, DEV, CALIBRATION only)

@pytest.fixture(scope="module")
def bank(comp_rs_data):
    def rows(relative):
        return dg.jsonl_rows((comp_rs_data / relative).read_bytes())
    train = rows("canonical/train.jsonl")
    plan = json.loads((comp_rs_data / dg.ATOMIC_S_MANIFEST).read_text(encoding="utf-8"))
    return {"train": train, "pool": dg.atomic_pool(train, plan["ids"]),
            "dev": rows("canonical/dev.jsonl"), "calibration": rows("canonical/calibration.jsonl"),
            "dev_oracles": {o["id"]: o for o in rows("oracles/dev.jsonl")}}


def test_real_rs_floor_dev(bank):
    result = dg.rs_floor(bank["dev"])
    assert result["rows"] == 250
    assert round(result["mean_p_blind"], 4) == 0.3437
    assert result["by_k"]["2"]["rows"] == 84 and round(result["by_k"]["2"]["mean_p_blind"], 4) == 0.4441
    assert result["by_k"]["3"]["rows"] == 166 and round(result["by_k"]["3"]["mean_p_blind"], 3) == 0.293
    assert result["first_lex_pass"] == 83


def test_real_triple_structure(bank):
    dev = dg.triple_structure(bank["dev"])
    assert dev["status"] == "PASS" and dev["by_signature"] == {"R": {"rows": 250, "pass": 250},
                                                              "RS": {"rows": 250, "pass": 250}}
    assert dev["descriptive"]["requested_aliases_in_resolved_index_order"] == 0
    # Canary: Amendment 1 C9 quotes these DEV ranks and chi-square (descriptive, not a gate).
    assert dev["descriptive"]["excluded_member_index_rank"] == {"lowest": 630, "middle": 702, "highest": 666}
    assert round(dev["descriptive"]["excluded_member_rank_chi_square"]["statistic"], 1) == 3.9
    calibration = dg.triple_structure(bank["calibration"])
    assert calibration["status"] == "PASS" and calibration["by_signature"] == {"R": {"rows": 150, "pass": 150}}


def test_real_s_rs_leak_dev(bank):
    result = dg.s_rs_leak(bank["dev"], bank["dev_oracles"])
    assert (result["rs_rows"], result["same_k_target"], result["s_first_reference_passes_rs"]) == (250, 62, 33)


def test_real_shortcut(bank):
    assert dg.shortcut(bank["dev"])["by_signature"]["S"]["first_indices_correct"] == 36
    assert dg.shortcut(bank["dev"])["by_signature"]["S"]["total"] == 250
    calibration = dg.shortcut(bank["calibration"])["by_signature"]["S"]
    assert (calibration["total"], calibration["first_indices_correct"]) == (150, 19)
    pool = dg.shortcut(bank["pool"])["by_signature"]["S"]
    assert (pool["total"], pool["first_indices_correct"]) == (3000, 424)
    # Canary: the dataset's own TRAIN diagnostic, every signature, reproduced exactly.
    recorded = json.loads(DIAGNOSTICS.read_text(encoding="utf-8"))["training_shortcut"]
    observed = {sig + "/" + key: v[key] for sig, v in dg.shortcut(bank["train"])["by_signature"].items()
                for key in ("total", "first_indices_correct", "unknown_correct")}
    assert observed == recorded


def test_real_overlap(bank):
    dev = dg.overlap(bank["pool"], bank["dev"])
    assert dev["pool_rows"] == 3000
    assert {s: (v["rows"], v["exposed"]) for s, v in dev["by_signature"].items()} == {"S": (250, 1), "RS": (250, 5)}
    assert dev["controls"]["status"] == "PASS"
    assert dev["controls"]["permuted_train_spec_recovers_its_row"] == 3000
    assert dev["controls"]["target_plus_one_recovers_its_row"] == 0
    calibration = dg.overlap(bank["pool"], bank["calibration"])
    assert calibration["by_signature"]["RS"]["rows"] == 0 and calibration["by_signature"]["S"]["rows"] == 150


def test_real_data_gates_cli(comp_rs_data, tmp_path, capsys):
    out = tmp_path / "gates.json"
    assert dg.main(["--data", str(comp_rs_data), "--out", str(out)]) == 0
    receipt = json.loads(out.read_text(encoding="utf-8"))
    manifest = json.loads((comp_rs_data / "manifest.json").read_text(encoding="utf-8"))
    assert receipt["roles"] == ["dev", "calibration"] and receipt["test_challenge_read"] is False
    assert set(receipt["files_sha256"]) == {"manifests/atomic_S.json", "canonical/train.jsonl",
                                            "canonical/dev.jsonl", "oracles/dev.jsonl",
                                            "canonical/calibration.jsonl", "oracles/calibration.jsonl"}
    assert all(manifest["files"][k]["sha256"] == v for k, v in receipt["files_sha256"].items())
    assert receipt["code_sha256"] == dg._IMPORT_CODE and receipt["code_bundle_sha256"] == digest(dg._IMPORT_CODE)
    assert receipt["train_atomic_S"]["shortcut"]["first_indices_correct"] == 424
    dev = receipt["gates"]["dev"]
    assert round(dev["rs_floor"]["mean_p_blind"], 4) == 0.3437 and dev["rs_floor"]["first_lex_pass"] == 83
    assert (dev["s_rs_leak"]["same_k_target"], dev["s_rs_leak"]["s_first_reference_passes_rs"]) == (62, 33)
    assert dev["shortcut"]["by_signature"]["S"]["first_indices_correct"] == 36
    assert {s: v["exposed"] for s, v in dev["overlap"]["by_signature"].items()} == {"S": 1, "RS": 5}
    assert dev["triple_structure"]["status"] == "PASS"
    calibration = receipt["gates"]["calibration"]
    assert calibration["shortcut"]["by_signature"]["S"]["first_indices_correct"] == 19
    assert calibration["rs_floor"]["rows"] == 0 and calibration["s_rs_leak"]["rs_rows"] == 0
    assert calibration["rs_floor"]["status"] == calibration["s_rs_leak"]["status"] == "NO_ROWS"
    assert calibration["rs_floor"]["first_lex_pass"] is None
    assert calibration["triple_structure"]["by_signature"] == {"R": {"rows": 150, "pass": 150}}
    assert b"\r\n" not in out.read_bytes()
    with pytest.raises(SystemExit, match="create-only"):
        dg.main(["--data", str(comp_rs_data), "--out", str(out)])
    with pytest.raises(SystemExit, match="protected"):
        dg.main(["--data", str(comp_rs_data), "--out", str(comp_rs_data / "gates.json")])


def test_real_calibration_scope_receipt(comp_rs_data, tmp_path):
    out = tmp_path / "CALIBRATION_SCOPE.json"
    assert sr.main(["--data", str(comp_rs_data), "--role", "calibration", "--out", str(out)]) == 0
    receipt = json.loads(out.read_text(encoding="utf-8"))
    signatures = receipt["signatures"]
    assert sorted(signatures) == ["P", "R", "S"] and receipt["absent_signatures"] == ["RS"]
    assert all(v["rows"] == 150 and v["parents"] == 150 for v in signatures.values())
    assert len({v["parents_sha256"] for v in signatures.values()}) == 1
    assert receipt["shared_parent_worlds"] == 150 and receipt["rows_in_bank"] == 1250
    assert receipt["test_challenge_read"] is False and receipt["fields_retained"] == ["id", "parent_id", "tag", "skills"]
    # Canary: the committed Q0 receipt's calibration digests (same digest convention).
    q0 = json.loads((PACKAGE / "evidence/Q0_RECEIPT.json").read_text(encoding="utf-8"))["calibration"]
    assert signatures["R"]["ids_sha256"] == q0["R"]["ids_sha256"]
    assert signatures["S"]["ids_sha256"] == q0["S"]["ids_sha256"]
    with pytest.raises(SystemExit, match="create-only"):
        sr.main(["--data", str(comp_rs_data), "--role", "calibration", "--out", str(out)])


def test_real_dev_scope_reproduces_q1_receipt(comp_rs_data):
    lines = (comp_rs_data / "canonical/dev.jsonl").read_text(encoding="utf-8").splitlines()
    scope = sr.signature_scope([sr.project(line) for line in lines])
    q1 = json.loads((PACKAGE / "evidence/Q1_DEV_SCOPE.json").read_text(encoding="utf-8"))["signatures"]
    for sig in ("R", "S", "RS"):
        assert scope["signatures"][sig] == q1[sig]
    assert scope["shared_parent_worlds"] == 250 and scope["rows"] == 4150
