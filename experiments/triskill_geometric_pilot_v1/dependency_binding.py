"""Narrow source binding for the pilot, including its renderer's DEV fixture.

Production paths are fixed here, not selected by a manifest or CLI argument.
Verified source bytes are compiled explicitly rather than imported through
sys.path or a .pyc cache. The existing renderer self-test consumes a pinned,
immutable snapshot of its existing DEV fixture; its other banks are not read.
This is application source binding, not a replacement for qualified containment.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import stat
import sys
import types
from pathlib import Path

SCHEMA = 1
PILOT_RELATIVE = Path("experiments/triskill_geometric_pilot_v1")
RENDERER_RELATIVE = Path("experiments/fresh_dual_skill_instrument_v1/score_arms.py")
RENDERER_FIXTURE = Path(
    "C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/"
    "qwen001-20260918/render-process/prompts.jsonl"
)
ROLE_KINDS = {
    "worker": "code", "geometry": "code", "benchmark": "code", "binding": "code",
    "renderer": "code", "renderer_fixture": "fixture",
}
LOCAL_LOAD_ROLES = frozenset(("geometry", "benchmark"))
_ALIASES = {
    "worker": "worker", "geometry": "geometry", "benchmark": "benchmark",
    "binding": "dependency_binding", "renderer": "score_arms",
}


def _absolute(path: Path) -> Path:
    # Deliberately do not resolve first: that would hide a substituted link.
    return Path(os.path.abspath(os.fspath(path)))


def _same_path(first: Path, second: Path) -> bool:
    return os.path.normcase(os.fspath(_absolute(first))) == os.path.normcase(os.fspath(_absolute(second)))


def reject_path_substitution(path: Path) -> None:
    """Reject symlinks and Windows reparse points in every path component."""
    path = _absolute(path)
    for component in reversed((path, *path.parents)):
        try:
            info = component.lstat()
        except OSError as exc:
            raise ValueError(f"bound dependency missing/inaccessible: {path}") from exc
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"bound dependency path substitution: {component}")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"bound dependency is not a regular file: {path}")


def _pinned_bytes(path: Path, expected: str) -> bytes:
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("invalid dependency SHA-256 pin")
    reject_path_substitution(path)
    data = path.read_bytes()
    reject_path_substitution(path)
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"bound dependency digest mismatch: {path}")
    return data


def _production_paths():
    here = _absolute(Path(__file__)).parent
    repo = here.parent.parent
    if here != repo / PILOT_RELATIVE:
        raise ValueError("binding module is outside the fixed pilot directory")
    paths = {role: here / f"{role}.py" for role in ("worker", "geometry", "benchmark")}
    paths.update(binding=here / "dependency_binding.py", renderer=repo / RENDERER_RELATIVE,
                 renderer_fixture=RENDERER_FIXTURE)
    return repo, paths


def _manifest_path(role: str, path: Path, repo: Path) -> str:
    return path.as_posix() if role == "renderer_fixture" else path.relative_to(repo).as_posix()


def build_dependency_manifest() -> dict:
    """Hash exactly the production allowlist; never enumerate external banks."""
    repo, paths = _production_paths()
    roles = {}
    for role, path in paths.items():
        reject_path_substitution(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        _pinned_bytes(path, digest)
        roles[role] = {"kind": ROLE_KINDS[role], "path": _manifest_path(role, path, repo), "sha256": digest}
    return {"schema": SCHEMA, "roles": roles}


def _validate_manifest(manifest: dict, repo: Path, paths: dict):
    """Private fixture-test seam; production callers cannot supply paths."""
    if not isinstance(manifest, dict) or set(manifest) != {"schema", "roles"} or manifest.get("schema") != SCHEMA:
        raise ValueError("incomplete/unsupported dependency binding manifest")
    roles = manifest["roles"]
    if not isinstance(roles, dict) or set(roles) != set(ROLE_KINDS) or set(paths) != set(ROLE_KINDS):
        raise ValueError("incomplete/extra dependency binding roles")
    snapshots = {}
    for role, path in paths.items():
        entry = roles[role]
        if not isinstance(entry, dict) or set(entry) != {"kind", "path", "sha256"}:
            raise ValueError(f"invalid dependency entry: {role}")
        if entry["kind"] != ROLE_KINDS[role] or entry["path"] != _manifest_path(role, path, repo):
            raise ValueError(f"bound dependency role/path mismatch: {role}")
        snapshots[role] = _pinned_bytes(path, entry["sha256"])
    return BoundDependencies(paths, roles, snapshots)


def validate_dependencies(manifest: dict):
    repo, paths = _production_paths()
    bound = _validate_manifest(manifest, repo, paths)
    bound.verify_cached_modules()
    return bound


def _check_cached_module(role: str, module, path: Path, digest: str):
    if module is None:
        return
    origin = getattr(module, "__file__", None)
    spec = getattr(module, "__spec__", None)
    spec_origin = getattr(spec, "origin", None)
    if not origin or not _same_path(Path(origin), path) or (spec_origin and not _same_path(Path(spec_origin), path)):
        raise ValueError(f"cached local module path mismatch: {role}")
    # Geometry, benchmark and renderer cannot inherit unverified executable
    # objects, even when __file__ claims the right source. Explicit loads below
    # carry the binding marker and are never obtained through the alias cache.
    if role in ("geometry", "benchmark", "renderer") and getattr(module, "_triskill_source_binding", None) != (str(path), digest):
        raise ValueError(f"unverified cached local module: {role}")


class _FixtureSnapshot:
    __slots__ = ("path", "data")

    def __init__(self, path, data):
        self.path, self.data = path, data

    def read_text(self, encoding="utf-8", errors=None):
        return self.data.decode(encoding, errors or "strict")

    def __fspath__(self):
        return os.fspath(self.path)


class BoundRenderer:
    def __init__(self, bound, module):
        self._bound, self._module = bound, module
        self._check_fixture_constant()
        # The original function's verified code executes with its original
        # globals except for the same-path, immutable fixture snapshot reader.
        namespace = dict(module.__dict__)
        namespace["RECORDED"] = _FixtureSnapshot(bound.paths["renderer_fixture"], bound.snapshots["renderer_fixture"])
        function = module.template_selftest
        self._selftest = types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)

    def _check_fixture_constant(self):
        recorded = getattr(self._module, "RECORDED", None)
        if not isinstance(recorded, Path) or recorded.as_posix() != self._bound.paths["renderer_fixture"].as_posix():
            raise ValueError("renderer fixture path override/substitution")

    def template_selftest(self):
        self._check_fixture_constant()
        self._bound.verify_current()
        self._selftest()
        self._check_fixture_constant()
        self._bound.verify_current()

    def render(self, system, user):
        return self._module.render(system, user)


class BoundDependencies:
    def __init__(self, paths, roles, snapshots):
        self.paths = dict(paths)
        self.roles = {role: dict(entry) for role, entry in roles.items()}
        self.snapshots = dict(snapshots)

    def verify_current(self):
        for role, path in self.paths.items():
            _pinned_bytes(path, self.roles[role]["sha256"])

    def verify_cached_modules(self):
        for role, alias in _ALIASES.items():
            _check_cached_module(role, sys.modules.get(alias), self.paths[role], self.roles[role]["sha256"])
        main = sys.modules.get("__main__")
        if main is not None and getattr(main, "__file__", None) and Path(main.__file__).name == "worker.py":
            _check_cached_module("worker", main, self.paths["worker"], self.roles["worker"]["sha256"])

    def _load_code(self, role):
        path, digest = self.paths[role], self.roles[role]["sha256"]
        _pinned_bytes(path, digest)
        _check_cached_module(role, sys.modules.get(_ALIASES[role]), path, digest)
        # Always create a new module from verified bytes. A preexisting private
        # import cache is not reused, and the loader does not inspect .pyc files.
        name = f"_triskill_bound_{role}_{digest}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None:
            raise ValueError(f"cannot create bound module: {role}")
        module = importlib.util.module_from_spec(spec)
        module._triskill_source_binding = (str(path), digest)
        prior = sys.modules.get(name)
        sys.modules[name] = module
        try:
            exec(compile(self.snapshots[role], str(path), "exec"), module.__dict__)
        except BaseException:
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior
            raise
        _pinned_bytes(path, digest)
        return module

    def load_local(self, role):
        if role not in LOCAL_LOAD_ROLES:
            raise ValueError("only geometry/benchmark are loadable local roles")
        return self._load_code(role)

    def load_worker(self):
        """Execute the pinned worker bytes; a public alias is bootstrap only."""
        return self._load_code("worker")

    def verify_worker_execution(self, module):
        path, digest = self.paths["worker"], self.roles["worker"]["sha256"]
        _check_cached_module("worker", module, path, digest)
        if module is None or getattr(module, "_triskill_source_binding", None) != (str(path), digest):
            raise ValueError("unverified executing native worker")

    def load_renderer(self):
        module = self._load_code("renderer")
        if not callable(getattr(module, "render", None)) or not callable(getattr(module, "template_selftest", None)):
            raise ValueError("renderer interface missing")
        return BoundRenderer(self, module)

    def receipt(self):
        return {"schema": SCHEMA, "roles": self.roles, "execution": "verified source bytes; immutable DEV fixture snapshot"}
