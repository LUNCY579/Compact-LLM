"""Sealed-bank guard for every test in this package (COMP-RS-001 Amendment 1, section 1).

No test may open a TEST/CHALLENGE data file. A path is sealed when both hold:
- its final name is a sealed-bank file or a sealed-role manifest (FORBIDDEN_NAMES), compared the way Windows
  opens it: case-folded, trailing dots and spaces dropped, any ':stream' suffix dropped;
- it lies under a directory named 'data', or under a guarded data root whatever that root is called: the
  default root, COMP_RS_DATA as it was when this file was imported, and COMP_RS_DATA as it is now.
Both the absolute form and the symlink-resolved form of the path are checked.

The guard is installed when this conftest is imported, before any test module is collected, and covers:
- builtins.open, io.open, pathlib.Path.open, Path.read_text and Path.read_bytes (wrapped);
- every route that reaches CPython's audited open, through a sys.addaudithook hook on the 'open' event (and on
  shutil.copyfile, os.link and os.rename): os.open, io.FileIO, _io.open, io.open_code, tokenize.open, numpy,
  shutil. The hook raises before the file is opened;
- the C++/Rust opens that the audit hook never sees, wrapped when their modules are imported:
  torch.from_file, torch.UntypedStorage.from_file (and so TypedStorage.from_file), and safetensors' safe_open
  (and so safetensors.torch/numpy.load_file), including names bound by `from safetensors import safe_open`.
A sealed open raises ForbiddenDataRead before any byte is read. Each attempt is also recorded, and an autouse
fixture fails the test at teardown when one happened during it, or at setup when one happened outside any test
(collection, or a wider-scoped fixture), so a caller that swallows the exception still fails.

Not covered: subprocesses (data-backed tests call the CLIs in-process), ctypes and other direct OS calls, and
torch._C internals reached without the public names.
"""
from __future__ import annotations

import builtins
import contextlib
import functools
import importlib.machinery
import io
import os
from pathlib import Path
import sys
import threading

import pytest

FORBIDDEN_NAMES = frozenset({
    "test.jsonl", "challenge.jsonl", "guard_test.jsonl", "guard_challenge.jsonl",
    # manifests/ of the sealed roles (they are listed in manifest['files'] with the banks)
    "test_core.json", "test_boundary.json", "challenge_core.json", "challenge_boundary.json",
    "guard_test.json", "guard_challenge.json"})
DEFAULT_DATA = Path(r"C:\Users\clclc\BusinessProject\Rineya-Dataset\data")
DATA_ENV = "COMP_RS_DATA"
AUDIT_EVENTS = {"open": (0,), "shutil.copyfile": (0, 1), "os.link": (0, 1), "os.rename": (0, 1)}


class ForbiddenDataRead(RuntimeError):
    """A test tried to open a sealed TEST/CHALLENGE data file."""


def _sealed_name(name: str) -> bool:
    return name.split(":", 1)[0].rstrip(" .").casefold() in FORBIDDEN_NAMES


def _root_forms(root) -> tuple[str, ...]:
    text = os.fsdecode(os.fspath(root))
    return tuple(dict.fromkeys(os.path.normcase(f(text)) for f in (os.path.abspath, os.path.realpath)))


_SESSION_ROOTS = _root_forms(DEFAULT_DATA) + (_root_forms(os.environ[DATA_ENV]) if os.environ.get(DATA_ENV) else ())
_env_roots_cache: dict[str, tuple[str, ...]] = {}


def guarded_roots() -> tuple[str, ...]:
    """Normalised data roots guarded whatever their folder name: the default, COMP_RS_DATA at import and now."""
    current = os.environ.get(DATA_ENV)
    if not current:
        return _SESSION_ROOTS
    if current not in _env_roots_cache:
        _env_roots_cache[current] = _root_forms(current)
    return tuple(dict.fromkeys(_SESSION_ROOTS + _env_roots_cache[current]))


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("\\/") + os.sep)


def _sealed(path: str, roots: tuple[str, ...]) -> bool:
    if not _sealed_name(os.path.basename(path)):
        return False
    parents = Path(path).parts[:-1]
    folded = os.path.normcase(path)
    return any(part.rstrip(" .").casefold() == "data" for part in parents) or any(_under(folded, r) for r in roots)


def is_forbidden(file) -> bool:
    """True iff `file` names a sealed-bank file under a 'data' tree or a guarded root (symlinks resolved)."""
    if isinstance(file, int):
        return False
    try:
        raw = os.fsdecode(os.fspath(file))
    except (TypeError, ValueError):
        return False
    roots = guarded_roots()
    for form in (os.path.abspath, os.path.realpath):
        try:
            if _sealed(form(raw), roots):
                return True
        except (OSError, RuntimeError, ValueError):
            continue
    return False


class Guard:
    error = ForbiddenDataRead
    is_forbidden = staticmethod(is_forbidden)
    guarded_roots = staticmethod(guarded_roots)
    default_data = DEFAULT_DATA

    def __init__(self):
        self.violations: list[str] = []
        self._local = threading.local()
        self._recorders: list[list[str]] = []

    def check(self, file) -> None:
        if is_forbidden(file):
            self.violations.append(str(file))
            raise ForbiddenDataRead("sealed TEST/CHALLENGE data file opened in a test: " + str(file))

    def _take(self) -> list[str]:
        taken = list(self.violations)
        del self.violations[:]
        return taken

    def per_test(self):
        """The per-test fixture body: fail at setup on attempts made outside any test, at teardown on its own."""
        stale = self._take()
        if stale:
            pytest.fail("sealed TEST/CHALLENGE data file(s) opened outside any test: " + ", ".join(stale),
                        pytrace=False)
        yield self
        new = self._take()
        if new:
            pytest.fail("sealed TEST/CHALLENGE data file(s) opened: " + ", ".join(new), pytrace=False)

    @contextlib.contextmanager
    def suspended(self):
        previous = getattr(self._local, "off", False)
        self._local.off = True
        try:
            yield
        finally:
            self._local.off = previous

    def real_open(self, file, mode, *args, **kwargs):
        """Create-only writes of synthetic fixtures in the guard's own tests: nothing existing is read or clobbered."""
        if "x" not in mode or "r" in mode or "+" in mode:
            raise ValueError("real_open writes new synthetic fixtures only (mode 'x' or 'xb')")
        with self.suspended():
            return _REAL["open"](file, mode, *args, **kwargs)

    @contextlib.contextmanager
    def record_opens(self):
        """Every audited open attempted meanwhile, as os.path.normcase(os.path.realpath(path))."""
        log: list[str] = []
        self._recorders.append(log)
        try:
            yield log
        finally:
            self._recorders.remove(log)

    def audit(self, event: str, args: tuple) -> None:
        positions = AUDIT_EVENTS.get(event)
        if positions is None or getattr(self._local, "off", False) or getattr(self._local, "busy", False):
            return
        self._local.busy = True
        try:
            targets = [args[i] for i in positions if i < len(args) and args[i] is not None]
            if event == "open" and self._recorders:
                for target in targets:
                    try:
                        seen = os.path.normcase(os.path.realpath(os.fsdecode(os.fspath(target))))
                    except (TypeError, ValueError, OSError):
                        continue  # a file descriptor, or a name no file can have
                    for log in self._recorders:
                        log.append(seen)
            for target in targets:
                self.check(target)  # raises ForbiddenDataRead, which aborts the open
        finally:
            self._local.busy = False


def _guarded(real, guard: Guard, param: str = "filename"):
    if getattr(real, "_sealed_guard_", False):
        return real

    @functools.wraps(real, updated=())
    def guarded(*args, **kwargs):
        guard.check(args[0] if args else kwargs.get(param))
        return real(*args, **kwargs)

    guarded._sealed_guard_ = True
    return guarded


def _patch_module(module, guard: Guard) -> None:
    """Wrap the file-opening names of torch and safetensors that bypass the audit hook."""
    name = getattr(module, "__name__", "")
    if name == "torch":
        if hasattr(module, "from_file"):
            module.from_file = _guarded(module.from_file, guard)
        storage = getattr(module, "UntypedStorage", None)
        if storage is not None and hasattr(storage, "from_file"):
            if not getattr(storage.from_file, "_sealed_guard_", False):
                storage.from_file = staticmethod(_guarded(storage.from_file, guard))
    elif name == "safetensors" or name.startswith("safetensors."):
        if "safe_open" in vars(module):
            module.safe_open = _guarded(module.safe_open, guard)


def _patch_loaded(guard: Guard) -> None:
    for name, module in list(sys.modules.items()):
        if module is not None and (name == "torch" or name == "safetensors" or name.startswith("safetensors.")):
            _patch_module(module, guard)


class _PatchOnImport:
    """Meta-path finder that wraps torch/safetensors names right after their module executes."""

    def __init__(self, guard: Guard):
        self.guard = guard

    def find_spec(self, fullname, path=None, target=None):
        if not (fullname == "torch" or fullname == "safetensors" or fullname.startswith("safetensors.")):
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        if isinstance(loader, (importlib.machinery.SourceFileLoader, importlib.machinery.SourcelessFileLoader,
                               importlib.machinery.ExtensionFileLoader)):
            real_exec, guard = loader.exec_module, self.guard

            def exec_module(module):
                real_exec(module)
                _patch_module(module, guard)

            loader.exec_module = exec_module
        return spec


_REAL = {"open": builtins.open, "io.open": io.open, "Path.open": Path.open, "Path.read_text": Path.read_text,
         "Path.read_bytes": Path.read_bytes}
GUARD = Guard()


def _install(guard: Guard) -> None:
    def guarded_open(file, *args, **kwargs):
        guard.check(file)
        return _REAL["open"](file, *args, **kwargs)

    def guarded_io_open(file, *args, **kwargs):
        guard.check(file)
        return _REAL["io.open"](file, *args, **kwargs)

    def guarded_path_open(self, *args, **kwargs):
        guard.check(self)
        return _REAL["Path.open"](self, *args, **kwargs)

    def guarded_read_text(self, *args, **kwargs):
        guard.check(self)
        return _REAL["Path.read_text"](self, *args, **kwargs)

    def guarded_read_bytes(self, *args, **kwargs):
        guard.check(self)
        return _REAL["Path.read_bytes"](self, *args, **kwargs)

    builtins.open, io.open = guarded_open, guarded_io_open
    Path.open, Path.read_text, Path.read_bytes = guarded_path_open, guarded_read_text, guarded_read_bytes
    sys.addaudithook(guard.audit)  # cannot be removed; it stays for the life of the test process
    sys.meta_path.insert(0, _PatchOnImport(guard))
    _patch_loaded(guard)


_install(GUARD)


@pytest.fixture(scope="session", autouse=True)
def _sealed_data_wrappers():
    _patch_loaded(GUARD)
    yield GUARD


@pytest.fixture(autouse=True)
def forbidden_data_guard(_sealed_data_wrappers):
    _patch_loaded(_sealed_data_wrappers)
    yield from _sealed_data_wrappers.per_test()


@pytest.fixture(scope="session")
def comp_rs_data() -> Path:
    """The Rineya dataset root (COMP_RS_DATA overrides); data-backed tests skip when it is absent."""
    root = Path(os.environ.get(DATA_ENV, str(DEFAULT_DATA)))
    if not (root / "manifest.json").is_file():
        pytest.skip("COMP-RS dataset absent: " + str(root))
    return root
