"""Regenerate the pre-S0 evidence in one audited process (COMP-RS-001 Amendment 1 section 1; review DS-8).

python -m experiments.maxrl_comp002_pretraining_v1.s0pre_evidence [--data DATA] [--out-dir DIR] [--reference-dir DIR]
python -m experiments.maxrl_comp002_pretraining_v1.s0pre_evidence --control-only [--data DATA]
python -m experiments.maxrl_comp002_pretraining_v1.s0pre_evidence --suite [--data DATA] [--out-dir DIR]

Receipts mode. Only standard-library modules are imported at module level. main() installs a sys.addaudithook
guard before any package module is imported, and runs a positive control before anything else:
- synthetic sealed-named files in a temporary directory (written before the hook exists) must be refused on every
  open route tried: builtins.open, io.open, os.open, io.FileIO, io.open_code, Path.read_bytes/read_text,
  shutil.copyfile, the Windows spellings TEST.JSONL, 'test.jsonl.' and 'test.jsonl::$DATA', and a sealed-role
  manifest;
- the same spellings of an allowed synthetic file must open, which shows that each refused spelling reaches a file;
- a read of the data root's manifest.json and a listing of the data root must be recorded by the same hook.
Then the package CLIs run in process, in this order, each writing its receipt create-only into --out-dir:
  scope_receipt --role calibration, --role dev      -> S0PRE_CALIBRATION_SCOPE.json, S0PRE_DEV_SCOPE.json
  data_gates --roles dev calibration                 -> S0PRE_DATA_GATES.json
  canary dev (R S P RS + guard_dev), calibration (R S P + guard_dev), guard_dev
                                                     -> S0PRE_CANARY_DEV.json, _CALIBRATION.json, _GUARD_DEV.json
  h3b_null_cell --role dev, 2,000 re-draws, 10^6 flips, seed 20260926 -> S0PRE_H3B_NULL_CELL_DEV.json
A composition canary is bound to the scope receipt written just before it (C8), pinned by that file's sha256.
--reference-dir names a directory holding earlier PASS canary receipts of the same banks under the same file
names; check (c) must then reproduce their derangement counts (canary TA-03).
Last, S0PRE_OPEN_LOG.json (create-only) records every file opened and every directory listed under the data root,
per step and with counts; every refusal (none is expected outside the control); any write or mutation under the
data root; the positive control; the sha256 of this runner and of every package source, re-checked at the end;
and the versions. The run stops at the first step that fails, and the log is written either way.

The hook refuses a sealed name wherever the file lives. The names are test.jsonl, challenge.jsonl,
guard_test.jsonl and guard_challenge.jsonl, plus the six sealed-role manifests of tests/conftest.py. They are
compared case-folded, with trailing dots and spaces and any ':stream' suffix dropped, on the raw, absolute and
symlink-resolved forms of the path. It sees the opens CPython audits (the 'open' event, which covers builtins, io,
os, pathlib, tokenize and numpy opens, plus shutil.copyfile, os.link and os.rename). It cannot see C, C++ or Rust
readers such as torch.from_file or safetensors, so the log records whether torch or safetensors was imported
(neither is expected).

Suite mode runs `python -m pytest -q experiments/maxrl_comp002_pretraining_v1/tests` from the worktree root twice,
one process at a time: with the dataset present, then with COMP_RS_DATA naming a path that does not exist (the
data-backed tests skip). It writes their exact output, line endings normalised to LF, with the versions and the
sha256 of every package file to S0PRE_TESTS_WINDOWS.txt, create-only. Each pytest process is guarded by
tests/conftest.py. Nothing here touches CUDA.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import datetime
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PACKAGE = "experiments.maxrl_comp002_pretraining_v1"
DEFAULT_DATA = Path(r"C:\Users\clclc\BusinessProject\Rineya-Dataset\data")
DATA_ENV = "COMP_RS_DATA"
EVIDENCE = HERE / "evidence"
MISSING_DATA = Path(r"C:\nonexistent\comp_rs_data")
SEALED_BANKS = frozenset({"test.jsonl", "challenge.jsonl", "guard_test.jsonl", "guard_challenge.jsonl"})
SEALED_MANIFESTS = frozenset({"test_core.json", "test_boundary.json", "challenge_core.json",
                              "challenge_boundary.json", "guard_test.json", "guard_challenge.json"})
SEALED_NAMES = SEALED_BANKS | SEALED_MANIFESTS
REFUSED_EVENTS = {"open": (0,), "shutil.copyfile": (0, 1), "os.link": (0, 1), "os.rename": (0, 1)}
LISTING_EVENTS = frozenset({"os.listdir", "os.scandir", "glob.glob"})
MUTATING_EVENTS = frozenset({"os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.truncate", "os.chmod",
                             "os.chown", "os.link", "os.symlink", "os.utime", "shutil.copyfile", "shutil.copymode",
                             "shutil.copystat", "shutil.copytree", "shutil.move", "shutil.rmtree", "shutil.chown"})
WATCHED = frozenset(REFUSED_EVENTS) | LISTING_EVENTS | MUTATING_EVENTS
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
# The data files these CLIs are expected to open; anything else opened is listed, not refused.
EXPECTED_DATA_FILES = frozenset({
    "manifest.json", "manifests/atomic_S.json", "canonical/train.jsonl", "canonical/dev.jsonl",
    "canonical/calibration.jsonl", "canonical/guard_dev.jsonl", "oracles/dev.jsonl", "oracles/calibration.jsonl",
    "oracles/guard_dev.jsonl", "policy/dev.jsonl", "policy/calibration.jsonl"})
NULL_CELL_SEED = 20260926  # C7: the DEV null cell at the registered counts
OUTPUTS = {"calibration_scope": "S0PRE_CALIBRATION_SCOPE.json", "dev_scope": "S0PRE_DEV_SCOPE.json",
           "data_gates": "S0PRE_DATA_GATES.json", "canary_dev": "S0PRE_CANARY_DEV.json",
           "canary_calibration": "S0PRE_CANARY_CALIBRATION.json", "canary_guard_dev": "S0PRE_CANARY_GUARD_DEV.json",
           "h3b_null_cell_dev": "S0PRE_H3B_NULL_CELL_DEV.json"}
OPEN_LOG = "S0PRE_OPEN_LOG.json"
TESTS_FILE = "S0PRE_TESTS_WINDOWS.txt"
SUITE_ARGS = ("-m", "pytest", "-q", "experiments/maxrl_comp002_pretraining_v1/tests")
SUITE_TIMEOUT = 3600
CONTROL_CONTENT = b'{"synthetic": "COMP-RS-001 evidence-runner guard control"}\n'
CONTROL_FILES = ("test.jsonl", "challenge.jsonl", "guard_test.jsonl", "guard_challenge.jsonl",
                 "manifests/test_core.json", "dev.jsonl")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def sealed_name(name: str) -> bool:
    """A final path component names a sealed file, compared the way Windows opens it."""
    return name.split(":", 1)[0].rstrip(" .").casefold() in SEALED_NAMES


def _text(value) -> str | None:
    if isinstance(value, (str, bytes, os.PathLike)):
        try:
            return os.fsdecode(os.fspath(value))
        except (TypeError, ValueError):
            return None
    return None


def _resolved(text: str) -> list[str]:
    """The absolute and symlink-resolved forms of a path (each only when it can be computed)."""
    forms = []
    for resolve in (os.path.abspath, os.path.realpath):
        try:
            forms.append(resolve(text))
        except (OSError, ValueError):
            pass
    return list(dict.fromkeys(forms))


def _is_sealed(text: str, resolved: list[str]) -> bool:
    """The raw, absolute or resolved form of a path ends in a sealed name."""
    return any(sealed_name(os.path.basename(form)) for form in [text, *resolved])


def _root_forms(root: Path) -> tuple[str, ...]:
    text = os.fspath(root)
    return tuple(dict.fromkeys(os.path.normcase(f(text)).rstrip("\\/") for f in (os.path.abspath, os.path.realpath)))


class SealedOpenRefused(PermissionError):
    """The evidence runner's audit hook refused a sealed file before it was opened."""


class Guard:
    """sys.addaudithook callback: refuses sealed names anywhere; records data-root opens, listings and writes."""

    def __init__(self, data: Path):
        self.data = Path(data)
        self.roots = _root_forms(self.data)
        self.step = "startup"
        self.opens: list[tuple[str, str, bool]] = []       # (step, path relative to the data root, write)
        self.listings: list[tuple[str, str, str]] = []     # (step, event, relative path)
        self.mutations: list[tuple[str, str, str]] = []    # (step, event, relative path)
        self.refused: list[dict] = []
        self._local = threading.local()

    def relative(self, text: str, resolved: list[str] | None = None) -> str | None:
        """The path relative to the data root (POSIX separators), or None when it lies outside."""
        for form in _resolved(text) if resolved is None else resolved:
            folded = os.path.normcase(form)
            for root in self.roots:
                if folded == root:
                    return "."
                if folded.startswith(root + os.sep):
                    return form[len(root) + 1:].replace("\\", "/")
        return None

    def __call__(self, event: str, args: tuple) -> None:
        if event not in WATCHED or getattr(self._local, "busy", False):
            return
        self._local.busy = True
        try:
            self._handle(event, args)
        finally:
            self._local.busy = False

    def _handle(self, event: str, args: tuple) -> None:
        texts = [_text(value) for value in args]
        wanted = range(len(texts)) if event in MUTATING_EVENTS else range(min(1, len(texts)))  # open/list: the path
        resolved = {i: _resolved(texts[i]) for i in wanted if texts[i] is not None}
        for position in REFUSED_EVENTS.get(event, ()):
            if position in resolved and _is_sealed(texts[position], resolved[position]):
                self.refused.append({"step": self.step, "event": event, "path": texts[position]})
                raise SealedOpenRefused("sealed file refused by the evidence runner's audit hook: " + texts[position])
        if event == "open":
            if 0 in resolved:
                relative = self.relative(texts[0], resolved[0])
                if relative is not None:
                    mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ""
                    flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
                    write = bool(flags & WRITE_FLAGS) or any(c in mode for c in "wax+")
                    self.opens.append((self.step, relative, write))
        elif event in LISTING_EVENTS:
            if 0 in resolved:
                relative = self.relative(texts[0], resolved[0])
                if relative is not None:
                    self.listings.append((self.step, event, relative))
        if event in MUTATING_EVENTS:
            for i, forms in resolved.items():
                relative = self.relative(texts[i], forms)
                if relative is not None:
                    self.mutations.append((self.step, event, relative))

    def opened(self, step: str | None = None) -> dict:
        return dict(sorted(Counter(r for s, r, _ in self.opens if step is None or s == step).items()))

    def listed(self, step: str | None = None) -> dict:
        return dict(sorted(Counter(f"{e} {r}" for s, e, r in self.listings if step is None or s == step).items()))


def package_modules_loaded() -> list[str]:
    """Package modules with code already imported (namespace packages have no __file__ and run no code)."""
    return sorted(name for name, module in list(sys.modules.items())
                  if name.startswith("experiments") and getattr(module, "__file__", None))


def install_guard(data: Path) -> tuple[Guard, list[str]]:
    guard = Guard(data)
    before = package_modules_loaded()
    sys.addaudithook(guard)
    return guard, before


def prepare_control(root: Path) -> None:
    """Write the synthetic control files; called before the hook exists (the hook refuses sealed writes too)."""
    (root / "manifests").mkdir()
    for name in CONTROL_FILES:
        (root / name).write_bytes(CONTROL_CONTENT)


def _close(handle) -> None:
    if isinstance(handle, int):
        os.close(handle)
    elif hasattr(handle, "close"):
        handle.close()


def run_control(guard: Guard, root: Path) -> dict:
    """The positive control: every sealed route refused by this hook, allowed spellings open, data opens recorded."""
    guard.step = "positive_control"
    test, challenge = root / "test.jsonl", root / "challenge.jsonl"
    guard_test, guard_challenge = root / "guard_test.jsonl", root / "guard_challenge.jsonl"
    sealed_routes = [
        ("builtins.open test.jsonl", lambda: open(test, "rb")),
        ("io.open challenge.jsonl", lambda: io.open(challenge, "rb")),
        ("os.open guard_test.jsonl", lambda: os.open(guard_test, os.O_RDONLY)),
        ("io.FileIO guard_challenge.jsonl", lambda: io.FileIO(guard_challenge)),
        ("io.open_code test.jsonl", lambda: io.open_code(os.fspath(test))),
        ("Path.read_bytes challenge.jsonl", lambda: challenge.read_bytes()),
        ("Path.read_text guard_test.jsonl", lambda: guard_test.read_text(encoding="utf-8")),
        ("shutil.copyfile test.jsonl", lambda: shutil.copyfile(test, root / "copied.bin")),
        ("builtins.open TEST.JSONL", lambda: open(root / "TEST.JSONL", "rb")),
        ("builtins.open 'test.jsonl.'", lambda: open(os.fspath(test) + ".", "rb")),
        ("builtins.open 'test.jsonl::$DATA'", lambda: open(os.fspath(test) + "::$DATA", "rb")),
        ("builtins.open manifests/test_core.json", lambda: open(root / "manifests" / "test_core.json", "rb")),
    ]
    routes = []
    for label, attempt in sealed_routes:
        before = len(guard.refused)
        try:
            handle = attempt()
        except SealedOpenRefused:
            routes.append({"route": label, "refused": len(guard.refused) == before + 1})
        except OSError as exc:  # an OS error is not the hook's refusal
            routes.append({"route": label, "refused": False, "error": type(exc).__name__ + ": " + str(exc)})
        else:
            _close(handle)
            routes.append({"route": label, "refused": False, "error": "opened"})
    allowed = []
    for label in ("dev.jsonl", "DEV.JSONL", "dev.jsonl.", "dev.jsonl::$DATA"):
        try:
            with open(os.fspath(root) + os.sep + label, "rb") as stream:
                opened = stream.read() == CONTROL_CONTENT
            allowed.append({"spelling": label, "opened": opened})
        except OSError as exc:
            allowed.append({"spelling": label, "opened": False, "error": type(exc).__name__})
    before_opens, before_listings = len(guard.opens), len(guard.listings)
    manifest_sha = sha256_bytes((guard.data / "manifest.json").read_bytes())
    names = os.listdir(guard.data)
    recorded_opens = guard.opens[before_opens:]
    recorded_listings = guard.listings[before_listings:]
    control_refusals = [r for r in guard.refused if r["step"] == "positive_control"]
    failures = [r["route"] + " not refused by the hook" for r in routes if not r["refused"]]
    if not allowed[0]["opened"]:
        failures.append("the allowed control file did not open")
    if [(r, w) for _, r, w in recorded_opens] != [("manifest.json", False)]:
        failures.append("the data-root manifest read was not recorded exactly once")
    if [(e, r) for _, e, r in recorded_listings] != [("os.listdir", ".")]:
        failures.append("the data-root listing was not recorded exactly once")
    if len(control_refusals) != sum(r["refused"] for r in routes):
        failures.append("refusal count differs from the refused routes")
    guard.step = "after_control"
    return {"status": "FAIL" if failures else "PASS", "failures": failures, "sealed_routes": routes,
            "allowed_spellings": allowed, "refusals": len(control_refusals),
            "recorder": {"opened": [r for _, r, _ in recorded_opens],
                         "listed": [f"{e} {r}" for _, e, r in recorded_listings],
                         "manifest_sha256": manifest_sha, "data_root_entries": len(names)},
            "note": "The sealed files are synthetic, in a temporary directory outside the data root; they were "
                    "written before the hook was installed. An allowed spelling that opens shows that the refused "
                    "spelling of a sealed name would have reached the file."}


def code_files() -> dict[str, Path]:
    """Every source the evidence depends on: this package's modules and the two packages they import."""
    files = {}
    for folder in (HERE, REPO / "experiments" / "maxrl_composable_skills_v2", REPO / "experiments" / "maxrl_verifiable_data_v1"):
        for path in sorted(folder.glob("*.py")):
            files[path.relative_to(REPO).as_posix()] = path
    lock = HERE / "source_lock.json"
    files[lock.relative_to(REPO).as_posix()] = lock
    return files


def code_sha256(files: dict[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in files.items()}


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              timeout=120).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _peak_rss_gib() -> float | None:
    """This process's peak working set (Windows), in GiB."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return None
    return round(counters.PeakWorkingSetSize / 2 ** 30, 3)


def _free_ram_gib() -> float | None:
    """Free physical memory (Windows), in GiB."""
    if os.name != "nt":
        return None
    import ctypes

    class Status(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = Status()
    status.dwLength = ctypes.sizeof(Status)
    if not ctypes.WinDLL("kernel32").GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return round(status.ullAvailPhys / 2 ** 30, 2)


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _display(path: Path) -> str:
    path = Path(path).resolve()
    return path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else str(path)


def _last_json(text: str):
    for line in reversed(text.splitlines()):
        if line.strip():
            try:
                return json.loads(line)
            except ValueError:
                return line
    return None


def run_step(guard: Guard, name: str, module, argv: list[str], output: Path) -> dict:
    """Run one CLI's main(argv) in process; its data-root opens are recorded under the step's name."""
    guard.step = name
    buffer = io.StringIO()
    start = time.perf_counter()
    code, error = 0, None
    try:
        with contextlib.redirect_stdout(buffer):
            code = module.main(argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        error = None if exc.code is None or isinstance(exc.code, int) else str(exc.code)
    except Exception as exc:  # a failing CLI is recorded, and the run stops
        code, error = 1, type(exc).__name__ + ": " + str(exc)
    wall = round(time.perf_counter() - start, 2)
    guard.step = "between_steps"
    printed = buffer.getvalue()
    sys.stdout.write(printed)
    record = {"step": name, "module": module.__name__, "argv": argv, "exit": code, "error": error,
              "wall_seconds": wall, "printed": _last_json(printed), "output": _display(output),
              "data_files_opened": guard.opened(name), "data_listings": guard.listed(name)}
    if output.exists():
        raw = output.read_bytes()
        record.update(output_sha256=sha256_bytes(raw), output_bytes=len(raw), output_has_cr=b"\r" in raw)
    else:
        record.update(output_sha256=None, output_bytes=None, output_has_cr=None)
    return record


def _plan(data: Path, out_dir: Path, reference_dir: Path | None, h3b) -> list[tuple[str, str, list]]:
    """(step, module, argv) in the registered order; '@pin:<step>' is replaced by that step's output sha256."""
    d = str(data)
    out = {key: str(out_dir / name) for key, name in OUTPUTS.items()}

    def reference(key):
        return [] if reference_dir is None else ["--reference-receipt", str(reference_dir / OUTPUTS[key])]

    return [
        ("calibration_scope", "scope_receipt", ["--data", d, "--role", "calibration", "--out", out["calibration_scope"]]),
        ("dev_scope", "scope_receipt", ["--data", d, "--role", "dev", "--out", out["dev_scope"]]),
        ("data_gates", "data_gates", ["--data", d, "--roles", "dev", "calibration", "--out", out["data_gates"]]),
        ("canary_dev", "canary", ["--data", d, "--role", "dev", "--signatures", "R", "S", "P", "RS",
                                  "--scope-receipt", out["dev_scope"], "--scope-receipt-sha256", "@pin:dev_scope",
                                  *reference("canary_dev"), "--out", out["canary_dev"]]),
        ("canary_calibration", "canary", ["--data", d, "--role", "calibration", "--signatures", "R", "S", "P",
                                          "--scope-receipt", out["calibration_scope"],
                                          "--scope-receipt-sha256", "@pin:calibration_scope",
                                          *reference("canary_calibration"), "--out", out["canary_calibration"]]),
        ("canary_guard_dev", "canary", ["--data", d, "--role", "guard_dev", *reference("canary_guard_dev"),
                                        "--out", out["canary_guard_dev"]]),
        ("h3b_null_cell_dev", "h3b_null_cell", ["--data", d, "--role", "dev",
                                                "--redraws", str(h3b.REGISTERED_MIN_REDRAWS),
                                                "--flips", str(h3b.REGISTERED_FLIPS), "--seed", str(NULL_CELL_SEED),
                                                "--out", out["h3b_null_cell_dev"]]),
    ]


def run_receipts(args) -> int:
    data = Path(args.data)
    out_dir = Path(args.out_dir)
    reference_dir = None if args.reference_dir is None else Path(args.reference_dir)
    targets = [out_dir / name for name in OUTPUTS.values()] + [out_dir / OPEN_LOG]
    if not args.control_only:
        if not out_dir.is_dir():
            raise SystemExit("output directory not found: " + str(out_dir))
        existing = [p.name for p in targets if p.exists()]
        if existing:
            raise SystemExit("evidence exists (create-only): " + ", ".join(existing))
        if reference_dir is not None:
            missing = [n for k, n in OUTPUTS.items() if k.startswith("canary") and not (reference_dir / n).is_file()]
            if missing:
                raise SystemExit("reference receipt not found: " + ", ".join(missing))
    runner_sha = sha256_file(Path(__file__))
    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    with tempfile.TemporaryDirectory(prefix="s0pre_guard_control_") as tmp:
        prepare_control(Path(tmp))
        guard, loaded_at_install = install_guard(data)  # before any package module is imported
        control = run_control(guard, Path(tmp))
    if args.control_only:
        print(json.dumps({"positive_control": control, "package_modules_loaded_at_install": loaded_at_install},
                         indent=2, sort_keys=True))
        return 0 if control["status"] == "PASS" and not loaded_at_install else 1
    failures = list(control["failures"])
    if loaded_at_install:
        failures.append("package modules were imported before the hook: " + ", ".join(loaded_at_install))
    files = code_files()
    code_start = code_sha256(files)
    steps, records = [], []
    if not failures:
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        guard.step = "import"
        modules = {name: importlib.import_module(PACKAGE + "." + name)
                   for name in ("preflight", "scope_receipt", "data_gates", "canary", "h3b_null_cell")}
        guard.step = "between_steps"
        if control["recorder"]["manifest_sha256"] != modules["preflight"].MANIFEST:
            failures.append("the data root's manifest.json is not the registered manifest")
        steps = _plan(data, out_dir, reference_dir, modules["h3b_null_cell"])
    written: dict[str, str] = {}
    for name, module_name, argv in steps:
        if failures:
            records.append({"step": name, "status": "NOT_RUN"})
            continue
        argv = [written[a[5:]] if a.startswith("@pin:") else a for a in argv]
        record = run_step(guard, name, modules[module_name], argv, out_dir / OUTPUTS[name])
        record["status"] = "PASS" if record["exit"] == 0 and record["output_sha256"] else "FAIL"
        if record["output_has_cr"]:
            record["status"] = "FAIL"
        records.append(record)
        if record["status"] != "PASS":
            failures.append(f"step {name} failed: exit {record['exit']}, {record['error']}")
        else:
            written[name] = record["output_sha256"]
    guard.step = "finish"
    code_end = code_sha256(files)
    runner_end = sha256_file(Path(__file__))
    run_refusals = [r for r in guard.refused if r["step"] != "positive_control"]
    run_opens = [(s, r, w) for s, r, w in guard.opens if s != "positive_control"]
    sealed_opened = sorted({r for _, r, _ in run_opens if sealed_name(r.rsplit("/", 1)[-1])})
    data_writes = sorted({f"{s} {r}" for s, r, w in run_opens if w})
    mutations = sorted({f"{s} {e} {r}" for s, e, r in guard.mutations})
    opened_all = dict(sorted(Counter(r for _, r, _ in run_opens).items()))
    native = [name for name in ("torch", "safetensors") if name in sys.modules]
    if run_refusals:
        failures.append("the hook refused %d opens during the run" % len(run_refusals))
    if sealed_opened or data_writes or mutations:
        failures.append("a sealed open, a write or a mutation under the data root was recorded")
    if code_end != code_start or runner_end != runner_sha:
        failures.append("a package source changed during the run")
    if native:
        failures.append("a native file reader was imported (the audit hook cannot see its opens): " + ", ".join(native))
    log = {
        "experiment": "COMP-RS-001", "stage": "S0PRE_OPEN_LOG", "kind": "s0pre_evidence_open_log",
        "amendment": "COMP_RS_001_AMENDMENT_1.md section 1 (read discipline); review finding DS-8",
        "status": "FAIL" if failures else "PASS", "failures": failures, "started": started,
        "finished": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "runner": {"path": _display(Path(__file__)), "sha256": runner_sha, "unchanged_at_end": runner_end == runner_sha,
                   "argv": sys.argv[1:]},
        "runtime": {"python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform(),
                    "numpy": sys.modules["numpy"].__version__ if "numpy" in sys.modules else _version("numpy"),
                    "torch": _version("torch"), "torch_source": "distribution metadata; torch was not imported",
                    "peak_working_set_gib": _peak_rss_gib()},
        "git": {"branch": _git("branch", "--show-current"), "head": _git("rev-parse", "HEAD"),
                "note": "the S0PRE sources are uncommitted working-tree files; their sha256 are in code_sha256"},
        "data_root": str(data),
        "hook": {"installed_before_package_import": not loaded_at_install,
                 "package_modules_loaded_at_install": loaded_at_install,
                 "sealed_names": sorted(SEALED_NAMES), "refused_events": sorted(REFUSED_EVENTS),
                 "recorded_events": sorted(WATCHED),
                 "rule": "refuse a sealed final name on any path (raw, absolute and resolved forms; case-folded; "
                         "trailing dots/spaces and ':stream' dropped); record every open, listing and mutation "
                         "under the data root",
                 "native_readers_imported": native},
        "positive_control": control,
        "steps": records,
        "data_files_opened": opened_all,
        "data_listings": guard.listed(),
        "data_files_opened_outside_steps": {k: v for k, v in Counter(
            r for s, r, _ in run_opens if s not in {n for n, _, _ in steps}).items()},
        "expected_data_files": sorted(EXPECTED_DATA_FILES),
        "unexpected_data_files_opened": sorted(set(opened_all) - EXPECTED_DATA_FILES),
        "refusals": run_refusals, "sealed_files_opened": sealed_opened,
        "data_root_writes": data_writes, "data_root_mutations": mutations,
        "outputs": {r["step"]: r.get("output_sha256") for r in records},
        "code_sha256": code_start, "code_unchanged_during_run": code_end == code_start,
        "reference_dir": None if reference_dir is None else str(reference_dir),
        "test_challenge_read": bool(sealed_opened), "model_outputs_read": False,
        "note": "CPU mechanics only: no model, tokenizer or GPU. Guard (preservation) banks are bound by their "
                "manifest-checked bank sha; a guard scope receipt is future work (C8)."}
    with open(out_dir / OPEN_LOG, "x", encoding="utf-8", newline="\n") as stream:
        json.dump(log, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": log["status"], "failures": failures, "open_log_sha256": sha256_file(out_dir / OPEN_LOG),
                      "data_files_opened": opened_all, "outputs": log["outputs"]}, indent=2))
    return 0 if not failures else 1


VERSION_PROBE = """import json, platform, sys
import numpy, pytest, safetensors, torch
try:
    import scipy
    scipy_version = scipy.__version__
except Exception:
    scipy_version = "absent"
print(json.dumps({"python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform(),
                  "torch": torch.__version__, "numpy": numpy.__version__, "safetensors": safetensors.__version__,
                  "pytest": pytest.__version__, "scipy": scipy_version}))
"""


def suite_files() -> dict[str, Path]:
    files = {path.relative_to(REPO).as_posix(): path for path in sorted(HERE.glob("*.py"))}
    files.update({path.relative_to(REPO).as_posix(): path for path in sorted((HERE / "tests").glob("*.py"))})
    files.update({path.relative_to(REPO).as_posix(): path for path in sorted(EVIDENCE.glob("S0PRE_*"))
                  if path.name != TESTS_FILE})
    files.update({path.relative_to(REPO).as_posix(): path for path in sorted(EVIDENCE.glob("Q*.json"))})
    return files


def _suite_run(env: dict) -> dict:
    free = _free_ram_gib()
    start = datetime.datetime.now().astimezone()
    tick = time.perf_counter()
    done = subprocess.run([sys.executable, *SUITE_ARGS], cwd=REPO, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=SUITE_TIMEOUT)
    output = done.stdout.decode("utf-8", errors="replace").replace("\r\n", "\n").rstrip("\n")
    return {"output": output, "exit": done.returncode, "wall": round(time.perf_counter() - tick, 1),
            "start": start.isoformat(timespec="seconds"),
            "end": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "free_ram_gib": free}


def run_suite(args) -> int:
    data = Path(args.data)
    out = Path(args.out_dir) / TESTS_FILE
    missing = Path(args.missing_data)
    if out.exists():
        raise SystemExit("test evidence exists (create-only): " + str(out))
    if missing.exists():
        raise SystemExit("the missing-data path exists: " + str(missing))
    if not (data / "manifest.json").is_file():
        raise SystemExit("dataset absent: " + str(data))
    probe = subprocess.run([sys.executable, "-c", VERSION_PROBE], capture_output=True, text=True, timeout=600)
    if probe.returncode != 0:
        raise SystemExit("version probe failed: " + probe.stderr[-2000:])
    versions = json.loads(probe.stdout.strip().splitlines()[-1])
    files = suite_files()
    before = code_sha256(files)
    present_env = dict(os.environ)
    if os.path.normcase(os.path.abspath(data)) == os.path.normcase(os.path.abspath(DEFAULT_DATA)):
        present_env.pop(DATA_ENV, None)
        present_note = "COMP_RS_DATA unset, so the default root " + str(DEFAULT_DATA)
    else:
        present_env[DATA_ENV] = str(data)
        present_note = "COMP_RS_DATA=" + str(data)
    present = _suite_run(present_env)
    absent = _suite_run(dict(os.environ, **{DATA_ENV: str(missing)}))
    after = code_sha256(files)
    hashes = "\n".join(f"{before[name]}  {name}" for name in sorted(before))
    changed = sorted(name for name in before if after.get(name) != before[name])
    command = "python " + " ".join(SUITE_ARGS)
    text = f"""COMP-RS-001 pre-S0 CPU qualification: package test suite (Windows)

Written by: {_display(Path(__file__))} --suite (sha256 {sha256_file(Path(__file__))})
Platform: {versions["platform"]}
Worktree: {REPO}
Branch: {_git("branch", "--show-current")}; HEAD {_git("rev-parse", "HEAD")}. The S0PRE files are uncommitted
working-tree files at this run; the sha256 of each at the time of the run is listed below.
Interpreter: {versions["executable"]}
Python {versions["python"]}; torch {versions["torch"]} (CPU use only); safetensors {versions["safetensors"]};
numpy {versions["numpy"]}; pytest {versions["pytest"]}; scipy {versions["scipy"]} (tests only, behind importorskip).

Command, from the worktree root, run twice, one process at a time:
{command}

Run 1, dataset present ({present_note}):
{present["start"]} to {present["end"]} (local time), {present["wall"]} s; free RAM at launch {present["free_ram_gib"]} GiB
{present["output"]}
exit={present["exit"]}

Run 2, dataset absent (COMP_RS_DATA={missing}, a path that does not exist; the data-backed tests skip):
{absent["start"]} to {absent["end"]} (local time), {absent["wall"]} s; free RAM at launch {absent["free_ram_gib"]} GiB
{absent["output"]}
exit={absent["exit"]}

Output is verbatim except that line endings are normalised to LF.
Package files changed during the runs: {", ".join(changed) if changed else "none"}.

tests/conftest.py guards every test in both runs: a sealed-bank file (test/challenge/guard_test/guard_challenge
.jsonl) or sealed-role manifest under a 'data' directory or a guarded data root (the default root and
COMP_RS_DATA) is refused on every audited open route and on the torch/safetensors file readers, and an attempt
fails the test. Sealed-role support is exercised only on synthetic temporary directories.

What this establishes: the CPU mechanics of the S0PRE code (scoped scorer, content/taxonomy repairs, exact
statistics, H3b null cell, FP32 arm construction and delta files on synthetic tensors, spec-only data gates,
scope receipts, scorer canary) pass their fixtures on this host.
What it does not establish: no model or tokenizer was loaded, nothing was generated, no GPU was used, and no
TEST, CHALLENGE or guard_test file was opened. It is not native qualification, FP32 re-qualification or
training readiness.

sha256 of the package files, the tests and the evidence receipts the tests read, at the start of run 1:
{hashes}
"""
    with open(out, "x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    print(json.dumps({"present": [present["exit"], present["output"].splitlines()[-1] if present["output"] else ""],
                      "absent": [absent["exit"], absent["output"].splitlines()[-1] if absent["output"] else ""],
                      "changed": changed, "sha256": sha256_file(out)}))
    return 0 if present["exit"] == 0 and absent["exit"] == 0 and not changed else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path(os.environ.get(DATA_ENV, str(DEFAULT_DATA))))
    ap.add_argument("--out-dir", type=Path, default=EVIDENCE)
    ap.add_argument("--reference-dir", type=Path, default=None,
                    help="earlier PASS canary receipts of the same banks (same file names)")
    ap.add_argument("--control-only", action="store_true", help="run only the hook's positive control; write nothing")
    ap.add_argument("--suite", action="store_true", help="run the test suite twice and write " + TESTS_FILE)
    ap.add_argument("--missing-data", type=Path, default=MISSING_DATA)
    args = ap.parse_args(argv)
    if args.suite:
        if args.control_only or args.reference_dir is not None:
            raise SystemExit("--suite takes no --control-only or --reference-dir")
        return run_suite(args)
    return run_receipts(args)


if __name__ == "__main__":
    raise SystemExit(main())
