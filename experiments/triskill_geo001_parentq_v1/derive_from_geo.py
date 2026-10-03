"""Derive parentq_supervise.py and parentq_launch_wmi.py from GEO's frozen originals.

Every change is an exact, asserted replacement (each old string occurs exactly once), so the
two lanes' supervisors differ only by the lines listed here. `--check` (used by the CPU tests)
re-derives in memory and requires byte equality with the committed files.

    python derive_from_geo.py          # write the two derived files
    python derive_from_geo.py --check  # compare only
"""
import sys
from pathlib import Path

PQ = Path(__file__).resolve().parent
GEO = PQ.parent / "triskill_geometric_pilot_v1"

SUP_DOC_OLD = '''"""Section H supervisor: GPU lease holder, external hard wall and own-lease release.

Replaces the original dependency on another lane's containment verifier (section H
item 2) with the programme's existing launch convention: the shared `gpu_lease` lock,
a watchdog with a hard wall, a tree kill of OUR worker only, and token-checked release
of OUR lease. Start it detached through launch_wmi.py so a session teardown cannot kill
it. It never waits for, reclaims from or terminates a foreign process or lock, and it
refuses to start without an explicit operator grant for this frozen screen.

Each launch gets its own attempt directory under native_attempts/. The screen output
(native_screen_sectionh.json) is written by the worker only once the model has loaded;
an attempt that stops earlier leaves the screen unspent (registered retry rule).
"""'''
SUP_DOC_NEW = '''"""PARENTQ supervisor: GPU lease holder, external hard wall and own-lease release.

GEO section H's reviewed supervisor (triskill_geometric_pilot_v1/supervise.py, frozen in
FREEZE_SECTIONH.json) with this lane's names, pins and retry rule; a CPU test diffs the two.
Start it detached through parentq_launch_wmi.py. It never waits for, reclaims from or
terminates a foreign process or lock, and it refuses to start without an explicit operator
grant for this frozen qualification.

Each launch gets its own attempt directory under native_attempts/. The create-only result
(native_parentq.json) is written by the worker only for a terminal verdict. A loaded attempt
that stops short leaves incomplete_result_parentq.json in its attempt directory; at most
MAX_LOADED_ATTEMPTS loaded attempts are ever made (PREREG_PARENTQ.md section 7).
"""'''

SUPERVISOR = [
    (SUP_DOC_OLD, SUP_DOC_NEW),
    ('LEASE_LABEL = "triskill-geo001-sectionh"       # equals worker.LEASE_LABEL (CPU test)',
     'LEASE_LABEL = "triskill-geo001-parentq"        # equals parentq_worker.LEASE_LABEL (CPU test)'),
    ('FREEZE_NAME = "FREEZE_SECTIONH.json"', 'FREEZE_NAME = "FREEZE_PARENTQ.json"'),
    ('START_NAME = "SUPERVISOR_START_SECTIONH.json"   # equals worker.START_NAME (CPU test)',
     'START_NAME = "SUPERVISOR_START_PARENTQ.json"    # equals parentq_worker.START_NAME (CPU test)'),
    ('RECEIPT_NAME = "SUPERVISOR_RECEIPT_SECTIONH.json"', 'RECEIPT_NAME = "SUPERVISOR_RECEIPT_PARENTQ.json"'),
    ('LOG_NAME = "native_worker_sectionh.log"', 'LOG_NAME = "native_worker_parentq.log"'),
    ('WORKER_PID_NAME = "WORKER_PID_SECTIONH.json"   # written by the worker itself',
     'WORKER_PID_NAME = "WORKER_PID_PARENTQ.json"    # written by the worker itself'),
    ('OUT_NAME = "native_screen_sectionh.json"',
     'OUT_NAME = "native_parentq.json"\nINCOMPLETE_NAME = "incomplete_result_parentq.json"\nMAX_LOADED_ATTEMPTS = 3'),
    ('PINNED_LOCAL = ("worker.py", "supervise.py", "dependency_binding.py", "benchmark.py", "geometry.py")',
     'PINNED_LOCAL = ("parentq_worker.py", "parentq_supervise.py", "parentq_rows.py", "parentq_decide.py",\n'
     '                "parentq_bank.json", "parentq_canary.json")\n'
     'GEO_FREEZE_SHA256 = "f18d48670e8928afb72b7182756f8f7bb6c8e241fb5988ae1daa91ebc534b687"'),
    ('        raise ValueError("STOP_FREEZE_MISSING_GPU_LEASE_PIN")\n    return freeze',
     '        raise ValueError("STOP_FREEZE_MISSING_GPU_LEASE_PIN")\n'
     '    geo_freeze = here.parent / "triskill_geometric_pilot_v1" / "FREEZE_SECTIONH.json"\n'
     '    if (freeze.get("geo_freeze", {}).get("sha256") != GEO_FREEZE_SHA256\n'
     '            or _sha256_bytes(geo_freeze.read_bytes()) != GEO_FREEZE_SHA256):\n'
     '        raise ValueError("STOP_GEO_FREEZE_MISMATCH")\n'
     '    return freeze'),
    ("""    spec = importlib.util.spec_from_file_location(f"_sectionh_gpu_lease_{entry['sha256'][:16]}", path)""",
     """    spec = importlib.util.spec_from_file_location(f"_parentq_gpu_lease_{entry['sha256'][:16]}", path)"""),
    ('def new_attempt_dir(here: Path = HERE) -> Path:',
     'def loaded_incomplete_attempts(here: Path = HERE) -> int:\n'
     '    folder = here / ATTEMPTS_DIR\n'
     '    return sum(1 for _ in folder.glob(f"*/{INCOMPLETE_NAME}")) if folder.exists() else 0\n\n\n'
     'def new_attempt_dir(here: Path = HERE) -> Path:'),
    ('    if out.exists():\n        raise FileExistsError(f"STOP_SCREEN_ALREADY_SPENT:{out.name}")',
     '    if out.exists():\n        raise FileExistsError(f"STOP_PARENTQ_ALREADY_DECIDED:{out.name}")\n'
     '    if loaded_incomplete_attempts(here) >= MAX_LOADED_ATTEMPTS:\n'
     '        raise RuntimeError("STOP_PARENTQ_ATTEMPTS_EXHAUSTED")'),
    ('    receipt = {"experiment": "TRISKILL-GEO-001-SECTIONH",', '    receipt = {"experiment": "TRISKILL-GEO-001-PARENTQ",'),
    ('    token = lease.acquire(LEASE_LABEL, "section H bounded zero-training screen (<=600 s)", timeout_seconds=0)',
     '    token = lease.acquire(LEASE_LABEL, "GEO parent qualification, BASE/T/H only (<=600 s)", timeout_seconds=0)'),
    ('        command = [python, "-u", str(here / "worker.py"), "run", "--freeze", str(freeze_path),',
     '        command = [python, "-u", str(here / "parentq_worker.py"), "run", "--freeze", str(freeze_path),'),
    ('                        "screen_output": out.name, "screen_output_exists": out.exists()})',
     '                        "result_output": out.name, "result_output_exists": out.exists()})'),
    ("""                        help="the operator's explicit grant for this frozen screen, recorded verbatim")""",
     """                        help="the operator's explicit grant for this frozen qualification, recorded verbatim")"""),
]

LAUNCHER = [
    ('"""Start supervise.py detached through WMI Win32_Process.Create.',
     '"""Start parentq_supervise.py detached through WMI Win32_Process.Create (GEO\'s launch_wmi.py,\n'
     'frozen in FREEZE_SECTIONH.json, with this lane\'s names; a CPU test diffs the two).'),
    ('START_NAME = "SUPERVISOR_START_SECTIONH.json"', 'START_NAME = "SUPERVISOR_START_PARENTQ.json"'),
    ('@{CommandLine=$env:SECTIONH_WMI_COMMAND; CurrentDirectory=$env:SECTIONH_WMI_CWD; ProcessStartupInformation=$si}; "',
     '@{CommandLine=$env:PARENTQ_WMI_COMMAND; CurrentDirectory=$env:PARENTQ_WMI_CWD; ProcessStartupInformation=$si}; "'),
    ('    return subprocess.list2cmdline([python, "-u", str(here / "supervise.py"), "--freeze", str(freeze),',
     '    return subprocess.list2cmdline([python, "-u", str(here / "parentq_supervise.py"), "--freeze", str(freeze),'),
    ('    env = {**os.environ, "SECTIONH_WMI_COMMAND": command, "SECTIONH_WMI_CWD": str(cwd)}',
     '    env = {**os.environ, "PARENTQ_WMI_COMMAND": command, "PARENTQ_WMI_CWD": str(cwd)}'),
    ('    parser.add_argument("--freeze", type=Path, default=HERE / "FREEZE_SECTIONH.json")',
     '    parser.add_argument("--freeze", type=Path, default=HERE / "FREEZE_PARENTQ.json")'),
]


PAIRS = ((GEO / "supervise.py", PQ / "parentq_supervise.py", SUPERVISOR),
         (GEO / "launch_wmi.py", PQ / "parentq_launch_wmi.py", LAUNCHER))


def derived(source: Path, replacements) -> bytes:
    text = source.read_bytes().decode("utf-8")
    for old, new in replacements:
        if text.count(old) != 1:
            raise SystemExit(f"{source.name}: expected one occurrence of {old[:70]!r}, found {text.count(old)}")
        text = text.replace(old, new)
    return text.encode("utf-8")


def check() -> bool:
    return all(target.read_bytes() == derived(source, replacements) for source, target, replacements in PAIRS)


if __name__ == "__main__":
    if "--check" in sys.argv:
        print("MATCH" if check() else "DIFFERS")
    else:
        for source, target, replacements in PAIRS:
            target.write_bytes(derived(source, replacements))
        print("derived", len(SUPERVISOR), "supervisor and", len(LAUNCHER), "launcher replacements")
