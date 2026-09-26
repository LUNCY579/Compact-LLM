"""ABST-001 -- does linear compensation lower the likelihood of the UNKNOWN reference on the recommended
V4_32768/20 student, and does it move V4/20 toward the teacher? (ABST001_PREREG.md, revision R1)

Cells on the SAME fresh RTG1-dev rows (256 UNKNOWN + 256 known), one forward pass per row (B = 1):
  depth configs  S20 plain / S20 LC / S18 plain / S18 LC / teacher (28 blocks)
  input vocab    V0, V4, V2 -- each pass read out under every output support (V0, V4, V2)
  matched cells  A = S20/V4  B = S20/V2  C = S18/V4  D = S18/V2  E = S20/V0  F = S18/V0  T0 T4 T2
The compensation is SMALLVD-001's (fit on SELECT; no free parameter).

REFUSAL CONTROLS, all BEFORE the first forward pass of the real teacher on the panel:
  R1 data   panel/manifest hashes, strata, references, id disjointness (incl. designer-seen rows),
            structure_id disjointness from every used row and RTG1 train (with a real positive control)
  R2 vocab  UNKNOWN and 0-3 are single tokens with the same old id in V0/V4/V2; <|im_end|> retained;
            every reference + <|im_end|> inside V4 and V2 (checked on text)
  R3 identity fold x2 (S20, S18), bitwise, on SELECT
  R4 fit canary x2: S20 vs HLCC-001's recorded fits, S18 vs LC_ARMS's, <= 1e-6
  R5 measurement canary, tier 1 (EXACT, the recorded code path vs.answer_bpb on SMALLVD-001's spent
     HOLDOUT reference rows): A-plain vs S6 V4/20, D-plain vs S6 V2/18, T4 vs S6 V4/28,
     D-LC vs LC_ARMS lc V2, F-LC vs LC_ARMS lc V0 -- every per-row nats within 1e-6
  R5b tier 2 (the NEW B = 1 path, same rows and cells): |nats - recorded| <= 1e-3 (sign agreement of d(D)
     is REPORTED, not gated -- once the nats clause holds it cannot fail)
  R6 measured throughput: the projected panel time must be <= --runtime-cap-min
  W  every down_proj weight byte-identical to its state at load (LC folds restored), before the boundary
READ-ONCE (R1 of the prereg; cold review M2/M3): an O_EXCL RUN.lock for the life of the run; the budget rules
are evaluated at start-up, again after the GPU lease, and again just before the BOUNDARY marker; for real runs
the attempts directory is keyed to the PANEL's records directory (never --out), and dry runs are refused there.
At most MAX_CROSSINGS boundary crossings; a new crossing is refused once two crossings have ended in recorded
(infrastructure) failures, or once any attempt record says the primary was written. A crossing that left no
attempt record was hard-killed (reboot, power loss, or a native crash) and persisted no panel values (soup
review: three operator restarts in 9.5 h on 2026-09-25/26). Liveness of a recorded pid compares the process
creation time with the recorded epoch, so a pid reused after a reboot is not the recorded process.
Code binding: the hashes of every module actually LOADED from the package and rab001 directories are taken at
start-up; the gated modules must load from the package (no shadowing from rab001_exploration, which is earlier
on sys.path); the files must be unchanged at the boundary; the marker records them, and a rerun must match.
After the boundary: PRIMARY cells first; ABST_PRIMARY.json (fsynced, written in a temporary directory and
renamed into place) before any other cell and before any RAM wait; RAM dips WAIT (bounded) instead of aborting,
but never between the last cell of a stage and that stage's write (primary, H3, result); ABST_H3.json before
generation; any failure writes attempts/ATTEMPT_*.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "rab001_exploration"))
import abst_describe as ad  # noqa: E402
import abst_measure as am  # noqa: E402
import abst_verdict as av  # noqa: E402
import gpu_lease  # noqa: E402
import hostbudget  # noqa: E402
import layer_ablation as la  # noqa: E402
import linear_compensation as lc  # noqa: E402
import param_accounting as pa  # noqa: E402
import run_layer_study as rls  # noqa: E402
import virtual_student as vs  # noqa: E402

CANARY_TOL = 1e-6
TIER2_TOL = 1e-3
PRE_BOUNDARY_RAM_GIB = {"real": 2.5, "dry": 0.5}
POST_BOUNDARY_WAIT_S, POST_BOUNDARY_POLL_S = 1800, 30
MAX_CROSSINGS = 3              # boundary crossings in total; none after two recorded failures
REGISTERED_PANEL_ROOT = Path(r"C:\Users\clclc\BusinessProject\rineya-dataset-retention-records\abst001-20260925")
REGISTERED_TEACHER_SHA256 = "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b"   # = HLCC_RESULT, LC_ARMS
GATED_FROM_PACKAGE = ("abst_describe", "abst_measure", "abst_verdict", "hlcc_verdict", "linear_compensation",
                      "virtual_student", "layer_ablation", "layer_select", "run_layer_study", "param_accounting",
                      "gpu_lease")
RAB = HERE.parents[1] / "experiments" / "rab001_exploration"
RESTART_EVENT_IDS = (1074, 41, 109, 6008)   # User32 shutdown/restart, Kernel-Power 41/109, EventLog 6008
PRIMARY_ORDER = (("S20_plain", "V4"), ("S20_lc", "V4"), ("T", "V4"))
REST_ORDER = (("S20_plain", "V2"), ("S20_lc", "V2"), ("S18_plain", "V4"), ("S18_lc", "V4"),
              ("S18_plain", "V2"), ("S18_lc", "V2"), ("S20_plain", "V0"), ("S20_lc", "V0"),
              ("S18_plain", "V0"), ("S18_lc", "V0"), ("T", "V0"), ("T", "V2"))
GEN_CELLS = (("A_plain", "S20_plain", "V4"), ("A_lc", "S20_lc", "V4"), ("T4", "T", "V4"), ("T0", "T", "V0"))


def parse_kept(s):
    return [int(x) for x in s.split(",")]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def jl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def code_hashes() -> dict:
    """sha256 of every module actually LOADED from the package or the rab001 directory, keyed '<dir>/<file>'
    (R1 review: hashing every *.py in the package was too strict -- an unrelated helper blocked a rerun -- and
    incomplete -- hostbudget.py was missing)."""
    out = {}
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f:
            continue
        q = Path(f).resolve()
        if q.suffix == ".py" and q.parent in (HERE, RAB.resolve()):
            out[f"{q.parent.name}/{q.name}"] = rls.sha256_file(q)
    return dict(sorted(out.items()))


def module_origins() -> dict:
    """Refuses unless every gated module was loaded from the package and hostbudget from rab001 (R1 review:
    rab001_exploration precedes the package on sys.path, so a same-named file there would shadow a gated one)."""
    got = {n: str(Path(sys.modules[n].__file__).resolve().parent) for n in GATED_FROM_PACKAGE + ("hostbudget",)}
    bad = {n: d for n, d in got.items() if Path(d) != (RAB.resolve() if n == "hostbudget" else HERE)}
    if bad:
        raise SystemExit(f"REFUSE: a gated module was loaded from the wrong directory: {bad}")
    return got


def pid_alive(pid: int, started_by: float | None = None) -> bool:
    """Is the recorded process running? A running process with this pid counts only if it was created no
    later than `started_by` (+2 s) -- the epoch at which the record was written -- so a pid reused after a
    reboot is not mistaken for it (soup review). If the creation time cannot be read, fail closed (alive)."""
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.OpenProcess.restype, k.OpenProcess.argtypes = wintypes.HANDLE, [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        h = k.OpenProcess(0x1000, False, int(pid))        # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            if ctypes.get_last_error() == 87:              # ERROR_INVALID_PARAMETER: no such process
                return False
            try:                                           # e.g. access denied: read the creation time without a handle
                import psutil
                created = psutil.Process(int(pid)).create_time()
            except Exception as e:  # noqa: BLE001
                return type(e).__name__ != "NoSuchProcess"     # unknown: fail closed
            return started_by is None or created <= float(started_by) + 2.0
        try:
            code = wintypes.DWORD()
            if not (k.GetExitCodeProcess(h, ctypes.byref(code)) and code.value == 259):     # STILL_ACTIVE
                return False
            if started_by is None:
                return True
            ft = [wintypes.FILETIME() for _ in range(4)]
            if not k.GetProcessTimes(h, *(ctypes.byref(x) for x in ft)):
                return True
            created = ((ft[0].dwHighDateTime << 32) | ft[0].dwLowDateTime) / 1e7 - 11644473600.0
            return created <= float(started_by) + 2.0
        finally:
            k.CloseHandle(h)
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def restart_events_since(epoch: float) -> dict:
    """DESCRIPTIVE evidence for a crossing that left no attempt record: System-log restart and power events
    since `epoch`. Never a gate. Read from the XML (SystemTime is true UTC; the text format prints LOCAL time
    with a 'Z' suffix -- measured 2026-09-26: 00:02:48 'Z' in text = 05:02:48Z in XML)."""
    if os.name != "nt":
        return {"queried": False}
    since = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(epoch))
    ids = " or ".join(f"EventID={i}" for i in RESTART_EVENT_IDS)
    q = f"*[System[({ids}) and TimeCreated[@SystemTime>='{since}']]]"
    try:
        r = subprocess.run(["wevtutil", "qe", "System", f"/q:{q}", "/f:xml", "/c:50"],
                           capture_output=True, text=True, timeout=60)
    except Exception as e:  # noqa: BLE001
        return {"queried": True, "since_utc": since, "error": repr(e)}
    ev = []
    for x in re.findall(r"<Event .*?</Event>", r.stdout, re.S):
        i = re.search(r"<EventID[^>]*>(\d+)</EventID>", x)
        t = re.search(r"<TimeCreated SystemTime=['\"]([^'\"]+)", x)
        pr = re.search(r"<Provider Name=['\"]([^'\"]+)", x)
        ev.append({"event_id": int(i.group(1)) if i else None, "utc": t.group(1) if t else None,
                   "provider": pr.group(1) if pr else None})
    return {"queried": True, "since_utc": since, "returncode": r.returncode, "events": ev}


def budget_check(att_dir: Path, primary_root: Path, when: str) -> list:
    """The read-once rules. Refuses (SystemExit) if an ABST_PRIMARY.json exists anywhere under the root (any
    depth); if MAX_CROSSINGS boundary crossings exist; if any earlier crossing ended in a non-infrastructure
    failure; if any attempt record says the primary was written; if two earlier crossings ended in recorded
    failures; if an earlier crossing's process is still alive; or if the loaded code differs from any earlier
    crossing's. A crossing without an attempt record was hard-killed and wrote no panel values. An unreadable
    (truncated) record raises, i.e. refuses. Returns the earlier markers."""
    done = sorted(primary_root.rglob("ABST_PRIMARY.json")) if primary_root.exists() else []
    if done:
        raise SystemExit(f"READ BUDGET SPENT ({when}): a primary already exists at {done[0]}")
    markers = sorted(att_dir.glob("BOUNDARY_*.json")) if att_dir.exists() else []
    if not markers:
        return []
    if len(markers) >= MAX_CROSSINGS:
        raise SystemExit(f"READ BUDGET SPENT ({when}): {len(markers)} earlier boundary crossings (cap {MAX_CROSSINGS})")
    prior = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(att_dir.glob("ATTEMPT_*.json"))]
    if any(p.get("classification_hint") != "infrastructure" for p in prior):
        raise SystemExit(f"READ BUDGET SPENT ({when}): an earlier crossing ended in a non-infrastructure failure")
    if any(p.get("primary_written") for p in prior):
        raise SystemExit(f"READ BUDGET SPENT ({when}): an earlier attempt record says the primary was written")
    if len(prior) >= 2:
        raise SystemExit(f"READ BUDGET SPENT ({when}): {len(prior)} earlier crossings ended in recorded failures")
    for mp in markers:
        mk = json.loads(mp.read_text(encoding="utf-8"))
        if mk.get("pid") and mk["pid"] != os.getpid() and pid_alive(mk["pid"], mk.get("epoch")):
            raise SystemExit(f"REFUSE ({when}): the process of an earlier crossing (pid {mk['pid']}) is still alive")
        if mk.get("code_sha256") != code_hashes():
            raise SystemExit(f"REFUSE ({when}): the code changed since an earlier crossing ({mp.name})")
    return markers


def prior_crossings(att_dir: Path, markers: list) -> dict:
    """Evidence for the record: each earlier crossing, whether it left an attempt record (matched by
    launch_id), and -- if any left none (a hard kill) -- System-log restart events since the earliest."""
    recs = {}
    for p in sorted(att_dir.glob("ATTEMPT_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        recs[d.get("launch_id")] = d.get("classification_hint")
    out = []
    for mp in markers:
        mk = json.loads(mp.read_text(encoding="utf-8"))
        lid = mk.get("launch_id")
        out.append({"marker": mp.name, "launch_id": lid, "pid": mk.get("pid"), "epoch": mk.get("epoch"),
                    "attempt_record": recs.get(lid, "NONE (hard kill: no panel values persisted)")})
    blind = [c["epoch"] for c in out if c["attempt_record"].startswith("NONE") and c["epoch"]]
    return {"crossings": out, "restart_events": restart_events_since(min(blind)) if blind else None}


class RunLock:
    """O_EXCL lock file with our pid; a lock held by a dead process is renamed aside (recorded) and retaken."""

    def __init__(self, path: Path):
        self.path, self.mine, self.stale = path, False, None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    d = json.loads(self.path.read_text(encoding="utf-8"))
                    holder, held_since = d.get("pid"), d.get("epoch")
                except (OSError, ValueError):
                    holder = held_since = None
                if holder and pid_alive(holder, held_since):
                    raise SystemExit(f"REFUSE: another ABST-001 launch holds {self.path} (pid {holder})")
                self.stale = self.path.with_name(f"RUN.lock.stale_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
                os.replace(self.path, self.stale)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"pid": os.getpid(), "epoch": time.time(),
                           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, fh)
            self.mine = True
            return
        raise SystemExit(f"REFUSE: could not take {self.path}")

    def release(self):
        if self.mine and self.path.exists():
            try:
                if json.loads(self.path.read_text(encoding="utf-8")).get("pid") == os.getpid():
                    self.path.unlink()
            except (OSError, ValueError):
                pass


def write_fsync(path: Path, obj) -> None:
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())


def wait_for_ram(min_gib: float, label: str) -> None:
    """After the boundary a RAM dip WAITS (bounded) instead of aborting: a resident, idle model loses nothing."""
    t0 = time.time()
    while hostbudget.free_gib() < min_gib:
        if time.time() - t0 > POST_BOUNDARY_WAIT_S:
            raise hostbudget.LowMemoryAbort(f"{label}: free RAM below {min_gib} GiB for {POST_BOUNDARY_WAIT_S} s")
        time.sleep(POST_BOUNDARY_POLL_S)


def structure_overlap(panel_ids, used_ids, dev_rows, train_rows) -> dict:
    """structure_id disjointness of the panel from every used row and RTG1 train. Positive controls (R1 review:
    a probe drawn from the banned set and tested for membership passes by construction): a used row's and a
    train row's structure_id are each APPENDED TO THE PANEL LIST and must raise the overlap count by exactly one;
    every train row must carry a structure_id (a schema drift would empty the train side silently)."""
    sid = {x["id"]: x.get("structure_id") for x in dev_rows}
    banned = {sid[i] for i in used_ids if i in sid and sid[i]} | {x.get("structure_id") for x in train_rows if x.get("structure_id")}
    ps = [sid.get(i) for i in panel_ids]
    count = lambda xs: sum(1 for s in xs if s and s in banned)
    base = count(ps)
    used_probe = next((sid[i] for i in sorted(used_ids) if sid.get(i)), None)
    train_probe = next((x.get("structure_id") for x in train_rows if x.get("structure_id")), None)
    caught = {k: q is not None and count(ps + [q]) == base + 1 for k, q in (("used", used_probe), ("train", train_probe))}
    return {"panel_structure_ids": len(ps), "distinct": len(set(ps)), "missing": sum(s is None for s in ps),
            "overlap_with_used_or_train": base, "train_rows": len(train_rows),
            "train_rows_without_structure_id": sum(1 for x in train_rows if not x.get("structure_id")),
            "positive_controls_caught": caught, "positive_control_caught": all(caught.values())}


def cells_for_verdict(records, ids, known_ids):
    g = lambda cfg, v, rows=ids: [records[i]["m"][cfg][v][v]["nats"] for i in rows]
    out = {"A": {"plain": g("S20_plain", "V4"), "lc": g("S20_lc", "V4")}, "T4": {"teacher": g("T", "V4")},
           "A_known": {"plain": g("S20_plain", "V4", known_ids), "lc": g("S20_lc", "V4", known_ids)},
           "T4_known": {"teacher": g("T", "V4", known_ids)}}
    have = lambda cfg, v: all(cfg in records[i]["m"] and v in records[i]["m"][cfg] for i in ids)
    for X, cfg, v in (("B", "S20", "V2"), ("C", "S18", "V4"), ("D", "S18", "V2")):
        if have(cfg + "_plain", v) and have(cfg + "_lc", v):
            out[X] = {"plain": g(cfg + "_plain", v), "lc": g(cfg + "_lc", v)}
    return out


def is_infrastructure(e: BaseException) -> bool:
    """Narrow (cold review): out-of-memory on the GPU or the HOST (R1 review: Python MemoryError, numpy's
    _ArrayMemoryError subclass, torch's CPU allocator 'not enough memory'), the Windows TDR signature, or a RAM
    stand-down. A KeyboardInterrupt or any other CUDA error (e.g. a device-side assert, which is deterministic)
    is NOT infrastructure."""
    msg = str(e).lower()
    return (isinstance(e, (hostbudget.LowMemoryAbort, MemoryError)) or type(e).__name__ == "OutOfMemoryError"
            or "out of memory" in msg or "not enough memory" in msg or "cuda error: unknown error" in msg)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, type=Path)
    ap.add_argument("--panels", required=True, type=Path, help="SMALLVD-001 LAYER_PANELS.jsonl (SELECT fit set, HOLDOUT canary rows)")
    ap.add_argument("--study", required=True, type=Path, help="SMALLVD-001 layer_study (S1 outputs, S6 recorded nats)")
    ap.add_argument("--smallvd-lc", required=True, type=Path, help="LC_ARMS.json (S18 fits + recorded LC nats)")
    ap.add_argument("--hlcc-result", required=True, type=Path, help="HLCC_RESULT.json (S20 recorded fits)")
    ap.add_argument("--abst-panel", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--confirm-panel", required=True, type=Path)
    ap.add_argument("--tokenizer-corpus", required=True, type=Path)
    ap.add_argument("--rtg1-sft", required=True, type=Path, help="RTG1 sft dir (dev.jsonl, train.jsonl) for the structure_id check")
    ap.add_argument("--vocab-v4", required=True, type=Path)
    ap.add_argument("--vocab-v2", required=True, type=Path)
    ap.add_argument("--s20-kept", required=True, type=parse_kept)
    ap.add_argument("--s18-kept", required=True, type=parse_kept)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--runtime-cap-min", type=float, default=60.0)
    ap.add_argument("--max-rows-per-stratum", type=int, default=0, help="dry runs only")
    ap.add_argument("--max-rows-per-category", type=int, default=0, help="must equal the study's (dry runs)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    panel_root = a.abst_panel.resolve().parent.parent            # the records directory of THIS panel
    reg = REGISTERED_PANEL_ROOT.resolve()
    out = a.out.resolve()
    if a.dry_run:
        if any(out == d or d in out.parents for d in (reg.parent, reg, panel_root)):
            raise SystemExit("--dry-run may never write under the records tree or the panel's directory")
        att_dir, primary_root, mode = out.parent / "attempts", out.parent, "dry"
    else:
        if panel_root != reg or out != reg / "result":
            raise SystemExit(f"a real run reads only the registered panel under {reg} and writes only to {reg / 'result'}")
        att_dir, primary_root, mode = reg / "attempts", reg, "real"
    if out.exists():
        raise SystemExit(f"{out} exists (create-only)")
    origins = module_origins()
    code_at_start = code_hashes()
    budget_check(att_dir, primary_root, "start-up")
    launch_id = uuid.uuid4().hex
    tmp_out = out.with_name(f"{out.name}.partial_{launch_id}")
    lock = RunLock(att_dir / "RUN.lock")
    lock.acquire()
    token = None
    panel_read, partial, t_start = False, {}, time.time()
    try:
        markers = budget_check(att_dir, primary_root, "after the run lock")
        cfg = json.loads((a.teacher / "config.json").read_text(encoding="utf-8"))
        n0 = pa.analytic(cfg, cfg["vocab_size"], cfg["num_hidden_layers"])["unique_physical_parameters"]
        if a.dry_run:
            if n0 >= 50_000_000 or a.device != "cpu":
                raise SystemExit("--dry-run is only for tiny CPU models")
            hostbudget.require(0.8, "ABST-001 DRY RUN")
            guard_gib, chunk = 0.4, 128
        else:
            if a.max_rows_per_stratum or a.max_rows_per_category:
                raise SystemExit("subsampling is for dry runs only")
            if a.device != "cuda" or n0 < 50_000_000:
                raise SystemExit("a real run is the registered teacher on cuda")
            hostbudget.require(3.6, "ABST-001")
            if rls.sha256_file(a.teacher / "model.safetensors") != REGISTERED_TEACHER_SHA256:
                raise SystemExit("a real run needs the registered teacher weights")
            token, guard_gib, chunk = gpu_lease.acquire(timeout_s=7200, label="abst001"), 1.5, 256
            markers = budget_check(att_dir, primary_root, "after the GPU lease")
            hostbudget.require(3.6, "ABST-001 after the GPU lease")   # the lease wait can last 2 h (cold review #35)
        import tokenizers
        import torch
        import transformers
        rt = la.configure_runtime(17, a.device)
        dev = a.device

        # ---------------- R1 data (CPU, no model)
        man = json.loads(a.manifest.read_text(encoding="utf-8"))
        if man["panel_sha256"] != sha(a.abst_panel):
            raise SystemExit("R1: the panel does not match its manifest")
        panel = jl(a.abst_panel)
        if a.max_rows_per_stratum:
            keep, seen = [], {}
            for r in panel:
                k = (r["stratum"], r["family"])
                if seen.get(k, 0) < a.max_rows_per_stratum:
                    keep.append(r)
                    seen[k] = seen.get(k, 0) + 1
            panel = keep
        strata = {}
        for r in panel:
            strata[(r["stratum"], r["family"])] = strata.get((r["stratum"], r["family"]), 0) + 1
        want = a.max_rows_per_stratum or 64
        if len(strata) != 8 or any(v != want for v in strata.values()):
            raise SystemExit(f"R1: strata {strata}")
        if any((r["reference"] == "UNKNOWN") != (r["stratum"] == "unk") for r in panel) or \
                any(r["stratum"] == "known" and r["reference"] not in ("0", "1", "2", "3") for r in panel):
            raise SystemExit("R1: a reference does not match its stratum")
        layer = jl(a.panels)
        used = {r["id"] for r in layer} | {r["id"] for r in jl(a.confirm_panel)} | {r["id"] for r in jl(a.tokenizer_corpus)}
        ids = [r["id"] for r in panel]
        reserve = {i for v in man["sealed_reserve_ids"].values() for i in v}
        if set(ids) & used or set(ids) & reserve or set(ids) & set(man["designer_seen_ids"]) or len(set(ids)) != len(ids):
            raise SystemExit("R1: panel rows overlap a used set, the sealed reserve or the designer-seen rows")
        so = structure_overlap(ids, used, jl(a.rtg1_sft / "dev.jsonl"), jl(a.rtg1_sft / "train.jsonl"))
        if (so["overlap_with_used_or_train"] or so["missing"] or so["distinct"] != len(ids) or not so["positive_control_caught"]
                or so["train_rows_without_structure_id"]):
            raise SystemExit(f"R1: structure_id check failed: {so}")
        if a.max_rows_per_category:
            keep, seen = [], {}
            for r in layer:
                k = (r["category"], r["split"])
                if seen.get(k, 0) < a.max_rows_per_category:
                    keep.append(r)
                    seen[k] = seen.get(k, 0) + 1
            layer = keep
        sel = [r for r in layer if r["split"] == "select"]
        ref_hold = [r for r in layer if r["split"] == "holdout" and r["kind"] == "chat" and r.get("reference")]
        S1 = json.loads((a.study / "S1_teacher_outputs.json").read_text(encoding="utf-8"))["rows"]
        S6 = json.loads((a.study / "S6_vocab_frontier.json").read_text(encoding="utf-8"))["configs"]
        LCA = json.loads(a.smallvd_lc.read_text(encoding="utf-8"))["arms"]["selected"]
        HL = json.loads(a.hlcc_result.read_text(encoding="utf-8"))["sets"]["primary"]
        if LCA["kept"] != a.s18_kept or HL["kept"] != a.s20_kept:
            raise SystemExit("R4: the recorded fits are for different kept sets than --s18-kept/--s20-kept")

        tok = tokenizers.Tokenizer.from_file(str(a.teacher / "tokenizer.json"))
        im_end, eot = tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")
        eos = [im_end, eot]
        model = la.load_model(a.teacher, dev)
        if next(model.parameters()).dtype != torch.float32:
            raise SystemExit("ABST requires an fp32 model")
        if token:
            gpu_lease.set_workers(token, [os.getpid()])
        L = model.config.num_hidden_layers
        full = list(range(L))

        def down_proj_fingerprint():
            return [hashlib.sha256(b.mlp.down_proj.weight.detach().cpu().numpy().tobytes()).hexdigest()
                    for b in model.model.layers]
        w_at_load = down_proj_fingerprint()
        V = {"V0": vs.Vocab(None, a.teacher / "tokenizer.json"), "V4": vs.Vocab(a.vocab_v4, a.teacher / "tokenizer.json"),
             "V2": vs.Vocab(a.vocab_v2, a.teacher / "tokenizer.json")}
        rows_v = model.lm_head.weight.shape[0]
        SUP = {k: am.support_index(v.retained, rows_v, dev) for k, v in V.items()}

        # ---------------- R2 vocabulary (text only)
        ans = am.answer_token_ids(V["V0"].teacher_tok, im_end)
        for k in ("V4", "V2"):
            keepset = set(V[k].retained.tolist())
            for s, i in ans.items():
                if i not in keepset:
                    raise SystemExit(f"R2: {s} (old id {i}) is not retained in {k}")
            for s in ("UNKNOWN", "0", "1", "2", "3"):
                if V[k].encode_old_ids(s) != [ans[s]]:
                    raise SystemExit(f"R2: {s!r} is not the single old id {ans[s]} under {k}")
            bad = [r["id"] for r in panel + ref_hold if not set(V[k].encode_old_ids(r["reference"]) + [im_end]) <= keepset]
            if bad:
                raise SystemExit(f"R2: {len(bad)} references leave {k}, e.g. {bad[:3]}")

        # ---------------- R3 identity folds, R4 fits + canaries (SELECT rows only)
        items_sel = rls.build_items(sel, tok, S1, im_end)
        sets = {"S20": a.s20_kept, "S18": a.s18_kept}
        recorded = {"S20": HL["fits"], "S18": LCA["fits"]}
        folds, fits, fit_canary = {}, {}, {}
        for nm in ("S18", "S20"):
            kept = sets[nm]
            ident = {r["p"]: torch.eye(model.config.hidden_size, device=dev) for r in lc.runs(kept, L) if r["p"] is not None}
            with la.keep_layers(model, kept):
                Hp = la.collect_positions(model, items_sel, eot, dev)
            with lc.folded(model, ident), la.keep_layers(model, kept):
                Hi = la.collect_positions(model, items_sel, eot, dev)
            if not all(torch.equal(x, y) for x, y in zip(Hp, Hi)):
                raise SystemExit(f"R3: {nm}: the identity fold does not reproduce the plain student bitwise")
            del Hp, Hi
            folds[nm], fits[nm] = lc.fit_compensation(model, kept, items_sel, eot, dev)
            hostbudget.guard(guard_gib, f"ABST {nm} fit")
            ref = [(f["p"], f["fit_residual_rel_fitted"]) for f in recorded[nm] if f.get("compensated")]
            got = [(f["p"], f["fit_residual_rel_fitted"]) for f in fits[nm] if f.get("compensated")]
            diff = max((abs(x[1] - y[1]) for x, y in zip(ref, got)), default=float("inf"))
            if [x[0] for x in ref] != [y[0] for y in got] or not diff <= CANARY_TOL:
                raise SystemExit(f"R4: FIT CANARY FAILED for {nm}: recorded {ref} vs now {got}")
            fit_canary[nm] = {"recorded": ref, "now": got, "max_abs_diff": diff}

        def configured(cfg_name):
            """(kept, folds or None) for a depth config name."""
            if cfg_name == "T":
                return full, None
            nm, kind = cfg_name.split("_")
            return sets[nm], (folds[nm] if kind == "lc" else None)

        # ---------------- R5 tier 1: the RECORDED code path on SMALLVD-001's spent HOLDOUT rows
        def rec_items(items):
            return {x["id"]: x["nats"] for x in items}
        L_name = f"{L}L"
        canary_cells = {
            "A_plain": ("S20_plain", "V4", rec_items(S6[f"{V['V4'].name}/{len(a.s20_kept)}L"]["answer_bpb"]["holdout_per_item"]),
                        S6[f"{V['V4'].name}/{len(a.s20_kept)}L"]["kept"]),
            "D_plain": ("S18_plain", "V2", rec_items(S6[f"{V['V2'].name}/{len(a.s18_kept)}L"]["answer_bpb"]["holdout_per_item"]),
                        S6[f"{V['V2'].name}/{len(a.s18_kept)}L"]["kept"]),
            "T4": ("T", "V4", rec_items(S6[f"{V['V4'].name}/{L_name}"]["answer_bpb"]["holdout_per_item"]), full),
            "D_lc": ("S18_lc", "V2", rec_items(LCA["lc"]["answer_bpb_V2"]["per_item"]), a.s18_kept),
            "F_lc": ("S18_lc", "V0", rec_items(LCA["lc"]["answer_bpb_V0"]["per_item"]), a.s18_kept),
        }
        tier1, tier2, t2_nats = {}, {}, {}
        t_tier2 = 0.0
        tok_count = 0
        for name, (cfg_name, v, recd, rec_kept) in canary_cells.items():
            kept, f = configured(cfg_name)
            if rec_kept != kept:
                raise SystemExit(f"R5: recorded {name} was for kept {rec_kept}, not {kept}")
            if set(recd) != {r["id"] for r in ref_hold}:
                raise SystemExit(f"R5: recorded {name} rows differ from the HOLDOUT reference rows")
            ctx = lc.folded(model, f) if f else _null()
            with ctx:
                now = rec_items(vs.answer_bpb(model, V[v], ref_hold, kept, im_end, dev, chunk=min(chunk, 256))["per_item"])
                t0 = time.time()
                new = am.measure(model, ref_hold, kept, V[v], {v: SUP[v]}, im_end, ans, dev)
                t_tier2 += time.time() - t0
            tok_count += sum(len(V[v].encode_old_ids(r["prompt_text"])) + 2 for r in ref_hold)
            d1 = max(abs(now[i] - recd[i]) for i in recd)
            if not d1 <= CANARY_TOL:
                raise SystemExit(f"R5: MEASUREMENT CANARY (tier 1) FAILED for {name}: max |diff| {d1}")
            t2_nats[name] = {i: new[i][v]["nats"] for i in recd}
            d2 = max(abs(t2_nats[name][i] - recd[i]) for i in recd)
            if not d2 <= TIER2_TOL:
                raise SystemExit(f"R5b: the B=1 path disagrees with the recorded nats for {name}: max |diff| {d2}")
            tier1[name], tier2[name] = d1, d2
            hostbudget.guard(guard_gib, f"ABST canary {name}")
        rd = {i: av.log_ratio(canary_cells["D_plain"][2][i], canary_cells["D_lc"][2][i]) for i in canary_cells["D_lc"][2]}
        nd = {i: av.log_ratio(t2_nats["D_plain"][i], t2_nats["D_lc"][i]) for i in rd}
        sign_disagree = sum(1 for i in rd if av.sgn(rd[i]) != av.sgn(nd[i]))
        if down_proj_fingerprint() != w_at_load:
            raise SystemExit("W: a down_proj weight differs from its state at load -- an LC fold was not restored")

        # ---------------- R6 measured throughput projection (generation priced as prefill + 4 decode
        # steps at 64 token-equivalents each: an upper bound on this card for B = 1)
        per_tok = t_tier2 / max(tok_count, 1)
        panel_tok = sum(len(V[v].encode_old_ids(r["prompt_text"])) + 2 for r in panel for v in V) * 5
        gen_tok = sum(len(V[v].encode_old_ids(r["prompt_text"])) + 4 * 64 for r in panel for _, _, v in GEN_CELLS)
        projected_min = (panel_tok + gen_tok) * per_tok / 60
        if projected_min > a.runtime_cap_min:
            raise SystemExit(f"R6: projected {projected_min:.1f} min > cap {a.runtime_cap_min}")
        controls = {"fit_canary": fit_canary, "tier1_max_abs_diff": tier1, "tier2_max_abs_diff": tier2,
                    "tier2_sign_disagreements_reported": sign_disagree, "projected_minutes": projected_min,
                    "panel_forward_tokens": panel_tok, "generation_token_equivalents": gen_tok,
                    "structure_id_check": so, "down_proj_restored": True, "mode": mode,
                    "stale_lock_replaced": str(lock.stale) if lock.stale else None, "launch_id": launch_id}
        partial["refusal_controls"] = controls

        # ---------------- the boundary: re-check the budget and the RAM, then write the marker
        markers = budget_check(att_dir, primary_root, "just before the boundary")
        if code_hashes() != code_at_start:
            raise SystemExit("REFUSE (just before the boundary): a loaded module's file changed since import")
        controls["prior_crossings"] = prior_crossings(att_dir, markers) if markers else None
        controls["module_origins"] = origins
        hostbudget.require(PRE_BOUNDARY_RAM_GIB[mode], "ABST-001 boundary")
        att_dir.mkdir(parents=True, exist_ok=True)
        write_fsync(att_dir / f"BOUNDARY_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{os.getpid()}.json",
                    {"record": "ABST_BOUNDARY", "pid": os.getpid(), "epoch": time.time(), "launch_id": launch_id,
                     "refusal_controls": controls,
                     "panel_sha256": sha(a.abst_panel), "code_sha256": code_at_start})
        panel_read = True
        print(f"[ABST] refusal controls passed (R1-R6, W; projected {projected_min:.1f} min); the panel is read now", flush=True)

        # ---------------- measurement (PRIMARY first)
        records = {r["id"]: {"stratum": r["stratum"], "family": r["family"], "tier": r.get("tier"),
                             "reference": r["reference"], "m": {}, "gen": {},
                             "cue_not_recorded": "not recorded" in r["prompt_text"].lower(),
                             "input_equal_V0": {v: V[v].encode_old_ids(r["prompt_text"]) == V["V0"].encode_old_ids(r["prompt_text"])
                                                for v in ("V4", "V2")}} for r in panel}
        partial["records"] = records

        def run_cfg(cfg_name, v, wait=True):
            kept, f = configured(cfg_name)
            ctx = lc.folded(model, f) if f else _null()
            sink = {}
            partial["in_progress"] = {"cell": f"{cfg_name}/{v}", "rows": sink}   # rows land here as measured
            with ctx:
                res = am.measure(model, panel, kept, V[v], SUP, im_end, ans, dev, top5_for=v, sink=sink)
            for i, rec in res.items():
                records[i]["m"].setdefault(cfg_name, {})[v] = rec
            partial["in_progress"] = None
            if f and down_proj_fingerprint() != w_at_load:
                raise SystemExit(f"W: down_proj not restored after {cfg_name}/{v}")
            if wait:
                wait_for_ram(guard_gib, f"ABST {cfg_name}/{v}")

        for k, (cfg_name, v) in enumerate(PRIMARY_ORDER):
            run_cfg(cfg_name, v, wait=k < len(PRIMARY_ORDER) - 1)     # no wait between the last primary cell and the write
        unk = [r["id"] for r in panel if r["stratum"] == "unk"]
        known_ids = [r["id"] for r in panel if r["stratum"] == "known"]
        primary = av.verdict(cells_for_verdict(records, unk, known_ids))
        tmp_out.mkdir(parents=True, exist_ok=False)
        prim = {"schema": 1, "record": "ABST_PRIMARY", "experiment": "ABST-001", "steps_1_2": {k: primary[k] for k in ("H1", "H2")},
                "panel_sha256": sha(a.abst_panel), "written_before_any_other_cell": True,
                "per_row_A_T4": {i: {"stratum": records[i]["stratum"],
                                     "A_plain": records[i]["m"]["S20_plain"]["V4"]["V4"]["nats"],
                                     "A_lc": records[i]["m"]["S20_lc"]["V4"]["V4"]["nats"],
                                     "T4": records[i]["m"]["T"]["V4"]["V4"]["nats"]} for i in unk + known_ids}}
        write_fsync(tmp_out / "ABST_PRIMARY.json.tmp", prim)
        os.replace(tmp_out / "ABST_PRIMARY.json.tmp", tmp_out / "ABST_PRIMARY.json")
        os.replace(tmp_out, out)
        print(f"[ABST] PRIMARY step 1: {primary['H1']['label']} ({primary['H1']['negative']} worse / "
              f"{primary['H1']['positive']} better) | step 2: {primary['H2']['label']}", flush=True)
        wait_for_ram(guard_gib, "ABST after the primary")
        for j, (cfg_name, v) in enumerate(REST_ORDER):
            run_cfg(cfg_name, v, wait=j < len(REST_ORDER) - 1)       # no wait between the last cell and ABST_H3
        h3_v = av.verdict(cells_for_verdict(records, unk, known_ids))
        write_fsync(out / "ABST_H3.json", {"schema": 1, "record": "ABST_H3", "experiment": "ABST-001",
                                          "verdict": h3_v, "written_before_generation": True})
        for j, (cell, cfg_name, v) in enumerate(GEN_CELLS):
            kept, f = configured(cfg_name)
            ctx = lc.folded(model, f) if f else _null()
            with ctx:
                g = vs.generate_rows(model, V[v], panel, kept, eos, eot, dev, 4, batch=1)
            for x in g:
                records[x["id"]]["gen"][cell] = x["text"]
            if f and down_proj_fingerprint() != w_at_load:
                raise SystemExit(f"W: down_proj not restored after generation {cell}")
            if j < len(GEN_CELLS) - 1:                                  # no wait between the last cell and ABST_RESULT
                wait_for_ram(guard_gib, f"ABST generation {cell}")
        full_v = h3_v
        if {k: full_v[k] for k in ("H1", "H2")} != prim["steps_1_2"]:
            raise SystemExit("steps 1-2 changed between ABST_PRIMARY and the full verdict")
        result = {"schema": 1, "record": "ABST_RESULT", "experiment": "ABST-001", "revision": "R1",
                  "REGISTERED": {"H1": full_v["H1"]["label"], "H2": full_v["H2"]["label"], "H3": full_v["H3"]["label"]},
                  "verdict": full_v, "descriptive": ad.summarise(records),
                  "refusal_controls": controls, "earlier_boundary_markers": [p.name for p in markers],
                  "fits": fits, "answer_token_ids": ans, "kept": sets,
                  "teacher_weight_sha256": rls.sha256_file(a.teacher / "model.safetensors"), "module_origins": origins,
                  "panel_sha256": sha(a.abst_panel), "manifest_sha256": sha(a.manifest),
                  "runtime": {"flags": rt, "python": sys.version.split()[0], "torch": torch.__version__,
                              "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
                              "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                              "seconds": round(time.time() - t_start, 1), "code_sha256": code_at_start,
                              "code_unchanged_at_end": code_hashes() == code_at_start},
                  "records": records, "quality_claim": False, "release_permitted": False}
        la.dump(out / "ABST_RESULT.json", result)
        print(f"[ABST] REGISTERED: H1 {full_v['H1']['label']} | H2 {full_v['H2']['label']} | H3 {full_v['H3']['label']}", flush=True)
        return 0
    except BaseException as e:
        if panel_read:
            import traceback
            rec = {"schema": 1, "record": "ABST_ATTEMPT", "experiment": "ABST-001", "launch_id": launch_id,
                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pid": os.getpid(),
                   "exception_type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc(),
                   "classification_hint": "infrastructure" if is_infrastructure(e) else "deterministic or unknown",
                   "primary_written": (out / "ABST_PRIMARY.json").exists() or (tmp_out / "ABST_PRIMARY.json").exists(),
                   "partial": partial, "code_sha256": code_at_start}
            att_dir.mkdir(parents=True, exist_ok=True)
            fn = att_dir / f"ATTEMPT_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{os.getpid()}.json"
            with open(fn, "x", encoding="utf-8", newline="\n") as fh:
                json.dump(rec, fh, ensure_ascii=False, default=str)
                fh.flush()
                os.fsync(fh.fileno())
            print(f"[ABST] attempt record written: {fn}", flush=True)
        raise
    finally:
        try:
            if token:
                gpu_lease.release(token)
        finally:
            lock.release()


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
