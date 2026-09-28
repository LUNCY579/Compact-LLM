"""ABST-001 dry harness (ABST001_PREREG.md R1; cold review M2, M3, M7; R1 review round). NOT measurement code and
not gated: it drives run_abst.main() on the tiny random CPU model (--dry-run) and checks the refusal and read-once
rules scenario by scenario. Every scenario has its own parent directory under --work and is isolated: a crash
inside one is recorded as that scenario's FAIL (with the traceback) and the rest still run. Nothing may be
written under the records tree: the runner refuses dry runs there, and a before/after snapshot of the panel's
records directory is compared (entries the harness could create: result/, attempts/, *.partial_*, ABST_*,
BOUNDARY_*, ATTEMPT_*, RUN.lock*).

Injection points (abst_measure.measure call numbers in one run): 1-5 = the pre-boundary tier-2 canary;
6, 7, 8 = the PRIMARY cells S20_plain/V4, S20_lc/V4, T/V4; ABST_PRIMARY.json is written after call 8;
9 = the first cell after the primary (S20_plain/V2).

  G   output keying: dry --out under the records tree / the panel dir / a new dir there / the records root, and a
      dry run with a COPIED panel into the real result dir -> refused; a real run with a copied panel -> refused
  NC  pre-boundary negative controls on the R1 code (registered in §5): a recorded LC fit residual +1e-5 (R4), a
      recorded S6 nats +1e-5 (R5 tier 1), a wrong kept set (R4) -> refused, no marker, nothing written
  E2E a clean end-to-end run: the primary equals the final steps 1-2; every row has all 15 config x vocabulary
      cells and 4 greedy cells; ABST_H3, the descriptive block (tracking, cue, digit anchor) and prior-crossing
      fields present
  A   attempt records: failure on call 6 (non-infrastructure, no primary), mid-cell after 1 row of call 7 (that
      row kept), on call 9 (primary with UNKNOWN and known rows stands)
  S1-S5 deterministic -> rerun refused; infrastructure -> one rerun completes -> sibling refused; clean -> same /
      sibling refused; two recorded failures -> third refused; infrastructure after the primary -> final, and
      S5b: with the result moved OUT of the tree, the attempt record's own primary flag still refuses
  HARD KILLS (soup review), approximated: an injected failure, then its attempt record removed, the marker's pid
  rewritten to a dead pid and a RUN.lock left holding that dead pid and the marker's epoch -- the files a reboot
  leaves (the rules read files; the in-memory process state of a reboot is not simulated)
  S6  two hard kills -> the third completes; both listed as hard kills; the restart query returned code 0
  S7  three hard kills -> the fourth refused on the cap (3)
  S8  one recorded failure + one hard kill -> the third completes; prior_crossings reads [infrastructure, NONE]
  L1  RUN.lock held by a live pid -> refused, lock untouched       L2  dead pid -> set aside, recorded
  L3  RUN.lock pid REUSED (a live process created after the lock's epoch) -> set aside, completes
  P1  the marker's pid rewritten to a live process started before the marker's epoch -> refused; P1b it ends ->
      completes        P2  a live process created AFTER the marker's epoch (pid reuse) -> does not block
  C1  the marker's code hashes altered -> refused; C1b restored -> completes
  K1  a loaded module's file reported changed between import and the boundary -> refused before the marker
  O1  a gated module reported as loaded from the wrong directory (shadowing) -> refused at start-up
  M3a/M3c a primary appearing under the parent (depth 1 / depth 3) during the canary -> refused just before the
      boundary, no marker       M3b a live concurrent crossing appearing during the canary -> refused
  B1  the pre-boundary RAM check refuses (the real SystemExit form) -> no marker, no record; B1b relaunch completes
  T1  a truncated attempt record -> the next launch refuses (raises) before any crossing
  W1  post-boundary dip waits and completes   W2 dip past the bound -> infrastructure record, no primary
  W3  dip right after the last primary cell -> the primary is still written first
  I   classification (no launch): MemoryError, 'not enough memory', OutOfMemoryError, TDR, LowMemoryAbort are
      infrastructure; KeyboardInterrupt, a device-side assert, SystemExit are not

usage: python -B design/abst_harness.py --fixtures DIR --records DIR --rtg1-sft DIR --work NEWDIR --log NEWFILE
  --fixtures holds tiny_teacher/, dry_20260925T035243Z/{study,lc/LC_ARMS.json}, dry_vocab_V2/,
  dry_hlcc_for_abst/result/HLCC_RESULT.json (the tiny dry artifacts; every file hashed into the log)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG.parents[1] / "experiments" / "rab001_exploration"))
import abst_measure as am  # noqa: E402
import hostbudget  # noqa: E402
import run_abst  # noqa: E402

ORIG_MEASURE = am.measure
ORIG_HB = run_abst.hostbudget
ORIG_WAIT = (run_abst.POST_BOUNDARY_WAIT_S, run_abst.POST_BOUNDARY_POLL_S)
ORIG_CODE_HASHES = run_abst.code_hashes
HARNESS_CAN_CREATE = ("result", "attempts", ".partial_", "ABST_", "BOUNDARY_", "ATTEMPT_", "RUN.lock")


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*")}


class BombRows(list):
    """Iterating past `after` rows raises: a failure in the middle of a cell."""

    def __init__(self, rows, after):
        super().__init__(rows)
        self.after = after

    def __iter__(self):
        for j, r in enumerate(list.__iter__(self)):
            if j == self.after:
                raise SystemExit("injected mid-cell failure")
            yield r


class HB:
    """Stand-in for run_abst's hostbudget: free_gib() can be scripted (it is read only by wait_for_ram), and
    require() can refuse one label the way the real one does (SystemExit). Everything else is the real module."""

    def __init__(self, free=None, refuse_label=None):
        self._free, self._refuse, self.low_polls = free, refuse_label, 0

    def __getattr__(self, k):
        return getattr(hostbudget, k)

    def free_gib(self):
        if self._free is None:
            return hostbudget.free_gib()
        v = self._free()
        self.low_polls += v < 1.0
        return v

    def require(self, gib, label="job"):
        if self._refuse is not None and label == self._refuse:
            raise SystemExit(f"REFUSING TO START {label}: needs {gib:.1f} GiB free physical memory (injected)")
        return hostbudget.require(gib, label)


def seq(values, then):
    it = iter(values)
    return lambda: next(it, then)


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


class Harness:
    def __init__(self, a):
        F, R = a.fixtures, a.records
        self.F, self.R = F, R
        self.D = F / "dry_20260925T035243Z"
        self.panel_root = R / "abst001-20260925"
        self.opts = {"--teacher": str(F / "tiny_teacher"),
                     "--panels": str(R / "smallvd001-20260924/calibration/LAYER_PANELS.jsonl"),
                     "--study": str(self.D / "study"), "--smallvd-lc": str(self.D / "lc/LC_ARMS.json"),
                     "--hlcc-result": str(F / "dry_hlcc_for_abst/result/HLCC_RESULT.json"),
                     "--abst-panel": str(self.panel_root / "panel/ABST_PANEL.jsonl"),
                     "--manifest": str(self.panel_root / "panel/ABST_MANIFEST.json"),
                     "--confirm-panel": str(R / "hlcc001-20260925/panel/CONFIRM_PANEL.jsonl"),
                     "--tokenizer-corpus": str(R / "smallvd001-20260924/calibration/TOKENIZER_CORPUS.jsonl"),
                     "--rtg1-sft": str(a.rtg1_sft),
                     "--vocab-v4": str(F / "dry_vocab_V2"), "--vocab-v2": str(F / "dry_vocab_V2"),
                     "--s20-kept": "0,1,3,4", "--s18-kept": "0,1,3", "--device": "cpu",
                     "--max-rows-per-stratum": "3", "--max-rows-per-category": "3"}
        files = sorted([*(F / "tiny_teacher").rglob("*"), *(F / "dry_vocab_V2").rglob("*"),
                        self.D / "study/S1_teacher_outputs.json", self.D / "study/S6_vocab_frontier.json",
                        self.D / "lc/LC_ARMS.json", F / "dry_hlcc_for_abst/result/HLCC_RESULT.json"])
        self.fixture_sha = {str(p.relative_to(F)).replace("\\", "/"): sha_file(p) for p in files if p.is_file()}
        self.work = a.work
        self.results = {}

    # ------------------------------------------------------------------ one launch
    def launch(self, out: Path, dry=True, fail_on=None, kind="det", hook=None, hb=None, wait=None, over=None):
        calls = {"n": 0}

        def measure(model, rows, *x, **k):
            calls["n"] += 1
            n = calls["n"]
            if hook:
                hook(n)
            if fail_on == n:
                if kind == "det":
                    raise SystemExit("injected deterministic failure")
                if kind == "infra":
                    raise hostbudget.LowMemoryAbort("injected stand-down")
                if kind == "midcell":
                    return ORIG_MEASURE(model, BombRows(rows, 1), *x, **k)
            return ORIG_MEASURE(model, rows, *x, **k)

        am.measure = measure
        if hb is not None:
            run_abst.hostbudget = hb
        if wait is not None:
            run_abst.POST_BOUNDARY_WAIT_S, run_abst.POST_BOUNDARY_POLL_S = wait
        opts = dict(self.opts, **(over or {}))
        argv = [x for kv in opts.items() for x in kv] + ["--out", str(out)] + (["--dry-run"] if dry else [])
        try:
            run_abst.main(argv)
            outcome = "completed"
        except BaseException as e:  # noqa: BLE001
            outcome = f"{type(e).__name__}: {e}"
        finally:
            am.measure = ORIG_MEASURE
            run_abst.hostbudget = ORIG_HB
            run_abst.POST_BOUNDARY_WAIT_S, run_abst.POST_BOUNDARY_POLL_S = ORIG_WAIT
            run_abst.code_hashes = ORIG_CODE_HASHES
        return {"outcome": outcome, "measure_calls": calls["n"]}

    @staticmethod
    def state(parent: Path, out_name="result"):
        att = parent / "attempts"
        recs = [jload(p) for p in sorted(att.glob("ATTEMPT_*.json"))] if att.exists() else []
        return {"boundary_markers": len(list(att.glob("BOUNDARY_*.json"))) if att.exists() else 0,
                "attempts": [{"hint": r["classification_hint"], "primary_written": r["primary_written"],
                              "exception_type": r["exception_type"], "launch_id": r.get("launch_id")} for r in recs],
                "primary": (parent / out_name / "ABST_PRIMARY.json").exists(),
                "h3": (parent / out_name / "ABST_H3.json").exists(),
                "result": (parent / out_name / "ABST_RESULT.json").exists(),
                "lock_left": (att / "RUN.lock").exists(),
                "partial_dirs": sorted(x.name for x in parent.glob("*.partial_*"))}

    @staticmethod
    def attempt(parent: Path, k=0):
        return jload(sorted((parent / "attempts").glob("ATTEMPT_*.json"))[k])

    def run_scenario(self, name, fn):
        try:
            steps, checks = fn()
        except BaseException as e:  # noqa: BLE001
            steps, checks = {"crash": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}, {"scenario ran": False}
        ok = bool(checks) and all(checks.values())
        self.results[name] = {"pass": ok, "checks": checks, "steps": steps}
        print(f"[harness] {name}: {'PASS' if ok else 'FAIL'} {[k for k, v in checks.items() if not v]}", flush=True)

    @staticmethod
    def live_process():
        return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])

    @staticmethod
    def dead_pid():
        for p in range(4_000_000, 3_000_000, -4):
            if not run_abst.pid_alive(p):
                return p
        raise SystemExit("no dead pid found")

    @staticmethod
    def edit_marker(parent: Path, fn, which=-1):
        mk = sorted((parent / "attempts").glob("BOUNDARY_*.json"))[which]
        d = jload(mk)
        old = json.loads(json.dumps(d))
        fn(d)
        mk.write_text(json.dumps(d, indent=1), encoding="utf-8", newline="\n")
        return mk, old

    def hard_kill(self, parent: Path):
        """An injected failure after the boundary, then the files a reboot leaves: no attempt record, the marker's
        pid dead, and a RUN.lock still holding that dead pid and the marker's epoch."""
        before = set((parent / "attempts").glob("ATTEMPT_*.json")) if (parent / "attempts").exists() else set()
        r = self.launch(parent / "result", fail_on=6, kind="infra")
        new = set((parent / "attempts").glob("ATTEMPT_*.json")) - before
        if len(new) != 1:
            raise RuntimeError(f"hard_kill: expected one new attempt record, got {len(new)}: {r}")
        new.pop().unlink()
        dead = self.dead_pid()
        mk, _ = self.edit_marker(parent, lambda d: d.update(pid=dead))
        (parent / "attempts" / "RUN.lock").write_text(json.dumps({"pid": dead, "epoch": jload(mk)["epoch"]}), encoding="utf-8")
        return r

    # ------------------------------------------------------------------ scenarios
    def run_all(self):
        W, PR = self.work, self.panel_root
        R = self.R
        refused = lambda st, s: st["outcome"].startswith("SystemExit") and s in st["outcome"]
        DRYMSG = "--dry-run may never write under the records tree or the panel's directory"

        def G():
            copy = W / "g_copy" / "abst001-20260925"
            (copy / "panel").mkdir(parents=True)
            for n in ("ABST_PANEL.jsonl", "ABST_MANIFEST.json"):
                shutil.copy2(PR / "panel" / n, copy / "panel" / n)
            probe = PR / "harness_probe_must_not_exist"
            probe2 = R / "harness_probe_must_not_exist"
            copied = {"--abst-panel": str(copy / "panel/ABST_PANEL.jsonl"), "--manifest": str(copy / "panel/ABST_MANIFEST.json")}
            st = {"G1_dry_into_result": self.launch(PR / "result"), "G1b_dry_new_dir_in_panel_dir": self.launch(probe / "result"),
                  "G1c_dry_elsewhere_in_records": self.launch(probe2 / "result"), "G2_dry_panel_dir": self.launch(PR),
                  "G1d_dry_copied_panel_into_real_result": self.launch(PR / "result", over=copied),
                  "G3_real_other_out": self.launch(W / "g3" / "result", dry=False),
                  "G4_real_copied_panel": self.launch(copy / "result", dry=False, over=copied)}
            ck = {f"{k} refused": refused(v, DRYMSG) for k, v in st.items() if k.startswith(("G1", "G2"))}
            ck["G3 refused"] = refused(st["G3_real_other_out"], "a real run reads only the registered panel")
            ck["G4 refused"] = refused(st["G4_real_copied_panel"], "a real run reads only the registered panel")
            ck["nothing under records"] = not (PR / "result").exists() and not (PR / "attempts").exists() \
                and not probe.exists() and not probe2.exists()
            ck["nothing next to the copied panel"] = sorted(x.name for x in copy.iterdir()) == ["panel"]
            return st, ck

        def NC():
            nc = W / "nc"
            nc.mkdir(parents=True)
            lca = jload(Path(self.opts["--smallvd-lc"]))
            f0 = next(f for f in lca["arms"]["selected"]["fits"] if f.get("compensated"))
            f0["fit_residual_rel_fitted"] += 1e-5
            (nc / "LC_ARMS_tampered.json").write_text(json.dumps(lca), encoding="utf-8")
            shutil.copytree(Path(self.opts["--study"]), nc / "study_tampered")
            s6p = nc / "study_tampered" / "S6_vocab_frontier.json"
            s6 = jload(s6p)
            key = f"{Path(self.opts['--vocab-v4']).name}/4L"          # the A_plain canary's recorded config (V4 name / 20-block stand-in)
            s6["configs"][key]["answer_bpb"]["holdout_per_item"][0]["nats"] += 1e-5
            s6p.write_text(json.dumps(s6), encoding="utf-8")
            st = {"NC1_fit_residual": self.launch(nc / "a" / "result", over={"--smallvd-lc": str(nc / "LC_ARMS_tampered.json")}),
                  "NC2_recorded_S6_nats": self.launch(nc / "b" / "result", over={"--study": str(nc / "study_tampered")}),
                  "NC3_wrong_kept_set": self.launch(nc / "c" / "result", over={"--s18-kept": "0,1,4"})}
            ck = {"NC1 refused by R4": refused(st["NC1_fit_residual"], "R4: FIT CANARY FAILED for S18"),
                  "NC2 refused by R5": refused(st["NC2_recorded_S6_nats"], "R5: MEASUREMENT CANARY (tier 1) FAILED for A_plain"),
                  "NC3 refused by R4": refused(st["NC3_wrong_kept_set"], "R4: the recorded fits are for different kept sets"),
                  "no marker or output in any": all(self.state(nc / x)["boundary_markers"] == 0 and not (nc / x / "result").exists()
                                                    for x in ("a", "b", "c")),
                  "S6 key tampered is the A_plain canary": key in jload(Path(self.opts["--study"]) / "S6_vocab_frontier.json")["configs"]}
            return st, ck

        def E2E():
            p = W / "e2e"
            s = self.launch(p / "result")
            st = self.state(p)
            res = jload(p / "result" / "ABST_RESULT.json") if st["result"] else {}
            prim = jload(p / "result" / "ABST_PRIMARY.json") if st["primary"] else {}
            h3 = jload(p / "result" / "ABST_H3.json") if st["h3"] else {}
            recs = res.get("records", {})
            cells = {(c, v) for r in recs.values() for c, vv in r["m"].items() for v in vv}
            full = all(len({(c, v) for c, vv in r["m"].items() for v in vv}) == 15 and len(r["m"][c][v]) == 3
                       for r in recs.values() for c in r["m"] for v in r["m"][c]) if recs else False
            desc = res.get("descriptive", {})
            return {"run": s, "state": st, "cells": sorted(map(list, cells)), "REGISTERED": res.get("REGISTERED")}, {
                "completed": s["outcome"] == "completed" and st["result"],
                "primary equals final steps 1-2": prim.get("steps_1_2") == {k: res["verdict"][k] for k in ("H1", "H2")} if res else False,
                "H3 file equals final verdict": h3.get("verdict") == res.get("verdict"),
                "24 rows, each with 15 config x input cells read under 3 supports": len(recs) == 24 and full,
                "4 greedy cells per row": all(len(r["gen"]) == 4 for r in recs.values()) and bool(recs),
                "descriptive has tracking, cue, digit anchor": all(k in desc for k in ("A_tracking", "cue_not_recorded", "A_known_digit_anchor")),
                "tracking is within family": all(desc.get("A_tracking", {}).get(s_, {}).get("within_family") for s_ in ("unk", "known")),
                "structure check with both controls": res.get("refusal_controls", {}).get("structure_id_check", {}).get("positive_controls_caught") == {"used": True, "train": True},
                "module origins recorded": bool(res.get("module_origins")),
                "code unchanged at end": res.get("runtime", {}).get("code_unchanged_at_end") is True,
                "no partial dir left": st["partial_dirs"] == [], "lock released": not st["lock_left"]}

        def A6():
            p = W / "a6"
            s = self.launch(p / "result", fail_on=6)
            st, rec = self.state(p), self.attempt(p)
            return {"run": s, "state": st}, {
                "failed with the injection": "injected deterministic failure" in s["outcome"],
                "one boundary, one non-infrastructure attempt": st["boundary_markers"] == 1 and [x["hint"] for x in st["attempts"]] == ["deterministic or unknown"],
                "attempt carries the launch_id of its marker": rec.get("launch_id") == jload(sorted((p / "attempts").glob("BOUNDARY_*.json"))[0]).get("launch_id"),
                "in_progress names S20_plain/V4": (rec["partial"].get("in_progress") or {}).get("cell") == "S20_plain/V4",
                "lock released": not st["lock_left"]}

        def A7m():
            p = W / "a7m"
            s = self.launch(p / "result", fail_on=7, kind="midcell")
            st, rec = self.state(p), self.attempt(p)
            ip = rec["partial"].get("in_progress") or {}
            done = sorted({c for v in rec["partial"]["records"].values() for c in v["m"]})
            return {"run": s, "state": st, "in_progress_rows": len(ip.get("rows", {}))}, {
                "failed mid-cell": "injected mid-cell failure" in s["outcome"], "in_progress is S20_lc/V4": ip.get("cell") == "S20_lc/V4",
                "exactly one finished row kept": len(ip.get("rows", {})) == 1, "completed cells = S20_plain only": done == ["S20_plain"]}

        def A9():
            p = W / "a9"
            s = self.launch(p / "result", fail_on=9)
            st, rec = self.state(p), self.attempt(p)
            prim = jload(p / "result" / "ABST_PRIMARY.json") if st["primary"] else {}
            done = sorted({c for v in rec["partial"]["records"].values() for c in v["m"]})
            return {"run": s, "state": st}, {
                "failed with the injection": "injected deterministic failure" in s["outcome"],
                "primary written, recorded as written": st["primary"] and st["attempts"][0]["primary_written"] is True,
                "primary holds UNKNOWN and known rows": sorted({v["stratum"] for v in prim.get("per_row_A_T4", {}).values()}) == ["known", "unk"],
                "primary has H1 and H2": sorted(prim.get("steps_1_2", {})) == ["H1", "H2"],
                "partial holds the three primary configs": done == ["S20_lc", "S20_plain", "T"],
                "no H3, no result, no partial dir": not st["h3"] and not st["result"] and st["partial_dirs"] == []}

        def S1():
            p = W / "s1"
            a, b = self.launch(p / "result", fail_on=6), self.launch(p / "result")
            return {"first": a, "rerun": b}, {"rerun refused": refused(b, "non-infrastructure failure"),
                                              "still one boundary": self.state(p)["boundary_markers"] == 1}

        def S2():
            p = W / "s2"
            a, b, c = self.launch(p / "result", fail_on=6, kind="infra"), self.launch(p / "result"), self.launch(p / "result_again")
            st = self.state(p)
            res = jload(p / "result" / "ABST_RESULT.json") if st["result"] else {}
            return {"first": a, "rerun": b, "third": c, "state": st}, {
                "first is infrastructure": [x["hint"] for x in st["attempts"]] == ["infrastructure"],
                "rerun completed": b["outcome"] == "completed" and st["result"],
                "result names the earlier crossing": len(res.get("earlier_boundary_markers", [])) == 1,
                "prior_crossings reads [infrastructure]": [x["attempt_record"] for x in ((res.get("refusal_controls") or {}).get("prior_crossings") or {}).get("crossings", [])] == ["infrastructure"],
                "third refused (primary exists)": refused(c, "a primary already exists"), "two boundaries": st["boundary_markers"] == 2}

        def S3():
            p = W / "s3"
            a, b, c = self.launch(p / "result"), self.launch(p / "result"), self.launch(p / "other")
            return {"first": a, "same_out": b, "sibling_out": c}, {
                "first completed": a["outcome"] == "completed", "same --out refused": refused(b, "exists (create-only)"),
                "sibling --out refused": refused(c, "a primary already exists")}

        def S4():
            p = W / "s4"
            a, b, c = (self.launch(p / "result", fail_on=6, kind="infra"), self.launch(p / "result", fail_on=6, kind="infra"),
                       self.launch(p / "result"))
            st = self.state(p)
            return {"first": a, "rerun": b, "third": c, "state": st}, {
                "rerun reached the boundary and failed": "injected stand-down" in b["outcome"],
                "two boundaries, two infrastructure attempts": st["boundary_markers"] == 2 and [x["hint"] for x in st["attempts"]] == ["infrastructure"] * 2,
                "third refused by the recorded-failures rule": refused(c, "2 earlier crossings ended in recorded failures"),
                "no primary": not st["primary"]}

        def S5():
            p = W / "s5"
            a, b, c = self.launch(p / "result", fail_on=9, kind="infra"), self.launch(p / "result"), self.launch(p / "result_again")
            st = self.state(p)
            deep = p / "archive" / "r1"
            deep.mkdir(parents=True)
            shutil.move(str(p / "result"), str(deep / "result"))
            d = self.launch(p / "result")                           # primary moved deeper in the tree (rglob)
            shutil.move(str(p / "archive"), str(W / "s5_outside_archive"))
            e = self.launch(p / "result")                           # primary moved OUT of the tree: the record's flag
            return {"first": a, "rerun_same": b, "rerun_sibling": c, "moved_deeper": d, "moved_out": e, "state": st}, {
                "primary written, infrastructure attempt": st["primary"] and [x["hint"] for x in st["attempts"]] == ["infrastructure"],
                "rerun into the same --out refused": refused(b, "exists (create-only)"),
                "rerun into a sibling refused": refused(c, "a primary already exists"),
                "primary moved deeper still refuses": refused(d, "a primary already exists"),
                "primary moved out: the attempt record's flag refuses": refused(e, "an earlier attempt record says the primary was written"),
                "one boundary": self.state(p)["boundary_markers"] == 1}

        def S6():
            p = W / "s6"
            k = [self.hard_kill(p), self.hard_kill(p)]
            s = self.launch(p / "result")
            st = self.state(p)
            res = jload(p / "result" / "ABST_RESULT.json") if st["result"] else {}
            pc = (res.get("refusal_controls") or {}).get("prior_crossings") or {}
            ev = pc.get("restart_events") or {}
            return {"kills": k, "third": s, "state": st, "prior_crossings": pc}, {
                "third completed": s["outcome"] == "completed" and st["result"],
                "stale lock set aside": bool((res.get("refusal_controls") or {}).get("stale_lock_replaced")),
                "both earlier crossings listed as hard kills": [c["attempt_record"][:4] for c in pc.get("crossings", [])] == ["NONE", "NONE"],
                "restart query ran with returncode 0": ev.get("queried") is True and ev.get("returncode") == 0 and "error" not in ev,
                "three boundaries": st["boundary_markers"] == 3}

        def S7():
            p = W / "s7"
            k = [self.hard_kill(p) for _ in range(3)]
            s = self.launch(p / "result")
            return {"kills": k, "fourth": s}, {"fourth refused on the cap": refused(s, "3 earlier boundary crossings (cap 3)"),
                                               "no primary": not self.state(p)["primary"]}

        def S8():
            p = W / "s8"
            a = self.launch(p / "result", fail_on=6, kind="infra")
            k = self.hard_kill(p)
            s = self.launch(p / "result")
            st = self.state(p)
            res = jload(p / "result" / "ABST_RESULT.json") if st["result"] else {}
            pc = ((res.get("refusal_controls") or {}).get("prior_crossings") or {}).get("crossings", [])
            return {"recorded": a, "kill": k, "third": s, "prior_crossings": pc}, {
                "third completed": s["outcome"] == "completed" and st["result"],
                "prior_crossings reads [infrastructure, NONE]": [c["attempt_record"][:14] for c in pc] == ["infrastructure", "NONE (hard kil"]}

        def L1():
            p = W / "l1"
            (p / "attempts").mkdir(parents=True)
            lk = p / "attempts" / "RUN.lock"
            lk.write_text(json.dumps({"pid": os.getpid(), "epoch": time.time()}), encoding="utf-8")
            before = lk.read_bytes()
            s = self.launch(p / "result")
            return {"run": s}, {"refused": refused(s, "another ABST-001 launch holds"), "lock untouched": lk.read_bytes() == before,
                                "no boundary": self.state(p)["boundary_markers"] == 0}

        def L2():
            p = W / "l2"
            (p / "attempts").mkdir(parents=True)
            dead = self.dead_pid()
            (p / "attempts" / "RUN.lock").write_text(json.dumps({"pid": dead}), encoding="utf-8")
            s = self.launch(p / "result")
            st = self.state(p)
            res = jload(p / "result" / "ABST_RESULT.json") if st["result"] else {}
            return {"run": s, "dead_pid": dead}, {
                "completed": s["outcome"] == "completed", "stale lock set aside": len(list((p / "attempts").glob("RUN.lock.stale_*"))) == 1,
                "recorded in refusal_controls": bool(res.get("refusal_controls", {}).get("stale_lock_replaced")), "own lock released": not st["lock_left"]}

        def L3():
            p = W / "l3"
            (p / "attempts").mkdir(parents=True)
            epoch = time.time() - 60
            child = self.live_process()
            try:
                (p / "attempts" / "RUN.lock").write_text(json.dumps({"pid": child.pid, "epoch": epoch}), encoding="utf-8")
                s = self.launch(p / "result")
            finally:
                child.kill()
                child.wait(timeout=60)
            return {"run": s, "child_pid": child.pid}, {
                "a reused pid does not hold the lock": s["outcome"] == "completed",
                "stale lock set aside": len(list((p / "attempts").glob("RUN.lock.stale_*"))) == 1}

        def P1():
            p = W / "p1"
            self.launch(p / "result", fail_on=6, kind="infra")
            child = self.live_process()
            try:
                time.sleep(1.0)
                self.edit_marker(p, lambda d: d.update(pid=child.pid, epoch=time.time()))
                a = self.launch(p / "result")
            finally:
                child.kill()
                child.wait(timeout=60)
            b = self.launch(p / "result")
            return {"refused_run": a, "after_exit": b}, {"refused while alive": refused(a, "is still alive"),
                                                        "completes once it ends": b["outcome"] == "completed"}

        def P2():
            p = W / "p2"
            self.launch(p / "result", fail_on=6, kind="infra")
            time.sleep(3.0)
            child = self.live_process()
            try:
                self.edit_marker(p, lambda d: d.update(pid=child.pid))
                a = self.launch(p / "result")
            finally:
                child.kill()
                child.wait(timeout=60)
            return {"run": a}, {"not mistaken for the recorded process": a["outcome"] == "completed"}

        def C1():
            p = W / "c1"
            self.launch(p / "result", fail_on=6, kind="infra")
            key = {}
            path, old = self.edit_marker(p, lambda d: key.update(k=next(iter(d["code_sha256"]))) or d["code_sha256"].__setitem__(key["k"], "0" * 64))
            a = self.launch(p / "result")
            path.write_text(json.dumps(old, indent=1), encoding="utf-8", newline="\n")
            b = self.launch(p / "result")
            return {"refused_run": a, "restored": b, "altered_key": key.get("k")}, {
                "refused on changed code": refused(a, "the code changed"), "completes when it matches": b["outcome"] == "completed"}

        def K1():
            p = W / "k1"
            def changed():                                 # armed after the start-up snapshot: every later read differs
                h = ORIG_CODE_HASHES()
                return dict(h, **{next(iter(h)): "f" * 64})

            def arm(n):
                if n == 1:                                 # the first tier-2 canary call: after start-up, before the boundary
                    run_abst.code_hashes = changed
            s = self.launch(p / "result", hook=arm)
            return {"run": s}, {"refused before the marker": refused(s, "a loaded module's file changed since import"),
                                "no boundary": self.state(p)["boundary_markers"] == 0}

        def O1():
            p = W / "o1"
            mod = sys.modules["hlcc_verdict"]
            real = mod.__file__
            mod.__file__ = str(Path(run_abst.RAB) / "hlcc_verdict.py")
            try:
                s = self.launch(p / "result")
            finally:
                mod.__file__ = real
            return {"run": s}, {"refused at start-up": refused(s, "a gated module was loaded from the wrong directory"),
                                "nothing created": not p.exists()}

        def M3a():
            p = W / "m3a"

            def plant(n):
                if n == 3:
                    (p / "other").mkdir(parents=True, exist_ok=True)
                    (p / "other" / "ABST_PRIMARY.json").write_text("{}", encoding="utf-8")
            s = self.launch(p / "result", hook=plant)
            st = self.state(p)
            return {"run": s, "state": st}, {"refused just before the boundary": refused(s, "(just before the boundary)"),
                                             "no boundary written": st["boundary_markers"] == 0}

        def M3c():
            p = W / "m3c"

            def plant(n):
                if n == 3:
                    (p / "a" / "b" / "c").mkdir(parents=True, exist_ok=True)
                    (p / "a" / "b" / "c" / "ABST_PRIMARY.json").write_text("{}", encoding="utf-8")
            s = self.launch(p / "result", hook=plant)
            return {"run": s}, {"refused just before the boundary (depth 3)": refused(s, "(just before the boundary)"),
                                "no boundary written": self.state(p)["boundary_markers"] == 0}

        def M3b():
            p = W / "m3b"
            child = self.live_process()
            try:
                def plant(n):
                    if n == 3:
                        (p / "attempts").mkdir(parents=True, exist_ok=True)
                        (p / "attempts" / "BOUNDARY_19700101T000000Z_0.json").write_text(
                            json.dumps({"pid": child.pid, "epoch": time.time(), "code_sha256": run_abst.code_hashes()}), encoding="utf-8")
                s = self.launch(p / "result", hook=plant)
            finally:
                child.kill()
                child.wait(timeout=60)
            st = self.state(p)
            return {"run": s, "state": st}, {
                "refused just before the boundary": refused(s, "REFUSE (just before the boundary)") and "is still alive" in s["outcome"],
                "only the planted marker": st["boundary_markers"] == 1, "no attempt record": st["attempts"] == []}

        def B1():
            p = W / "b1"
            a = self.launch(p / "result", hb=HB(refuse_label="ABST-001 boundary"))
            st1 = self.state(p)
            b = self.launch(p / "result")
            st = self.state(p)
            res = jload(p / "result" / "ABST_RESULT.json") if st["result"] else {}
            return {"refused": a, "relaunch": b, "state_after_refusal": st1}, {
                "refused at the boundary RAM check": refused(a, "REFUSING TO START ABST-001 boundary"),
                "no boundary, no attempt after the refusal": st1["boundary_markers"] == 0 and st1["attempts"] == [],
                "relaunch completes with no earlier crossing": b["outcome"] == "completed" and res.get("earlier_boundary_markers") == []}

        def T1():
            p = W / "t1"
            self.launch(p / "result", fail_on=6, kind="infra")
            f = sorted((p / "attempts").glob("ATTEMPT_*.json"))[0]
            f.write_bytes(f.read_bytes()[:200])
            s = self.launch(p / "result")
            return {"run": s}, {"refused (raised) on the truncated record": s["outcome"].startswith("JSONDecodeError"),
                                "no new crossing": self.state_markers(p) == 1}

        def W1():
            p = W / "w1"
            hb = HB(free=seq([0.0] * 5, 99.0))
            s = self.launch(p / "result", hb=hb, wait=(30.0, 0.01))
            st = self.state(p)
            return {"run": s, "low_polls": hb.low_polls, "state": st}, {
                "completed": s["outcome"] == "completed" and st["result"], "waited through 5 low polls": hb.low_polls == 5}

        def W2():
            p = W / "w2"
            s = self.launch(p / "result", hb=HB(free=lambda: 0.0), wait=(0.05, 0.01))
            st = self.state(p)
            return {"run": s, "state": st}, {
                "aborted after the bound": "LowMemoryAbort" in s["outcome"] and "S20_plain/V4" in s["outcome"],
                "infrastructure attempt, no primary": [x["hint"] for x in st["attempts"]] == ["infrastructure"] and not st["primary"]}

        def W3():
            p = W / "w3"
            s = self.launch(p / "result", hb=HB(free=seq([99.0, 99.0], 0.0)), wait=(0.05, 0.01))
            st = self.state(p)
            return {"run": s, "state": st}, {
                "aborted in the wait after the primary": "LowMemoryAbort" in s["outcome"] and "after the primary" in s["outcome"],
                "primary written first": st["primary"] and [x["primary_written"] for x in st["attempts"]] == [True],
                "classified infrastructure": [x["hint"] for x in st["attempts"]] == ["infrastructure"]}

        def I():
            class OutOfMemoryError(RuntimeError):
                pass
            infra = {"MemoryError": MemoryError(), "torch CPU allocator": RuntimeError("DefaultCPUAllocator: not enough memory: you tried to allocate 1 bytes"),
                     "OutOfMemoryError": OutOfMemoryError("CUDA out of memory"), "TDR": RuntimeError("CUDA error: unknown error"),
                     "LowMemoryAbort": hostbudget.LowMemoryAbort("stand-down")}
            not_infra = {"KeyboardInterrupt": KeyboardInterrupt(), "device-side assert": RuntimeError("CUDA error: device-side assert triggered"),
                         "SystemExit": SystemExit("injected deterministic failure")}
            got = {k: run_abst.is_infrastructure(e) for k, e in {**infra, **not_infra}.items()}
            return {"classified": got}, {**{f"{k} is infrastructure": got[k] for k in infra},
                                         **{f"{k} is not infrastructure": not got[k] for k in not_infra}}

        for name, fn in (("G_output_keying", G), ("NC_negative_controls", NC), ("E2E_clean_run", E2E),
                         ("A6_failure_on_first_primary_cell", A6), ("A7m_failure_mid_cell", A7m), ("A9_failure_after_primary", A9),
                         ("S1_deterministic_then_rerun", S1), ("S2_infrastructure_one_rerun", S2), ("S3_clean_then_relaunch", S3),
                         ("S4_two_recorded_failures_then_third", S4), ("S5_infrastructure_after_primary", S5),
                         ("S6_two_hard_kills_then_third", S6), ("S7_three_hard_kills_then_fourth", S7),
                         ("S8_recorded_failure_plus_hard_kill", S8), ("L1_live_lock", L1), ("L2_stale_lock", L2),
                         ("L3_reused_pid_lock", L3), ("P1_earlier_process_alive", P1), ("P2_pid_reused_after_the_marker", P2),
                         ("C1_code_changed", C1), ("K1_file_changed_before_boundary", K1), ("O1_module_shadowed", O1),
                         ("M3a_primary_appears_before_boundary", M3a), ("M3c_deep_primary_appears_before_boundary", M3c),
                         ("M3b_concurrent_crossing_before_boundary", M3b), ("B1_pre_boundary_ram_refusal", B1),
                         ("T1_truncated_attempt_record", T1), ("W1_dip_waits_and_completes", W1),
                         ("W2_dip_past_the_bound", W2), ("W3_dip_after_last_primary_cell", W3), ("I_classification", I)):
            self.run_scenario(name, fn)

    @staticmethod
    def state_markers(p):
        return len(list((p / "attempts").glob("BOUNDARY_*.json"))) if (p / "attempts").exists() else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True, type=Path)
    ap.add_argument("--records", required=True, type=Path)
    ap.add_argument("--rtg1-sft", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--log", required=True, type=Path)
    a = ap.parse_args(argv)
    a.work, a.records = a.work.resolve(), a.records.resolve()
    if a.work.exists() or a.log.exists():
        raise SystemExit("--work and --log are create-only")
    if a.records in a.work.parents or a.records in a.log.resolve().parents:
        raise SystemExit("the harness never writes under the records tree")
    h = Harness(a)
    PR = h.panel_root
    if (PR / "result").exists() or any((PR / "attempts").glob("BOUNDARY_*.json")):
        raise SystemExit(f"{PR} has a result or a boundary marker: the harness is for BEFORE the real read")
    snap0 = snapshot(PR)
    a.work.mkdir(parents=True)
    t0 = time.time()
    h.run_all()
    snap1 = snapshot(PR)
    diff = sorted(k for k in set(snap0) | set(snap1) if snap0.get(k) != snap1.get(k))
    touched = [k for k in diff if any(t in k for t in HARNESS_CAN_CREATE) or k.startswith("panel")]
    import torch
    log = {"schema": 1, "record": "ABST_HARNESS_PASS", "experiment": "ABST-001", "revision": "R1",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0, 1),
           "harness_sha256": sha_file(Path(__file__).resolve()), "run_abst_code_sha256": run_abst.code_hashes(),
           "fixtures_sha256": h.fixture_sha, "python": sys.version.split()[0], "torch": torch.__version__,
           "records_panel_dir_changes": diff, "records_panel_dir_touched_by_harness": touched,
           "scenarios": h.results}
    log["all_pass"] = all(r["pass"] for r in h.results.values()) and not touched
    with open(a.log, "x", encoding="utf-8", newline="\n") as f:
        json.dump(log, f, indent=1, default=str)
    print(json.dumps({k: v["pass"] for k, v in h.results.items()} | {"records_untouched": not touched,
                                                                      "all_pass": log["all_pass"]}, indent=1))
    return 0 if log["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
