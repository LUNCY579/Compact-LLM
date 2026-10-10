"""One exposed, cooperative Windows CPU invocation; no automatic retry.

Only stdlib and direct-file pinned instruments/scoring_v2 code are loaded.
The create-once marker remains OUTCOME_UNKNOWN; RESULT resolves it without erasure.
"""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PREREG_SHA = "cabf16f5f96a19b741015b9c7286dd22a803d7f325ca01db2937be7779d7aa58"
SCHEDULE = [pair + variant for pair in "ABCD" for variant in "ab"]
START = time.monotonic()
LIMIT_BYTES = 1048576


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bounded_json(data):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(data, object_pairs_hook=unique, parse_constant=lambda word: (_ for _ in ()).throw(ValueError(word)))
    nodes = [0]
    def visit(item, depth):
        nodes[0] += 1
        require(depth <= 16 and nodes[0] <= 20000, "JSON depth/node bound")
        if type(item) is dict:
            require(len(item) <= 512 and all(type(k) is str and len(k) <= 256 for k in item), "JSON key bound")
            for child in item.values(): visit(child, depth + 1)
        elif type(item) is list:
            require(len(item) <= 512, "JSON list bound")
            for child in item: visit(child, depth + 1)
        elif type(item) is str:
            require(len(item) <= 131072, "JSON string bound")
        else:
            require(item is None or type(item) is bool or type(item) is int and abs(item) <= 1000000, "JSON scalar type/bound")
    visit(value, 0)
    return value


def confined(relative):
    require(type(relative) is str and not Path(relative).is_absolute(), "relative file reference")
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT), "repository reference boundary")
    return path


def read_pin(path, expected):
    require(type(expected) is str and len(expected) == 64 and all(c in "0123456789abcdef" for c in expected), "SHA256 shape")
    require(path.is_file() and path.stat().st_size <= 131072, "bounded physical file")
    data = path.read_bytes()
    require(len(data) <= 131072 and sha(data) == expected, "physical source/data/approval drift: " + path.name)
    return data


class Journal:
    def __init__(self, output):
        require(not output.exists(), "existing output/attempt denies invocation; reconcile only")
        output.mkdir(parents=True, exist_ok=False)
        self.output, self.used, self.lock = output, 0, threading.RLock()
        self.write_once("ATTEMPT.json", {"study_id": "CRS-SEMANTIC-GUARD-CONTRAST-001", "state": "OUTCOME_UNKNOWN",
                                       "pid": os.getpid(), "scheduled": SCHEDULE, "started_monotonic": START,
                                       "no_retry": True, "cooperative_not_OS_containment": True})
        self.stream = (output / "PARTIAL.jsonl").open("xb")

    def reserve(self, size):
        if self.used + size > LIMIT_BYTES:
            os._exit(125)
        self.used += size

    def write_once(self, name, value):
        data = encoded(value)
        with self.lock:
            self.reserve(len(data))
            with (self.output / name).open("xb") as handle:
                handle.write(data); handle.flush(); os.fsync(handle.fileno())

    def append(self, value):
        data = encoded(value)
        with self.lock:
            self.reserve(len(data)); self.stream.write(data); self.stream.flush(); os.fsync(self.stream.fileno())

    def fatal(self, reason, observation=None, code=124):
        if self.lock.acquire(timeout=0.02):
            try:
                self.append({"event": "FATAL_STOP", "reason": reason, "observation": observation,
                             "wall_seconds": time.monotonic() - START, "attempt_remains_unknown": True})
            finally:
                self.lock.release()
        os._exit(code)


class Guard:
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    def __init__(self, journal):
        require(os.name == "nt", "actual Windows memory collection required")
        self.journal, self.maximum, self.samples, self.done = journal, 0, 0, threading.Event()
        kernel, psapi = ctypes.WinDLL("kernel32", use_last_error=True), ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        self.process, self.collect = kernel.GetCurrentProcess(), psapi.GetProcessMemoryInfo
        self.collect.argtypes = [wintypes.HANDLE, ctypes.POINTER(self.Counters), wintypes.DWORD]
        self.collect.restype = wintypes.BOOL
        self.timer = threading.Timer(max(0.001, 30 - (time.monotonic() - START)), journal.fatal, args=("deadline",))
        self.timer.daemon = True; self.timer.start()
        self.check()
        self.sampler = threading.Thread(target=self.monitor, daemon=True, name="cooperative-memory-sampler")
        self.sampler.start()

    def check(self):
        counters = self.Counters(); counters.cb = ctypes.sizeof(counters)
        if not self.collect(self.process, ctypes.byref(counters), counters.cb):
            self.journal.fatal("memory_collection_unavailable", {"last_error": ctypes.get_last_error()}, 126)
        self.samples += 1; self.maximum = max(self.maximum, counters.PeakWorkingSetSize, counters.WorkingSetSize)
        observation = {"current_working_set_bytes": counters.WorkingSetSize, "peak_working_set_bytes": counters.PeakWorkingSetSize,
                       "sampled_maximum_bytes": self.maximum, "sample_count": self.samples}
        if self.maximum > 256 * 1024 * 1024:
            self.journal.fatal("memory_breach", observation, 126)
        if time.monotonic() - START > 30:
            self.journal.fatal("deadline_overshoot", observation)
        return observation

    def monitor(self):
        try:
            while not self.done.wait(0.05): self.check()
        except BaseException as exc:
            self.journal.fatal("monitor_failed", {"error": type(exc).__name__}, 126)

    def close(self):
        self.done.set(); self.timer.cancel(); self.sampler.join(timeout=0.2)


def load_file(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-sha", required=True)
    parser.add_argument("--acceptance", required=True, help="repository-relative Root disposition receipt")
    parser.add_argument("--acceptance-sha", required=True)
    args = parser.parse_args()
    # No prediction/scorer/load occurs unless these frozen references agree.
    freeze_bytes = read_pin(HERE / "FREEZE.json", args.freeze_sha)
    freeze = bounded_json(freeze_bytes)
    acceptance_bytes = read_pin(confined(args.acceptance), args.acceptance_sha)
    acceptance = bounded_json(acceptance_bytes)
    require(acceptance.get("study_id") == "CRS-SEMANTIC-GUARD-CONTRAST-001" and acceptance.get("status") == "FROZEN_SOURCE_ACCEPTED"
            and acceptance.get("freeze_sha256") == args.freeze_sha and acceptance.get("preregistration_sha256") == PREREG_SHA, "source disposition bindings")
    reviews = acceptance.get("reviews")
    require(type(reviews) is list and len(reviews) == 2 and {r.get("role") for r in reviews} == {"SCI", "OPS"}, "two independent source reviews")
    for review in reviews: read_pin(confined(review["path"]), review["sha256"])
    require(freeze["preregistration_sha256"] == PREREG_SHA and freeze["ordered_case_ids"] == SCHEDULE, "frozen preregistration/schedule")
    require(set(freeze["files"]) == {"run_once.py", "instruments.py", "PUBLIC_INPUTS.json", "MANUAL_EXPECTATIONS.json", "README.md"}, "exact local file membership")
    for name, pin in freeze["files"].items(): read_pin(HERE / name, pin)
    for reference in freeze["dependencies"]: read_pin(confined(reference["path"]), reference["sha256"])
    journal = Journal(HERE / "outputs" / "actual_cpu_001")
    guard = None
    predictions = {arm: {} for arm in ("F", "L", "P")}
    counts = {"F": 0, "L": 0, "P": 0, "aggregate_scorer_entry_calls": 0, "standalone_scorer_entry_calls": 0, "nested_item_scorings": 0}
    phase, scores = "resource_admission", {}
    try:
        guard = Guard(journal)
        journal.append({"event": "ADMISSION", "memory": guard.check(), "freeze_sha256": args.freeze_sha,
                        "acceptance_sha256": args.acceptance_sha, "preregistration_sha256": PREREG_SHA,
                        "exposure": "OPEN_EXPOSED_DEV; builder/reviewers saw gold; logical interface separation only"})
        phase = "load_frozen_code_and_data"
        instruments = load_file(HERE / "instruments.py", "semantic_contrast_instruments")
        scorer = load_file(confined(freeze["scorer_path"]), "semantic_contrast_scorer_v2")
        public = bounded_json(read_pin(HERE / "PUBLIC_INPUTS.json", freeze["files"]["PUBLIC_INPUTS.json"]))
        gold = bounded_json(read_pin(HERE / "MANUAL_EXPECTATIONS.json", freeze["files"]["MANUAL_EXPECTATIONS.json"]))
        require(type(public) is list and len(public) == 8 and [row["case_id"] for row in public] == SCHEDULE, "literal eight membership/order")
        require(type(gold) is dict and set(gold) == set(SCHEDULE), "full manual expectation membership")
        for row in public: require(set(row) == {"case_id", "input"} and set(row["input"]) == {"source", "query", "rules"}, "public projection group")
        phase = "precall_invariants"
        structures = [instruments.public_structure(row["input"]) for row in public]
        for offset in range(0, 8, 2):
            a, b = structures[offset:offset + 2]
            ia, ib = public[offset]["input"], public[offset + 1]["input"]
            require(a["token_bag"] == b["token_bag"] and len(ia["source"].encode()) == len(ib["source"].encode())
                    and len(a["records"]) == len(b["records"]) and a["candidates"] == b["candidates"]
                    and a["inspected_records"] == b["inspected_records"] and ia["query"] == ib["query"] and ia["rules"] == ib["rules"], "matched pair/control observation invariant")
            require(gold[SCHEDULE[offset]] != gold[SCHEDULE[offset + 1]], "pair expected divergence")
            journal.append({"event": "PAIR_INVARIANTS", "pair": SCHEDULE[offset][0], "records": len(a["records"]),
                            "whitespace_tokens": sum(a["token_bag"].values()), "source_bytes_each": len(ia["source"].encode()),
                            "inspected_records": a["inspected_records"], "candidate_count": 6,
                            "raw_transfer_count": sum(r["kind"] == "TRANSFER" for r in a["records"])})
        guard.check()
        for arm in ("F", "L", "P"):
            phase = "predict_" + arm
            for row in public:
                guard.check()
                require(sum(counts[a] for a in ("F", "L", "P")) < 24 and counts[arm] < 8, "prediction entry cap")
                counts[arm] += 1
                started = time.monotonic()
                journal.append({"event": "PREDICTION_START", "arm": arm, "case_id": row["case_id"], "counts": dict(counts)})
                trace = instruments.full_semantic(row["input"]) if arm == "F" else instruments.watched(row["input"], arm)
                elapsed = time.monotonic() - started
                answer = trace["answer"]
                raw = json.dumps(answer, sort_keys=True, separators=(",", ":"), allow_nan=False)
                require(type(answer) is dict and set(answer) == {"status", "value"}
                        and answer["status"] in {"SUPPORTED", "REFUTED", "UNKNOWN", "CONTRADICTORY"}
                        and (type(answer["value"]) is str and bool(answer["value"]) if answer["status"] == "SUPPORTED" else answer["value"] is None), "malformed primary prediction")
                predictions[arm][row["case_id"]] = raw
                journal.append({"event": "PREDICTION_END", "arm": arm, "case_id": row["case_id"], "raw_prediction": raw,
                                "trace": trace, "nested_seconds": elapsed, "memory": guard.check()})
                require(answer == gold[row["case_id"]] if arm == "F" else answer == {"status": "REFUTED", "value": None}, "prospective F/control disagreement; stop without repair")
            if arm in {"L", "P"}:
                require(all(predictions[arm][SCHEDULE[i]] == predictions[arm][SCHEDULE[i + 1]] for i in range(0, 8, 2)), "unexpected control pair divergence")
        for arm in ("F", "L", "P"):
            phase = "score_" + arm; guard.check()
            counts["aggregate_scorer_entry_calls"] += 1; counts["nested_item_scorings"] += 8
            started = time.monotonic()
            journal.append({"event": "SCORER_START", "arm": arm, "counts": dict(counts)})
            score = scorer.score(predictions[arm], gold)
            scores[arm] = score
            journal.append({"event": "SCORER_END", "arm": arm, "score": score, "nested_seconds": time.monotonic() - started,
                            "memory": guard.check()})
            require(score["scheduled_cases"] == 8 and score["counts"]["semantic_exact"] == (8 if arm == "F" else 1)
                    and score["counts"]["malformed"] == 0 and score["membership_valid"], "unexpected aggregate result")
        phase = "four_scorer_integrity_probes"
        probes = [("correct", '{"status":"SUPPORTED","value":"Nobb"}', {"status": "SUPPORTED", "value": "Nobb"}),
                  ("wrong_owner", '{"status":"SUPPORTED","value":"Nodd"}', {"status": "SUPPORTED", "value": "Nobb"}),
                  ("unknown_for_contradictory", '{"status":"UNKNOWN","value":null}', {"status": "CONTRADICTORY", "value": None}),
                  ("duplicate_key", '{"status":"SUPPORTED","status":"SUPPORTED","value":"Nobb"}', {"status": "SUPPORTED", "value": "Nobb"})]
        for name, answer, target in probes:
            guard.check(); counts["standalone_scorer_entry_calls"] += 1; counts["nested_item_scorings"] += 1
            journal.append({"event": "SCORER_PROBE_START", "probe": name, "counts": dict(counts)})
            started = time.monotonic(); result = scorer.score_one(answer, target)
            journal.append({"event": "SCORER_PROBE_END", "probe": name, "raw_answer": answer, "gold": target, "result": result,
                            "nested_seconds": time.monotonic() - started, "memory": guard.check()})
            require(result["semantic_exact"] == (name == "correct") and result["format_valid"] == (name != "duplicate_key")
                    and (result["parse_error"] == "ValueError" if name == "duplicate_key" else result["parse_error"] is None), "failed expected-negative scorer integrity criterion")
        require(counts == {"F": 8, "L": 8, "P": 8, "aggregate_scorer_entry_calls": 3,
                           "standalone_scorer_entry_calls": 4, "nested_item_scorings": 28}, "exact dose final")
        for name, pin in freeze["files"].items(): read_pin(HERE / name, pin)
        for reference in freeze["dependencies"]: read_pin(confined(reference["path"]), reference["sha256"])
        memory = guard.check()
        journal.write_once("RESULT.json", {"state": "COMPLETED_OBSERVATION_PENDING_INDEPENDENT_DISPOSITION", "scheduled": SCHEDULE,
                          "counts": counts, "scores": scores, "freeze_sha256": args.freeze_sha, "acceptance_sha256": args.acceptance_sha,
                          "wall_seconds_before_result_write": time.monotonic() - START, "memory": memory, "native_cost_seconds": 0,
                          "logical_invariant_bound": "4/8 on these eight only", "pair_parents": 4, "independent_recipe_clusters": 1,
                          "no_acquisition_learning_or_native_qualification": True, "cooperative_not_OS_containment": True,
                          "output_bytes_before_result": journal.used, "clocks_are_nested_do_not_sum": True})
        guard.check()
        journal.append({"event": "FINAL_MONITOR", "wall_seconds": time.monotonic() - START, "memory": guard.check(),
                        "output_bytes_before_this_record": journal.used})
        print(json.dumps({"result": str(journal.output / "RESULT.json"), "counts": counts, "wall_seconds": time.monotonic() - START}))
    except BaseException as exc:
        journal.append({"event": "STOP", "phase": phase, "error": type(exc).__name__, "message": str(exc)[:2048],
                        "counts": counts, "scheduled": SCHEDULE, "observed_ids": {a: list(v) for a, v in predictions.items()},
                        "wall_seconds": time.monotonic() - START, "attempt_remains_unknown": True})
        raise
    finally:
        if guard is not None: guard.close()
        journal.stream.close()


if __name__ == "__main__":
    main()
