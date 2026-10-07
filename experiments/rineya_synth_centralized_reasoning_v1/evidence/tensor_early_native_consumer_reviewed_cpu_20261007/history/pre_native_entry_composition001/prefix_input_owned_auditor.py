"""Independent bounded replay of one owned tensor diagnostic's preserved bytes.

CPU evidence never becomes native authority. Native entry is location-gated
before paths and uses separately registered runtime/source and current-death
checks only after reviewed integration into the direct experiment package.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
import stat


HERE = Path(__file__).resolve().parent
PACKAGE_NAME = "rineya_synth_centralized_reasoning_v1"
PACKAGE = HERE.parents[1] if HERE.parent.name == "artifacts" else HERE
CPU_SCOPE = "CPU_OWNED_TENSOR_PIPELINE_SIMULATION"
CORE_SCOPE = "CPU_INJECTED_PREFIX_INPUT_ONLY"
NATIVE_SCOPE = "ONE_EXPOSED_PREFIX_CUDA_INT64_INPUT_MASK_ONLY"
GIB = 1024**3
MAX_JSON = 16 * 1024**2
MAX_LEDGER = 8 * 1024**2
MAX_LINE = 65536
MAX_MARKERS = 128
PREFIX_SHA = "e806d781c2c21c15180b6ec1bc714ed5acc7104948c2859ec6b411761a4a19d1"
INPUT_FILE = "195f1b3a658959ccbe517ce5aefaef1d0d2c298db0095ace08f744bd6488feb5"
INPUT_BODY = "a22c5b27225fc75ae697a5f06a96da18c1cb995a7bf7c6d03eb05f0cb6f06ec9"
RAW_CORE = ("PHASE_OBSERVATIONS.jsonl", "ownership_verified.json", "input_check_summary.json",
            "worker_result.json", "PHASE_ACCOUNTING_FINAL.json")
RAW_NAMES = set(RAW_CORE) | {"OWNED_TERMINAL.json"}
STARTUP_PHASES = ("startup_source_qualification", "startup_owned_metadata_profile",
    "startup_ctypes_observer_binding", "startup_registered_roots_loader",
    "startup_native_provider_binding", "startup_owned_cleanup_publication")
STARTUP_COST_KEYS = {"source_reads", "source_bytes", "cache_reads", "cache_bytes", "API_calls", "IO_calls"}
ENTRY_REPORT_ADDITIONS = {"entry_bootstrap_sources", "startup_subcosts", "whole_scope_order",
    "whole_scope_success_validated", "whole_scope_closure_error", "entry_errors", "selected_boundary_counters",
    "selected_boundary_source_pins"}
TIMELINE = ("admission_first", "admission_second", "worker_created_suspended", "Job_assigned",
            "before_resume_admission", "worker_resumed", "worker_returned", "worker_dead",
            "samplers_joined", "NVML_closed", "Job_closed", "reservation_released", "lease_terminal")
GUARDS = ("pre_import", "after_import", "after_growth", "before_context_init", "after_context_init",
          "before_input_construction", "after_input_validation", "complete")
OPS = ("initial_config_preflight", "worker_ownership_verify", "publication_ownership_verified",
       "disk_admission", "torch_import", "torch_version_read", "post_import_growth_observation",
       "cuda_context_init", "pre_input_direct_CUDA_mem_get_info", "original_precision_thread_configuration",
       "Python_seed_101", "Torch_CPU_seed_101", "Torch_CUDA_seed_all_101", "CUDA_reset_peak_memory_stats",
       "input_tensor_create", "mask_ones_like_create", "input_is_tensor", "input_shape_matches",
       "input_dtype_long", "input_device_cuda0", "input_exact_tolist", "mask_is_tensor", "mask_shape_matches",
       "mask_dtype_long", "mask_device_cuda0", "mask_exact_tolist", "input_reference_release",
       "mask_reference_release", "final_config_preflight", "terminal_cuda_is_initialized",
       "terminal_cuda_max_memory_allocated", "terminal_cuda_max_memory_reserved", "terminal_cuda_mem_get_info",
       "publication_input_check_summary", "publication_worker_result")
ROOT_ORDER = (
    *OPS[:3], "guard_pre_import", *OPS[3:5], "guard_after_import", *OPS[5:7],
    "guard_after_growth", "guard_before_context_init", OPS[7], "guard_after_context_init", *OPS[8:14],
    "guard_before_input_construction", *OPS[14:26], "guard_after_input_validation", *OPS[26:33],
    "guard_complete", *OPS[33:])
PRE_ROOT = ROOT_ORDER[:ROOT_ORDER.index("input_reference_release")]
QUERY_NAMES = {"guard_cuda_is_initialized", "guard_cuda_mem_get_info"}
PREDICATES = {"input_is_tensor", "input_shape_matches", "input_dtype_long", "input_device_cuda0",
              "input_exact_tolist", "mask_is_tensor", "mask_shape_matches", "mask_dtype_long",
              "mask_device_cuda0", "mask_exact_tolist"}
RESOURCES = {"CPU_sample_max_age_seconds": .2, "GPU_free_at_admission_GiB": 4.0,
    "GPU_free_during_work_GiB": .5, "GPU_probe_interval_seconds": .5,
    "GPU_probe_timeout_seconds": 2.0, "GPU_sample_max_age_seconds": 2.0,
    "admission_poll_seconds": 2.0, "admission_seconds": 30,
    "disk_free_GiB": 4.0, "external_watchdog_milliseconds": 100,
    "preworker_available_GiB": 3.3125, "system_available_reserve_GiB": 1.25,
    "whole_scope_seconds": 600, "worker_process_tree_RSS_GiB": 2.0}
PERMISSION = {"allocations": 2, "dtype": "int64", "shape": [1, 422], "logical_bytes": 6752,
              "model": False, "forward": False, "backward": False, "optimizer": False,
              "decoder": False, "scoring": False, "M0_membership": False}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("ascii")).hexdigest()


def seal(value):
    body = {key: deepcopy(item) for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": digest(body)}


def sha(value):
    return type(value) is str and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def fields(value, names, label):
    require(type(value) is dict and set(value) == set(names), "exact " + label + " schema")


def pairs(items):
    value = {}
    for key, item in items:
        require(key not in value, "duplicate JSON key")
        value[key] = item
    return value


def floating(value):
    result = float(value)
    require(math.isfinite(result), "nonfinite or overflowing JSON float")
    return result


def parse(raw):
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_float=floating,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON constant")))
    except (UnicodeError, RecursionError) as error:
        raise ValueError("invalid bounded JSON") from error


def checked_seal(value, expected=None):
    require(type(value) is dict and sha(value.get("sha256")), "sealed document required")
    body = digest({key: item for key, item in value.items() if key != "sha256"})
    require(value["sha256"] == body and (expected is None or expected == body), "original external body anchor")


class Reader:
    def __init__(self, root):
        self.root = Path(root).absolute()
        self.original = {}

    def read(self, path, expected, maximum=MAX_JSON, owned=True):
        fields(expected, {"bytes", "sha256"}, "raw byte descriptor")
        require(integer(expected["bytes"]) and expected["bytes"] <= maximum and sha(expected["sha256"]), "typed bounded external byte anchor")
        path = Path(path)
        require(path.is_absolute() and path.resolve() == path, "literal physical selected path")
        require(not owned or path.is_relative_to(self.root), "selected owned file cannot escape")
        for entry in (path, *path.parents):
            meta = entry.lstat()
            require(not entry.is_symlink() and not(getattr(meta, "st_file_attributes", 0) & 0x400), "non-reparse selected path")
        before = path.lstat()
        require(stat.S_ISREG(before.st_mode) and before.st_size == expected["bytes"], "regular selected file exact size before read")
        with path.open("rb") as stream:
            raw = stream.read(before.st_size + 1)
        after = path.lstat()
        require(len(raw) == before.st_size and (before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), "stable bounded file bytes")
        require(hashlib.sha256(raw).hexdigest() == expected["sha256"], "original selected raw file anchor")
        key = str(path)
        require(key not in self.original or self.original[key] == expected, "first read anchor cannot be refreshed")
        self.original.setdefault(key, deepcopy(expected))
        return raw

    def verify(self):
        for path, expected in list(self.original.items()):
            self.read(Path(path), expected, MAX_JSON if Path(path).suffix != ".jsonl" else MAX_LEDGER,
                      owned=Path(path).is_relative_to(self.root))


def parse_ledger(raw):
    require(len(raw) <= MAX_LEDGER and raw.endswith(b"\n"), "complete bounded phase ledger")
    lines = raw.splitlines(keepends=True)
    require(0 < len(lines) <= MAX_MARKERS and all(len(line) <= MAX_LINE for line in lines), "bounded exact phase lines")
    return [parse(line) for line in lines]


def zero_counts():
    return {"attempted": 0, "returned": 0, "failed": 0, "pending": 0}


def replay(records):
    allowed = set(OPS) | set(STARTUP_PHASES) | {"guard_" + name for name in GUARDS} | QUERY_NAMES | {"bounded_original_exception_frames_capture"}
    stack, next_ordinal, previous_time = [], 1, None
    counts, returns, roots, guard_children = {}, {}, [], {}
    for record in records:
        event = record.get("event")
        names = {"operation", "event", "ordinal", "monotonic", "utc"}
        if event != "attempt":
            names |= {"result", "error"}
        fields(record, names, "phase marker")
        name, ordinal = record["operation"], record["ordinal"]
        require(name in allowed and event in ("attempt", "returned", "failed") and integer(ordinal, 1), "declared typed phase operation")
        clock = record["monotonic"]
        require(finite(clock) and (previous_time is None or clock >= previous_time), "ordered finite phase time")
        require(type(record["utc"]) is str and 0 < len(record["utc"]) <= 64, "bounded UTC marker")
        previous_time = clock
        item = counts.setdefault(name, zero_counts())
        if event == "attempt":
            require(ordinal == next_ordinal, "exact original attempt ordinal/order")
            next_ordinal += 1
            if stack:
                require(name in QUERY_NAMES and stack[-1][0] in {"guard_" + value for value in GUARDS}, "only declared guard query nesting")
                guard_children.setdefault(stack[-1][1], []).append((name, ordinal))
            else:
                require(name not in QUERY_NAMES, "guard query must be inside its guard")
                roots.append(name)
            stack.append((name, ordinal))
            item["attempted"] += 1
            item["pending"] += 1
        else:
            require(stack and stack[-1] == (name, ordinal), "exact LIFO returned/failed pair")
            stack.pop()
            item["pending"] -= 1
            item[event] += 1
            require((record["error"] is None) == (event == "returned"), "truthful phase error state")
            returns[ordinal] = record
    require(not stack, "no omitted terminal phase marker")
    require(all(item["attempted"] == item["returned"] + item["failed"] and item["pending"] == 0 for item in counts.values()), "conserved exact phase costs")
    for name in QUERY_NAMES:
        require(counts.get(name, zero_counts())["attempted"] <= 7, "original per-query guard ceiling")
    require(sum(counts.get(name, zero_counts())["attempted"] for name in QUERY_NAMES) <= 14, "full guard query ceiling")
    by_name = {}
    for record in records:
        if record["event"] != "attempt":
            by_name.setdefault(record["operation"], []).append(record)
    for children in guard_children.values():
        require(children[0][0] == "guard_cuda_is_initialized" and len(children) <= 2, "original guard query order/count")
        initialized = returns[children[0][1]]
        if initialized["event"] == "returned":
            require(type(initialized["result"]) is bool, "typed actual guard initialization observation")
            require(len(children) == (2 if initialized["result"] else 1), "guard free-memory query follows actual initialized state")
        if len(children) == 2:
            require(children[1][0] == "guard_cuda_mem_get_info", "exact second guard query")
            observed = returns[children[1][1]]
            if observed["event"] == "returned":
                require(memory(observed["result"], .5 * GIB), "typed actual loaded-guard GPU floor")
    for record in records:
        if record["event"] == "attempt" and record["operation"] in {"guard_" + name for name in GUARDS}:
            children = guard_children.get(record["ordinal"], [])
            terminal = returns[record["ordinal"]]
            if record["operation"] == "guard_pre_import":
                require(not children, "no Torch guard queries before import")
            elif terminal["event"] == "returned":
                require(children, "all returned loaded guards retain actual query children")
    return {"counts": counts, "roots": roots, "by_name": by_name, "first_clock": records[0]["monotonic"],
            "last_clock": records[-1]["monotonic"]}


def value_of(replayed, name):
    rows = replayed["by_name"].get(name, [])
    require(len(rows) == 1 and rows[0]["event"] == "returned", "one actual returned " + name)
    return rows[0]["result"]


def memory(value, floor):
    return type(value) is list and len(value) == 2 and all(integer(item) for item in value) and floor <= value[0] <= value[1]


def identity(value):
    fields(value, {"pid", "creation_filetime_100ns"}, "owned PID/birth")
    require(all(integer(item, 1) for item in value.values()), "positive nonboolean owned identity")
    return value


def _manifest(manifest, expected, source_pins, mode):
    require(sha(expected), "external immutable manifest body selection required")
    checked_seal(manifest, expected)
    scope = CPU_SCOPE if mode == "CPU" else NATIVE_SCOPE
    require(manifest["scope"] == scope and manifest["permission"] == PERMISSION and manifest["resources"] == RESOURCES, "exact selected purpose/permission/resources")
    require(manifest["input_file_sha256"] == INPUT_FILE and manifest["input_body_sha256"] == INPUT_BODY and
            manifest["prefix_ids_sha256"] == PREFIX_SHA, "exact exposed input anchors")
    if mode == "CPU":
        require(manifest["kind"] == CPU_SCOPE and manifest["source_pins"] == source_pins and manifest["runtime_pins"] == {}, "exact CPU/source namespace")
        require(all(manifest[key] is False for key in ("native_runtime_closure_complete", "cache_execution_policy_qualified",
                "external_dependencies_qualified", "native_launch_authorized")), "CPU proof cannot open native authority")


def _input(reader):
    path = PACKAGE / "evidence/exception_frame_helper_reviewed_cpu_20261005/root/ROOT_FAILURE_PREFIX_INPUT_001.json"
    require(path.lstat().st_size <= 16384, "bounded exposed input before read")
    expected = {"bytes": path.lstat().st_size, "sha256": INPUT_FILE}
    raw = reader.read(path, expected, 16384, owned=False)
    value = parse(raw)
    checked_seal(value, INPUT_BODY)
    prefix = value["prefix_ids"]
    require(type(prefix) is list and len(prefix) == 422 and all(integer(item) and item < 151936 for item in prefix)
            and value["attention_mask"] == [1] * 422 and digest(prefix) == PREFIX_SHA, "exact original422 prompt-only prefix")
    require(value["native_launch_authorized"] is value["M0_population_membership_allowed"] is value["training_or_grading_allowed"] is False, "input confers no study authority")
    return prefix


def _terminal(terminal, manifest, successful, phase_first, phase_last):
    fields(terminal, {"kind", "scope", "status", "manifest_sha256", "worker_identity", "owned_process_identities",
        "timeline", "terminal_flags", "current_death", "resource_samples", "admission_observations", "worker_report",
        "worker_report_sha256", "error", "primary_message_exact", "secondary_errors", "fixture_only",
        "native_authority_issued", "sha256"}, "owned terminal")
    checked_seal(terminal)
    require(terminal["scope"] == manifest["scope"] and terminal["manifest_sha256"] == manifest["sha256"], "same selected terminal manifest")
    owner = identity(terminal["worker_identity"])
    owners = terminal["owned_process_identities"]
    require(type(owners) is list and owners and all(identity(item) for item in owners) and owner in owners
            and len({(item["pid"], item["creation_filetime_100ns"]) for item in owners}) == len(owners), "complete unique owned process/birth population")
    if manifest["scope"] == CPU_SCOPE:
        require(owners == [owner], "one declared injected CPU worker birth")
    flags = terminal["terminal_flags"]
    fields(flags, {"job_closed", "reservation_released", "lease_terminal", "samplers_joined", "NVML_closed", "worker_dead"}, "owned terminal flags")
    require(all(value is True for value in flags.values()), "all owned terminal actions returned")
    if successful:
        require(terminal["error"] is None and terminal["secondary_errors"] == [] and terminal["primary_message_exact"] is True,
                "no hidden admission/cleanup/terminal error")
    deaths = terminal["current_death"]
    require(type(deaths) is list and len(deaths) == len(owners), "all owned current-death observations")
    for observed, wanted in zip(deaths, owners):
        fields(observed, {"identity", "currently_alive"}, "current death")
        require(observed["identity"] == wanted and observed["currently_alive"] is False, "same owned birth currently dead")
    timeline = terminal["timeline"]
    require(type(timeline) is list and len(timeline) == len(TIMELINE), "complete owned event population")
    events, previous = {}, None
    for wanted, event in zip(TIMELINE, timeline):
        require(type(event) is dict and set(event) in ({"event", "monotonic"}, {"event", "monotonic", "identity"})
                and event["event"] == wanted and finite(event["monotonic"]), "exact ordered owned event")
        require(previous is None or event["monotonic"] >= previous, "owned times do not reverse")
        if "identity" in event:
            require(event["identity"] == owner, "timeline exact PID/birth identity")
        events[wanted] = event["monotonic"]
        previous = event["monotonic"]
    require(previous - events["admission_first"] <= 600 and events["before_resume_admission"] - events["admission_first"] <= 30,
            "original whole-scope/admission time bounds")
    require(events["before_resume_admission"] <= events["worker_resumed"] <= phase_first <= phase_last <= events["worker_returned"], "actual phase work inside owned resume/return interval")
    samples = terminal["resource_samples"]
    fields(samples, {"CPU", "GPU"}, "resource sample populations")
    for domain, names in (("CPU", {"monotonic", "available_bytes", "worker_RSS_bytes"}),
                          ("GPU", {"monotonic", "free_bytes", "total_bytes"})):
        records = samples[domain]
        require(type(records) is list and 0 < len(records) <= 65536, "bounded nonempty original resource samples")
        last = None
        for sample in records:
            fields(sample, names, "typed " + domain + " sample")
            require(finite(sample["monotonic"]) and (last is None or sample["monotonic"] >= last), "resource sample chronological clock")
            require(all(integer(value) for key, value in sample.items() if key != "monotonic"), "typed actual resource byte amounts")
            if domain == "GPU":
                require(sample["free_bytes"] <= sample["total_bytes"], "GPU free/total conservation")
            if successful:
                if domain == "CPU":
                    require(sample["available_bytes"] >= 1.25 * GIB and sample["worker_RSS_bytes"] <= 2 * GIB, "original CPU reserve/RSS floors")
                else:
                    require(sample["free_bytes"] >= .5 * GIB, "original during-work GPU floor")
            last = sample["monotonic"]
    if successful:
        observations = terminal["admission_observations"]
        require(type(observations) is list and len(observations) == 3, "two admissions plus before-resume observation")
        gpu_identity = None
        for clock_name, observation in zip(("admission_first", "admission_second", "before_resume_admission"), observations):
            fields(observation, {"monotonic", "available_bytes", "worker_RSS_bytes", "GPU_monotonic", "free_bytes",
                "total_bytes", "GPU_identity", "disk_free_bytes"}, "full actual admission observation")
            now = events[clock_name]
            require(finite(observation["monotonic"]) and 0 <= now - observation["monotonic"] <= .2 and
                    finite(observation["GPU_monotonic"]) and 0 <= now - observation["GPU_monotonic"] <= 2.0,
                    "actual CPU/GPU admission freshness")
            require(all(integer(value) for key, value in observation.items() if key not in ("monotonic", "GPU_monotonic", "GPU_identity")), "typed admission bytes")
            require(observation["available_bytes"] >= 3.3125 * GIB and observation["worker_RSS_bytes"] <= 2 * GIB and
                    4 * GIB <= observation["free_bytes"] <= observation["total_bytes"] and observation["disk_free_bytes"] >= 4 * GIB,
                    "original full admission floors")
            require(type(observation["GPU_identity"]) is str and len(observation["GPU_identity"]) == 40 and
                    observation["GPU_identity"].startswith("GPU-") and (gpu_identity is None or gpu_identity == observation["GPU_identity"]), "same exact admitted GPU identity")
            gpu_identity = observation["GPU_identity"]
        require(observations[1]["monotonic"] - observations[0]["monotonic"] >= 2 and
                observations[1]["GPU_monotonic"] > observations[0]["GPU_monotonic"], "two distinct current original2s admissions")
        for domain, age in (("CPU", .2), ("GPU", 2.0)):
            times = [sample["monotonic"] for sample in samples[domain]]
            during = [clock for clock in times if events["worker_resumed"] <= clock <= events["worker_dead"]]
            require(during and during[0] - events["worker_resumed"] <= age and
                    events["worker_dead"] - during[-1] <= age, "resource sampling covers resume/first and last/death edges")
            require(any(phase_first <= clock <= phase_last for clock in during), "actual during-work resource sample required")
            for left, right in zip(during, during[1:]):
                require(right - left <= age, "no unobserved original resource sampling gap")
    return owner


def _entry_costs(final, report, replayed, successful):
    """Replay durable startup costs, never infer actual native API coverage."""
    costs = final.get("startup_subcosts")
    require(type(costs) is dict and set(costs) <= set(STARTUP_PHASES), "finite registered startup cost phases")
    require(report.get("startup_subcosts") == costs and report.get("whole_scope_order") == final.get("whole_scope_order"), "returned/final startup cost and order joins")
    require(final.get("whole_scope_order") == [name for name in replayed["counts"] if not name.startswith("guard_")], "actual first-attempt whole scope order")
    for name, values in costs.items():
        require(type(values) is dict and set(values) == STARTUP_COST_KEYS and all(integer(value) and value < 2**63 for value in values.values()), "exact bounded startup cost counters")
        entries = replayed["by_name"].get(name, [])
        require(len(entries) == 1, "one actual terminal startup phase")
        if entries[0]["event"] == "returned":
            result = entries[0]["result"]
            require(type(result) is dict and result.get("startup_subcosts") == values, "durable returned marker startup costs")
    boundaries = final.get("selected_boundary_counters")
    if boundaries is not None:
        require(boundaries == report.get("selected_boundary_counters") and type(boundaries) is dict and set(boundaries) <= set(STARTUP_PHASES), "same finite selected nested boundary counters")
        for phase, values in boundaries.items():
            require(type(values) is dict and len(values) <= 64, "bounded nested boundary population")
            derived = {key: 0 for key in STARTUP_COST_KEYS}
            for name, counts in values.items():
                require(type(name) is str and 0 < len(name) <= 256 and type(counts) is dict and set(counts) == {"attempted", "returned", "failed", "pending", "bytes", "category"}, "exact categorized nested boundary counts")
                require(all(integer(counts[key]) and counts[key] < 2**63 for key in ("attempted", "returned", "failed", "pending", "bytes")) and counts["attempted"] == counts["returned"] + counts["failed"] + counts["pending"], "conserved bounded nested boundary attempts")
                category = counts["category"]
                require(category in ("source", "cache", "API", "IO", "boundary"), "finite source-owned boundary categories")
                expected = {"contract.read_bytes.source": "source", "contract.read_bytes.cache": "cache",
                    "contract.file_descriptor.source": "source", "contract.file_descriptor.cache": "cache",
                    "contract.sealed_json": "boundary", "contract.physical_roster": "IO",
                    "worker.exact_module_read.source": "source", "git.disk_read.source": "source",
                    "git.immutable_blob.source": "source", "git.metadata_command": "IO"}.get(name)
                if expected is None and name.startswith("API."):
                    require(name in {"API.NativeLibraryObserver.abi_binding", "API.NativeLibraryObserver.bootstrap",
                        "API.NativeLibraryObserver.begin", "API.NativeLibraryObserver.events", "API.NativeLibraryObserver.close",
                        "API.ManagedWin32LibraryProvider.begin", "API.ManagedWin32LibraryProvider.scope",
                        "API.ManagedWin32LibraryProvider.libraries", "API.ManagedWin32LibraryProvider.events",
                        "API.ManagedWin32LibraryProvider.close"}, "finite selected observer/provider method counter")
                    expected = "API"
                require(expected == category, "counter category bound to selected source wrapper")
                if category in ("source", "cache"):
                    require(name.endswith("." + category), "distinct source/cache counter keys")
                    derived[category + "_reads"] += counts["returned"]
                    derived[category + "_bytes"] += counts["bytes"]
                else:
                    require(counts["bytes"] == 0, "only selected read/hash boundaries account bytes")
                if category in ("source", "cache", "IO"): derived["IO_calls"] += counts["attempted"]
                if category == "API": derived["API_calls"] += counts["attempted"]
                if successful: require(counts["failed"] == counts["pending"] == 0, "complete nested boundary costs")
            require(costs.get(phase) == derived, "startup costs independently derived from categorized selected boundaries")
        require(all(not any(values.values()) for name, values in costs.items() if name not in boundaries), "no aggregate costs without selected boundary population")
    pins = final.get("selected_boundary_source_pins", {})
    require(pins == report.get("selected_boundary_source_pins", {}) and type(pins) is dict and len(pins) <= 64, "same finite selected boundary source pins")
    for path, descriptor in pins.items():
        require(type(path) is str and 0 < len(path) <= 32768, "bounded selected boundary source path")
        fields(descriptor, {"bytes", "sha256", "role"}, "selected boundary source descriptor")
        require(integer(descriptor["bytes"]) and descriptor["bytes"] <= MAX_JSON and sha(descriptor["sha256"]) and descriptor["role"] == "python_source", "bounded pinned boundary source bytes")
    if successful:
        require(set(costs) == set(STARTUP_PHASES) and final.get("whole_scope_success_validated") is report.get("whole_scope_success_validated") is True, "all successful startup costs and producer closure")
        for name in (*STARTUP_PHASES, *OPS):
            require(replayed["counts"].get(name) == {"attempted": 1, "returned": 1, "failed": 0, "pending": 0}, "exact single successful startup/core operation multiplicity")
    else:
        require(final.get("whole_scope_success_validated") is report.get("whole_scope_success_validated") is False, "failed entry cannot claim successful closure")

def _early_entry_core(data, records, mode):
    """Truthful pre-core failure; missing tensor/core facts remain absent."""
    final = data["PHASE_ACCOUNTING_FINAL.json"]
    terminal = data["OWNED_TERMINAL.json"]
    report = terminal["worker_report"]
    require(type(report) is dict and terminal["worker_report_sha256"] == digest(report), "actual early entry report digest")
    expected_status = "CPU_PREFIX_INPUT_CHECKS_FAILED" if mode == "CPU" else "NATIVE_INPUT_CHECKS_FAILED"
    require(report.get("status") == final["status"] == expected_status and final["closed"] is True and final["native_PASS"] is False, "failed closed entry outcome only")
    require(final.get("scope") == (CORE_SCOPE if mode == "CPU" else NATIVE_SCOPE), "early failure retains selected CPU/native scope")
    replayed = replay(records)
    roots = replayed["roots"]
    require(roots and set(roots) <= set(STARTUP_PHASES) and roots[-1] == STARTUP_PHASES[-1], "pre-core failure entered cleanup and no core work")
    entered = roots[:-1]
    require(entered == list(STARTUP_PHASES[:len(entered)]) and any(value["failed"] for value in replayed["counts"].values()), "actual failed startup prefix")
    require(final["operations"] == replayed["counts"] and final["markers"] == len(records), "early failure phase/marker counts exact")
    require(report.get("operation_accounting") == final["operations"] and report.get("IO_accounting") == final["IO"] and
            report.get("publication_errors") == final["publication_errors"], "early failure actual returned count/IO/error joins")
    require(report["final_phase_summary"] == final and report.get("entry_errors"), "early failure retains final summary and original errors")
    _entry_costs(final, report, replayed, False)
    require(not any(name in data for name in RAW_CORE if name not in ("PHASE_OBSERVATIONS.jsonl", "PHASE_ACCOUNTING_FINAL.json")), "no invented pre-core tensor/ownership/result files")
    return False, replayed

def _core(data, records, manifest, prefix, mode):
    entry = any(record.get("operation") in STARTUP_PHASES for record in records)
    require(not (entry and mode == "NATIVE"), "shared-entry native terminal integration remains closed pending registered current run")
    if entry and "worker_result.json" not in data:
        return _early_entry_core(data, records, mode)
    worker, final = data["worker_result.json"], data["PHASE_ACCOUNTING_FINAL.json"]
    terminal_report = data["OWNED_TERMINAL.json"]["worker_report"]
    require(type(worker) is dict and type(terminal_report) is dict and
            digest(terminal_report) == data["OWNED_TERMINAL.json"]["worker_report_sha256"], "actual returned worker report bytes/body")
    status = terminal_report.get("status")
    success_status = "CPU_PREFIX_INPUT_CHECKS_COMPLETE_UNQUALIFIED" if mode == "CPU" else "NATIVE_INPUT_CHECKS_RETURNED_PENDING_TERMINAL_AUDIT"
    failure_status = "CPU_PREFIX_INPUT_CHECKS_FAILED" if mode == "CPU" else "NATIVE_INPUT_CHECKS_FAILED"
    require(status in (success_status, failure_status), "source-defined whole worker outcome")
    successful = status == success_status
    scope = CORE_SCOPE if mode == "CPU" else NATIVE_SCOPE
    require(worker["scope"] == final["scope"] == terminal_report["scope"] == scope, "no CPU/native evidence conversion")
    require(data["ownership_verified.json"] == {"scope": scope}, "exact durable core ownership marker")
    require(worker["requires_complete_final_phase_accounting"] is terminal_report["requires_complete_final_phase_accounting"] is True, "provisional worker requires final authority")
    for name in ("model_loaded", "model_constructed", "M0_membership", "native_launch_authorized", "scoring_authorized", "native_fit_established", "cause_established"):
        require(worker[name] is terminal_report[name] is False, "all unsupported model/fit/study claims closed")
    for name in ("forwards", "backwards", "optimizer_steps", "decodes", "grades"):
        require(type(worker[name]) is type(terminal_report[name]) is int and worker[name] == terminal_report[name] == 0, "zero forbidden work")
    require(worker["logical_tensor_bytes"] == terminal_report["logical_tensor_bytes"] == 6752, "exact2x422x8 logical bytes")
    replayed = replay(records)
    require(final["status"] == status and final["closed"] is True and final["native_PASS"] is False,
            "paired final phase decision, never preliminary worker alone")
    require(final["operations"] == replayed["counts"] == terminal_report["operation_accounting"], "independent actual operation count replay")
    require(final["IO"] == terminal_report["IO_accounting"] and final["publication_errors"] == terminal_report["publication_errors"], "exact final returned IO/error joins")
    require(final["markers"] == len(records), "all original markers retained")
    require(terminal_report["final_phase_summary"] == final, "actual final summary retained in returned report")
    additions = {"final_phase_summary", "operation_accounting", "IO_accounting", "publication_errors"} | (ENTRY_REPORT_ADDITIONS if entry else set())
    if entry:
        _entry_costs(final, terminal_report, replayed, successful)
    if successful:
        require({key: item for key, item in terminal_report.items() if key not in additions} == worker, "unchanged durable successful provisional worker")
        expected_roots = list(STARTUP_PHASES[:5]) + list(ROOT_ORDER) + [STARTUP_PHASES[-1]] if entry else list(ROOT_ORDER)
        require(replayed["roots"] == expected_roots and all(item["failed"] == 0 for item in replayed["counts"].values()), "complete exact successful root order")
        require(worker["error"] is worker["terminal_error"] is None and worker["secondary_errors"] == [] and final["publication_errors"] == [], "no hidden first/terminal/publication failure")
        for name in PREDICATES:
            require(value_of(replayed, name) is True, "typed actual successful predicate " + name)
        require(value_of(replayed, "disk_admission") >= 4 * GIB and integer(value_of(replayed, "disk_admission")), "original typed disk admission")
        growth = value_of(replayed, "post_import_growth_observation")
        fields(growth, {"available_bytes", "worker_RSS_bytes"}, "actual growth observation")
        require(all(integer(value) for value in growth.values()) and growth["worker_RSS_bytes"] <= 2 * GIB and
            growth["available_bytes"] >= 1.25 * GIB + max(0, 2 * GIB - growth["worker_RSS_bytes"]), "original full growth reserve")
        require(memory(value_of(replayed, "pre_input_direct_CUDA_mem_get_info"), 4 * GIB), "actual pre-input GPU admission")
        require(value_of(replayed, "terminal_cuda_is_initialized") is True, "actual initialized terminal")
        allocated, reserved = value_of(replayed, "terminal_cuda_max_memory_allocated"), value_of(replayed, "terminal_cuda_max_memory_reserved")
        require(integer(allocated) and integer(reserved) and allocated <= reserved and
                memory(value_of(replayed, "terminal_cuda_mem_get_info"), .5 * GIB), "typed actual terminal allocator/free floor")
        require(value_of(replayed, "initial_config_preflight") == value_of(replayed, "final_config_preflight"), "original immutable initial/final source/config observations")
        summary = data["input_check_summary.json"]
        require(summary == worker["input_summary"] == terminal_report["input_summary"], "exact successful input summary file/report")
        require(summary["prefix_ids_sha256"] == PREFIX_SHA and summary["observed_ids_host_values_sha256"] == digest([prefix])
            and summary["mask_values_sha256"] == digest([[1] * 422]) and summary["shape"] == summary["mask_shape"] == summary["second_ids_shape"] == [1, 422]
            and summary["dtype_long_observation"] is summary["mask_dtype_long_observation"] is True
            and summary["device"] == summary["mask_device"] == "cuda:0" and summary["logical_tensor_bytes"] == 6752
            and summary["logical_bytes_not_allocator_reservation"] is True, "actual observed IDs/mask dimensions/dtype/device/values")
        require(final["IO"] == {"ledger_create": {"attempted": 1, "returned": 1, "failed": 0, "pending": 0},
            "marker_write": {"attempted": len(records), "returned": len(records), "failed": 0, "pending": 0},
            "ledger_close": {"attempted": 1, "returned": 1, "failed": 0, "pending": 0},
            "final_summary_write": {"attempted": 1, "returned": 1, "failed": 0, "pending": 0}}, "complete conserved successful durable IO")
    else:
        roots = replayed["roots"]
        if "bounded_original_exception_frames_capture" in roots:
            split = roots.index("bounded_original_exception_frames_capture")
            expected_prefix = list(STARTUP_PHASES[:5]) + list(PRE_ROOT) if entry else list(PRE_ROOT)
            require(roots[:split] == expected_prefix[:split], "honest original failed core prefix")
            require(worker["error"] is not None, "first failure retained")
        else:
            require(worker["terminal_error"] is not None or final["publication_errors"] or
                    (entry and (terminal_report.get("entry_errors") or terminal_report.get("whole_scope_closure_error"))), "failure cannot be invented after successful evidence")
        require(data["input_check_summary.json"] == worker.get("input_summary", {"partial_failure": True}), "truthful partial input summary")
    return successful, replayed


def _CPU_shared_entry_bytes(preserved_directory, *, original_directory, raw_files, report_file,
                            report_descriptor, prefix, selected_source_pins=None):
    """Private CPU fixture seam: actual selected bytes/replay, no terminal grant.

    This intentionally supplies no Job, resource, death, library or native
    interface qualification. Missing final publication remains a failed,
    partial record, rather than an invented worker result or final summary.
    """
    original_root, preserved_root = Path(original_directory).resolve(), Path(preserved_directory).resolve()
    require(original_root != preserved_root and not original_root.is_relative_to(preserved_root) and
            not preserved_root.is_relative_to(original_root), "separate original and preserved CPU roots")
    require(type(raw_files) is dict and "PHASE_OBSERVATIONS.jsonl" in raw_files and
            set(raw_files) <= set(RAW_CORE), "finite explicit CPU entry raw roster")
    require(type(prefix) is list and len(prefix) == 422 and all(integer(value) for value in prefix), "bounded injected CPU prefix")
    readers = (Reader(original_root), Reader(preserved_root))
    data, raw = {}, {}
    for name, descriptor in raw_files.items():
        values = [reader.read(reader.root / name, descriptor,
            MAX_LEDGER if name.endswith(".jsonl") else MAX_JSON) for reader in readers]
        require(values[0] == values[1], "preserved CPU raw byte identity")
        raw[name] = values[0]
        if not name.endswith(".jsonl"): data[name] = parse(values[0])
    require(type(report_file) is str and report_file == "CPU_ENTRY_RETURN.json", "explicit fixture report name")
    names = set(raw_files) | {report_file}
    for root in (original_root, preserved_root):
        require(set(path.name for path in root.iterdir()) == names, "exact original/preserved CPU entry raw roster")
    returned = [reader.read(reader.root / report_file, report_descriptor) for reader in readers]
    require(returned[0] == returned[1], "preserved CPU report byte identity")
    report = parse(returned[0])
    require(type(report) is dict and report.get("status") in
            ("CPU_PREFIX_INPUT_CHECKS_COMPLETE_UNQUALIFIED", "CPU_PREFIX_INPUT_CHECKS_FAILED"), "CPU source-defined outcome only")
    data["OWNED_TERMINAL.json"] = {"worker_report": report, "worker_report_sha256": digest(report)}
    require(type(report.get("entry_bootstrap_sources")) is list and len(report["entry_bootstrap_sources"]) == 3,
            "three selected writer bootstrap source descriptors")
    bootstrap = {}
    for item in report["entry_bootstrap_sources"]:
        fields(item, {"path", "bytes", "sha256"}, "entry bootstrap source descriptor")
        require(type(item["path"]) is str and item["path"] not in bootstrap, "unique selected writer bootstrap paths")
        bootstrap[item["path"]] = {"bytes": item["bytes"], "sha256": item["sha256"]}
    boundaries = report.get("selected_boundary_source_pins", {})
    require(type(boundaries) is dict, "selected source pin map")
    selected = {} if selected_source_pins is None else selected_source_pins
    require(type(selected) is dict and set(bootstrap) | set(boundaries) <= set(selected), "external selected source pin map covers actual called sources")
    for path, actual in {**bootstrap, **boundaries}.items():
        require(path in selected and actual["bytes"] == selected[path]["bytes"] and actual["sha256"] == selected[path]["sha256"], "actual called source pin matches frozen selected source")
        readers[0].read(Path(path), {"bytes": actual["bytes"], "sha256": actual["sha256"]}, owned=False)
    records = parse_ledger(raw["PHASE_OBSERVATIONS.jsonl"])
    require(any(row.get("operation") in STARTUP_PHASES for row in records), "shared entry fixture journal required")
    final = data.get("PHASE_ACCOUNTING_FINAL.json")
    if final is None:
        replayed = replay(records)
        require(report["status"] == "CPU_PREFIX_INPUT_CHECKS_FAILED" and report.get("final_summary_error") is not None and
                report.get("whole_scope_success_validated") is False, "missing final publication is failed partial evidence")
        successful = False
    else:
        require(final.get("marker_bytes") == len(raw["PHASE_OBSERVATIONS.jsonl"]), "exact durable journal byte count")
        successful, replayed = _core(data, records, {}, prefix, "CPU")
    for reader in readers: reader.verify()
    for root in (original_root, preserved_root):
        require(set(path.name for path in root.iterdir()) == names, "CPU raw roster stable through replay")
    return seal({"kind": "CPU_SHARED_ENTRY_DURABLE_REPLAY_ONLY", "fixture_only": True,
        "source_outcome_completed": successful, "durable_final_summary_present": final is not None,
        "final_returned_accounting_join_complete": final is not None,
        "markers": len(records), "operations": replayed["counts"], "raw_files": deepcopy(raw_files),
        "report_descriptor": deepcopy(report_descriptor), "whole_native_terminal_consumer_qualified": False,
        "resource_Job_death_or_library_evidence_validated": False, "native_PASS": False,
        "native_authority_issued": False, "native_launch_authorized": False, "scoring_authorized": False})


def _audit_body(preserved_directory, *, original_directory, manifest, expected_manifest_sha256, source_pins,
                expected_preservation_file_sha256, expected_preservation_body_sha256, mode="CPU"):
    require(mode == "CPU" or _direct_package(), "native evidence cannot be audited from ignored stage")
    _manifest(manifest, expected_manifest_sha256, source_pins, mode)
    root = Path(preserved_directory).absolute()
    reader = Reader(root)
    original_root = Path(original_directory).absolute()
    require(original_root != root and original_root.resolve() == original_root, "externally selected original/copy directory pair")
    path = root / "PRESERVATION.json"
    meta = path.lstat()
    raw = reader.read(path, {"bytes": meta.st_size, "sha256": expected_preservation_file_sha256})
    preservation = parse(raw)
    fields(preservation, {"kind", "scope", "manifest_sha256", "raw_files", "source_files", "source_snapshot_files",
            "fixture_only", "native_authority_issued", "sha256"}, "preservation")
    checked_seal(preservation, expected_preservation_body_sha256)
    require(preservation["manifest_sha256"] == expected_manifest_sha256 and preservation["scope"] == manifest["scope"]
            and preservation["source_files"] == source_pins, "external manifest/source preservation joins")
    if mode == "CPU":
        require(preservation["kind"] == "CPU_PRESERVED_OWNED_TENSOR_SIMULATION" and preservation["fixture_only"] is True
                and preservation["native_authority_issued"] is False, "CPU preservation cannot issue native authority")
    names = set(preservation["raw_files"])
    require(("OWNED_TERMINAL.json" in names and names <= RAW_NAMES) if mode == "CPU" else RAW_NAMES <= names,
            "declared exact original raw population")
    require(all(type(name) is str and Path(name).name == name and name not in (".", "..") for name in names), "literal flat raw file names")
    original_roster = set(entry.name for entry in original_root.iterdir())
    preserved_roster = set(entry.name for entry in root.iterdir())
    require(original_roster == names and preserved_roster == names | {"PRESERVATION.json", "source_snapshots"}, "exact original/preserved raw rosters")
    require(type(source_pins) is dict and 0 < len(source_pins) <= 4096, "finite nonempty actual source map")
    for source, expected in source_pins.items():
        require(Path(source).suffix == ".py", "auditor source map reads Python source only")
        require(type(expected) is dict and set(expected) in ({"bytes", "sha256"}, {"bytes", "sha256", "role"}), "exact original source descriptor")
        reader.read(Path(source), {key: expected[key] for key in ("bytes", "sha256")}, owned=False)
    snapshots = preservation["source_snapshot_files"]
    require(type(snapshots) is dict and set(snapshots) == set(source_pins), "all original source snapshots preserved")
    snapshot_names = set()
    for source, snapshot in snapshots.items():
        fields(snapshot, {"path", "bytes", "sha256", "role"}, "source snapshot descriptor")
        wanted_name = hashlib.sha256(source.encode("utf-8")).hexdigest() + ".py"
        require(snapshot["path"] == "source_snapshots/" + wanted_name and snapshot["role"] == "python_source"
                and snapshot["bytes"] == source_pins[source]["bytes"] and snapshot["sha256"] == source_pins[source]["sha256"], "same original/source-snapshot bytes and exact literal roster")
        snapshot_names.add(wanted_name)
        reader.read(root / "source_snapshots" / wanted_name, {key: snapshot[key] for key in ("bytes", "sha256")})
    require(set(path.name for path in (root / "source_snapshots").iterdir()) == snapshot_names, "complete original snapshot roster")
    raw_files, data = {}, {}
    for name in sorted(names):
        raw_files[name] = reader.read(root / name, preservation["raw_files"][name], MAX_LEDGER if name.endswith(".jsonl") else MAX_JSON)
        require(reader.read(original_root / name, preservation["raw_files"][name], MAX_LEDGER if name.endswith(".jsonl") else MAX_JSON, owned=False) == raw_files[name], "exact original/preserved raw pair")
        if name in RAW_NAMES and not name.endswith(".jsonl"):
            data[name] = parse(raw_files[name])
    if mode == "CPU" and names != RAW_NAMES:
        terminal = data["OWNED_TERMINAL.json"]
        checked_seal(terminal)
        require(terminal["kind"] == "CPU_OWNED_TERMINAL_SIMULATION" and terminal["scope"] == CPU_SCOPE and
                terminal["status"] == "CPU_OWNED_TERMINAL_FAILED" and terminal["manifest_sha256"] == expected_manifest_sha256 and
                terminal["fixture_only"] is True and terminal["native_authority_issued"] is False,
                "missing core files retain failed CPU metadata only")
        require(terminal["worker_report"] is None and terminal["worker_report_sha256"] == digest(None) and
                (terminal["error"] is not None or terminal["secondary_errors"]), "honest pre-core admission/launch failure")
        reader.verify()
        require(set(path.name for path in original_root.iterdir()) == original_roster and
                set(path.name for path in root.iterdir()) == preserved_roster and
                set(path.name for path in (root / "source_snapshots").iterdir()) == snapshot_names,
                "partial failure original/copy/source rosters stable")
        return seal({"schema_version": 1, "kind": "INDEPENDENT_OWNED_TENSOR_EVIDENCE_AUDIT", "scope": CPU_SCOPE,
            "status": "CPU_AUDIT_FAILURE_PRESERVED", "manifest_sha256": expected_manifest_sha256,
            "missing_core_files": sorted(RAW_NAMES - names), "independent_replay_complete": False,
            "failed_metadata_preserved": True, "fixture_only": True, "native_authority_issued": False,
            "native_PASS": False, "native_fit_established": False, "scoring_authorized": False,
            "M0_membership": False, "cause_established": False})
    records = parse_ledger(raw_files["PHASE_OBSERVATIONS.jsonl"])
    prefix = _input(reader)
    successful, replayed = _core(data, records, manifest, prefix, mode)
    terminal = data["OWNED_TERMINAL.json"]
    require(terminal["kind"] == ("CPU_OWNED_TERMINAL_SIMULATION" if mode == "CPU" else "NATIVE_OWNED_TENSOR_TERMINAL")
            and terminal["status"] == (("CPU_OWNED_TERMINAL_COMPLETE" if successful else "CPU_OWNED_TERMINAL_FAILED") if mode == "CPU" else
                ("NATIVE_OWNED_TERMINAL_COMPLETE" if successful else "NATIVE_OWNED_TERMINAL_FAILED")), "truthful whole terminal outcome/kind")
    owner = _terminal(terminal, manifest, successful, replayed["first_clock"], replayed["last_clock"])
    require(data["PHASE_ACCOUNTING_FINAL.json"]["marker_bytes"] == len(raw_files["PHASE_OBSERVATIONS.jsonl"]), "exact durable ledger byte dose")
    reader.verify()
    require(set(entry.name for entry in root.iterdir()) == preserved_roster and
            set(entry.name for entry in original_root.iterdir()) == original_roster and
            set(path.name for path in (root / "source_snapshots").iterdir()) == snapshot_names, "final original/raw/source rosters unchanged")
    return seal({"schema_version": 1, "kind": "INDEPENDENT_OWNED_TENSOR_EVIDENCE_AUDIT",
        "scope": manifest["scope"], "status": ("CPU_AUDIT_COMPLETE" if successful else "CPU_AUDIT_FAILURE_PRESERVED") if mode == "CPU" else
            ("NATIVE_TENSOR_ONLY_TERMINAL_AUDIT_COMPLETE" if successful else "NATIVE_TENSOR_ONLY_FAILURE_PRESERVED"),
        "preservation_file_sha256": expected_preservation_file_sha256, "preservation_body_sha256": expected_preservation_body_sha256,
        "manifest_sha256": expected_manifest_sha256, "worker_identity": owner, "raw_files": deepcopy(preservation["raw_files"]),
        "actual_phase_counts": replayed["counts"], "independent_replay_complete": True, "fixture_only": mode == "CPU",
        "native_authority_issued": False, "native_PASS": False, "native_fit_established": False,
        "scoring_authorized": False, "M0_membership": False, "cause_established": False})


def audit_cpu(preserved_directory, *, original_directory, manifest, expected_manifest_sha256, source_pins,
              expected_preservation_file_sha256, expected_preservation_body_sha256):
    return _audit_body(preserved_directory, original_directory=original_directory, manifest=manifest, expected_manifest_sha256=expected_manifest_sha256,
        source_pins=source_pins, expected_preservation_file_sha256=expected_preservation_file_sha256,
        expected_preservation_body_sha256=expected_preservation_body_sha256, mode="CPU")


def _direct_package():
    return HERE.name == PACKAGE_NAME and HERE.parent.name == "experiments"


def native_audit(*args, **kwargs):
    require(_direct_package(), "ignored staged auditor has no native paths/probes/authority")
    return _integrated_native_audit(*args, **kwargs)


def _CPU_native_schema_capabilities():
    issued = set()
    class Capability:
        __slots__ = ()
    def mint():
        value = Capability()
        issued.add(value)
        return value
    def check(value):
        require(type(value) is Capability and value in issued, "private CPU native-schema fixture factory identity")
    return mint, check


_mint_CPU_native_schema_fixture, _check_CPU_native_schema_fixture = _CPU_native_schema_capabilities()


def _integrated_native_audit(preserved_directory, *, original_directory, config_path, expected_config_file, expected_config_body,
        registration_path, expected_registration_file, expected_registration_body,
        runtime_path, expected_runtime_file, expected_runtime_body, expected_head,
        expected_contract_source_sha256, expected_preservation_file_sha256, expected_preservation_body_sha256):
    require(_direct_package(), "native integration required before any evidence or live query")
    # This block is unreachable in ignored/preserved stages. A separately selected
    # direct-package contract checks current clean HEAD and the entire actual
    # source/runtime/cache/binary closure before any owned live-death query.
    import importlib.util
    import sys
    selected = Reader(Path(preserved_directory).absolute())
    contract_path = HERE / "prefix_tensor_contract.py"
    require(sha(expected_contract_source_sha256), "external contract source selection required")
    raw = selected.read(contract_path, {"bytes": contract_path.lstat().st_size,
        "sha256": expected_contract_source_sha256}, owned=False)
    spec = importlib.util.spec_from_file_location("owned_terminal_exact_native_contract", contract_path)
    contract = importlib.util.module_from_spec(spec)
    exec(compile(raw.decode("utf-8"), str(contract_path), "exec"), contract.__dict__)
    context = contract.validate_native_manifest(config_path, expected_config_file=expected_config_file,
        expected_config_body=expected_config_body, registration_path=registration_path,
        expected_registration_file=expected_registration_file, expected_registration_body=expected_registration_body,
        runtime_path=runtime_path, expected_runtime_file=expected_runtime_file,
        expected_runtime_body=expected_runtime_body, expected_head=expected_head)
    qualified = contract.context_value(context)
    manifest = qualified["config"]
    source_pins = {path: {"bytes": item["bytes"], "sha256": item["sha256"]} for path, item in manifest["source_files"].items()}
    require(source_pins[str(contract_path)]["sha256"] == expected_contract_source_sha256, "qualified actual contract source binding")
    require(str(Path(__file__).absolute()) in source_pins, "actual integrated auditor included in exact source closure")
    selected.read(Path(__file__).absolute(), source_pins[str(Path(__file__).absolute())], owned=False)
    result = _native_preserved_body(preserved_directory, original_directory=original_directory, manifest=manifest,
        source_pins=source_pins, expected_preservation_file_sha256=expected_preservation_file_sha256,
        expected_preservation_body_sha256=expected_preservation_body_sha256, expected_head=expected_head,
        anchors={"config_path": str(Path(config_path).absolute()), "anchors": qualified["anchors"]}, _selected_context=qualified)
    # Use the original independently birth-aware read-only death verifier, after
    # verifying its selected source and actual imported module binding. No signal,
    # kill, worker launch, model import, or tensor construction exists here.
    original_path = HERE / "native_supervisor.py"
    require(str(original_path) in source_pins, "original death verifier source qualification")
    original_raw = selected.read(original_path, source_pins[str(original_path)], owned=False)
    spec = importlib.util.spec_from_file_location("native_supervisor", original_path)
    original = importlib.util.module_from_spec(spec)
    prior = sys.modules.get("native_supervisor")
    if prior is not None:
        require(Path(prior.__file__).absolute() == original_path, "no original supervisor module alias")
    loader_path = HERE / "prefix_registered_loader.py"
    require(str(loader_path) in source_pins, "qualified source/cache loader present")
    loader_raw = selected.read(loader_path, source_pins[str(loader_path)], owned=False)
    loader_spec = importlib.util.spec_from_file_location("owned_audit_registered_source_loader", loader_path)
    loader = importlib.util.module_from_spec(loader_spec)
    exec(compile(loader_raw.decode("utf-8"), str(loader_path), "exec"), loader.__dict__)
    finder = loader.install(context, contract)
    try:
        exec(compile(original_raw.decode("utf-8"), str(original_path), "exec"), original.__dict__)
        require(Path(original.__file__).absolute() == original_path, "exact current death verifier origin")
        for owner in result["owned_process_identities"]:
            alive = original._pid_alive(deepcopy(owner))
            require(type(alive) is bool and alive is False, "actual current same PID/FILETIME owned birth dead")
        result["actual_audit_runtime_module_observations"] = deepcopy(finder.observed)
    finally:
        loader.remove(finder)
    # Recheck the qualified current head and all byte closures after live-query
    # callbacks; neither source nor runtime can be rebased to a later baseline.
    require(contract.context_value(context) == qualified and contract.revalidate_context(context) == qualified,
            "same original qualified native factory context after callbacks")
    for mapping in (manifest["source_files"], qualified["runtime"]["files"]):
        for path, descriptor in mapping.items():
            contract.file_descriptor(path, descriptor)
    selected.verify()
    result.update(current_exact_death_verified=True, actual_source_module_bindings_verified=True,
                  native_tensor_permission_verified=True)
    return seal(result)


def _library_clock_window_ns(resume,deadline):
    require(finite(resume) and finite(deadline) and resume<=deadline,'finite original owned clock window')
    lower_n,lower_d=math.nextafter(float(resume),-math.inf).as_integer_ratio()
    upper_n,upper_d=math.nextafter(float(deadline),math.inf).as_integer_ratio()
    return (lower_n*10**9)//lower_d,-((-upper_n*10**9)//upper_d)

def _audit_library_evidence_bytes(raw, *, selected_context, owner_receipt, executing_birth, reader):
    """Independent selected-source replay of the actual worker sidecar bytes.

    Reads and source maps must already be bound to the original qualified context.
    This validates recorded evidence; it neither observes DLLs nor grants authority.
    """
    value=selected_context;runtime=value['runtime'];manifest=value['config']
    require(type(runtime.get('library_boundary_plan')) is list and runtime['library_boundary_plan'],'independently selected library boundary plan required')
    policy=runtime['library_search_policy'].get('enforced_boundary')
    require(type(policy) is dict,'selected enforced library policy for terminal evidence')
    envelope=parse(raw)
    expected_fields={'schema_version','kind','source_commit','original_scope','context_sha256','external_anchors','policy_sha256','provider_source','owner_receipt_sha256','raw_packet','raw_packet_sha256','main_errors','fixture_only','native_PASS','native_authority_issued','sha256'}
    fields(envelope,expected_fields,'exact worker library evidence envelope')
    fixture=value.get('fixture_only',False)
    require(type(fixture) is bool and type(envelope['schema_version']) is int and envelope['schema_version']==1 and envelope['kind']=='OWNED_LIBRARY_BOUNDARY_EVIDENCE_V1','selected library evidence envelope version')
    require(envelope['fixture_only'] is fixture and envelope['native_PASS'] is envelope['native_authority_issued'] is False,'library evidence provenance cannot grant authority')
    checked_seal(envelope,envelope['sha256'])
    require(envelope['source_commit']==value['expected_head'] and envelope['original_scope']==NATIVE_SCOPE and envelope['context_sha256']==digest(value),'original selected source/scope/context library evidence join')
    require(envelope['external_anchors']==value['anchors'] and envelope['policy_sha256']==digest(policy) and envelope['provider_source']==policy['provider_source'],'original external anchors and policy/provider library joins')
    require(envelope['owner_receipt_sha256']==digest(owner_receipt),'original owner receipt library evidence join')
    packet=envelope['raw_packet'];require(type(packet) is dict and envelope['raw_packet_sha256']==digest(packet),'actual embedded raw packet byte-value digest')
    errors=envelope['main_errors'];require(type(errors) is list and len(errors)<=32,'bounded main library publication errors')
    for row in errors:
        fields(row,{'stage','error'},'typed ordered main error')
        require(row['stage'] in {'install','ownership','execute','owned_return','cleanup','export'},'known main error stage')
        error=row['error'];fields(error,{'type','message','message_exact'},'bounded original main error')
        require(type(error['type']) is str and 0<len(error['type'])<=256 and type(error['message_exact']) is bool and ((type(error['message']) is str and len(error['message'])<=4096) or (error['message'] is None and error['message_exact'] is False)),'plain bounded main error metadata')
    provider=policy['provider_source']
    worker_path=str(Path(value['config_path']).parent.parent/'prefix_input_owned_worker.py')
    worker_source=manifest['source_files'].get(worker_path)
    require(type(worker_source) is dict and worker_source.get('role')=='python_source','original selected worker evidence publication source')
    expected_identity={'source_commit':value['expected_head'],'original_scope':NATIVE_SCOPE,
        'owned_birth':executing_birth,'context_sha256':digest(value),'producer_source':{'path':worker_path,'descriptor':worker_source},
        'fixture_only':fixture,'native_PASS':False,'native_authority_issued':False,
        'owned_birth_status':'UNAVAILABLE' if executing_birth is None else 'VALIDATED',
        'owner_receipt_sha256':digest(owner_receipt)}
    if executing_birth is not None:identity(executing_birth)
    from types import ModuleType
    path=Path(value['config_path']).parent.parent/'prefix_library_evidence.py'
    entry=manifest['source_files'].get(str(path));require(type(entry) is dict and entry.get('role')=='python_source','independently selected library evidence consumer source')
    source=reader.read(path,{k:entry[k] for k in ('bytes','sha256')},owned=False)
    require(hashlib.sha256(source).hexdigest()==entry['sha256'],'exact independent library consumer source bytes')
    consumer=ModuleType('terminal_selected_library_evidence');consumer.__file__=str(path)
    exec(compile(source.decode('utf-8'),str(path),'exec'),consumer.__dict__)
    packet_bytes=(canonical(packet)+'\n').encode('ascii')
    replay=consumer.validate_observed_packet(packet_bytes,policy=policy,expected_identity=expected_identity,
        selected_file_map=policy['library_files'],selected_source_map=manifest['source_files'],
        expected_boundaries=runtime['library_boundary_plan'],expected_file_sha256=hashlib.sha256(packet_bytes).hexdigest())
    require(replay['native_PASS'] is replay['native_authority_issued'] is False and replay['fixture_only'] is fixture,'independent library replay preserves original provenance')
    return {'replay':replay,'main_errors':deepcopy(errors),'envelope_sha256':envelope['sha256'],
        'raw_packet_sha256':envelope['raw_packet_sha256'],'consumer_source_sha256':entry['sha256'],
        'checkpoint_clocks_ns':[row['time_ns'] for row in packet['observations'] if type(row['time_ns']) is int],
        'conservation_complete':replay['conservation_complete'] is True and not errors,
        'actual_packet_bytes_consumed':True,'native_observer_qualified':False,
        'native_PASS':False,'native_authority_issued':False}

def _native_preserved_body(preserved_directory, *, original_directory, manifest, source_pins,
                          expected_preservation_file_sha256, expected_preservation_body_sha256,
                          expected_head, anchors, _CPU_fixture_capability=None, _selected_context=None):
    """Adapt real original supervisor/lease/resource receipts, never CPU flags."""
    require(_direct_package(), "native adapter requires reviewed direct-package integration")
    if _CPU_fixture_capability is not None:
        _check_CPU_native_schema_fixture(_CPU_fixture_capability)
    root, original = Path(preserved_directory).absolute(), Path(original_directory).absolute()
    require(root != original, "separate exact native original/preserved directories")
    reader = Reader(root)
    preservation_path = root / "PRESERVATION.json"
    preservation = parse(reader.read(preservation_path, {"bytes": preservation_path.lstat().st_size,
        "sha256": expected_preservation_file_sha256}))
    fields(preservation, {"kind", "scope", "manifest_sha256", "raw_files", "source_files", "source_snapshot_files",
            "lease_files", "current_death_observation", "fixture_only", "native_authority_issued", "sha256"}, "native preservation")
    checked_seal(preservation, expected_preservation_body_sha256)
    require(preservation["kind"] == "PRESERVED_REGISTERED_ONE_PREFIX_NATIVE_TENSOR_DIAGNOSTIC" and
            preservation["scope"] == NATIVE_SCOPE and preservation["manifest_sha256"] == manifest["sha256"] and
            preservation["fixture_only"] is (_CPU_fixture_capability is not None) and preservation["native_authority_issued"] is False,
            "actual selected native preservation is not permission")
    require(preservation["source_files"] == manifest["source_files"], "exact actual native original source inventory")
    required = set(RAW_CORE) | {"OWNED_WORKER_RETURN.json", "OWNED_EXTERNAL_ANCHORS.json", "SUPERVISOR_RECEIPT.json",
                                "RESOURCE_OBSERVATIONS.json", "supervisor_ownership.json"}
    names = set(preservation["raw_files"])
    require(required <= names and all(Path(name).name == name and name not in (".", "..") for name in names), "all original native supervisor/core/log raw files")
    initial_original = set(path.name for path in original.iterdir())
    require(initial_original == names, "native original entire raw roster")
    raw, data = {}, {}
    for name, descriptor in preservation["raw_files"].items():
        raw[name] = reader.read(root / name, descriptor, MAX_LEDGER if name.endswith(".jsonl") else MAX_JSON)
        require(reader.read(original / name, descriptor, MAX_LEDGER if name.endswith(".jsonl") else MAX_JSON, owned=False) == raw[name], "native all-original/all-copy byte equality")
        if name.endswith(".json"):
            data[name] = parse(raw[name])
    require(data["OWNED_EXTERNAL_ANCHORS.json"] == anchors, "actual original native selected external anchors")
    snapshots = preservation["source_snapshot_files"]
    require(set(snapshots) == set(source_pins), "complete native source snapshot population")
    snapshot_names = set()
    for source, descriptor in source_pins.items():
        reader.read(Path(source), {key:descriptor[key] for key in ('bytes','sha256')}, owned=False)
        snapshot = snapshots[source]
        name = hashlib.sha256(source.encode("utf-8")).hexdigest() + Path(source).suffix
        fields(snapshot, {"path", "bytes", "sha256", "role"}, "native source snapshot")
        require(snapshot["path"] == "source_snapshots/" + name and snapshot["bytes"] == descriptor["bytes"] and
                snapshot["sha256"] == descriptor["sha256"] and snapshot["role"] == manifest["source_files"][source]["role"], "native original/source-copy byte identity")
        reader.read(root / "source_snapshots" / name, {key: snapshot[key] for key in ("bytes", "sha256")})
        snapshot_names.add(name)
    require(set(path.name for path in (root / "source_snapshots").iterdir()) == snapshot_names, "native exact source copy roster")
    supervisor = data["SUPERVISOR_RECEIPT.json"]
    require(supervisor["config_sha256"] == anchors["anchors"]["expected_config_file"] and
            supervisor["preflight"]["source_commit"] == expected_head and supervisor["preflight"]["resources"] == RESOURCES,
            "original supervisor selected config/source/resource binding")
    require(supervisor["worker_attempted"] is supervisor["worker_launched"] is supervisor["worker_resumed"] is True and
            supervisor["preflight_revalidated_before_current_observations"] is True and
            supervisor["job_closed_kill_on_close"] is supervisor["sampler_threads_terminal_verified"] is
            supervisor["gpu_telemetry_cleanup_verified"] is supervisor["partial_outputs_frozen"] is True and
            supervisor["owned_worker_identity_currently_alive"] is False and supervisor["unjoined_sampler_threads"] == [],
            "actual original owned launch/containment/samplers/NVML/frozen cleanup")
    require(finite(supervisor["elapsed_seconds"]) and supervisor["elapsed_seconds"] <= 600,
            "actual original native whole-scope deadline")
    admission = supervisor["admission_observations"]
    require(type(admission) is list and len(admission) == 2, "two original actual admission observations")
    for sample in admission + [supervisor["dispatch_observation"]]:
        require(all(integer(sample[key]) for key in ("host_available_bytes", "worker_tree_rss_bytes", "disk_free_bytes", "gpu_free_bytes", "gpu_total_bytes")),
                "typed actual native admission byte amounts")
        require(sample["host_available_bytes"] >= 3.3125 * GIB and sample["worker_tree_rss_bytes"] <= 2 * GIB and
                sample["disk_free_bytes"] >= 4 * GIB and 4 * GIB <= sample["gpu_free_bytes"] <= sample["gpu_total_bytes"],
                "original native admission CPU/GPU/disk floors")
        require(all(finite(sample[key]) for key in ("sample_started_monotonic", "sample_completed_monotonic", "gpu_sample_started_monotonic", "gpu_sample_completed_monotonic")) and
                sample["sample_started_monotonic"] <= sample["sample_completed_monotonic"] and
                sample["gpu_sample_started_monotonic"] <= sample["gpu_sample_completed_monotonic"], "actual ordered native admission sample clocks")
        require(type(sample["GPU_index"]) is int and sample["GPU_index"] == 0 and type(sample["gpu_uuid"]) is str and sample["gpu_uuid"],
                "actual same-device admission identity")
    require(admission[1]["sample_started_monotonic"] - admission[0]["sample_started_monotonic"] >= 2 and
            admission[1]["gpu_sample_started_monotonic"] > admission[0]["gpu_sample_started_monotonic"] and
            admission[1]["gpu_observation_id"] != admission[0]["gpu_observation_id"] and
            admission[0]["gpu_uuid"] == admission[1]["gpu_uuid"] == supervisor["dispatch_observation"]["gpu_uuid"],
            "distinct actual original two-second same-GPU admissions")
    dispatch = supervisor["dispatch_observation"]
    resume_clock = supervisor["worker_resume_monotonic"]
    require(finite(resume_clock) and dispatch["sample_started_monotonic"] <= resume_clock and
            dispatch["gpu_sample_started_monotonic"] <= resume_clock and
            dispatch["sample_started_monotonic"] - admission[0]["sample_started_monotonic"] <= 30,
            "original actual dispatch ordering and admission deadline")
    owner_raw = supervisor["worker_identity"]
    owner = identity({key: owner_raw[key] for key in ("pid", "creation_filetime_100ns")})
    require(owner_raw["assigned_before_resume"] is owner_raw["job_kill_on_close"] is True and owner_raw["breakaway_allowed"] is False,
            "actual suspended worker assigned before owned resume")
    require(data["supervisor_ownership.json"]["worker_pid"] == owner["pid"] and
            data["supervisor_ownership.json"]["job_name"] == owner_raw["job_name"] and
            data["supervisor_ownership.json"]["config_sha256"] == supervisor["config_sha256"], "same original Job/worker/config ownership")
    lease = preservation["lease_files"]
    require(set(lease) == {"reservation.json", "worker.json", "resume.json", "terminal.json"}, "complete original reservation/worker/resume/terminal lease population")
    lease_data, lease_copy_names = {}, set()
    for name, descriptor in lease.items():
        fields(descriptor, {"source_path", "path", "bytes", "sha256"}, "native lease copy descriptor")
        require(descriptor["source_path"] == str(Path(supervisor["lease_path"]) / name) and descriptor["path"] == "lease_" + name,
                "original exact lease pointer/copy name")
        expected = {key: descriptor[key] for key in ("bytes", "sha256")}
        value = reader.read(root / descriptor["path"], expected)
        require(reader.read(Path(descriptor["source_path"]), expected, owned=False) == value, "actual original/preserved lease bytes")
        lease_data[name] = parse(value)
        lease_copy_names.add(descriptor["path"])
    require(lease_data["terminal.json"] == supervisor and lease_data["worker.json"] == owner_raw and
            lease_data["resume.json"]["worker_identity"] == owner_raw and lease_data["resume.json"]["external_monitor_ready_before_resume"] is True and
            lease_data["resume.json"]["resumed_monotonic"] == supervisor["worker_resume_monotonic"], "actual lease terminal/birth/resume/monitor joins")
    require(lease_data["reservation.json"]["config_sha256"] == supervisor["config_sha256"], "same original reservation configuration")
    container = data["OWNED_WORKER_RETURN.json"]
    fields(container, {"report", "runtime_execution_modules", "owned_membership_verified", "IO", "native_PASS"}, "actual native returned caller container")
    require(container["native_PASS"] is False, "returned worker container cannot issue native audit approval")
    membership = container["owned_membership_verified"]
    fields(membership, {"actual_worker_pid", "actual_parent_pid", "launcher_root_pid", "supervisor_pid", "job_name",
        "owned_job_membership_verified", "query_job_handle_closed_before_framework_import"}, "actual worker Job membership")
    require(all(integer(membership[key], 1) for key in ("actual_worker_pid", "actual_parent_pid", "launcher_root_pid", "supervisor_pid")) and
            membership["launcher_root_pid"] == owner["pid"] and membership["supervisor_pid"] == data["supervisor_ownership.json"]["supervisor_pid"] and
            membership["job_name"] == owner_raw["job_name"] and membership["owned_job_membership_verified"] is
            membership["query_job_handle_closed_before_framework_import"] is True and container["IO"] == {"owned_return_write":
            {"attempted": 1, "returned": 1, "failed": 0, "pending": 0}}, "actual returned caller membership/durable IO")
    execution = container["runtime_execution_modules"]
    require(type(execution) is dict and execution, "actual registered runtime module execution observations")
    actual_known = {**manifest["runtime_files"], **manifest["source_files"]}
    for module_name, observed in execution.items():
        require(type(module_name) is str and type(observed) is dict and observed["path"] in actual_known and
                observed["sha256"] == actual_known[observed["path"]]["sha256"], "actual consumed registered runtime source/binary identities")
    report = container["report"]
    data["OWNED_TERMINAL.json"] = {"worker_report": report, "worker_report_sha256": digest(report)}
    records = parse_ledger(raw["PHASE_OBSERVATIONS.jsonl"])
    prefix = _input(reader)
    successful, replayed = _core(data, records, manifest, prefix, "NATIVE")
    require(value_of(replayed, "worker_ownership_verify") == membership,
            "actual original ownership verifier projection equals returned typed membership")
    require(not successful or supervisor["status"] == "SUPERVISED_WORKER_EXITED_ZERO" and supervisor["returncode"] == 0,
            "actual successful native worker exited zero under original supervisor")
    require(finite(supervisor["worker_resume_monotonic"]) and finite(supervisor["deadline_monotonic"]) and
            supervisor["worker_resume_monotonic"] <= replayed["first_clock"] <= replayed["last_clock"] <= supervisor["deadline_monotonic"],
            "actual native phase work inside original resume/deadline")
    observations = data["RESOURCE_OBSERVATIONS.json"]
    require(type(observations) is list and observations, "actual complete original resource sampler observations")
    births = {(owner["pid"], owner["creation_filetime_100ns"]): owner}
    clocks = {"CPU": [], "GPU": []}
    for sample in observations:
        require(type(sample) is dict and sample.get("kind") in clocks, "actual CPU/GPU sampler kind")
        domain = sample["kind"]
        clock = sample["sample_started_monotonic"] if domain == "CPU" else sample["gpu_sample_started_monotonic"]
        require(finite(clock) and (not clocks[domain] or clock >= clocks[domain][-1]), "native chronological sampler clock")
        clocks[domain].append(clock)
        if domain == "CPU":
            require(integer(sample["host_available_bytes"]) and integer(sample["worker_tree_rss_bytes"]) and
                    sample["host_available_bytes"] >= 1.25 * GIB and sample["worker_tree_rss_bytes"] <= 2 * GIB, "actual native CPU reserve/tree RSS")
            for found in sample["owned_process_identities"]:
                found = identity({key: found[key] for key in ("pid", "creation_filetime_100ns")})
                births[(found["pid"], found["creation_filetime_100ns"])] = found
        else:
            require(integer(sample["gpu_free_bytes"]) and integer(sample["gpu_total_bytes"]) and
                    .5 * GIB <= sample["gpu_free_bytes"] <= sample["gpu_total_bytes"] and sample["gpu_uuid"] == admission[0]["gpu_uuid"],
                    "actual native same-GPU during-work floor")
    for domain, age in (("CPU", .2), ("GPU", 2.0)):
        admission_clocks = [sample["sample_started_monotonic"] if domain == "CPU" else sample["gpu_sample_started_monotonic"]
                            for sample in admission + [dispatch]]
        before_resume = [clock for clock in clocks[domain] + admission_clocks if clock <= supervisor["worker_resume_monotonic"]]
        require(before_resume and supervisor["worker_resume_monotonic"] - max(before_resume) <= age,
                "actual current pre-resume monitor/admission observation")
        require(clocks[domain] and any(replayed["first_clock"] <= clock <= replayed["last_clock"] for clock in clocks[domain]),
                "actual native during-work sampling coverage")
        during = [clock for clock in clocks[domain] if supervisor["worker_resume_monotonic"] <= clock <= supervisor["deadline_monotonic"]]
        require(during and during[0] - supervisor["worker_resume_monotonic"] <= age and
                replayed["last_clock"] - max(clock for clock in clocks[domain] if clock <= replayed["last_clock"]) <= age,
                "native resume/work-end sampling edges")
        require(all(right - left <= age for left, right in zip(during, during[1:])), "native original sampler cadence")
    require(membership["actual_worker_pid"] in {found["pid"] for found in births.values()}, "actual interpreter birth present in complete owned Job sampler roster")
    death=preservation['current_death_observation']
    fields(death,{'observed_monotonic','observed_utc','supervisor_file_sha256','config_file_sha256','observations','death_verifier_source_sha256'},'preservation current death observation')
    require(finite(death['observed_monotonic']) and death['observed_monotonic']>=replayed['last_clock'] and type(death['observed_utc']) is str and len(death['observed_utc'])<=64,'bounded finite actual preservation death time')
    from datetime import datetime
    require(datetime.fromisoformat(death['observed_utc']).utcoffset() is not None,'explicit timezone-qualified death observation')
    require(death['supervisor_file_sha256']==preservation['raw_files']['SUPERVISOR_RECEIPT.json']['sha256'] and death['config_file_sha256']==supervisor['config_sha256'] and death['death_verifier_source_sha256']==manifest['helper_pins']['native_supervisor.py'],'actual preservation verifier/SUP/config source joins')
    require(type(death['observations']) is list and len(death['observations'])==len(births),'complete preserved current all-birth death observations')
    observed={}
    for item in death['observations']:
        fields(item,{'identity','currently_alive'},'preserved current owned birth');found=identity(item['identity']);key=(found['pid'],found['creation_filetime_100ns'])
        require(key not in observed and item['currently_alive'] is False,'unique actually dead preserved birth');observed[key]=found
    require(observed==births,'no omitted/substituted descendant preservation death proof')
    require(data["PHASE_ACCOUNTING_FINAL.json"]["marker_bytes"] == len(raw["PHASE_OBSERVATIONS.jsonl"]), "actual native original ledger byte dose")
    wanted_preserved = names | lease_copy_names | {"PRESERVATION.json", "source_snapshots"}
    require(set(path.name for path in root.iterdir()) == wanted_preserved, "exact full native preserved raw/lease/source roster")
    reader.verify()
    require(set(path.name for path in original.iterdir()) == initial_original and
            set(path.name for path in root.iterdir()) == wanted_preserved and
            set(path.name for path in (root / "source_snapshots").iterdir()) == snapshot_names, "native final whole-original/copy/source roster conservation")
    library_evidence=None
    if _selected_context is not None and 'library_boundary_plan' in _selected_context['runtime']:
        require('LIBRARY_BOUNDARY_EVIDENCE.json' in raw,'actual worker library raw evidence file required')
        actual_pid=membership['actual_worker_pid'];matching=[]
        for observed in execution.values():
            topology=observed.get('topology') if type(observed) is dict else None
            if type(topology) is dict and topology.get('membership')==membership:
                for born in topology.get('owned_birth_roster',[]):
                    if type(born) is dict and born.get('pid')==actual_pid:
                        matching.append(identity({k:born[k] for k in ('pid','creation_filetime_100ns')}))
        require(matching and all(born==matching[0] for born in matching),'actual executing interpreter birth in original topology')
        executing_birth=matching[0]
        require((executing_birth['pid'],executing_birth['creation_filetime_100ns']) in births,'library observer executing birth in independent owned terminal population')
        library_evidence=_audit_library_evidence_bytes(raw['LIBRARY_BOUNDARY_EVIDENCE.json'],selected_context=_selected_context,
            owner_receipt=data['supervisor_ownership.json'],executing_birth=executing_birth,reader=reader)
        # The same system-wide monotonic clock is represented as integer ns
        # by the guard and float seconds by the original supervisor. Widen only
        # by the adjacent representable float interval, never a time allowance.
        lower,upper=_library_clock_window_ns(supervisor['worker_resume_monotonic'],supervisor['deadline_monotonic'])
        require(all(lower<=clock<=upper for clock in library_evidence['checkpoint_clocks_ns']),'library raw checkpoints inside original owned resume/deadline')
        library_evidence['original_owned_clock_window_joined']=True
        require(not successful or library_evidence['conservation_complete'],'successful tensor scope requires complete library evidence conservation')
    else:require(_CPU_fixture_capability is not None,'native terminal needs original context and external library boundary plan')
    reader.verify()
    fixture = _CPU_fixture_capability is not None
    return seal({"schema_version": 1, "kind": "CPU_NATIVE_SCHEMA_ORCHESTRATION_TEST" if fixture else "INDEPENDENT_REGISTERED_NATIVE_TENSOR_TERMINAL_AUDIT",
        "scope": "CPU_NATIVE_SCHEMA_ORCHESTRATION_TEST" if fixture else NATIVE_SCOPE,
        "status": ("CPU_NATIVE_SCHEMA_COMPLETE" if successful else "CPU_NATIVE_SCHEMA_FAILURE_PRESERVED") if fixture else
            ("NATIVE_TENSOR_ONLY_TERMINAL_AUDIT_COMPLETE" if successful else "NATIVE_TENSOR_ONLY_FAILURE_PRESERVED"),
        "worker_identity": owner, "owned_process_identities": list(births.values()), "raw_files": preservation["raw_files"],
        "manifest_sha256": manifest["sha256"], "preservation_file_sha256": expected_preservation_file_sha256,
        "preservation_body_sha256": expected_preservation_body_sha256, "actual_phase_counts": replayed["counts"],
        "native_authority_issued": False, "native_PASS": False, "native_fit_established": False, "scoring_authorized": False,
        "M0_membership": False, "cause_established": False, "fixture_only": fixture, "library_evidence":library_evidence,
        "library_terminal_evidence_join_complete":library_evidence is not None and library_evidence['actual_packet_bytes_consumed']})

def _CPU_integrated_native_audit_fixture(preserved_directory,*,original_directory,contract,context,_capability,death_verifier,expected_preservation_file_sha256,expected_preservation_body_sha256):
    """Actual outer qualification/read/replay body; explicit CPU capability only.

    The current-host byte and live-death boundaries are injected by the isolated
    fixture environment. Selected JSON, source/runtime files, snapshots, raw and
    lease readsets remain the actual independent auditor implementation.
    """
    contract._check_CPU_native_context_fixture(_capability);value=contract.context_value(context)
    require(value.get('fixture_only') is True and value.get('native_PASS') is False and value.get('native_authority_issued') is False,'sealed CPU outer context only')
    original=deepcopy(value);selected=Reader(Path(preserved_directory).absolute())
    manifest=value['config'];source_pins=manifest['source_files']
    called=globals().get('__compiled_source_descriptor__');wanted=source_pins[str(contract.PACKAGE/'prefix_input_owned_auditor.py')]
    require(type(called) is dict and called.get('bytes')==wanted['bytes'] and called.get('sha256')==wanted['sha256'] and Path(__file__).absolute()==contract.PACKAGE/'prefix_input_owned_auditor.py','actual selected auditor compiled bytes/origin')
    for filename in ('prefix_tensor_contract.py','prefix_input_owned_auditor.py'):
        path=contract.PACKAGE/filename;entry=source_pins[str(path)];selected.read(path,{k:entry[k] for k in ('bytes','sha256')},owned=False)
    fresh=contract._native_manifest_value(value['config_path'],**value['anchors'])
    for key in ('fixture_only','native_PASS','native_authority_issued','status'):original.pop(key,None)
    require(fresh==original,'actual original outer anchors/schema/factory source identity')
    source_code=selected.read(contract.PACKAGE/'prefix_input_owned_auditor.py',{k:source_pins[str(contract.PACKAGE/'prefix_input_owned_auditor.py')][k] for k in ('bytes','sha256')},owned=False)
    require(hashlib.sha256(source_code).hexdigest()==source_pins[str(contract.PACKAGE/'prefix_input_owned_auditor.py')]['sha256'],'selected actual auditor implementation bytes')
    token=_mint_CPU_native_schema_fixture();result=_native_preserved_body(preserved_directory,original_directory=original_directory,manifest=manifest,source_pins=source_pins,expected_preservation_file_sha256=expected_preservation_file_sha256,expected_preservation_body_sha256=expected_preservation_body_sha256,expected_head=value['expected_head'],anchors={'config_path':value['config_path'],'anchors':value['anchors']},_CPU_fixture_capability=token,_selected_context=value)
    for owner in result['owned_process_identities']:require(death_verifier(deepcopy(owner)) is False,'inert current exact all-birth death boundary')
    require(contract.context_value(context)==value,'opaque CPU context cannot rebase')
    refreshed=contract._native_manifest_value(value['config_path'],**value['anchors']);require(refreshed==fresh,'final unchanged actual outer qualification/readset')
    selected.verify();body_status=result['status'];result.update(kind='CPU_INTEGRATED_NATIVE_SOURCE_WORKFLOW_TEST',scope='CPU_INTEGRATED_NATIVE_SOURCE_WORKFLOW_TEST',source_body_status=body_status,status='CPU_INTEGRATED_SOURCE_WORKFLOW_COMPLETE' if body_status=='CPU_NATIVE_SCHEMA_COMPLETE' else 'CPU_INTEGRATED_SOURCE_FAILURE_PRESERVED',fixture_only=True,native_PASS=False,native_authority_issued=False,current_exact_death_fixture_verified=True)
    return seal(result)
