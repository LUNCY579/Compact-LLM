"""Registered short greedy capture controls. Stdlib only; no model executor."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import subprocess

CONFIG_RELATIVE = "configs/CAPTURE_GREEDY_SHORT_QUAL_V1.json"
CONFIG_SHA256 = "cab513b8006a8f4212be2e3f8fe1d9862abca96a53ab58ab033165e395a82b27"
PASS_STATUS = "PASS_FROZEN_GREEDY_SHORT_CAPTURE_SCOPED"
PACKAGE = Path(__file__).resolve().parents[1]
NEW_SOURCE_FILES = tuple("experiments/post_training_mimo_centralized_v1/scripts/" + name + ".py"
                         for name in ("greedy_capture", "qualify_capture", "supervise_capture"))
CPU_TEST_SOURCE = "experiments/post_training_mimo_centralized_v1/tests/test_greedy_capture.py"

def require(value, message):
    if not value:
        raise RuntimeError(message)

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")

def token_hash(ids):
    require(type(ids) is list and all(type(token) is int and 0 <= token < 151936 for token in ids), "invalid token IDs")
    return hashlib.sha256(canonical(ids)).hexdigest()

def file_identity(path):
    path = Path(path)
    before = path.stat(); h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "source changed during hash")
    return {"sha256": h.hexdigest(), "bytes": after.st_size}

def parse(raw):
    def unique(pairs):
        result = {}
        for name, value in pairs:
            require(name not in result, "duplicate JSON key")
            result[name] = value
        return result
    def invalid(value):
        raise RuntimeError("nonfinite JSON " + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)

def registered_config():
    raw = (PACKAGE / CONFIG_RELATIVE).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == CONFIG_SHA256, "frozen capture config changed")
    return parse(raw)

def validate_config(cfg):
    require(type(cfg) is dict and canonical(cfg) == canonical(registered_config()), "unregistered capture config")

def load_config(path):
    path = Path(path).resolve()
    require(path == (PACKAGE / CONFIG_RELATIVE).resolve(), "only registered capture config path")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == CONFIG_SHA256, "capture config byte identity")
    result = parse(raw); validate_config(result)
    return result

def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value) + b"\n")

def source_snapshot(root, cfg):
    """Bind current/committed sources before any reused/native imports."""
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=15).strip()
    require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True, timeout=15).strip(),
            "commit all source before native capture")
    pins = dict(cfg["reused_source_files"])
    for relative in (*NEW_SOURCE_FILES, CPU_TEST_SOURCE, "experiments/post_training_mimo_centralized_v1/" + CONFIG_RELATIVE):
        pins[relative] = file_identity(root / relative)
    for relative, pin in pins.items():
        require(file_identity(root / relative) == pin, "capture source drift: " + relative)
        historical = subprocess.run(["git", "show", head + ":" + relative], cwd=root, capture_output=True, timeout=15, check=False)
        require(historical.returncode == 0 and {"sha256": hashlib.sha256(historical.stdout).hexdigest(), "bytes": len(historical.stdout)} == pin,
                "uncommitted capture dependency: " + relative)
    installed = cfg["installed_runtime_source_files"]
    for path, pin in installed.items():
        require(file_identity(path) == pin, "installed runtime source drift")
    return {"code_sha": head, "dirty": False, "source_files": pins, "installed_runtime_source_files": installed}

def recheck_sources(root, snapshot):
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=15).strip() == snapshot["code_sha"], "capture HEAD changed")
    for relative, pin in snapshot["source_files"].items():
        require(file_identity(root / relative) == pin, "capture source changed: " + relative)
    for path, pin in snapshot["installed_runtime_source_files"].items():
        require(file_identity(path) == pin, "installed runtime source changed")

def admission_required(cfg, rss_gib):
    require(type(rss_gib) in (int, float) and math.isfinite(rss_gib) and rss_gib >= 0, "invalid worker RSS")
    return cfg["host_free_gib"] + max(0.0, cfg["worker_rss_limit_gib"] - rss_gib)

def greedy_index(scores):
    """CPU control for the raw finite argmax; ties keep the lowest token ID."""
    require(type(scores) is list and bool(scores) and all(type(value) in (int, float) and math.isfinite(value) for value in scores), "nonfinite/invalid logits")
    return max(range(len(scores)), key=lambda index: scores[index])

def validate_choice(token, selected_logit, maximum_logit, first_maximum_token):
    require(type(token) is int and 0 <= token < 151936 and type(first_maximum_token) is int, "invalid greedy choice")
    require(all(type(value) in (int, float) and math.isfinite(value) for value in (selected_logit, maximum_logit)), "nonfinite greedy logits")
    require(token == first_maximum_token and selected_logit == maximum_logit, "choice is not verified raw lowest-ID argmax")
    return 0.0  # Deterministic point-mass behavior, distinct from model logprob.

def validate_sequence(ids, cap, eos_ids):
    token_hash(ids)
    require(type(cap) is int and cap > 0 and 1 <= len(ids) <= cap, "invalid generation length")
    require(not any(token in eos_ids for token in ids[:-1]), "tokens after first EOS")
    stop = "eos" if ids[-1] in eos_ids else "length"
    require(stop == "eos" or len(ids) == cap, "generation stopped before EOS/cap")
    return stop

def termination_after_emit(prompt_ids, ids, cap, eos_ids):
    token_hash(prompt_ids); token_hash(ids)
    require(type(cap) is int and cap > 0 and 1 <= len(ids) <= cap, "invalid emission count")
    require(not any(token in eos_ids for token in ids[:-1]), "tokens emitted after first EOS")
    return "eos" if ids[-1] in eos_ids else ("length" if len(ids) == cap else None)

def render_response(codec, ids, cap, eos_ids):
    stop = validate_sequence(ids, cap, eos_ids)
    visible_ids = ids[:-1] if stop == "eos" else ids[:]
    kwargs = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    raw, visible = codec.decode(ids, **kwargs), codec.decode(visible_ids, **kwargs)
    require(type(raw) is str and type(visible) is str, "decoder must return exact text")
    return {"stop_reason": stop, "terminal_eos_id": ids[-1] if stop == "eos" else None,
            "visible_ids": visible_ids, "raw_response": raw, "visible_response": visible,
            "raw_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "visible_response_sha256": hashlib.sha256(visible.encode("utf-8")).hexdigest()}

def offline_indices(generated_length, logits_rows):
    require(type(generated_length) is int and 1 <= generated_length <= 128 and type(logits_rows) is int
            and logits_rows == generated_length + 1, "offline logits must be exactly N+1")
    return list(range(generated_length))  # First row predicts first generated ID.

def likelihood_measurement(online, offline, maximum=.03125, mean=.01):
    require(type(online) is list and type(offline) is list and len(online) == len(offline) and bool(online), "likelihood lengths differ")
    require(all(type(value) in (int, float) for value in online + offline), "invalid logprob dtype")
    finite = all(math.isfinite(value) and value <= 0 for value in online + offline)
    errors = [abs(first - second) for first, second in zip(online, offline)] if finite else None
    max_abs, mean_abs = (max(errors), sum(errors) / len(errors)) if finite else (None, None)
    return {"finite": finite, "absolute_errors": errors, "max_abs": max_abs, "mean_abs": mean_abs,
            "passed": bool(finite and max_abs <= maximum and mean_abs <= mean)}

def safe_numbers(value):
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, dict): return {name: safe_numbers(item) for name, item in value.items()}
    if isinstance(value, list): return [safe_numbers(item) for item in value]
    return value

def expected_counts(first, second):
    require(all(type(value) is int and 1 <= value <= 128 for value in (first, second)), "invalid actual rollout lengths")
    positions = 1280 + 129 * first + 385 * second + first * (first - 1) // 2 + second * (second - 1) // 2
    return {"generated_tokens": first + second, "forward_calls": 4 + first + second,
            "logical_calls": 316 + first + second, "forward_input_positions": positions,
            "resource_samples": 7 + first + second, "journal_records": 645 + 4 * (first + second)}

def run_cpu_controls():
    """Actual computed synthetic controls; these are never native observations."""
    cases = []
    class Codec:
        def decode(self, ids, **kwargs):
            require(kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}, "decode options drift")
            return "".join({7: " \n", 8: "<other-special>", 9: " tail ", 151645: "<|im_end|>", 151643: "<|endoftext|>"}.get(token, str(token)) for token in ids)
    def case(name, inputs, observed, expected):
        passed = canonical(observed) == canonical(expected)
        cases.append({"case_id": name, "designation": "SYNTHETIC_MECHANICS_ONLY_NOT_NATIVE_OR_FORCED_MODEL_EOS",
                      "input": inputs, "output": observed, "expected": expected, "passed": passed,
                      "input_sha256": hashlib.sha256(canonical(inputs)).hexdigest(),
                      "output_sha256": hashlib.sha256(canonical(observed)).hexdigest()})
        require(passed, "CPU capture control failed: " + name)
    scores = [-2.0, 4.0, 4.0]
    case("raw_greedy_tie_lowest_id", {"scores": scores}, greedy_index(scores), 1)
    eos_ids = [151645, 151643]
    for eos in eos_ids:
        inputs = {"prompt_ids": [eos, 10], "generated_ids": [7, 8, 9, eos], "cap": 128, "eos_ids": eos_ids}
        observed = render_response(Codec(), inputs["generated_ids"], 128, eos_ids)
        case("terminal_only_render_" + str(eos), inputs,
             {"stop": termination_after_emit(inputs["prompt_ids"], inputs["generated_ids"], 128, eos_ids),
              "visible_ids": observed["visible_ids"], "raw": observed["raw_response"], "visible": observed["visible_response"]},
             {"stop": "eos", "visible_ids": [7, 8, 9], "raw": " \n<other-special> tail " + ("<|im_end|>" if eos == 151645 else "<|endoftext|>"),
              "visible": " \n<other-special> tail "})
    case("prompt_EOS_does_not_stop_new_nonEOS", {"prompt_ids": [151645], "generated_ids": [7], "cap": 128},
         termination_after_emit([151645], [7], 128, eos_ids), None)
    case("EOS_at_cap_has_EOS_priority", {"generated_ids": [7] * 127 + [151643], "cap": 128},
         termination_after_emit([151645], [7] * 127 + [151643], 128, eos_ids), "eos")
    case("exact_cap_stop_without_EOS", {"generated_ids": [7] * 128, "cap": 128},
         termination_after_emit([151645], [7] * 128, 128, eos_ids), "length")
    case("offline_Nplus1_firstN_shift", {"generated_length": 2, "kept_logits_rows": 3}, offline_indices(2, 3), [0, 1])
    case("verified_greedy_behavior_zero_is_not_model_logprob", {"token_id": 1, "selected": 4.0, "maximum": 4.0, "first_maximum": 1,
                                                               "model_logprob": -.5}, validate_choice(1, 4.0, 4.0, 1), 0.0)
    return {"status": "PASS_CPU_CAPTURE_MECHANICS_ONLY", "cases": cases,
            "native_EOS_observations": [], "model_calls": 0, "training_authorized": False, "E1_authorized": False}

class BudgetedCalls:
    def __init__(self, accounting, cfg):
        self.accounting, self.ceilings = accounting, cfg["ceilings"]
    def call(self, operation, function, *, stage, input_positions=0, training=False):
        require(operation in ("model_construct", "checkpoint_load", "tensor_copy", "forward") and training is False, "capture permits no training/mutation calls")
        snapshot = self.accounting.snapshot()
        require(snapshot["last_call_id"] < self.ceilings["logical_calls"], "CAPTURE_LOGICAL_CALL_CEILING")
        if operation == "forward":
            require(snapshot["operations"]["forward"]["attempted"] < self.ceilings["forward_calls"], "CAPTURE_FORWARD_CEILING")
            require(snapshot["forward_input_positions_attempted"] + input_positions <= self.ceilings["forward_input_positions"], "CAPTURE_POSITION_CEILING")
        return self.accounting.call(operation, function, stage=stage, input_positions=input_positions, training=False)
