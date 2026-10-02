"""Exact-prefix reader policy. Stdlib only; import never initializes a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from .operations import canonical

PACKAGE = Path(__file__).resolve().parent
CONTRACT = PACKAGE / "contracts/READER_QUALIFICATION_V1.json"
SCOPE = "CRS_EXACT_PREFIX_GREEDY_SERVING_576_V1"
EOS_IDS = (151645, 151643)
VOCAB = 151936
BLOCKS = (
    ("direct_wrapper_parity", 6, 2432),
    ("historical_natural_owned_stock_replay", 9, 1161),
    ("long_natural_owned_stock_replay", 192, 104352),
    ("forced_prefix_shape_mechanics", 65, 35360),
    ("bulk_diagnostics_not_serving_gates", 2, 707),
    ("max_shape_suffix_causality", 4, 2304),
)


class ReaderBlocked(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise ReaderBlocked(message)


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON field")
            result[key] = value
        return result
    value = json.loads(Path(path).read_bytes(), object_pairs_hook=unique)
    canonical(value)
    return value


def validate_request(request):
    require(type(request) is dict and set(request) == {"input_ids", "attention_mask", "max_new_tokens"},
            "reader accepts only prompt token fields; scorer/gold/metadata forbidden")
    ids, mask, cap = request["input_ids"], request["attention_mask"], request["max_new_tokens"]
    require(type(ids) is list and 1 <= len(ids) <= 512
            and all(type(token) is int and 0 <= token < VOCAB for token in ids), "supported single prompt1..512 required")
    require(type(mask) is list and all(type(v) is int for v in mask) and mask == [1] * len(ids),
            "unpadded single sequence with all-one mask required")
    require(type(cap) is int and 1 <= cap <= 64 and len(ids) + cap <= 576, "reader output/total envelope exceeded")
    return {"input_ids": list(ids), "attention_mask": list(mask), "max_new_tokens": cap}


def manual_greedy(request, forward):
    """Trusted callback supplies actual raw argmax evidence, never model code."""
    request = validate_request(request)
    generated, observations = [], []
    for index in range(request["max_new_tokens"]):
        prefix = request["input_ids"] + generated
        observation = forward(prefix, 1, "owned_step_" + str(index))
        token = observation.get("argmax_id")
        require(type(token) is int and 0 <= token < VOCAB and observation.get("lowest_id_argmax_verified") is True,
                "raw lowest-ID argmax evidence required")
        require(observation.get("input_ids_sha256") == digest(prefix)
                and observation.get("input_length") == len(prefix), "prefix geometry/identity mismatch")
        lp = observation.get("selected_model_logprob")
        require(type(lp) in (int, float) and math.isfinite(lp) and lp <= 0, "finite model softmax logprob required")
        generated.append(token); observations.append(observation)
        if token in EOS_IDS:
            stop = "eos"
            break
    else:
        stop = "cap"
    return {"prompt_ids": request["input_ids"], "generated_ids": generated,
            "visible_ids": generated[:-1] if generated[-1] in EOS_IDS else list(generated),
            "terminal_eos_id": generated[-1] if generated[-1] in EOS_IDS else None,
            "stop_reason": stop, "steps": observations, "behavior_logprobs": [0.0] * len(generated),
            "generation_count": len(generated), "model_logprobs": [o["selected_model_logprob"] for o in observations]}


def qualifier_plan(cfg):
    require(cfg["scope"] == SCOPE, "reader qualification namespace mismatch")
    prompt = cfg["historical_prompt_ids"]
    long_prompt = cfg["long_prompt_ids"]
    continuation = cfg["mechanical_continuation_ids"]
    require(len(prompt) == 128 and len(long_prompt) == 512 and len(continuation) == 64,
            "frozen mechanical input/continuation shapes mismatch")
    for ids in (prompt, long_prompt, continuation):
        require(all(type(v) is int and 0 <= v < VOCAB for v in ids), "mechanical token identity invalid")
    require(sum(n for _, n, _ in BLOCKS) == 278 and sum(p for _, _, p in BLOCKS) == 146316,
            "schedule ceiling arithmetic mismatch")
    return {"blocks": [{"id": name, "max_forwards": calls, "max_input_positions": positions}
                        for name, calls, positions in BLOCKS],
            "max_forwards": 278, "max_input_positions": 146316, "max_logical_calls": 590,
            "max_natural_emissions_owned_plus_stock": 134,
            "forced_mechanical_prefixes": [long_prompt + continuation[:index] for index in range(65)],
            "mechanical_is_model_generation": False}


def likelihood_measurement(online, replay):
    require(type(online) is list and type(replay) is list and len(online) == len(replay) > 0,
            "complete matching token likelihood vectors required")
    require(all(type(v) in (int, float) and math.isfinite(v) for v in online + replay), "nonfinite likelihood")
    errors = [abs(a - b) for a, b in zip(online, replay)]
    return {"max_abs": max(errors), "mean_abs": sum(errors) / len(errors),
            "passed": max(errors) <= .03125 and sum(errors) / len(errors) <= .01,
            "tolerances": {"max_abs": .03125, "mean_abs": .01}}


def require_same_prefix(owned, replay):
    require(len(owned) == len(replay) > 0, "missing replay prefix")
    for left, right in zip(owned, replay):
        require(left["input_ids_sha256"] == right["input_ids_sha256"]
                and left["raw_logits_sha256"] == right["raw_logits_sha256"]
                and left["raw_hidden_sha256"] == right["raw_hidden_sha256"], "same-prefix raw parity failed")
    return likelihood_measurement([x["selected_model_logprob"] for x in owned],
                                  [x["selected_model_logprob"] for x in replay])


def verify_qualification(reference, expected_binding):
    """Caller status dictionaries cannot open serving; require immutable bundle."""
    require(type(reference) is dict and set(reference) == {"path", "sha256"}, "immutable reader qualification reference required")
    path = Path(reference["path"]).resolve()
    require(path.is_relative_to(PACKAGE / "results") and sha(path) == reference["sha256"], "qualification path/hash mismatch")
    receipt = read_json(path)
    require(receipt.get("scope") == SCOPE and receipt.get("status") == "PASS_EXACT_PREFIX_SERVING_576_SCOPED"
            and receipt.get("binding") == expected_binding, "exact reader source/model/runtime qualification mismatch")
    require(receipt.get("supervisor_fully_verified") is True and receipt.get("natural_eos_observed") is True,
            "full contained-path/natural EOS qualification required")
    require(receipt.get("bulk_capture_qualified") is False and receipt.get("training_qualified") is False,
            "reader qualification cannot expand bulk/training operations")
    for name in ("worker_result", "supervisor_result", "independent_review"):
        ref = receipt.get(name)
        require(type(ref) is dict and set(ref) == {"path", "sha256"}, "missing qualification dependency")
        child = Path(ref["path"]).resolve()
        require(child.is_relative_to(PACKAGE / "results") and sha(child) == ref["sha256"], "qualification dependency drift")
        content = read_json(child)
        if name == "worker_result":
            require(content.get("status") == "COMPLETE_READER_QUALIFICATION_WORKER_ONLY" and content.get("binding") == expected_binding,
                    "reader qualification worker proof mismatch")
        elif name == "supervisor_result":
            require(content.get("status") == "SUCCEEDED" and content.get("exit_code") == 0
                    and content.get("owned_tree_exit", {}).get("owned_tree_exited") is True, "reader supervisor proof mismatch")
        else:
            require(content.get("status") == "APPROVED_FOR_READER_QUALIFICATION" and content.get("binding") == expected_binding
                    and type(content.get("reviewer")) is str and content["reviewer"].rstrip("/").rsplit("/", 1)[-1] != "runtime_audit",
                    "independent reader review proof mismatch")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--qualify", action="store_true")
    mode.add_argument("--worker", action="store_true")
    mode.add_argument("--serve", action="store_true")
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--host-db", type=Path)
    parser.add_argument("--job-key"); parser.add_argument("--epoch")
    parser.add_argument("--owner"); parser.add_argument("--fence", type=int)
    parser.add_argument("--go-path", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    if args.host_db is None:
        from .operations import host_state_path
        args.host_db = host_state_path()
    from . import reader_execution
    if args.preflight:
        print(json.dumps(reader_execution.preflight(args.contract_sha256, args.review), indent=2)); return 0
    cfg = reader_execution.load_contract(args.contract_sha256)
    if args.worker:
        return reader_execution.worker(args, cfg)
    if args.serve:
        raise ReaderBlocked("DEV/serving dispatch remains separately registered; qualified endpoint is exposed by ReaderSession only")
    return reader_execution.supervise(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
