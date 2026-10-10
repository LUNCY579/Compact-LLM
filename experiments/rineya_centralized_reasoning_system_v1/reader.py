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
CONTRACT = PACKAGE / "contracts/READER_QUALIFICATION_OWNER_EXCEPTION_001_V2.json"
SCOPE = "CRS_EXACT_PREFIX_GREEDY_SERVING_576_V1"
EOS_IDS = (151645, 151643)
VOCAB = 151936
EVIDENCE_NAMES = ("worker_result", "events", "completion_verification", "supervisor_result", "source_launch_review",
                  "containment", "tree_containment", "go", "runtime", "model", "tokenizer", "job_outcome", "native_outcome",
                  "launch_intent", "launch_attempt")
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


def independent_reviewer_id(value):
    """Owner exception001's fixed non-builder leaves; this is not authentication."""
    require(type(value) is str and value.strip(), "identified independent reviewer required")
    parts = [part.strip().casefold() for part in value.strip().replace('\\', '/').split('/') if part.strip()]
    require(len(parts) == 1 or (len(parts) == 2 and parts[0] == 'root'), "unknown reviewer namespace")
    require(parts[-1] in {'exception_science', 'exception_ops'},
            "owner exception001 requires a registered non-builder reviewer")
    return '/root/' + parts[-1]


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


def actual_forward_plan(cfg, long_generated):
    """Exact source/token/retention schedule, conditional only on realized EOS."""
    require(type(long_generated) is list and 1 <= len(long_generated) <= 64
            and all(type(v) is int and 0 <= v < VOCAB for v in long_generated), "natural generated IDs invalid")
    plan = []
    def add(block, stage, ids, keep=1):
        plan.append({"block": block, "stage": stage, "input_ids": ids, "logits_to_keep": keep})
    full = cfg["long_prompt_ids"] + cfg["mechanical_continuation_ids"]
    for length, ids in ((128, cfg["historical_prompt_ids"]), (512, cfg["long_prompt_ids"]), (576, full)):
        add(0, "direct_" + str(length), ids); add(0, "wrapper_" + str(length), ids)
    for block, label, prompt, generated in ((1, "historical", cfg["historical_prompt_ids"], cfg["historical_generated_ids"]),
                                          (2, "long", cfg["long_prompt_ids"], long_generated)):
        for kind in ("owned", "stock", "replay"):
            for i in range(len(generated)):
                stage = label + "_owned_owned_step_" + str(i) if kind == "owned" else label + "_" + kind + "_" + str(i)
                add(block, stage, prompt + generated[:i])
    for i in range(65): add(3, "FORCED_MECHANICAL_NOT_GENERATED_" + str(i), cfg["long_prompt_ids"] + cfg["mechanical_continuation_ids"][:i])
    for label, prompt, generated in (("historical", cfg["historical_prompt_ids"], cfg["historical_generated_ids"]),
                                    ("long", cfg["long_prompt_ids"], long_generated)):
        add(4, label + "_bulk_DIAGNOSTIC_ONLY", prompt + generated, len(generated) + 1)
    add(5, "max576_baseline", full, 4)
    for start in (573, 574, 575):
        mutated = list(full)
        for index in range(start, 576): mutated[index] = (mutated[index] + 1) % VOCAB
        add(5, "max576_mutate_" + str(start), mutated, 4)
    return plan


def validate_termination(prompt, generated, cap, stop, terminal_eos=None):
    require(type(generated) is list and 1 <= len(generated) <= cap
            and all(type(v) is int and 0 <= v < VOCAB for v in generated), "complete bounded natural output required")
    first = next((i for i, token in enumerate(generated) if token in EOS_IDS), None)
    if first is not None:
        require(first == len(generated) - 1 and stop == "eos" and terminal_eos == generated[-1], "natural EOS branch corrupt")
    else:
        require(len(generated) == cap and stop == "cap" and terminal_eos is None, "natural cap branch corrupt")
    return True


def require_same_prefix(owned, replay):
    require(len(owned) == len(replay) > 0, "missing replay prefix")
    for left, right in zip(owned, replay):
        require(left["input_ids_sha256"] == right["input_ids_sha256"]
                and left["raw_logits_sha256"] == right["raw_logits_sha256"]
                and left["raw_hidden_sha256"] == right["raw_hidden_sha256"], "same-prefix raw parity failed")
    return likelihood_measurement([x["selected_model_logprob"] for x in owned],
                                  [x["selected_model_logprob"] for x in replay])


def validate_native_evidence_review(review, expected_binding, evidence_refs):
    require(type(review) is dict and review.get('status') == 'APPROVED_READER_NATIVE_EVIDENCE'
            and review.get('review_kind') == 'POST_RUN_NATIVE_EVIDENCE' and review.get('actual_native_evidence_reviewed') is True
            and review.get('binding') == expected_binding and review.get('evidence_refs') == evidence_refs,
            'source-only or other-run review cannot approve actual reader evidence')
    independent_reviewer_id(review.get('reviewer'))
    return True


def verify_qualification(reference, expected_binding):
    """Caller status dictionaries cannot open serving; require immutable bundle."""
    require(type(reference) is dict and set(reference) == {"path", "sha256"}, "immutable reader qualification reference required")
    from . import reader_execution
    require(type(expected_binding) is dict and type(expected_binding.get('contract_sha256')) is str,
            'complete expected source binding required')
    cfg = reader_execution.load_contract(expected_binding['contract_sha256'])
    require(expected_binding == reader_execution.binding(cfg, expected_binding['contract_sha256'], expected_binding.get('source_commit')),
            'complete expected contract binding required')
    require(cfg['inherited_supervision_prerequisite']['status'] == 'QUALIFIED_FULL_PATH',
            'quarantined inherited supervisor cannot be bypassed by serving receipt')
    path = Path(reference["path"]).resolve()
    require(path.is_relative_to(PACKAGE / "results") and sha(path) == reference["sha256"], "qualification path/hash mismatch")
    receipt = read_json(path)
    require(receipt.get("scope") == SCOPE and receipt.get("status") == "PASS_EXACT_PREFIX_SERVING_576_SCOPED"
            and receipt.get("binding") == expected_binding, "exact reader source/model/runtime qualification mismatch")
    require(receipt.get("supervisor_fully_verified") is True and receipt.get("natural_eos_observed") is True,
            "full contained-path/natural EOS qualification required")
    require(receipt.get("bulk_capture_qualified") is False and receipt.get("training_qualified") is False,
            "reader qualification cannot expand bulk/training operations")
    dependencies = {}
    for name in (*EVIDENCE_NAMES, "independent_review"):
        ref = receipt.get(name)
        require(type(ref) is dict and set(ref) == {"path", "sha256"}, "missing qualification dependency")
        child = Path(ref["path"]).resolve()
        require(child.is_relative_to(PACKAGE / "results") and sha(child) == ref["sha256"], "qualification dependency drift")
        dependencies[name] = ref
        if name == "events": continue
        content = read_json(child)
        if name == "worker_result":
            require(content.get("status") == "COMPLETE_READER_QUALIFICATION_WORKER_ONLY" and content.get("binding") == expected_binding,
                    "reader qualification worker proof mismatch")
        elif name == "supervisor_result":
            require(content.get("status") == "SUCCEEDED" and content.get("exit_code") == 0
                    and content.get("owned_tree_exit", {}).get("owned_tree_exited") is True, "reader supervisor proof mismatch")
        elif name == "independent_review":
            require(content.get("status") == "APPROVED_READER_NATIVE_EVIDENCE" and content.get("binding") == expected_binding
                    and independent_reviewer_id(content.get('reviewer')),
                    "independent post-run reader evidence review mismatch")
        elif name == "source_launch_review":
            require(content.get("status") == "APPROVED_FOR_READER_QUALIFICATION" and content.get("binding") == expected_binding,
                    "reader source-launch review mismatch")
        elif name == "completion_verification":
            require(content.get("reader_scope") == SCOPE and content.get("durable_journal_verified") is True,
                    "reader completion verification missing")
        elif name in {'runtime','model'}:
            require(content.get('binding') == expected_binding, "native runtime/model artifact binding mismatch")
        elif name == 'tokenizer':
            require(content.get('tokenizer_sha256') == expected_binding['tokenizer_sha256']
                    and content.get('template_sha256') == expected_binding['template_sha256'], "tokenizer artifact binding mismatch")
        elif name == 'containment':
            require(content.get('in_exact_supervisor_job') is True and content.get('query_handle_closed_before_framework') is True,
                    "explicit native containment evidence mismatch")
    evidence_review = read_json(dependencies["independent_review"]["path"])
    validate_native_evidence_review(evidence_review,expected_binding,{k:v for k,v in dependencies.items() if k!='independent_review'})
    worker = read_json(dependencies['worker_result']['path'])
    require(all(canonical(receipt.get(k)) == canonical(worker.get(k)) for k in
                ('run_identity','worker_identity','runtime_worker_identity')), 'serving receipt/native worker run mismatch')
    reader_execution.validate_native_artifacts({name: read_json(dependencies[name]['path'])
        for name in EVIDENCE_NAMES if name != 'events'}, cfg, expected_binding, dependencies)
    verified = reader_execution.validate_qualification_result(worker,cfg,expected_binding['source_commit'],
                                reader_execution.read_event_journal(dependencies['events']['path']))
    require(reader_execution.read_event_journal(dependencies['events']['path'])[-1]['time']
            <= read_json(dependencies['job_outcome']['path'])['job']['finished'], 'reader journal extends beyond recorded outcome')
    completion = read_json(dependencies['completion_verification']['path'])
    require(all(completion.get(k) == value for k,value in verified.items())
            and completion.get('events_sha256') == dependencies['events']['sha256'], 'completion verifier/native journal mismatch')
    model = read_json(dependencies['model']['path'])
    require(model.get('unique_parameters') == 596049920 and model.get('tied') is True and model.get('trainable_parameters') == 0
            and model.get('dtype') == 'bfloat16' and model.get('loader',{}).get('serialized_tensors') == 310
            and type(model.get('tensor_sha256')) is dict and len(model['tensor_sha256']) == 310
            and all(type(v) is str and re.fullmatch(r'[0-9a-f]{64}',v) for v in model['tensor_sha256'].values()),
            'full native loader/tensor identity evidence required')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--qualify", action="store_true")
    mode.add_argument("--worker", action="store_true")
    mode.add_argument("--serve", action="store_true")
    mode.add_argument("--publish-reviewed", action="store_true")
    mode.add_argument("--recover-publication", action="store_true")
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--host-db", type=Path)
    parser.add_argument("--job-key"); parser.add_argument("--epoch")
    parser.add_argument("--owner"); parser.add_argument("--fence", type=int)
    parser.add_argument("--run-id"); parser.add_argument("--request-hash")
    parser.add_argument("--go-path", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--evidence-review", type=Path)
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
    if args.publish_reviewed:
        require(args.output is not None and args.evidence_review is not None, "output and independent native evidence review required")
        print(json.dumps(reader_execution.publish_after_evidence_review(args.output,args.evidence_review),indent=2)); return 0
    if args.recover_publication:
        print(json.dumps(reader_execution.recover_pending_publication(args,cfg),indent=2)); return 0
    if args.serve:
        raise ReaderBlocked("DEV/serving dispatch remains separately registered; qualified endpoint is exposed by ReaderSession only")
    return reader_execution.supervise(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
