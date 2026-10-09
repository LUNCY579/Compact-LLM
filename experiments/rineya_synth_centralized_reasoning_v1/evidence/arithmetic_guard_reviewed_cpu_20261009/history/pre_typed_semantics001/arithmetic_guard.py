"""New public arithmetic guard candidates, adapted from pinned central semantics.

No central module/population is imported. Policy and canonical parent allocation
precede solving. CPU candidates and scorer fixtures cannot enroll M0 or qualify
native scoring, tokenization, disjointness or model improvement.
"""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import random

HERE = Path(__file__).resolve().parent
POLICY = HERE / "SYNTH_ARITHMETIC_GUARD_POLICY001.json"
PARENT_FIELDS = {"kind", "domain_id", "domain_class", "source_id", "parent_id", "policy_sha256",
                 "canonical_expression", "expression_sha256", "depth", "candidate_index"}
ROW_FIELDS = {"kind", "domain_id", "domain_class", "source_id", "parent_id", "row_id", "policy_sha256",
              "parent", "condition", "rendering", "expression", "prompt", "prompt_sha256", "answer"}


def require(value, message):
    if not value: raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key"); result[key] = value
    return result


def _bad(value):
    raise ValueError("nonfinite JSON")


def policy():
    value = json.loads(POLICY.read_bytes(), object_pairs_hook=_pairs, parse_constant=_bad)
    require(value["sha256"] == digest({k: v for k, v in value.items() if k != "sha256"}), "policy content checksum")
    require(value["kind"] == "SYNTH_ARITHMETIC_GUARD_POLICY001" and type(value["schema_version"]) is int
            and value["schema_version"] == 1 and value["domain_id"] == "arithmetic"
            and value["domain_class"] == "task" and value["source_id"] == "SYNTHARITHGUARD001"
            and value["seed"] == 202610091 and value["parent_count"] == 128
            and value["parents_per_depth"] == 32 and value["depths"] == [1, 2, 3, 4]
            and value["rows_per_parent"] == 4 and value["row_count"] == 512, "fixed closed arithmetic guard scope")
    require(all(type(value[key]) is int for key in ("seed", "parent_count", "parents_per_depth", "rows_per_parent", "row_count"))
            and type(value["depths"]) is list and all(type(n) is int for n in value["depths"]), "exact integer policy counts/depths")
    require(value["semantics"] == {"integer_leaf_min": -50, "integer_leaf_max": 50,
        "operations": ["add", "sub", "mul"], "binary_only": True, "maximum_depth": 4,
        "intermediate_absolute_limit": 100000, "response_maximum_characters": 4096,
        "response_exact_keys": ["answer"], "response_exact_answer_type": "int_without_bool"}, "frozen arithmetic semantics")
    require(value["selection"]["before_solving"] is True and value["selection"]["before_M0"] is True
            and value["selection"]["finite_candidates_per_depth"] == 128
            and value["selection"]["no_output_based_filter"] is True
            and value["selection"]["resampling_after_M0"] is False
            and value["selection"]["training_augmentation"] is False, "frozen prebaseline finite selection")
    require(type(value["selection"]["finite_candidates_per_depth"]) is int
            and value["selection"]["filters"] == ["duplicate_canonical_content", "equal_root_operands_prevent_distinct_equivalent_layout"]
            and value["selection"]["candidate_grammar"] == {"root_op": "add", "leaf_values": [-50, 50],
                "full_binary_depths": [1, 2, 3, 4], "internal_ops_at_height1": ["add", "sub", "mul"],
                "internal_ops_at_greater_heights": ["add", "sub"], "proved_maximum_intermediate_absolute_value": 20000,
                "overflow_action": "fail whole compile; no answer-dependent resampling"}
            and value["renderings"] == ["expression_json", "fully_parenthesized_infix"]
            and value["conditions"] == ["canonical_operand_order", "commutative_operand_reversal"], "frozen generator and equivalent layout axes")
    for key in ("native_launch_authorized", "native_scoring_authorized", "M0_enrollment_performed",
                "tokenization_performed", "actual_model_quality_established"):
        require(value[key] is False, "CPU policy cannot acquire native authority")
    return value


def normalize(node, depth=0):
    require(depth <= 4, "expression depth exceeds4")
    if type(node) is int:
        require(-50 <= node <= 50, "integer leaf outside[-50,50]")
        return node
    require(type(node) is dict and set(node) == {"op", "args"} and node["op"] in ("add", "sub", "mul")
            and type(node["args"]) is list and len(node["args"]) == 2, "exact binary arithmetic schema")
    args = [normalize(child, depth + 1) for child in node["args"]]
    if node["op"] in ("add", "mul"): args.sort(key=lambda item: canonical(item).encode())
    return {"op": node["op"], "args": args}


def depth(node):
    return 0 if type(node) is int else 1 + max(depth(child) for child in node["args"])


def recursive(node):
    value = normalize(node)
    def walk(item):
        if type(item) is int: return item
        left, right = (walk(child) for child in item["args"])
        if item["op"] == "add": result = left + right
        elif item["op"] == "sub": result = left - right
        else: result = left * right
        require(-100000 <= result <= 100000, "intermediate arithmetic overflow")
        return result
    return walk(value)


def postfix(node):
    """Independent iterative validation and stack execution; no recursive solver."""
    todo, stack = [(node, 0, False)], []
    while todo:
        item, level, visited = todo.pop()
        require(level <= 4, "postfix expression depth exceeds4")
        if type(item) is int:
            require(-50 <= item <= 50, "postfix integer leaf range")
            stack.append(item)
        elif not visited:
            require(type(item) is dict and set(item) == {"op", "args"}
                    and item["op"] in ("add", "sub", "mul") and type(item["args"]) is list
                    and len(item["args"]) == 2, "postfix exact binary arithmetic schema")
            todo.extend([(item, level, True), (item["args"][1], level + 1, False),
                         (item["args"][0], level + 1, False)])
        else:
            right, left = stack.pop(), stack.pop()
            if item["op"] == "add": result = left + right
            elif item["op"] == "sub": result = left - right
            else: result = left * right
            require(-100000 <= result <= 100000, "postfix intermediate arithmetic overflow")
            stack.append(result)
    require(len(stack) == 1, "postfix exact final stack")
    return stack[0]


def _tree(rng, height):
    if height == 0: return rng.randint(-50, 50)
    # Multiplication only of leaves guarantees the finite pool's bound without
    # solving or conditioning selection on any target/teacher/student output.
    op = rng.choice(("add", "sub", "mul") if height == 1 else ("add", "sub"))
    return {"op": op, "args": [_tree(rng, height - 1), _tree(rng, height - 1)]}


def allocate(frozen):
    require(frozen == policy(), "selected policy changed before allocation")
    rng = random.Random(frozen["seed"]); parents = []; audit = []
    for height in frozen["depths"]:
        candidates = {}; duplicate = equal_operands = 0
        for index in range(frozen["selection"]["finite_candidates_per_depth"]):
            tree = normalize({"op": "add", "args": [_tree(rng, height - 1), _tree(rng, height - 1)]})
            if tree["args"][0] == tree["args"][1]: equal_operands += 1; continue
            expression_sha = digest(tree)
            if expression_sha in candidates: duplicate += 1; continue
            candidates[expression_sha] = (tree, index)
        require(len(candidates) >= 32, "fixed finite pool lacks32 unique parents; no resampling")
        for expression_sha in sorted(candidates)[:32]:
            tree, index = candidates[expression_sha]
            identity = digest({"source_id": frozen["source_id"], "canonical_expression": tree})
            parents.append({"kind": "SYNTH_ARITHMETIC_GUARD_PARENT001", "domain_id": "arithmetic",
                "domain_class": "task", "source_id": frozen["source_id"], "parent_id": identity,
                "policy_sha256": frozen["sha256"], "canonical_expression": tree,
                "expression_sha256": expression_sha, "depth": height, "candidate_index": index})
        audit.append({"depth": height, "candidates": 128, "unique_eligible_canonical": len(candidates),
                      "duplicate_canonical": duplicate, "equal_operand_layout": equal_operands,
                      "selected": 32, "output_or_answer_filter_performed": False})
    require(len(parents) == 128 and len({p["parent_id"] for p in parents}) == 128, "128 unique allocated parents")
    return parents, audit


@lru_cache(maxsize=1)
def _allocated_roster(policy_sha256):
    frozen = policy(); require(frozen["sha256"] == policy_sha256, "selected policy byte/value identity")
    parents, _ = allocate(frozen)
    return canonical({parent["parent_id"]: parent for parent in parents})


def validate_parent(parent, frozen):
    require(type(parent) is dict and set(parent) == PARENT_FIELDS
            and parent["kind"] == "SYNTH_ARITHMETIC_GUARD_PARENT001" and parent["domain_id"] == "arithmetic"
            and parent["domain_class"] == "task" and parent["source_id"] == frozen["source_id"]
            and parent["policy_sha256"] == frozen["sha256"], "exact allocated arithmetic parent schema")
    tree = normalize(parent["canonical_expression"])
    require(tree == parent["canonical_expression"] and parent["expression_sha256"] == digest(tree)
            and parent["parent_id"] == digest({"source_id": frozen["source_id"], "canonical_expression": tree})
            and type(tree) is dict and tree["op"] == "add" and tree["args"][0] != tree["args"][1]
            and type(parent["depth"]) is int and parent["depth"] in (1, 2, 3, 4)
            and parent["depth"] == depth(tree) and type(parent["candidate_index"]) is int
            and 0 <= parent["candidate_index"] < 128, "parent content/identity/depth binding")
    roster = json.loads(_allocated_roster(frozen["sha256"]))
    require(roster.get(parent["parent_id"]) == parent, "parent is outside registered finite allocation")
    return tree


def _infix(tree):
    if type(tree) is int: return str(tree)
    symbol = {"add": "+", "sub": "-", "mul": "*"}[tree["op"]]
    return "(" + _infix(tree["args"][0]) + " " + symbol + " " + _infix(tree["args"][1]) + ")"


def render(parent, condition, rendering, frozen):
    tree = validate_parent(parent, frozen)
    require(type(condition) is int and condition in (0, 1) and rendering in frozen["renderings"], "registered crossed rendering axes")
    if condition: tree["args"].reverse()  # All allocated roots are commutative add.
    require(normalize(tree) == parent["canonical_expression"], "meaning-preserving operand variant")
    payload = canonical(tree) if rendering == "expression_json" else _infix(tree)
    format_rule = ("In the expression JSON, add/sub/mul mean integer addition/subtraction/multiplication."
                   if rendering == "expression_json" else "Evaluate the fully parenthesized integer expression; * is multiplication.")
    prompt = format_rule + '\nReturn exactly one JSON object with key "answer" and an integer value.\nExpression: ' + payload
    return tree, prompt


def compile_population(frozen):
    parents, allocation_audit = allocate(frozen)  # Every parent is fixed before any solver call.
    rows = []
    for parent in parents:
        answer = recursive(parent["canonical_expression"])
        require(type(answer) is int and answer == postfix(parent["canonical_expression"]), "independent solvers disagree")
        for condition in (0, 1):
            for rendering in frozen["renderings"]:
                expression, prompt = render(parent, condition, rendering, frozen)
                require(postfix(expression) == answer, "rendered variant changed answer")
                row_id = digest({"parent_id": parent["parent_id"], "condition": condition, "rendering": rendering,
                                 "policy_sha256": frozen["sha256"]})
                rows.append({"kind": "SYNTH_ARITHMETIC_GUARD_ROW001", "domain_id": "arithmetic",
                    "domain_class": "task", "source_id": frozen["source_id"], "parent_id": parent["parent_id"],
                    "row_id": row_id, "policy_sha256": frozen["sha256"], "parent": deepcopy(parent),
                    "condition": condition, "rendering": rendering, "expression": expression,
                    "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "answer": answer})
    value = {"kind": "SYNTH_ARITHMETIC_GUARD_CPU_POPULATION001", "policy_sha256": frozen["sha256"],
        "parents": parents, "rows": rows, "parent_order_sha256": digest([p["parent_id"] for p in parents]),
        "row_order_sha256": digest([r["row_id"] for r in rows]), "allocation_audit": allocation_audit,
        "independent_parent_units": 128, "M0_eligible_parent_count": None, "precision_established": False,
        "tokenization_performed": False, "native_scoring_authorized": False, "M0_enrollment_performed": False,
        "disjointness_qualified": False, "actual_model_quality_established": False}
    value["sha256"] = digest(value)
    return value


def validate_row(row, frozen):
    require(type(row) is dict and set(row) == ROW_FIELDS and row["kind"] == "SYNTH_ARITHMETIC_GUARD_ROW001"
            and row["domain_id"] == "arithmetic" and row["domain_class"] == "task"
            and row["source_id"] == frozen["source_id"] and row["policy_sha256"] == frozen["sha256"], "exact arithmetic guard row")
    expression, prompt = render(row["parent"], row["condition"], row["rendering"], frozen)
    expected_id = digest({"parent_id": row["parent"]["parent_id"], "condition": row["condition"],
                          "rendering": row["rendering"], "policy_sha256": frozen["sha256"]})
    require(row["parent_id"] == row["parent"]["parent_id"] and row["row_id"] == expected_id
            and row["expression"] == expression and row["prompt"] == prompt
            and row["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest(), "row content/prompt/identity binding")
    answer = postfix(expression)
    require(type(row["answer"]) is int and row["answer"] == answer == recursive(expression), "row target independent verification")
    return answer


def score(row, response, frozen):
    answer = validate_row(row, frozen)  # Host corruption never becomes a malformed model answer.
    try:
        require(type(response) is str and len(response) <= 4096, "bounded JSON response")
        parsed = json.loads(response, object_pairs_hook=_pairs, parse_constant=_bad)
        require(type(parsed) is dict and set(parsed) == {"answer"} and type(parsed["answer"]) is int,
                "exact integer answer JSON")
    except (ValueError, TypeError, RecursionError, OverflowError):
        return {"correct": False, "malformed": True}
    return {"correct": parsed["answer"] == answer, "malformed": False}
