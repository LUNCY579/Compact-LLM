"""Separate closed arithmetic512 plan; original short native gates retained."""
from copy import deepcopy
from pathlib import Path
from types import ModuleType
import hashlib
import importlib
import json
import sys

HERE = Path(__file__).resolve().parent
PACKAGE = next(p for p in HERE.parents if p.name == "rineya_synth_centralized_reasoning_v1")
EXPECTED = PACKAGE / "arithmetic_capture_v1"
KIND = "arithmetic_guard_DEV"
PLAN_PATH = HERE / "ARITHMETIC_CAPTURE_PLAN001.json"
PLAN_REF = None
ORIGINAL_CONTRACT_SHA256 = "1d66d5823d7485864a4f3d887870e62ed68978f08df90f5f7698c4766d2d365e"


def require(value, message):
    if not value: raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def integrated():
    require(HERE == EXPECTED, "staged arithmetic capture has no native authority")


def build_plan(prompt_rows, metadata, *, measurement_reference, prompt_reference, policy_reference, ledger_reference, fixture_only=False):
    if str(PACKAGE) not in sys.path: sys.path.insert(0, str(PACKAGE))
    from inference_component import validate_row
    require(len(prompt_rows) == len(metadata) == 512, "exact512 arithmetic source/metadata rows")
    rows = []; identities = []
    groups={}
    for original, meta in zip(prompt_rows, metadata):
        require(type(meta) is dict and set(meta)=={'row_id','component_id','mechanism','variant','rendering'}
                and meta['mechanism']=='bounded_integer_arithmetic' and type(meta['variant']) is int
                and meta['variant'] in (0,1) and meta['rendering'] in ('expression_json','fully_parenthesized_infix'),
                'exact arithmetic condition/rendering metadata')
        groups.setdefault(meta['component_id'],[]).append((meta['variant'],meta['rendering']))
        require(original["row_id"] == meta["row_id"] and original["parent_id"] == meta["component_id"], "arithmetic parent/row origin join")
        require(original["prompt_token_sha256"] == digest(original["input_ids"]), "prompt token origin")
        row = {"row_id": original["row_id"], "prompt_sha256": original["prompt_sha256"],
            "prompt_ids": deepcopy(original["input_ids"]), "prompt_tokens": original["prompt_tokens"],
            "attention_mask": deepcopy(original["attention_mask"])}
        validate_row(row); rows.append(row)
        identities.append({"row_id": row["row_id"], "component_id": meta["component_id"],
            "message_sha256": digest(original["messages"]), "rendered_chat_sha256": original["prompt_sha256"],
            "prompt_token_sha256": original["prompt_token_sha256"]})
    require(len({r["row_id"] for r in rows}) == 512 and len({m["component_id"] for m in metadata}) == 128, "arithmetic512/128 denominator")
    pairs={(c,r) for c in (0,1) for r in ('expression_json','fully_parenthesized_infix')}
    require(all(len(g)==4 and set(g)==pairs for g in groups.values()),'four registered siblings per arithmetic parent')
    batches = []
    for ordinal in range(1, 9):
        selected = rows[(ordinal - 1) * 64:ordinal * 64]
        positions = sum(32 * row["prompt_tokens"] + 496 for row in selected)
        batches.append({"ordinal": ordinal, "source_kind": KIND, "start": (ordinal - 1) * 64, "stop": ordinal * 64,
            "rows": selected, "source_packet_sha256": digest(selected), "maximum_capture_forwards": 2048,
            "maximum_capture_input_positions": positions, "maximum_zero_parity_forwards": 2,
            "maximum_zero_parity_input_positions": 2 * selected[0]["prompt_tokens"], "maximum_total_forwards": 2050,
            "maximum_total_input_positions": positions + 2 * selected[0]["prompt_tokens"],
            "native_launch_authorized": False, "scoring_authorized": False})
    value = {"kind": "SYNTH_ARITHMETIC_CAPTURE_PLAN001", "fixture_only": fixture_only, "rows": 512,
        "parent_components": 128, "rows_per_batch": 64, "context_cap": 512, "maximum_generated_tokens": 32,
        "batches": {KIND: batches}, "metadata": deepcopy(metadata), "origins": identities,
        "row_order_sha256": digest([r["row_id"] for r in rows]), "measurement_reference": measurement_reference,
        "prompt_reference": prompt_reference, "policy_reference": policy_reference, "ledger_reference": ledger_reference,
        "source_pins": {KIND: [prompt_reference["path"], prompt_reference["sha256"], 512]},
        "native_launch_authorized": False, "scoring_authorized": False, "production_training_authorized": False,
        "language_qualification_established": False, "long_context_qualification_established": False,
        "prior_correct_enrollment_performed": False, "protected_bank_access": False, "confirmation_eligible": False}
    value["sha256"] = digest(value)
    return value


def select_batch(body, kind, ordinal):
    require(kind == KIND and type(ordinal) is int and 1 <= ordinal <= 8, "registered arithmetic batch1..8")
    return deepcopy(body["batches"][KIND][ordinal - 1])


def plan():
    require(PLAN_REF is not None, "independently selected arithmetic plan reference absent")
    raw = PLAN_PATH.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == PLAN_REF["sha256"], "selected arithmetic plan byte pin")
    body = json.loads(raw)
    require(body["sha256"] == digest({k: v for k, v in body.items() if k != "sha256"})
            and body["fixture_only"] is False and body["rows"] == 512 and len(body["batches"][KIND]) == 8,
            "closed whole arithmetic plan")
    def read(ref):
        require(type(ref) is dict and set(ref)=={'path','bytes','sha256'} and type(ref['bytes']) is int
                and 0<ref['bytes']<=32*1024**2,'exact bound arithmetic data reference')
        path=Path(ref['path']);require(path.is_absolute() and path.is_file() and not path.is_symlink(),'selected arithmetic JSON file')
        with path.open('rb') as stream:payload=stream.read(ref['bytes']+1)
        require(len(payload)==ref['bytes'] and hashlib.sha256(payload).hexdigest()==ref['sha256'],'arithmetic source bytes drift')
        def pairs(items):
            value={}
            for k,v in items:require(k not in value,'duplicate arithmetic source field');value[k]=v
            return value
        return json.loads(payload,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
    measured=read(body['measurement_reference'])
    measured_policy=measured['policy_file'];declared_policy=body['prompt_reference']
    require(set(measured_policy)=={'path','bytes','sha256'} and measured_policy['bytes']==declared_policy['bytes']
            and measured_policy['sha256']==declared_policy['sha256']
            and Path(measured_policy['path']).resolve()==Path(declared_policy['path']).resolve(), 'corrected measurement exact prompt-file join')
    require(measured.get('sha256')==digest({k:v for k,v in measured.items() if k!='sha256'})
            and measured.get('projection_kind')=='SYNTH_ARITHMETIC_TOKEN_PROJECTION001'
            and type(measured.get('projection_schema_version')) is int and measured['projection_schema_version']==1
            and type(measured.get('full_result_sha256')) is str and len(measured['full_result_sha256'])==64
            and all(c in '0123456789abcdef' for c in measured['full_result_sha256']),
            'self-sealed corrected projection and explicit full result binding')
    prompts=read(body['prompt_reference']);policy=read(body['policy_reference']);ledger=read(body['ledger_reference'])
    require(prompts['native_scoring_authorized'] is False and measured['status']=='CPU_CONTEXT_MEASURED_NATIVE_UNADMITTED'
            and measured['measured_rows']==512 and measured['unmeasured_rows']==0 and measured['overflows']==[]
            and measured['policy_sha256']==policy['sha256']==ledger['policy_sha256']
            and measured['parent_order_sha256']==ledger['selected_parent_order_sha256'], 'exact measured512 policy/ledger cohort')
    expected=build_plan(prompts['rows'],body['metadata'],measurement_reference=body['measurement_reference'],
        prompt_reference=body['prompt_reference'],policy_reference=body['policy_reference'],ledger_reference=body['ledger_reference'])
    require(canonical(expected)==canonical(body),'exact rebuilt arithmetic source/order/origins/dose plan')
    require(measured['row_order_sha256']==body['row_order_sha256'],'corrected measurement complete512 order')
    return body


def _original():
    if str(PACKAGE) not in sys.path: sys.path.insert(0, str(PACKAGE))
    raw = (PACKAGE / "short_collection_contract.py").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ORIGINAL_CONTRACT_SHA256, "exact original short contract bytes before execution")
    module = ModuleType("selected_original_arithmetic_short_contract"); module.__file__ = str(PACKAGE / "short_collection_contract.py")
    import ast
    tree = ast.parse(raw)
    # Only data-plan code reference changes; all qualification/model/runtime
    # resource/numeric checks remain the original selected function bodies.
    class PlanCode(ast.NodeTransformer):
        def visit_BinOp(self, node):
            if isinstance(node.op, ast.Div) and isinstance(node.left, ast.Name) and node.left.id == "PLAN_DIR" \
                    and isinstance(node.right, ast.Constant) and node.right.value == "collection_plan.py":
                return ast.copy_location(ast.Name(id="ARITHMETIC_PLAN_CODE", ctx=ast.Load()), node)
            return self.generic_visit(node)
    tree = ast.fix_missing_locations(PlanCode().visit(tree))
    exec(compile(tree, module.__file__, "exec"), module.__dict__)
    module.plan = plan; module.select_batch = select_batch
    module.HERE = HERE; module.SOURCE_NAMES = ("arithmetic_capture_contract.py", "arithmetic_capture_pipeline.py", "arithmetic_capture_owned.py")
    module.PLAN_PATH = PLAN_PATH; module.PLAN_DIR = HERE
    module.ARITHMETIC_PLAN_CODE = Path(__file__).resolve()
    module.integrated = integrated
    module.PURPOSE = "arithmetic_M0_raw_collection_one_fixed_batch_only"
    module.AUTHORIZATION = {**module.AUTHORIZATION, "scope": module.PURPOSE}
    return module


def build_scope(kind, ordinal):
    integrated()
    module = _original()  # Full original qualification, runtime, model, resource and numerical checks.
    return module.build_scope(kind, ordinal)


def preflight(path):
    integrated()
    module = _original(); module.build_scope = build_scope
    value = module.preflight(path)
    value["worker_path"] = str(HERE / "arithmetic_capture_owned.py")
    return value
