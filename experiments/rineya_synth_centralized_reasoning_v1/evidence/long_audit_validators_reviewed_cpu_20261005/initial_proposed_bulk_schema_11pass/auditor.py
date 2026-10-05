"""Staged independent native-long audit validators. No execution or authority."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,math,sys
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1] if HERE.name=="long_native_audit_work" and HERE.parent.name=="artifacts" else HERE
sys.path.insert(0,str(PACKAGE))
import long_native_contract as contract
import long_native_core as core
from rs_contract import canonical,digest,require
SIGNAL="ROOT_TERMINAL_AND_PRESERVED"
INVENTORY_KIND="SYNTH_LONG_NATIVE_INDEPENDENT_AUDIT_INVENTORY_V1"
MAX_JSON=16*1024**2
EOS={151643,151645}
def hash64(value):return type(value) is str and len(value)==64 and all(c in "0123456789abcdef" for c in value)
def checksum(value):
    require(type(value) is dict and hash64(value.get("sha256"))
        and value["sha256"]==digest({k:v for k,v in value.items() if k!="sha256"}),"canonical typed body hash")
def validate_anchors(signal,anchors):
    require(signal==SIGNAL,"root terminal and preservation signal required before heavy reads")
    names={"inventory_file_sha256","config_file_sha256","result_file_sha256","supervisor_file_sha256",
        "preservation_body_sha256","encoder_registration_file_sha256","recipe_sha256","source_commit","context_cap","role"}
    require(type(anchors) is dict and set(anchors)==names,"all external independently selected trust anchors")
    require(all(hash64(anchors[k]) for k in names-{"source_commit","context_cap","role"})
        and type(anchors["source_commit"]) is str and len(anchors["source_commit"])==40
        and all(c in "0123456789abcdef" for c in anchors["source_commit"])
        and type(anchors["context_cap"]) is int and anchors["context_cap"] in (1024,2048,4096)
        and anchors["role"] in contract.ROLES,"typed external SHA/HEAD/context/role anchors")
    return True
def counts(value,*,complete=True):
    require(type(value) is dict,"named operation counts")
    for name,item in value.items():
        require(type(name) is str and name and type(item) is dict
            and set(item)=={"attempted","returned","failed","pending"}
            and all(type(n) is int and n>=0 for n in item.values())
            and item["attempted"]==item["returned"]+item["failed"]+item["pending"],"conserved typed actual counts")
        if complete:require(item["failed"]==item["pending"]==0,"partial/failure cannot issue successful native audit")
    return deepcopy(value)
def delta(current,previous):
    counts(current);counts(previous)
    require(all(name in current for name in previous),"no erased previous primitive costs")
    result={}
    for name,item in current.items():
        old=previous.get(name,dict(attempted=0,returned=0,failed=0,pending=0))
        require(all(item[key]>=old[key] for key in item),"no cumulative primitive rollback")
        value={key:item[key]-old[key] for key in item}
        if any(value.values()):result[name]=value
    counts(result);return result
def validate_trace(scope,capsule):
    role=scope["role"];row=scope["row"];trace=capsule["trace"]
    require(type(trace) is list and 1<=len(trace)<=32,"complete actual generated trace")
    for index,item in enumerate(trace):
        require(type(item) is dict and set(item)=={"token","logprob","prefix_length"}
            and type(item["token"]) is int and 0<=item["token"]<151936
            and type(item["logprob"]) in (int,float) and math.isfinite(item["logprob"]) and item["logprob"]<=0
            and type(item["prefix_length"]) is int and item["prefix_length"]==row["prompt_tokens"]+index,
            "every generated token/LP/prefix typed and finite")
    if role=="prefix_stress":
        declared=scope["recipe"]["prefix_stress_ids"]
        require(type(declared) is list and len(declared)==31 and all(type(t) is int and 0<=t<151936 and t not in EOS for t in declared)
            and any(row["prompt_ids"][i:i+31]==declared for i in range(len(row["prompt_ids"])-30)),"separate frozen TRAIN nonEOS stress span")
        require(len(trace)==32 and capsule["greedy_EOS_respected"] is False and capsule["full32_greedy_coverage"] is False
            and capsule["stress_is_greedy"] is False,"stress32 is not fabricated greedy coverage")
    else:
        require(not any(item["token"] in EOS for item in trace[:-1]) and (len(trace)==32 or trace[-1]["token"] in EOS),
            "exact earlyEOS/generation-cap finish")
        require(capsule["greedy_EOS_respected"] is True and capsule["full32_greedy_coverage"] is (len(trace)==32)
            and capsule["stress_is_greedy"] is False,"truthful actual greedy coverage")
    return {"generated_steps":len(trace),"finish":"declared_prefix_stress" if role=="prefix_stress" else "eos" if trace[-1]["token"] in EOS else "generation_cap",
        "full32_greedy_coverage":role!="prefix_stress" and len(trace)==32}
def validate_physical(scope,capsule):
    trace=capsule["trace"];role=scope["role"];row=scope["row"];short=scope["short_parity_row"]["prompt_ids"]
    require(len(short)==419 and scope["dose"]==contract.work(row,role),"exact source-defined419 parity/role dose")
    roster=[("long_native_dense_parity_forward",short),("long_native_zeroB_parity_forward",short)]
    if role=="prefix_stress":
        roster += [("long_native_declared_prefix_stress_forward",row["prompt_ids"]+scope["recipe"]["prefix_stress_ids"][:i]) for i in range(32)]
    elif role=="bulk":
        roster.append(("long_native_bulk_forward",row["prompt_ids"]+[item["token"] for item in trace[:-1]]))
    else:
        roster += [("long_native_greedy_forward",row["prompt_ids"]+[item["token"] for item in trace[:i]]) for i in range(len(trace))]
    expected={"attempted_calls":len(roster),"input_positions":sum(len(ids) for _,ids in roster),
        "records":[{"operation":name,"input_ids_sha256":digest(ids),"input_positions":len(ids)} for name,ids in roster]}
    require(capsule["physical_dose"]==expected,"all actual physical operations/prefix hashes/positions exact")
    require(expected["attempted_calls"]<=scope["dose"]["maximum_forward_calls"]
        and expected["input_positions"]<=scope["dose"]["maximum_forward_input_positions"]<=134144
        and all(len(ids)<=scope["dose"]["context_cap"] for _,ids in roster),"unchanged per-scope physical caps")
    return expected
def validate_chain_memory(scope,result,previous):
    checksum(result);node=result["capsule_chain"];checksum(node)
    capsule=core.validate_chain_memory(node,scope["recipe"]);checksum(capsule)
    require(node["kind"]==contract.KIND and capsule["kind"]==contract.KIND and capsule["role"]==scope["role"]
        and capsule["CPU_fixture"] is False and result["execution_scope"]==contract.NATIVE
        and result["status"]=="COMPLETE_NATIVE_LONG_SCOPE_PENDING_INDEPENDENT_AUDIT","actual completed same-purpose native scope only")
    require(all(capsule.get(k) is False for k in ("native_launch_authorized","native_long_context_qualified","study_training_authorized","scoring_authorized"))
        and all(result.get(k) is False for k in ("native_launch_authorized","native_long_context_qualified","study_training_authorized",
            "guard_enrollment_authorized","scoring_authorized")),"qualification never study/enrollment/scoring authority")
    expected_roles=[] if scope["role"]=="trace1" else list(contract.ROLES[:contract.ROLES.index(scope["role"])])
    require([item["role"] for item in previous]==expected_roles,"all actual audited previous roles in order")
    old=previous[-1]["capsule_chain"]["cumulative_costs"] if previous else {"operations":{},"physical_calls":0,"input_positions":0}
    require(capsule["previous_capsule_sha256"]==(previous[-1]["capsule_chain"]["sha256"] if previous else None),"latest actual capsule lineage no rollback")
    require(result["actual_costs"]==node["cumulative_costs"] and result["partial_trace"]==capsule["trace"]
        and result["physical_dose"]==capsule["physical_dose"],"terminal trace/physical/cumulative exact")
    for costs in (capsule["posted_scope_costs"],node["terminal"]["costs"],node["cumulative_costs"]):
        require(type(costs) is dict and set(costs)=={"operations","physical_calls","input_positions"},"closed cumulative cost schema")
        counts(costs["operations"])
        require(all(type(costs[k]) is int and costs[k]>=old[k] for k in ("physical_calls","input_positions")),"cumulative physical costs cannot rollback")
        require(costs["physical_calls"]==old["physical_calls"]+capsule["physical_dose"]["attempted_calls"]
            and costs["input_positions"]==old["input_positions"]+capsule["physical_dose"]["input_positions"],"whole cumulative physical dose includes every role")
    return {"capsule":capsule,"current_operations":delta(node["cumulative_costs"]["operations"],old["operations"])}
def validate_bulk_records(scope,capsule,previous):
    """Every bulk LP requires durable per-step measurements; no summary-only PASS."""
    require(scope["role"]=="bulk" and len(previous)==2,"bulk requires both audited trace scopes")
    traces=[item["capsule_chain"]["capsule"]["trace"] for item in previous]
    require(traces[0]==traces[1]==capsule["trace"],"cross-process actual complete greedy repeats identical")
    comparison=capsule["comparison"];records=comparison.get("records")
    require(type(records) is list and len(records)==len(traces[0]),"durable every-step actual bulk token/LP/error records required")
    errors=[];different=[]
    for index,(actual,reference) in enumerate(zip(records,traces[0])):
        require(type(actual) is dict and set(actual)=={"index","position","reference_token","reference_logprob","actual_argmax_token","actual_selected_logprob","absolute_error"}
            and actual["index"]==index and actual["position"]==scope["row"]["prompt_tokens"]-1+index
            and type(actual["reference_token"]) is int and actual["reference_token"]==reference["token"]
            and actual["reference_logprob"]==reference["logprob"] and type(actual["actual_argmax_token"]) is int
            and 0<=actual["actual_argmax_token"]<151936
            and type(actual["actual_selected_logprob"]) in (int,float) and math.isfinite(actual["actual_selected_logprob"])
            and actual["actual_selected_logprob"]<=0 and type(actual["absolute_error"]) in (int,float)
            and math.isfinite(actual["absolute_error"]) and actual["absolute_error"]>=0,"typed exact per-generated-step bulk measurement")
        require(actual["absolute_error"]==abs(actual["actual_selected_logprob"]-reference["logprob"]),"independent error arithmetic from stored actual bulk LP")
        errors.append(actual["absolute_error"])
        if actual["actual_argmax_token"]!=reference["token"]:different.append(index)
    require(comparison["actual_compared_steps"]==len(records) and comparison["argmax_differences"]==different
        and comparison["maximum_error"]==max(errors) and comparison["mean_error"]==sum(errors)/len(errors)
        and not different and max(errors)<=.03125 and sum(errors)/len(errors)<=.01,"complete bulk max/mean/argmax numerical gates")
    return True

