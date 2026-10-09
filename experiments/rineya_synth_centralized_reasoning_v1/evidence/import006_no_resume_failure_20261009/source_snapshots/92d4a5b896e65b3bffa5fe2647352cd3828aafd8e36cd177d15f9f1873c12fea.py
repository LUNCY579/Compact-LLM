"""Staged independent native-long audit validators. No execution or authority."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,math,sys
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1] if HERE.name in ("long_native_audit_work","long_v2_audit_adaptation_work") and HERE.parent.name=="artifacts" else HERE
sys.path.insert(0,str(PACKAGE))
import long_native_contract as contract
from rs_contract import canonical,digest,require
SIGNAL="ROOT_TERMINAL_AND_PRESERVED"
INVENTORY_KIND="SYNTH_LONG_NATIVE_INDEPENDENT_AUDIT_INVENTORY_V1"
MAX_JSON=16*1024**2
EOS={151643,151645}
NATIVE_DRIVER_READY=False

def v2_merge(*maps):
    result={}
    for mapping in maps:
        require(type(mapping) is dict,"runtime hash map")
        for path,pin in mapping.items():
            require(type(path) is str and Path(path).is_absolute() and hash64(pin),"absolute typed runtime SHA")
            require(path not in result or result[path]==pin,"conflicting inherited/additive hash")
            result[path]=pin
    return dict(sorted(result.items()))
def validate_v2_registration_schema(scope):
    from stdlib_inventory import BASE,PHYSICAL_BASE,VERSION,PYTHON_VERSION
    reg=scope.get("encoder_runtime_registration");checksum(reg)
    fields={"schema_version","kind","policy","evidence_anchors","encoder_inventory","stdlib_inventory","additional_runtime_pins","runtime_pins",
        "prompt_registration_preview","inherited_runtime_pins","consumption_source_pins","complete_declared_encoder_execution_scope",
        "complete_machine_trust_claimed","native_launch_authorized","native_qualification_established","study_training_authorized","scoring_authorized","sha256"}
    require(set(reg)==fields and type(reg["schema_version"]) is int and reg["schema_version"]==2
        and reg["kind"]=="SYNTH_LONG_ENCODER_STDLIB_RUNTIME_REGISTRATION_V2"
        and reg["policy"]=="ENCODER45_FINITE_HERMES_STDLIB_EXECUTION_SCOPE_V2","exact V2 declaration schema")
    require(reg["complete_declared_encoder_execution_scope"] is True and all(reg[k] is False for k in
        ("complete_machine_trust_claimed","native_launch_authorized","native_qualification_established","study_training_authorized","scoring_authorized")),"declared finite closure only")
    anchors=reg["evidence_anchors"]
    require(type(anchors) is dict and set(anchors)=={"package","stdlib","package_review","stdlib_review"}
        and all(type(item) is dict and set(item)=={"path","file_sha256","body_sha256"} and Path(item["path"]).is_absolute()
            and hash64(item["file_sha256"]) and hash64(item["body_sha256"]) for item in anchors.values()),"four complete external evidence anchors")
    enc,std=reg["encoder_inventory"],reg["stdlib_inventory"]
    require(enc["versions"]=={"tokenizers":"0.22.2","jinja2":"3.1.6","markupsafe":"3.0.3"} and set(enc["packages"])=={"tokenizers","jinja2","markupsafe"},"three exact encoder versions")
    require(set(enc["runtime_pins"])==set(enc["paths"])==set(enc["files"]) and len(enc["runtime_pins"])==45
        and all(enc["files"][path]["path"]==path and enc["files"][path]["sha256"]==pin for path,pin in enc["runtime_pins"].items())
        and {name:len(item["files"]) for name,item in enc["packages"].items()}=={"tokenizers":17,"jinja2":25,"markupsafe":3}
        and set(enc["runtime_pins"])=={path for item in enc["packages"].values() for path in item["files"]},"complete exact encoder45 roster/file map")
    alias={"logical_base_prefix":str(BASE.absolute()),"physical_base_prefix":str(PHYSICAL_BASE.absolute()),"file_attributes":0x410,"reparse_tag":0xa0000003,"is_symlink":False}
    require(std["fixture"] is False and std["policy"]==VERSION and std["python_version"]==PYTHON_VERSION=="3.11.15"
        and std["approved_alias_binding"]==alias and std["base_prefix"]==alias["logical_base_prefix"] and std["physical_base_prefix"]==alias["physical_base_prefix"]
        and std["excluded_directory_names"]==["site-packages","__pycache__"]
        and std["Windows_system_driver_scope"]=="inherited separately; no machine closure claim","exact finite policy/known alias")
    require(std["optional_stdlib_zip"]=={"path":str(PHYSICAL_BASE/"python311.zip"),"present":False},"exact absent optional ZIP")
    require(set(std["runtime_pins"])==set(std["paths"])==set(std["files"]) and len(std["runtime_pins"])==779
        and all(std["files"][path]["path"]==path and std["files"][path]["sha256"]==pin for path,pin in std["runtime_pins"].items())
        and {category:list(std["paths"].values()).count(category) for category in set(std["paths"].values())}
            =={"stdlib_python_source":738,"stdlib_extension_runtime":34,"interpreter_root_executable_runtime":6,"executing_Hermes_launcher":1},"complete exact finite779 categories/files")
    inherited=reg["inherited_runtime_pins"]
    require(type(inherited) is dict and len(inherited)==65 and inherited==contract.base_runtime_pins(),"every exact qualified current inherited65 pin retained")
    additional=v2_merge(enc["runtime_pins"],std["runtime_pins"]);union=v2_merge(inherited,additional)
    require(reg["additional_runtime_pins"]==additional and len(additional)==824 and reg["runtime_pins"]==union==scope["runtime_pins"] and len(union)==887,"entire conflict-free runtime887 union")
    preview=reg["prompt_registration_preview"];checksum(preview)
    require(preview["runtime_pins"]==enc["runtime_pins"] and preview["dependency_versions"]==enc["versions"] and preview["executing_package_roster"]==enc["packages"],"prompt/package declaration joined")
    from encoder_registration_v2 import expected_sources
    require(reg["consumption_source_pins"]==expected_sources(),"all four exact current reviewed V2 consumption source pins")
    require(scope["recipe"]["declared_execution_scope_registration_sha256"]==reg["sha256"],"recipe binds selected declaration body")
    return reg
def validate_runtime_observation_v2(scope,maps,phase):
    reg=validate_v2_registration_schema(scope)
    require(type(maps) is dict and set(maps)=={"source","runtime","data","model","encoder_assets","encoder_bindings","runtime_observation"},"exact actual admitted7-section maps")
    proof=maps["runtime_observation"]
    require(type(proof) is dict and set(proof)=={"policy","phase","actual_additional_runtime_pins","stdlib_roster","encoder_versions","native_qualification_established","complete_machine_trust_claimed"}
        and proof["policy"]=="ADMITTED_ENCODER45_FINITE_STDLIB_BYTE_OBSERVATION_V2" and proof["phase"]==phase and phase in ("before_import","after_math")
        and proof["native_qualification_established"] is False and proof["complete_machine_trust_claimed"] is False,"exact V2 actual observation phase/schema/closed flags")
    require(proof["actual_additional_runtime_pins"]==reg["additional_runtime_pins"] and proof["encoder_versions"]==reg["encoder_inventory"]["versions"],"all actual admitted824 hashes/versions")
    std=reg["stdlib_inventory"]
    require(proof["stdlib_roster"]=={"paths":std["paths"],"approved_alias_binding":std["approved_alias_binding"],"optional_stdlib_zip":std["optional_stdlib_zip"]},"exact actual alias/roster/ZIP")
    expected={"source":scope["source_pins"],"runtime":scope["runtime_pins"],"data":scope["data_pins"],"model":scope["model_pins"],
        "encoder_assets":{str(Path(scope["tokenizer_binding"]["model_dir"])/name):item["sha256"] for name,item in scope["tokenizer_binding"]["assets"].items()},"encoder_bindings":scope["tokenizer_binding"]["binding_sources"]}
    require({key:value for key,value in maps.items() if key!="runtime_observation"}==expected,"all actual observed six maps including887 runtime")
    return deepcopy(proof)
def normalize_runtime_phase_only(maps):
    value=deepcopy(maps);require("runtime_observation" in value and "phase" in value["runtime_observation"],"actual observation phase required")
    value["runtime_observation"].pop("phase")
    return value


def validate_native_admission(scope,result,sup):
    """Recorded proof mirrors the literal owned callback; no new resource probe."""
    proof=result.get("native_admission")
    request={"purpose":contract.PURPOSE,"recipe_sha256":scope["recipe"]["sha256"],"role":scope["role"],"seed":101}
    require(type(proof) is dict and set(proof)=={"binding","owned_supervision_verified","model_load_and_constructor_in_scope",
        "observed_monotonic","expires_monotonic"} and proof["binding"]==request
        and proof["owned_supervision_verified"] is True and proof["model_load_and_constructor_in_scope"] is True,
        "durable exact owned native admission before factory")
    observed,expires=proof["observed_monotonic"],proof["expires_monotonic"]
    require(type(observed) in (int,float) and type(expires) in (int,float) and math.isfinite(observed) and math.isfinite(expires)
        and sup["worker_resume_monotonic"]<=observed<=expires<=observed+.2 and expires<sup["deadline_monotonic"],
        "typed finite admission within unchanged200ms/resume/deadline")
    return True

def audit(attempt_path,preserved_path,config_path,inventory_path,*,terminal_signal,expected,output_path=None):
    """Future real terminal entrypoint, intentionally closed at this milestone.

    Pure validators below do not authenticate externally selected artifacts.
    Complete encoder+stdlib closure and persisted per-factory admission must be
    independently reviewed before implementing/enabling native PASS emission.
    """
    validate_anchors(terminal_signal,expected)
    require(NATIVE_DRIVER_READY,"native-long audit driver remains pending complete independently reviewed encoder/stdlib execution closure and durable admission provenance; CPU fixtures cannot emit native PASS")
    raise ValueError("native-long terminal driver not implemented; no native audit or resume authority")
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

def validate_observed_state(scope,capsule,owner):
    """Native byte-hash observations are bound; raw RNG/tensor decoding is absent."""
    start=capsule["start_proof"];end=capsule["end_proof"];common=start["identity_common"]
    require(start["kind"]=="NATIVE_OBSERVED_LONG_START" and end["kind"]=="NATIVE_OBSERVED_LONG_END"
        and end["identity_common"]==common and start["owned_scope_identity"]==owner,"actual fresh owned start/end identity")
    validate_runtime_observation_v2(scope,start["actual_pin_maps"],"before_import")
    validate_runtime_observation_v2(scope,end["actual_pin_maps_after"],"after_math")
    require(normalize_runtime_phase_only(start["actual_pin_maps"])==normalize_runtime_phase_only(end["actual_pin_maps_after"])
        and common["source_byte_maps_sha256"]==digest(start["actual_pin_maps"]),"complete maps agree excluding ONLY phase; before digest retains original observation")
    require(common["recipe_sha256"]==scope["recipe"]["sha256"] and common["seed"]==101,"same frozen native recipe/seed")
    for rng in (common["dense_start_rng"],common["inference_start_rng"],end["RNG_after"]):
        require(type(rng) is dict and set(rng)=={"python_sha256","cpu","cuda0"} and all(hash64(v) for v in rng.values()),"three typed observed RNG hashes")
    require(end["RNG_unchanged"] is True and end["RNG_after"]==common["inference_start_rng"],"Python/CPU/CUDA0 RNG observed hash preservation")
    encoder=start["encoder"]
    require(encoder=={"short_hash_kind":"SHA256_RENDERED_UTF8_CHAT_TEMPLATE","long_hash_kind":"SHA256_CANONICAL_MESSAGES_JSON",
        "short_tokens":419,"long_tokens":scope["row"]["prompt_tokens"],"full_retokenization_matches":True,
        "tokenizer_binding_sha256":digest(scope["tokenizer_binding"]),"native_long_context_qualified":False}
        and common["encoder_proof_sha256"]==digest(encoder),"complete actual419/full-long encoding with distinct typed prompt hash conventions")
    from native_loader import _source_shapes,_adapter_shapes
    loader=start["loader"];source=loader["source_tensor_sha256"];shapes=_adapter_shapes();names=list(shapes)
    require(set(source)==set(_source_shapes()) and len(source)==310 and all(hash64(pin) for pin in source.values())
        and common["model_source_map_sha256"]==digest(source),"exact310 native base source hash roster")
    require(loader["unique_base_tensors"]==loader["serialized_tensors"]==310 and loader["unique_base_parameters"]==596049920
        and loader["base_dtype"]=="bfloat16" and loader["base_frozen"] is True and loader["embedding_head_tie"] is True
        and loader["memory_mapped"] is False and loader["cpu_model_allocated"] is False and loader["FP32_base_copy_allocated"] is False
        and loader["blocking_chunk_copies"] is True,"qualified exact dense BF16 loader/storage/tie profile")
    require(set(loader["asset_hashes"])==set(scope["model_pins"]) and all(loader["asset_hashes"][name]["sha256"]==pin for name,pin in scope["model_pins"].items()),"loader all7 bound model assets")
    adapters=start["adapters"];inventory=[{"name":name,"shape":shapes[name],"dtype":"float32","device":"cuda:0","requires_grad":True,
        "parameters":math.prod(shapes[name])} for name in names]
    require(adapters["inventory"]==inventory and adapters["rank"]==8 and adapters["alpha"]==16 and adapters["dropout"]==0.
        and adapters["qv_site_count"]==56 and adapters["layer_count"]==28 and adapters["trainable_tensors"]==112
        and adapters["trainable_parameters"]==1146880,"all112 actual native q/v shapes/order/F32/flags/rank/dose")
    before=start.get("adapter_byte_hashes_before");after=end.get("adapter_byte_hashes_after")
    require(type(before) is dict and len(before)==112 and set(before)==set(shapes) and all(hash64(pin) for pin in before.values())
        and after==before and common["initial_adapter_sha256"]==digest(before),"all112 individual observed adapter hashes before-after")
    require(start["actual_dense_zeroB_parity"]=={"equal":True,"maximum_error":0.}
        and all(proof["all_B_zero"] is True and proof["qv_site_count"]==56 and proof["trainable_tensors"]==112
            for proof in (start["zeroB"],end["zeroB_after"])) and end["zeroB_unchanged"] is True,"actual fresh dense versus zeroB output parity and zeroB preserved")
    compute=start["compute"]
    require(compute["base_parameter_objects_and_pointers_preserved"] is True and compute["unique_base_tensors"]==310
        and compute["unique_base_parameters"]==596049920 and compute["base_storage_dtype"]=="bfloat16"
        and compute["activation_dtype"]=="float32" and compute["linear_site_count"]==197 and compute["embedding_site_count"]==1,
        "unchanged qualified BF16/F32 compute-site inventory")
    post=end["base_postcheck"]
    require(end["base310_unchanged"] is True and post["verified_source_tensors"]==310 and post["frozen_base_unchanged"] is True
        and post["unique_base_parameters"]==596049920 and post["base_dtype"]=="bfloat16" and post["embedding_head_tie"] is True,"all310 actual post-native content check")
    active=end["activation_observations"]
    expected={"model.norm","model.rotary_emb"}
    for index in range(28):
        expected.update("model.layers.%d%s"%(index,suffix) for suffix in ("",".input_layernorm",".post_attention_layernorm",
            ".self_attn.q_norm",".self_attn.k_norm",".self_attn",".mlp"))
    require(type(active) is dict and set(active)==expected and len(active)==198
        and all(type(v) is int and v==capsule["physical_dose"]["attempted_calls"] for v in active.values()),"every198 FP32 activation site at every physical model call")
    return {"adapter_hashes":112,"base_hashes":310,"activation_sites":198,"RNG_hash_fields":3,
        "raw_RNG_encoding_or_restore_verified":False,"independent_resident_tensor_reread_by_auditor":False,
        "limits":"These are source-bound actual native hash/query observations and returned verification calls; no raw RNG state/tensor bytes are saved in this capsule and the auditor performs no replay/restore/model reads."}
def file_info(path,*,reader=None):
    if reader is not None:return reader.info(path)
    path=Path(path);require(path.is_file() and not path.is_symlink(),"regular immutable audit file")
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda:stream.read(1024**2),b""):value.update(data)
    return {"bytes":path.stat().st_size,"sha256":value.hexdigest()}
def finite_float(value):
    parsed=float(value);require(math.isfinite(parsed),"finite selected JSON float required");return parsed

def read_json(path,*,reader=None):
    if reader is not None:return reader.json(path)
    path=Path(path);require(path.is_file() and not path.is_symlink(),"regular JSON evidence")
    with path.open("rb") as stream:raw=stream.read(MAX_JSON+1)
    require(len(raw)<=MAX_JSON,"bounded audit JSON evidence")
    return json.loads(raw.decode("utf-8"),object_pairs_hook=contract.native._pairs,
        parse_constant=contract.native._bad_constant,parse_float=finite_float)
def owned_file(path,leaf):
    path=Path(path);require(path.is_absolute() and path.is_file() and not path.is_symlink()
        and path.resolve().parent==Path(leaf).resolve(),"evidence file belongs to exact owned immutable leaf")
    return path
def validate_event_journal(path,leaf,*,reader=None):
    require((reader.info(path,32*1024**2) if reader is not None else file_info(path))["bytes"]<=32*1024**2,"bounded registered API event stream")
    calls={};operations={};active=[];publications=[];measurements=[];ordered=[];seen_files=set();last_id=0
    with Path(path).open("rb") as stream:
        for line,raw in enumerate(stream,1):
            require(len(raw)<=65536 and raw.endswith(b"\n"),"complete bounded event line")
            event=json.loads(raw.decode(),object_pairs_hook=contract.native._pairs,
                parse_constant=contract.native._bad_constant,parse_float=finite_float)
            kind=event.get("event")
            if kind=="attempt":
                identity,name=event["id"],event["operation"]
                require(type(identity) is int and identity==last_id+1 and type(name) is str and name,"contiguous named registered API id")
                last_id=identity;calls[identity]=name;active.append(identity)
                item=operations.setdefault(name,dict(attempted=0,returned=0,failed=0,pending=0));item["attempted"]+=1;item["pending"]+=1
                ordered.append((identity,name,kind,line))
            elif kind in ("returned","failed"):
                identity,name=event["id"],event["operation"]
                require(active and active.pop()==identity and calls.get(identity)==name,"exact once synchronous matching API return")
                operations[name]["pending"]-=1;operations[name][kind]+=1;ordered.append((identity,name,kind,line))
            elif kind=="long_measurement_preserved":
                descriptor=event["publication"];path=owned_file(descriptor["path"],leaf)
                require(str(path) not in seen_files,"unique advertised publication file");seen_files.add(str(path))
                payload=read_json(path,reader=reader);verify_publication(descriptor,payload,reader=reader)
                publications.append({"name":event["name"],"publication":descriptor,"payload":payload,"line":line})
            elif kind=="measurement_preserved":
                name=event["path"];require(type(name) is str and "/" not in name and "\\" not in name,"bounded stage evidence name")
                path=owned_file(str(Path(leaf)/name),leaf);require(str(path) not in seen_files and file_info(path,reader=reader)["sha256"]==event["sha256"],"unique exact stage measurement bytes")
                seen_files.add(str(path));measurements.append({"path":path.name,"line":line,"sha256":event["sha256"]})
            else:raise ValueError("unknown/failure journal entry cannot establish native success")
    require(not active,"no pending outer API");counts(operations)
    return {"operations":operations,"calls":last_id,"publications":publications,"measurements":measurements,"ordered":ordered}
def validate_publications(scope,result,previous,journal,leaf):
    capsule=chain_payload(result["capsule_chain"],scope["recipe"])
    node=result["capsule_chain"];trace=capsule["trace"];role=scope["role"]
    prefixes=["long_token_%02d"%i for i in range(1,len(trace)+1)] if role in ("trace1","trace2") else[]
    expected=prefixes+["long_capsule_publish","long_terminal_publish","long_capsule_promote"]
    require([item["name"] for item in journal["publications"]]==expected,"every actual token prefix and closed publication sequence")
    old=previous[-1]["capsule_chain"]["cumulative_costs"] if previous else {"operations":{},"physical_calls":0,"input_positions":0}
    for index,item in enumerate(journal["publications"][:len(prefixes)],1):
        payload=item["payload"];checksum(payload)
        require(payload["purpose"]==contract.PURPOSE and payload["recipe_sha256"]==scope["recipe"]["sha256"] and payload["role"]==role
            and payload["prefix"]==trace[:index],"exact every-step preserved complete trace prefix")
        token_costs=payload["costs"];counts(token_costs["operations"])
        require(token_costs["physical_calls"]==old["physical_calls"]+2+index
            and token_costs["input_positions"]==old["input_positions"]+838+index*scope["row"]["prompt_tokens"]+index*(index-1)//2,"every token cost/physical dose includes actual previous lineage")
    by_name={item["name"]:item for item in journal["publications"]}
    for name,key,payload in (("long_capsule_publish","capsule_publication",capsule),("long_terminal_publish","terminal_publication",node["terminal"]),
        ("long_capsule_promote","promotion_publication",node["promotion"])):
        require(by_name[name]["publication"]==node[key] and by_name[name]["payload"]==payload,"journalâ†’immutable capsule/terminal/promotion bytes")
        owned_file(node[key]["path"],leaf)
    proof=validate_chain_memory(scope,result,previous)
    require(proof["current_operations"]==journal["operations"],"whole current registered API delta includes construction/imports/hash/parity/cleanup/preservation")
    require(result["operation_accounting"]=={"scope":"registered_API_boundaries;internal_ATen_operations_not_individually_metered",
        "operations":journal["operations"],"last_call":journal["calls"]},"worker original Journal current cost accounting exact")
    return proof

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
    capsule=chain_payload(node,scope["recipe"]);checksum(capsule)
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
    comparison=capsule["comparison"];records=comparison.get("bulk_steps")
    require(type(records) is list and len(records)==len(traces[0]),"durable every-step actual bulk token/LP/error records required")
    errors=[];different=[]
    for index,(actual,reference) in enumerate(zip(records,traces[0])):
        require(type(actual) is dict and set(actual)=={"step","position","argmax_token","argmax_logprob","selected_token","selected_logprob","trace_logprob","absolute_error"}
            and type(actual["step"]) is int and actual["step"]==index and type(actual["position"]) is int
            and actual["position"]==scope["row"]["prompt_tokens"]-1+index
            and type(actual["selected_token"]) is int and actual["selected_token"]==reference["token"]
            and type(actual["trace_logprob"]) in (int,float) and actual["trace_logprob"]==reference["logprob"]
            and type(actual["argmax_token"]) is int and 0<=actual["argmax_token"]<151936
            and all(type(actual[k]) in (int,float) and math.isfinite(actual[k]) and actual[k]<=0 for k in ("argmax_logprob","selected_logprob","trace_logprob"))
            and type(actual["absolute_error"]) in (int,float)
            and math.isfinite(actual["absolute_error"]) and actual["absolute_error"]>=0,"typed exact per-generated-step bulk measurement")
        require(actual["absolute_error"]==abs(actual["selected_logprob"]-reference["logprob"]),"independent error arithmetic from stored actual bulk LP")
        errors.append(actual["absolute_error"])
        if actual["argmax_token"]!=reference["token"]:different.append(index)
        else:require(actual["argmax_logprob"]==actual["selected_logprob"],"same argmax/reference token exact selected LP")
    require(comparison["actual_compared_steps"]==len(records) and comparison["argmax_differences"]==different
        and comparison["maximum_error"]==max(errors) and comparison["mean_error"]==sum(errors)/len(errors)
        and not different and max(errors)<=.03125 and sum(errors)/len(errors)<=.01,"complete bulk max/mean/argmax numerical gates")
    return True

def project_costs(costs,operation):
    require(type(costs) is dict and set(costs)=={"operations","physical_calls","input_positions"},"closed whole cumulative cost snapshot")
    counts(costs["operations"]);value=deepcopy(costs)
    require(all(type(value[k]) is int and value[k]>=0 for k in ("physical_calls","input_positions")),"typed whole cumulative physical costs")
    for suffix in ("write","verify"):
        name=operation+"_"+suffix
        item=value["operations"].setdefault(name,dict(attempted=0,returned=0,failed=0,pending=0))
        item["attempted"]+=1;item["returned"]+=1
    counts(value["operations"]);return value
def chain_payload(node,recipe):
    checksum(node)
    require(type(node) is dict and set(node)=={"kind","recipe_sha256","capsule","capsule_publication","terminal","terminal_publication",
        "promotion","promotion_publication","cumulative_costs","sha256"} and node["kind"]==contract.KIND
        and node["recipe_sha256"]==recipe["sha256"],"closed same-recipe native capsule chain")
    payload=node["capsule"];checksum(payload)
    require(payload["purpose"]==contract.PURPOSE and payload["recipe_sha256"]==recipe["sha256"],"qualification-only typed capsule recipe/purpose")
    terminal=node["terminal"];promotion=node["promotion"]
    require(type(terminal) is dict and set(terminal)=={"capsule_sha256","costs"}
        and terminal["capsule_sha256"]==payload["sha256"],"exact capsuleâ†’terminal body linkage")
    projected=project_costs(project_costs(payload["posted_scope_costs"],"long_capsule_publish"),"long_terminal_publish")
    require(terminal["costs"]==projected,"independent postedâ†’capsuleâ†’terminal actual primitive count projection")
    require(type(promotion) is dict and set(promotion)=={"capsule_sha256","terminal_sha256","costs"}
        and promotion["capsule_sha256"]==payload["sha256"] and promotion["terminal_sha256"]==digest(terminal)
        and promotion["costs"]==node["cumulative_costs"]==project_costs(terminal["costs"],"long_capsule_promote"),
        "independent terminalâ†’promotion exact cumulative costs")
    return deepcopy(payload)
def verify_publication(descriptor,payload,*,reader=None):
    require(type(descriptor) is dict and set(descriptor)=={"path","bytes","sha256","payload_sha256","durable"}
        and descriptor["durable"] is True and descriptor["payload_sha256"]==digest(payload)
        and hash64(descriptor["sha256"]) and type(descriptor["bytes"]) is int and 0<descriptor["bytes"]<=MAX_JSON,"closed durable typed publication")
    info=file_info(descriptor["path"],reader=reader);require(info=={"bytes":descriptor["bytes"],"sha256":descriptor["sha256"]},"independent exact publication file byte count/hash")
    require(canonical(read_json(descriptor["path"],reader=reader))==canonical(payload),"independent exact publication payload")
    return True

def verify_resources(scope,sup,leaf,*,reader=None):
    bounds={"worker_process_tree_RSS_GiB":2.0,"system_available_reserve_GiB":1.25,"preworker_available_GiB":3.3125,
        "GPU_free_at_admission_GiB":4.0,"GPU_free_during_work_GiB":.5,"disk_free_GiB":4.0,"whole_scope_seconds":600,
        "external_watchdog_milliseconds":100}
    cadence={"CPU_watchdog_seconds":.1,"CPU_freshness_seconds":.2,"GPU_query_interval_seconds":.5,
        "GPU_query_timeout_seconds":2.0,"GPU_freshness_seconds":2.0,"admission_interval_seconds":2.,
        "admission_deadline_seconds":30.,"admission_samples":2}
    require(sup.get("bounds")==bounds and sup.get("cadence")==cadence,"unchanged exact supervised amounts/cadence/freshness")
    require(sup.get("admission_only") is False and all(sup.get(k) is True for k in (
        "worker_attempted","worker_launched","worker_resumed","owned_handle_alive_verified_after_launch",
        "preflight_revalidated_before_current_observations","job_closed_kill_on_close",
        "sampler_threads_terminal_verified","gpu_telemetry_cleanup_verified")) and
        sup.get("owned_worker_identity_currently_alive") is False and sup.get("unjoined_sampler_threads")==[],
        "one completed owned launch/Job/sampler/NVML terminal closure")
    require(not any(sup.get(k) for k in ("failure","termination_error","containment_failure","sampler_shutdown_error","gpu_telemetry_cleanup_error")),
        "no contradictory terminal cleanup failure")
    admissions=sup.get("admission_observations")
    require(type(admissions) is list and len(admissions)==2,"two fresh native admissions")
    uuid=admissions[0]["gpu_uuid"]
    def cpu_fresh(row,now):
        vals=[row.get(k) for k in ("sample_started_monotonic","sample_completed_monotonic")]
        require(all(type(v) in (int,float) and math.isfinite(v) for v in vals)
            and vals[0]<=vals[1]<=now and 0<=now-vals[0]<=.2
            and all(type(row.get(k)) is int and row[k]>=0 for k in ("host_available_bytes","worker_tree_rss_bytes","disk_free_bytes")),"fresh typed chronological CPU envelope")
    def gpu_fresh(row,now):
        vals=[row.get(k) for k in ("gpu_sample_started_monotonic","gpu_sample_completed_monotonic")]
        require(all(type(v) in (int,float) and math.isfinite(v) for v in vals) and vals[0]<=vals[1]<=now and 0<=now-vals[0]<=2
            and row.get("GPU_index")==0 and type(row.get("GPU_index")) is int
            and type(row.get("gpu_uuid")) is str and bool(row["gpu_uuid"]) and row["gpu_uuid"]==uuid
            and type(row.get("gpu_observation_id")) is str and bool(row["gpu_observation_id"])
            and type(row.get("gpu_free_bytes")) is int
            and type(row.get("gpu_total_bytes")) is int and 0<=row["gpu_free_bytes"]<=row["gpu_total_bytes"],"fresh typed pinned GPU envelope")
    for row in admissions+[sup["dispatch_observation"]]:
        now=max(row["sample_completed_monotonic"],row["gpu_sample_completed_monotonic"])
        cpu_fresh(row,now);gpu_fresh(row,now)
        require(row["gpu_free_bytes"]>=4*1024**3 and row["host_available_bytes"]>=3.3125*1024**3
            and row["disk_free_bytes"]>=4*1024**3 and row["worker_tree_rss_bytes"]==0,"preworker amount floors")
    require(admissions[1]["sample_started_monotonic"]-admissions[0]["sample_started_monotonic"]>=2.
        and sup["dispatch_observation"]["gpu_observation_id"]==admissions[-1]["gpu_observation_id"],"fixed two-observation admission")
    owner=read_json(Path(leaf)/"supervisor_ownership.json",reader=reader);identity=sup["worker_identity"]
    ownership=read_json(Path(leaf)/"stage_worker_ownership_verified.json",reader=reader)
    require(identity["assigned_before_resume"] is True and identity["job_kill_on_close"] is True
        and identity["breakaway_allowed"] is False and owner["worker_pid"]==identity["pid"]
        and owner["job_name"]==identity["job_name"]==ownership["job_name"]
        and owner["config_sha256"]==sup["config_sha256"] and ownership["supervisor_pid"]==owner["supervisor_pid"]
        and ownership["launcher_root_pid"]==identity["pid"] and ownership["owned_job_membership_verified"] is True
        and ownership["query_job_handle_closed_before_framework_import"] is True,"owned launcher/actual child/Job before framework")
    resources=read_json(Path(leaf)/"RESOURCE_OBSERVATIONS.json",reader=reader)
    require(type(resources) is list and resources and all(row.get("kind") in ("CPU","GPU") for row in resources),"complete bounded CPU/GPU resource record inventory")
    cpu=[r for r in resources if r["kind"]=="CPU"];gpu=[r for r in resources if r["kind"]=="GPU"]
    require(len(cpu)>=2 and len(gpu)>=2 and cpu[0].get("phase")=="worker_suspended_startup"
        and cpu[0]["sample_completed_monotonic"]<=sup["worker_resume_monotonic"],"external ready monitor while owned worker suspended")
    observed={}
    for row in cpu:
        cpu_fresh(row,row["sample_completed_monotonic"])
        require(row["host_available_bytes"]>=1.25*1024**3 and row["worker_tree_rss_bytes"]<=2*1024**3
            and row["disk_free_bytes"]>=4*1024**3,"all returned CPU/RSS/disk amount gates")
        keys=("pid","create_time","creation_filetime_100ns")
        require([{k:item[k] for k in keys} for item in row["owned_process_samples"]]==row["owned_process_identities"]
            and sum(item["rss_bytes"] for item in row["owned_process_samples"])==row["worker_tree_rss_bytes"],"owned identity/RSS sum exact")
        for item in row["owned_process_samples"]:
            require(all(type(item[k]) is int and item[k]>0 for k in ("pid","creation_filetime_100ns"))
                and type(item["create_time"]) in (int,float) and math.isfinite(item["create_time"])
                and type(item["rss_bytes"]) is int and item["rss_bytes"]>=0
                and row["sample_started_monotonic"]<=item["sample_started_monotonic"]<=item["sample_completed_monotonic"]<=row["sample_completed_monotonic"],
                "all per-process identities and timing typed")
            previous=observed.setdefault(item["pid"],item["create_time"]);require(previous==item["create_time"],"no reused identity silently merged")
    for row in gpu:
        gpu_fresh(row,row["gpu_sample_completed_monotonic"]);require(row["gpu_free_bytes"]>=.5*1024**3,"all runtime GPU amount gates")
    for rows,start,end,age in ((cpu,"sample_started_monotonic","sample_completed_monotonic",.2),(gpu,"gpu_sample_started_monotonic","gpu_sample_completed_monotonic",2.)):
        require(all(b[start]>=a[start] and b[end]-a[start]<=age for a,b in zip(rows,rows[1:])),
            "returned sample cadence/publication gaps remain within registered freshness")
    hb=[];heartbeat=Path(leaf)/"supervisor_heartbeat.jsonl"
    require(file_info(heartbeat,reader=reader)["bytes"]<=1024**2,"bounded heartbeat")
    with heartbeat.open("rb") as stream:
        for raw in stream:
            require(len(raw)<=1024 and raw.endswith(b"\n"),"complete heartbeat line")
            value=json.loads(raw,object_pairs_hook=contract.native._pairs,parse_constant=contract.native._bad_constant,
                parse_float=finite_float);require(value["supervisor_pid"]==owner["supervisor_pid"],"heartbeat supervisor identity");hb.append(value)
    require(len(hb)>=2 and all(0<=b["unix_seconds"]-a["unix_seconds"]<=.2 for a,b in zip(hb,hb[1:])),"returned heartbeat cadence within CPU freshness")
    direct=read_json(Path(leaf)/"worker_GPU_final.json",reader=reader)
    require(direct.get("CUDA_initialized") is True and not direct.get("diagnostic_error")
        and all(type(direct.get(k)) is int and direct[k]>=0 for k in ("peak_allocated_bytes","peak_reserved_bytes","free_bytes_final"))
        and direct["peak_reserved_bytes"]>=direct["peak_allocated_bytes"] and direct["free_bytes_final"]>=.5*1024**3,
        "direct successful terminal GPU diagnostics and same work floor")
    lease=Path(sup["lease_path"]);require(lease.is_dir() and not lease.is_symlink() and lease.resolve().parent==(PACKAGE/"artifacts/_native_gpu_leases").resolve(),"owned durable lease directory")
    if reader is not None:reader.directory(lease)
    require(read_json(lease/"worker.json",reader=reader)==identity,"lease worker identity")
    resumed=read_json(lease/"resume.json",reader=reader)
    require(resumed["worker_identity"]==identity and resumed["external_monitor_ready_before_resume"] is True
        and resumed["resumed_monotonic"]==sup["worker_resume_monotonic"] and resumed["deadline_monotonic"]==sup["deadline_monotonic"],"assigned-before-resume/ready-before-execute lease")
    require(file_info(lease/"terminal.json",reader=reader)==file_info(Path(leaf)/"SUPERVISOR_RECEIPT.json",reader=reader),"durable terminal lease same exact supervisor bytes")
    if reader is not None:read_json(lease/"terminal.json",reader=reader)
    reservation=read_json(lease/"reservation.json",reader=reader)
    require(reservation["config_sha256"]==sup["config_sha256"] and reservation["output_dir"]==str(Path(leaf))
        and reservation["lease_id"]==lease.name and reservation["GPU_index"]==0
        and reservation["supervisor"]=={"pid":owner["supervisor_pid"],"create_time":owner["supervisor_create_time"]},"same durable scope reservation")
    observed[owner["supervisor_pid"]]=owner["supervisor_create_time"]
    require(ownership["actual_worker_pid"] in observed,"actual child independently observed in owned Job samples")
    return {"CPU_samples":len(cpu),"GPU_samples":len(gpu),"minimum_sampled_GPU_free_bytes":min(r["gpu_free_bytes"] for r in gpu),
        "minimum_host_available_bytes":min(r["host_available_bytes"] for r in cpu),"maximum_owned_tree_RSS_bytes":max(r["worker_tree_rss_bytes"] for r in cpu),
        "minimum_disk_free_bytes":min(r["disk_free_bytes"] for r in cpu),"direct_terminal_GPU":direct,"identities":observed,
        "CPU_max_sample_seconds":max(r["sample_completed_monotonic"]-r["sample_started_monotonic"] for r in cpu),
        "GPU_max_sample_seconds":max(r["gpu_sample_completed_monotonic"]-r["gpu_sample_started_monotonic"] for r in gpu),
        "CPU_max_gap_seconds":max(b["sample_started_monotonic"]-a["sample_started_monotonic"] for a,b in zip(cpu,cpu[1:])),
        "GPU_max_gap_seconds":max(b["gpu_sample_started_monotonic"]-a["gpu_sample_started_monotonic"] for a,b in zip(gpu,gpu[1:])),
        "heartbeat_rows":len(hb),"whole_scope_seconds":sup["elapsed_seconds"]}
