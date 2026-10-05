"""V2 metadata and whole-frame CPU fixtures. No actual native observations."""
from copy import deepcopy
from pathlib import Path
import hashlib,importlib.util,json,sys,tempfile,unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
STAGED=HERE.name=="long_v2_audit_adaptation_work" and HERE.parent.name=="artifacts"
PACKAGE=HERE.parents[1] if STAGED else HERE.parent
CODE=HERE if STAGED else PACKAGE
sys.path.insert(0,str(PACKAGE))
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
d=module("explicit_long_terminal_driver_v2",CODE/"long_terminal_driver_v2.py")
a=d.v
legacy=module("explicit_unchanged_terminal_fixture_builder",PACKAGE/"tests/test_long_terminal_driver.py")
FIXTURE=HERE/"fixtures/V2_DECLARATION_METADATA_ONLY.json" if STAGED else PACKAGE/"evidence/long_v2_audit_adaptation_reviewed_cpu_20261005/fixtures/V2_DECLARATION_METADATA_ONLY.json"
assert Path(d.__file__).resolve()==(CODE/"long_terminal_driver_v2.py").resolve()
DECLARATION=json.loads(FIXTURE.read_text(encoding="utf-8"))
assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest()=="f2a768b4fc062e8b86d5126d179e5a6a250008189b3fba64f22f877bd6f43fa4"
def resign(value):return d.seal({k:v for k,v in value.items() if k!="sha256"})
def rebuild(frame,anchors):
    """Resign only fixture envelopes: adversarial signatures are not authority."""
    scope=frame["scope"];reg=resign(scope["encoder_runtime_registration"]);scope["encoder_runtime_registration"]=reg
    scope["recipe"]["declared_execution_scope_registration_sha256"]=reg["sha256"]
    scope["recipe"]=resign(scope["recipe"]);sha=scope["recipe"]["sha256"];frame["scope"]=resign(scope)
    result=frame["result"];node=result["capsule_chain"];cap=node["capsule"]
    for proof in (cap["start_proof"],cap["end_proof"]):proof["identity_common"]["recipe_sha256"]=sha
    cap["recipe_sha256"]=sha;cap=resign(cap);node["capsule"]=cap;node["recipe_sha256"]=sha
    node["terminal"]["capsule_sha256"]=cap["sha256"];node["promotion"]["capsule_sha256"]=cap["sha256"]
    node["promotion"]["terminal_sha256"]=d.digest(node["terminal"]);result["capsule_chain"]=resign(node)
    result["recipe_sha256"]=sha;result["native_admission"]["binding"]["recipe_sha256"]=sha;frame["result"]=resign(result)
    anchors["recipe_sha256"]=sha;inventory=frame["inventory"];inventory["recipe_sha256"]=sha
    inventory["maps"]["runtime"]=deepcopy(scope["runtime_pins"]);inventory["counts"]["runtime"]=len(scope["runtime_pins"])
    frame["inventory"]=resign(inventory)
def frame():
    value,anchors=legacy.Controls().frame();scope=value["scope"];reg=deepcopy(DECLARATION)
    scope["runtime_pins"]=deepcopy(reg["runtime_pins"]);scope["encoder_runtime_registration"]=reg
    maps=value["result"]["capsule_chain"]["capsule"]["start_proof"]["actual_pin_maps"]
    maps["runtime"]=deepcopy(reg["runtime_pins"])
    std=reg["stdlib_inventory"]
    maps["runtime_observation"]={"policy":"ADMITTED_ENCODER45_FINITE_STDLIB_BYTE_OBSERVATION_V2","phase":"before_import",
        "actual_additional_runtime_pins":deepcopy(reg["additional_runtime_pins"]),"encoder_versions":deepcopy(reg["encoder_inventory"]["versions"]),
        "stdlib_roster":{"paths":deepcopy(std["paths"]),"approved_alias_binding":deepcopy(std["approved_alias_binding"]),"optional_stdlib_zip":deepcopy(std["optional_stdlib_zip"])},
        "native_qualification_established":False,"complete_machine_trust_claimed":False}
    cap=value["result"]["capsule_chain"]["capsule"];cap["start_proof"]["actual_pin_maps"]=maps
    cap["start_proof"]["identity_common"]["source_byte_maps_sha256"]=d.digest(maps)
    cap["end_proof"]["identity_common"]=deepcopy(cap["start_proof"]["identity_common"])
    cap["end_proof"]["actual_pin_maps_after"]=deepcopy(maps);cap["end_proof"]["actual_pin_maps_after"]["runtime_observation"]["phase"]="after_math"
    value["inventory"]["validator_file_sha256"]=d.VALIDATOR_SHA256;rebuild(value,anchors);return value,anchors
def proofs(value):
    cap=value["result"]["capsule_chain"]["capsule"]
    return cap["start_proof"]["actual_pin_maps"],cap["end_proof"]["actual_pin_maps_after"]
class Controls(unittest.TestCase):
    def reject(self,value,anchors):
        rebuild(value,anchors)
        with self.assertRaises((ValueError,KeyError)):d.validate_frame(value,anchors,CPU_fixture_mode=True)
    def test_complete_V2_whole_frame_exact887_only_CPU_status(self):
        value,anchors=frame();report=d.validate_frame(value,anchors,CPU_fixture_mode=True)
        self.assertEqual(d.CPU_KIND,report["kind"]);self.assertFalse(report["native_AUDIT_KIND_PASS_issued"]);self.assertFalse(report["native_ready"])
        self.assertFalse(report["raw_RNG_restore_or_tensor_reread_verified"]);self.assertEqual(887,len(value["scope"]["runtime_pins"]))
    def test_phase_only_normalization_preserves_original_before_digest(self):
        value,_=frame();before,after=proofs(value)
        self.assertNotEqual(before,after);self.assertEqual(a.normalize_runtime_phase_only(before),a.normalize_runtime_phase_only(after))
        self.assertEqual("before_import",before["runtime_observation"]["phase"]);self.assertEqual("after_math",after["runtime_observation"]["phase"])
        self.assertNotEqual(d.digest(before),d.digest(a.normalize_runtime_phase_only(before)))
    def test_missing_seventh_section_rejected_even_resigned(self):
        value,anchors=frame()
        for maps in proofs(value):maps.pop("runtime_observation")
        self.reject(value,anchors)
    def test_wrong_same_or_reversed_phase_rejected(self):
        for phase in ("before_import","unknown",None):
            value,anchors=frame();proofs(value)[1]["runtime_observation"]["phase"]=phase;self.reject(value,anchors)
        value,anchors=frame();before,after=proofs(value);before["runtime_observation"]["phase"]="after_math";after["runtime_observation"]["phase"]="before_import";self.reject(value,anchors)
    def test_no_other_observation_field_can_be_normalized(self):
        for key in ("encoder_versions","stdlib_roster","actual_additional_runtime_pins"):
            value,anchors=frame();proof=proofs(value)[1]["runtime_observation"];proof[key].pop(next(iter(proof[key])));self.reject(value,anchors)
    def test_observation_extra_field_or_native_flags_rejected(self):
        for key,new in (("extra",1),("native_qualification_established",True),("complete_machine_trust_claimed",True)):
            value,anchors=frame();proofs(value)[1]["runtime_observation"][key]=new;self.reject(value,anchors)
    def test_normalized_instead_of_actual_before_map_digest_rejected(self):
        value,anchors=frame();cap=value["result"]["capsule_chain"]["capsule"]
        for item in (cap["start_proof"],cap["end_proof"]):item["identity_common"]["source_byte_maps_sha256"]=d.digest(a.normalize_runtime_phase_only(proofs(value)[0]))
        self.reject(value,anchors)
    def test_V1_or_boolean_only_authority_rejected(self):
        for kind in ("SYNTH_LONG_ENCODER_RUNTIME_REGISTRATION_V1",None):
            value,anchors=frame();value["scope"]["encoder_runtime_registration"]["kind"]=kind;self.reject(value,anchors)
        value,anchors=frame();value["scope"]["encoder_runtime_registration"]["native_launch_authorized"]=True;self.reject(value,anchors)
    def test_exact_inherited65_not_just_count_union(self):
        value,anchors=frame();reg=value["scope"]["encoder_runtime_registration"];key=next(k for k in reg["inherited_runtime_pins"] if k not in reg["additional_runtime_pins"])
        reg["inherited_runtime_pins"][key]="f"*64;reg["runtime_pins"][key]="f"*64;value["scope"]["runtime_pins"][key]="f"*64
        self.reject(value,anchors)
    def test_missing_or_substituted_current_consumption_source_rejected(self):
        for change in ("missing","substitute"):
            value,anchors=frame();pins=value["scope"]["encoder_runtime_registration"]["consumption_source_pins"];key=next(iter(pins))
            if change=="missing":pins.pop(key)
            else:pins[key]="f"*64
            self.reject(value,anchors)
    def test_registered_union_conflict_or_omission_rejected(self):
        for change in ("conflict","omission"):
            value,anchors=frame();reg=value["scope"]["encoder_runtime_registration"]
            if change=="conflict":key=next(k for k in reg["inherited_runtime_pins"] if k in reg["additional_runtime_pins"]);reg["inherited_runtime_pins"][key]="f"*64
            else:reg["additional_runtime_pins"].pop(next(iter(reg["additional_runtime_pins"])))
            self.reject(value,anchors)
    def test_alias_reparse_optionalZIP_policy_mutation_rejected(self):
        for kind in ("alias","zip","category","version"):
            value,anchors=frame();std=value["scope"]["encoder_runtime_registration"]["stdlib_inventory"]
            if kind=="alias":std["approved_alias_binding"]["reparse_tag"]=0
            elif kind=="zip":std["optional_stdlib_zip"]["present"]=True
            elif kind=="category":std["paths"][next(iter(std["paths"]))]="arbitrary"
            else:std["python_version"]="3.12"
            self.reject(value,anchors)
    def test_encoder_version_or_roster_filehash_rejected(self):
        for kind in ("version","filehash","roster"):
            value,anchors=frame();enc=value["scope"]["encoder_runtime_registration"]["encoder_inventory"]
            if kind=="version":enc["versions"]["tokenizers"]="0.0"
            elif kind=="filehash":enc["files"][next(iter(enc["files"]))]["sha256"]="f"*64
            else:enc["packages"]["tokenizers"]["files"].pop()
            self.reject(value,anchors)
    def test_failed_terminal_or_wrong_ROLE_HEAD_expired_admission_rejected(self):
        for kind in ("failed","role","head","admission"):
            value,anchors=frame()
            if kind=="failed":value["supervisor"]["status"]="OWNED_WORKER_ABORTED"
            elif kind=="role":value["result"]["native_admission"]["binding"]["role"]="bulk"
            elif kind=="head":value["supervisor"]["preflight"]["source_commit"]="f"*40
            else:value["result"]["native_admission"]["expires_monotonic"]=10.3
            self.reject(value,anchors)
    def test_actual_runtime_observation_hash_drift_rejected(self):
        value,anchors=frame();maps=proofs(value)[1];key=next(iter(maps["runtime_observation"]["actual_additional_runtime_pins"]))
        maps["runtime_observation"]["actual_additional_runtime_pins"][key]="f"*64;self.reject(value,anchors)
    def test_staged_metadata_fixture_is_not_final_registration(self):
        import encoder_registration_v2 as r
        with self.assertRaises(ValueError):r.registration({"path":str(FIXTURE.resolve()),"sha256":hashlib.sha256(FIXTURE.read_bytes()).hexdigest()},a.contract.base_runtime_pins())
    def test_fixture_or_pending_cost_cannot_be_native(self):
        for kind in ("fixture","cost"):
            value,anchors=frame()
            if kind=="fixture":value["result"]["capsule_chain"]["capsule"]["CPU_fixture"]=True
            else:value["result"]["actual_costs"]["operations"]["model_construct"].update(returned=0,pending=1)
            self.reject(value,anchors)
    def test_native_entry_remains_denied_before_IO(self):
        value,anchors=frame()
        with self.assertRaisesRegex(ValueError,"staged native terminal dispatch forbidden"):d.audit_terminal("UNUSED","UNUSED","UNUSED","UNUSED",terminal_signal=a.SIGNAL,anchors=anchors)
        self.assertFalse(d.NATIVE_PASS_ENABLED);self.assertTrue(d.ENCODER_V2_CONSUMPTION_REVIEWED)


class FilesystemControls(unittest.TestCase):
    """Real temporary byte IO and current contracts; all device probes injected."""
    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((d.canonical(value)+"\n").encode());return d.bytes_info(path)
    def fixture(self,root):
        package=Path(root).resolve();_,leaf,sup,_=legacy.fixture_module.ResourceFixtureControls().fixture(package)
        preserved=package/"evidence/retained";preserved.mkdir(parents=True)
        registration=package/"artifacts/long_native_audit_registrations/registration.json"
        registration.parent.mkdir();registration.write_bytes(FIXTURE.read_bytes())
        binding={"path":str(registration),"sha256":d.bytes_info(registration)["sha256"]}
        with patch.object(a.contract,"encoder_registration",return_value=deepcopy(DECLARATION)):
            scope=d.seal(a.contract.build_scope(1024,"trace1",encoder=binding,authorize=True))
        config=package/"configs/SCOPE.json";config_hash=self.write(config,scope)["sha256"]
        value,anchors=frame();cap=value["result"]["capsule_chain"]["capsule"]
        owner=d.selected_json(leaf/"supervisor_ownership.json");owner["config_sha256"]=config_hash;self.write(leaf/"supervisor_ownership.json",owner)
        recipe=scope["recipe"]["sha256"]
        maps={"source":scope["source_pins"],"runtime":scope["runtime_pins"],"model":scope["model_pins"],"data":scope["data_pins"],
            "encoder_assets":{str(Path(scope["tokenizer_binding"]["model_dir"])/name):item["sha256"] for name,item in scope["tokenizer_binding"]["assets"].items()},
            "encoder_bindings":scope["tokenizer_binding"]["binding_sources"],"runtime_observation":deepcopy(proofs(value)[0]["runtime_observation"])}
        cap["start_proof"]["owned_scope_identity"]=owner;cap["start_proof"]["actual_pin_maps"]=maps
        cap["start_proof"]["encoder"].update(long_tokens=scope["row"]["prompt_tokens"],tokenizer_binding_sha256=d.digest(scope["tokenizer_binding"]))
        common=cap["start_proof"]["identity_common"];common.update(recipe_sha256=recipe,source_byte_maps_sha256=d.digest(maps),
            encoder_proof_sha256=d.digest(cap["start_proof"]["encoder"]))
        cap["end_proof"]["identity_common"]=deepcopy(common);cap["end_proof"]["actual_pin_maps_after"]=deepcopy(maps)
        cap["end_proof"]["actual_pin_maps_after"]["runtime_observation"]["phase"]="after_math"
        cap["start_proof"]["loader"]["asset_hashes"]={path:{"sha256":pin} for path,pin in scope["model_pins"].items()}
        cap["recipe_sha256"]=recipe
        p=scope["row"]["prompt_tokens"]
        for i,item in enumerate(cap["trace"]):item["prefix_length"]=p+i
        roster=[("long_native_dense_parity_forward",scope["short_parity_row"]["prompt_ids"]),
            ("long_native_zeroB_parity_forward",scope["short_parity_row"]["prompt_ids"])]
        roster.extend(("long_native_greedy_forward",scope["row"]["prompt_ids"]+[item["token"] for item in cap["trace"][:i]]) for i in range(32))
        cap["physical_dose"]={"attempted_calls":34,"input_positions":sum(len(ids) for _,ids in roster),
            "records":[{"operation":name,"input_ids_sha256":d.digest(ids),"input_positions":len(ids)} for name,ids in roster]}
        ops={"model_construct":{"attempted":1,"returned":1,"failed":0,"pending":0}}
        events=[];last=0
        def event_calls(name):
            nonlocal last
            last+=1;events.extend([{"event":"attempt","id":last,"operation":name},{"event":"returned","id":last,"operation":name}])
        event_calls("model_construct")
        def publish(name,payload):
            path=leaf/(name+".json");info=self.write(path,payload)
            descriptor={"path":str(path),**info,"payload_sha256":d.digest(payload),"durable":True}
            for suffix in ("write","verify"):event_calls(name+"_"+suffix)
            events.append({"event":"long_measurement_preserved","name":name,"publication":descriptor})
            return descriptor
        for i in range(1,33):
            name="long_token_%02d"%i
            for suffix in ("write","verify"):ops[name+"_"+suffix]={"attempted":1,"returned":1,"failed":0,"pending":0}
            payload=d.seal({"purpose":a.contract.PURPOSE,"recipe_sha256":recipe,"role":"trace1","prefix":cap["trace"][:i],
                "costs":{"operations":deepcopy(ops),"physical_calls":2+i,"input_positions":838+i*p+i*(i-1)//2}})
            publish(name,payload)
        cap["posted_scope_costs"]={"operations":ops,"physical_calls":34,"input_positions":cap["physical_dose"]["input_positions"]};cap=resign(cap)
        capsule_publication=publish("long_capsule_publish",cap)
        terminal={"capsule_sha256":cap["sha256"],"costs":a.project_costs(a.project_costs(cap["posted_scope_costs"],"long_capsule_publish"),"long_terminal_publish")}
        terminal_publication=publish("long_terminal_publish",terminal)
        promotion={"capsule_sha256":cap["sha256"],"terminal_sha256":d.digest(terminal),"costs":a.project_costs(terminal["costs"],"long_capsule_promote")}
        promotion_publication=publish("long_capsule_promote",promotion)
        node=d.seal({"kind":a.contract.KIND,"recipe_sha256":recipe,"capsule":cap,"capsule_publication":capsule_publication,
            "terminal":terminal,"terminal_publication":terminal_publication,"promotion":promotion,"promotion_publication":promotion_publication,
            "cumulative_costs":promotion["costs"]})
        result=value["result"];result.update(recipe_sha256=recipe,config_sha256=config_hash,capsule_chain=node,
            actual_costs=promotion["costs"],physical_dose=cap["physical_dose"],partial_trace=cap["trace"],
            operation_accounting={"scope":"registered_API_boundaries;internal_ATen_operations_not_individually_metered","operations":promotion["costs"]["operations"],"last_call":last})
        result["native_admission"]["binding"]["recipe_sha256"]=recipe
        result["native_admission"].update(observed_monotonic=10.1,expires_monotonic=10.2)
        result=resign(result);self.write(leaf/"worker_result.json",result)
        (leaf/"events.jsonl").write_bytes(b"".join((d.canonical(item)+"\n").encode() for item in events))
        sup.update(status="SUPERVISED_WORKER_EXITED_ZERO",returncode=0,partial_outputs_frozen=True,output_dir=str(leaf),config_sha256=config_hash,
            preflight={**scope,"config_path":str(config),"config_sha256":config_hash,"source_commit":anchors["source_commit"],"worker_path":str(PACKAGE/"long_native_worker.py")})
        sup["partial_outputs"]=[{"path":path.name,**d.bytes_info(path)} for path in sorted(leaf.iterdir()) if path.name!="SUPERVISOR_RECEIPT.json"]
        self.write(leaf/"SUPERVISOR_RECEIPT.json",sup)
        lease=Path(sup["lease_path"]);reservation=d.selected_json(lease/"reservation.json");reservation["config_sha256"]=config_hash
        self.write(lease/"reservation.json",reservation);(lease/"terminal.json").write_bytes((leaf/"SUPERVISOR_RECEIPT.json").read_bytes())
        inv=d.seal({"kind":a.INVENTORY_KIND,"source_commit":anchors["source_commit"],"recipe_sha256":recipe,"role":"trace1","context_cap":1024,
            "maps":{key:item for key,item in maps.items() if key!="runtime_observation"},
            "counts":{key:len(item) for key,item in maps.items() if key!="runtime_observation"},"previous_references":None,
            "encoder_registration":binding,"validator_file_sha256":d.VALIDATOR_SHA256,"terminal_driver_file_sha256":d.bytes_info(Path(d.__file__))["sha256"]})
        inventory=preserved/"AUDIT_INVENTORY.json";self.write(inventory,inv)
        raw=preserved/"raw";raw.mkdir();source_dir=preserved/"source_snapshot";source_dir.mkdir();source_files={};source_map={}
        for path in leaf.iterdir():(raw/path.name).write_bytes(path.read_bytes())
        for i,path in enumerate(scope["source_pins"]):
            source_files[path]=d.bytes_info(path);name="%03d_%s"%(i,Path(path).name);source_map[path]=name
            (source_dir/name).write_bytes(Path(path).read_bytes())
        (preserved/"CONFIG_SNAPSHOT.json").write_bytes(config.read_bytes())
        pres=d.seal({"attempt_path":str(leaf),"preserved_path":str(preserved),"raw_files":{path.name:d.bytes_info(path) for path in leaf.iterdir()},
            "config_snapshot":d.bytes_info(config),"audit_inventory_snapshot":d.bytes_info(inventory),"source_files":source_files,"source_snapshot_map":source_map})
        self.write(preserved/"PRESERVATION.json",pres)
        anchors.update(config_file_sha256=config_hash,inventory_file_sha256=d.bytes_info(inventory)["sha256"],
            result_file_sha256=d.bytes_info(leaf/"worker_result.json")["sha256"],supervisor_file_sha256=d.bytes_info(leaf/"SUPERVISOR_RECEIPT.json")["sha256"],
            recipe_sha256=recipe,preservation_body_sha256=pres["sha256"],encoder_registration_file_sha256=binding["sha256"])
        return {"package":package,"attempt":leaf,"preserved":preserved,"config":config,"inventory":inventory,"anchors":anchors,
            "terminal_signal":a.SIGNAL,"identity_alive":lambda identity:False,"asset_checker":lambda scope:True,
            "commit_checker":lambda require_clean:anchors["source_commit"]}
    def refresh_raw(self,kw):
        leaf,preserved=kw["attempt"],kw["preserved"];sup=d.selected_json(leaf/"SUPERVISOR_RECEIPT.json")
        sup["partial_outputs"]=[{"path":path.name,**d.bytes_info(path)} for path in sorted(leaf.iterdir()) if path.name!="SUPERVISOR_RECEIPT.json"]
        self.write(leaf/"SUPERVISOR_RECEIPT.json",sup);(Path(sup["lease_path"])/"terminal.json").write_bytes((leaf/"SUPERVISOR_RECEIPT.json").read_bytes())
        for path in leaf.iterdir():(preserved/"raw"/path.name).write_bytes(path.read_bytes())
        pres=d.selected_json(preserved/"PRESERVATION.json");pres["raw_files"]={path.name:d.bytes_info(path) for path in leaf.iterdir()};pres=resign(pres)
        self.write(preserved/"PRESERVATION.json",pres);kw["anchors"]["preservation_body_sha256"]=pres["sha256"]
        kw["anchors"]["result_file_sha256"]=d.bytes_info(leaf/"worker_result.json")["sha256"]
        kw["anchors"]["supervisor_file_sha256"]=d.bytes_info(leaf/"SUPERVISOR_RECEIPT.json")["sha256"]
    def test_complete_filesystem_reconstruction_publication_resource_state_identity_cost_CPU_kind(self):
        with tempfile.TemporaryDirectory() as root:
            kw=self.fixture(root);seen=[];kw["identity_alive"]=lambda identity:seen.append(identity) or False
            report=d.check_filesystem_fixture(**kw,output_path=kw["preserved"]/"CPU_REPORT.json")
            self.assertEqual(d.CPU_KIND,report["kind"]);self.assertFalse(report["native_AUDIT_KIND_PASS_issued"])
            self.assertTrue(report["exact_registration_config_recipe_reconstructed"]);self.assertEqual(3,len(seen))
            self.assertEqual(71,report["journal_calls"]);self.assertEqual(34,report["physical"]["attempted_calls"])
            self.assertEqual(report,d.selected_json(kw["preserved"]/"CPU_REPORT.json"))
    def test_whole_path_wrong_signal_anchor_live_identity_asset_or_HEAD_denied(self):
        for kind in ("signal","anchor","live","asset","HEAD","registration_drift"):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as root:
                kw=self.fixture(root)
                if kind=="signal":kw["terminal_signal"]="NOT_TERMINAL"
                elif kind=="anchor":kw["anchors"]["result_file_sha256"]="f"*64
                elif kind=="live":kw["identity_alive"]=lambda identity:True
                elif kind=="asset":kw["asset_checker"]=lambda scope:False
                elif kind=="HEAD":kw["commit_checker"]=lambda required:"f"*40
                else:
                    def change_selected_registration(scope):
                        p=Path(scope["encoder_registration"]["path"]);p.write_bytes(p.read_bytes()+b" ");return True
                    kw["asset_checker"]=change_selected_registration
                with self.assertRaises(ValueError):d.check_filesystem_fixture(**kw)
    def test_whole_path_omitted_extra_raw_and_source_snapshot_changed_reject(self):
        for kind in ("omitted","extra","source"):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as root:
                kw=self.fixture(root)
                if kind=="omitted":(kw["preserved"]/"raw/long_token_01.json").unlink()
                elif kind=="extra":(kw["attempt"]/"unadvertised.json").write_bytes(b"{}")
                else:
                    p=next((kw["preserved"]/"source_snapshot").iterdir());p.write_bytes(p.read_bytes()+b"changed")
                with self.assertRaises((ValueError,FileNotFoundError)):d.check_filesystem_fixture(**kw)
    def test_whole_path_resigned_resource_FILETIME_journal_publication_and_LP_fail(self):
        for kind in ("resource","FILETIME","journal","publication","LP","finite"):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as root:
                kw=self.fixture(root);leaf=kw["attempt"]
                if kind in ("resource","FILETIME"):
                    rows=d.selected_json(leaf/"RESOURCE_OBSERVATIONS.json")
                    if kind=="resource":rows[-1]["gpu_free_bytes"]=1
                    else:
                        rows[1]["owned_process_samples"][0]["creation_filetime_100ns"]=99
                        rows[1]["owned_process_identities"][0]["creation_filetime_100ns"]=99
                    self.write(leaf/"RESOURCE_OBSERVATIONS.json",rows)
                elif kind=="journal":
                    p=leaf/"events.jsonl";p.write_bytes(p.read_bytes().replace(b'"id":1,',b'"id":9,',1))
                elif kind=="finite":
                    p=leaf/"events.jsonl";p.write_bytes(p.read_bytes().replace(b'"event":"attempt",',b'"event":"attempt","ignored":1e999,',1))
                elif kind=="publication":
                    p=leaf/"long_token_01.json";body=d.selected_json(p);body["costs"]["physical_calls"]-=1;self.write(p,resign(body))
                else:
                    result=d.selected_json(leaf/"worker_result.json");result["capsule_chain"]["capsule"]["trace"][0]["logprob"]=.1
                    self.write(leaf/"worker_result.json",resign(result))
                self.refresh_raw(kw)
                with self.assertRaises(ValueError):d.check_filesystem_fixture(**kw)
    def test_reconstruction_rejects_resigned_config_message_precision_data_or_registration_source(self):
        for kind in ("messages","precision","data","registration"):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as root:
                kw=self.fixture(root);scope=d.selected_json(kw["config"])
                if kind=="messages":scope["messages"][0]["content"]+="CHANGED"
                elif kind=="precision":scope["precision"][next(iter(scope["precision"]))]="arbitrary"
                elif kind=="data":scope["data_pins"].pop(next(iter(scope["data_pins"])))
                else:scope["encoder_runtime_registration"]["consumption_source_pins"].pop(next(iter(scope["encoder_runtime_registration"]["consumption_source_pins"])))
                scope=resign(scope)
                with self.assertRaises(ValueError):
                    if kind=="registration":d.reconstruct_registration(scope,scope["encoder_registration"]["path"])
                    else:d.reconstruct_scope(scope)
    def test_finite_strict_bounded_selected_JSON_and_portable_nonlocal_absolute_leaf(self):
        with tempfile.TemporaryDirectory() as root:
            kw=self.fixture(root);report=d.check_filesystem_fixture(**kw)
            self.assertEqual(d.CPU_KIND,report["kind"]);p=Path(root)/"bad.json"
            for raw in (b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":1e999}',b'{"a":Infinity}'):
                p.write_bytes(raw)
                with self.assertRaises(ValueError):d.selected_json(p)
            p.write_bytes(b" "*101)
            with self.assertRaises(ValueError):d.selected_json(p,maximum=100)
    def test_identity_probe_error_is_not_dead_and_fixture_cannot_native_via_alias(self):
        with tempfile.TemporaryDirectory() as root:
            kw=self.fixture(root)
            def denied(identity):raise PermissionError("CPU injection access denied")
            kw["identity_alive"]=denied
            with self.assertRaises(PermissionError):d.check_filesystem_fixture(**kw)
        with patch.object(d,"STAGED",False):
            with self.assertRaisesRegex(ValueError,"staged native terminal dispatch forbidden"):
                d.audit_terminal("UNUSED","UNUSED","UNUSED","UNUSED",terminal_signal=a.SIGNAL,anchors=frame()[1])

    def test_final_read_closure_rejects_snapshot_lease_and_directory_drift(self):
        for kind in ("source_snapshot","lease_reservation","lease_worker","lease_resume","lease_terminal","extra_raw","extra_snapshot"):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as root:
                kw=self.fixture(root)
                if kind=="source_snapshot":path=next((kw["preserved"]/"source_snapshot").iterdir())
                elif kind.startswith("lease_"):
                    sup=d.selected_json(kw["attempt"]/"SUPERVISOR_RECEIPT.json")
                    path=Path(sup["lease_path"])/(kind.removeprefix("lease_")+".json")
                elif kind=="extra_raw":path=kw["attempt"]/"late_extra.json"
                else:path=kw["preserved"]/"source_snapshot/late_extra.py"
                def changed_after_semantics(scope):
                    path.write_bytes(path.read_bytes()+b" " if path.exists() else b"{}");return True
                kw["asset_checker"]=changed_after_semantics
                with self.assertRaisesRegex(ValueError,"read-set bytes changed|directory roster changed"):
                    d.check_filesystem_fixture(**kw)

    def test_registration_first_read_baseline_cannot_rebase_after_semantics(self):
        with tempfile.TemporaryDirectory() as root:
            kw=self.fixture(root);calls=0
            path=Path(d.selected_json(kw["config"])["encoder_registration"]["path"])
            def changed_at_first_commit(require_clean):
                nonlocal calls
                calls+=1
                if calls==1:path.write_bytes(path.read_bytes()+b" ")
                return kw["anchors"]["source_commit"]
            kw["commit_checker"]=changed_at_first_commit
            with self.assertRaisesRegex(ValueError,"read-set bytes changed"):
                d.check_filesystem_fixture(**kw)

    def test_ignored_nonfinite_lease_JSON_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            kw=self.fixture(root);sup=d.selected_json(kw["attempt"]/"SUPERVISOR_RECEIPT.json")
            path=Path(sup["lease_path"])/"resume.json";raw=path.read_bytes()
            path.write_bytes(raw[:-2]+b',"ignored":1e999}\n')
            with self.assertRaisesRegex(ValueError,"finite JSON float required"):
                d.check_filesystem_fixture(**kw)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"selected.json";path.write_bytes(b'{"ignored":1e999}')
            with self.assertRaisesRegex(ValueError,"finite selected JSON float required"):a.read_json(path)

    def test_trace2_prior_files_remain_immutable_and_fixture_stays_CPU(self):
        for kind in ("unchanged","result","supervisor","audit"):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as root:
                kw,paths=trace2_fixture(root)
                if kind=="unchanged":
                    report=d.check_filesystem_fixture(**kw)
                    self.assertEqual(d.CPU_KIND,report["kind"]);self.assertEqual("trace2",report["role"])
                    self.assertFalse(report["native_AUDIT_KIND_PASS_issued"])
                    self.assertTrue(report["immutable_read_closure"]["first_read_anchors_rechecked"])
                else:
                    path=paths[kind]
                    def changed_prior_after_semantics(scope):path.write_bytes(path.read_bytes()+b" ");return True
                    kw["asset_checker"]=changed_prior_after_semantics
                    with self.assertRaisesRegex(ValueError,"read-set bytes changed"):
                        d.check_filesystem_fixture(**kw)

    def test_selected_evidence_read_set_first_hash_and_final_bytes_are_exact(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"evidence.json";path.write_bytes(b'{"value":1}\n')
            reader=d.ReadSet();initial=d.bytes_info(path)["sha256"]
            self.assertEqual({"value":1},reader.json(path,initial))
            path.write_bytes(b'{"value":1}\n ')
            with self.assertRaisesRegex(ValueError,"read-set bytes changed"):reader.json(path)
            with self.assertRaisesRegex(ValueError,"read-set bytes changed"):reader.unchanged()

def reseal(value):
    return d.seal({key: item for key, item in value.items() if key != "sha256"})

def reseal_chain(result):
    node = result["capsule_chain"]
    cap = reseal(node["capsule"])
    node["capsule"] = cap
    node["terminal"] = {"capsule_sha256": cap["sha256"], "costs": a.project_costs(a.project_costs(cap["posted_scope_costs"], "long_capsule_publish"), "long_terminal_publish")}
    node["promotion"] = {"capsule_sha256": cap["sha256"], "terminal_sha256": d.digest(node["terminal"]), "costs": a.project_costs(node["terminal"]["costs"], "long_capsule_promote")}
    node["cumulative_costs"] = node["promotion"]["costs"]
    result["capsule_chain"] = reseal(node)
    result["actual_costs"] = node["cumulative_costs"]
    return reseal(result)

def add_costs(old, new):
    output = deepcopy(new)
    for key in ("physical_calls", "input_positions"):
        output[key] += old[key]
    for name, counts in old["operations"].items():
        target = output["operations"].setdefault(name, dict(attempted=0, returned=0, failed=0, pending=0))
        for key, value in counts.items():
            target[key] += value
    return output

def trace2_fixture(root):
    c = FilesystemControls()
    kw = c.fixture(root)
    leaf, retained = kw["attempt"], kw["preserved"]
    scope = d.selected_json(kw["config"])
    result = d.selected_json(leaf / "worker_result.json")
    sup = d.selected_json(leaf / "SUPERVISOR_RECEIPT.json")
    prior_result, prior_sup = deepcopy(result), deepcopy(sup)
    prior_job = prior_result["capsule_chain"]["capsule"]["start_proof"]["owned_scope_identity"]["job_name"] + "_PREVIOUS"
    prior_result["capsule_chain"]["capsule"]["start_proof"]["owned_scope_identity"]["job_name"] = prior_job
    prior_sup["worker_identity"]["job_name"] = prior_job
    prior_result = reseal_chain(prior_result)
    prior_dir = kw["package"] / "evidence/prior_cpu_fixture"
    rp, sp, ap = [prior_dir / name for name in ("RESULT.json", "SUPERVISOR.json", "AUDIT.json")]
    rh = c.write(rp, prior_result)["sha256"]
    sh = c.write(sp, prior_sup)["sha256"]
    prior_audit = d.seal({"kind": a.contract.AUDIT_KIND, "status": "PASS_NATIVE_LONG_SCOPE_NUMERICAL_MECHANICS_ONLY", "purpose": a.contract.PURPOSE,
        "role": "trace1", "recipe_sha256": scope["recipe"]["sha256"], "result_sha256": rh, "supervisor_sha256": sh,
        "config_sha256": prior_result["config_sha256"], "source_commit": prior_result["source_commit"],
        "study_training_authorized": False, "scoring_authorized": False, "validator_file_sha256": d.VALIDATOR_SHA256,
        "terminal_driver_file_sha256": sha(Path(d.__file__)), "native_AUDIT_KIND_PASS_issued": True,
        "CPU_injected_identity_asset_commit_probes": False})
    ah = c.write(ap, prior_audit)["sha256"]
    refs = [{"result_path": str(rp), "result_sha256": rh, "supervisor_path": str(sp), "supervisor_sha256": sh, "audit_path": str(ap), "audit_sha256": ah}]
    scope.update(role="trace2", references=refs)
    scope = reseal(scope)
    ch = c.write(kw["config"], scope)["sha256"]
    old = prior_result["capsule_chain"]["cumulative_costs"]
    cap = result["capsule_chain"]["capsule"]
    cap.update(role="trace2", previous_capsule_sha256=prior_result["capsule_chain"]["sha256"])
    cap["posted_scope_costs"] = add_costs(old, cap["posted_scope_costs"])
    cap["start_proof"]["owned_scope_identity"]["config_sha256"] = ch
    result.update(role="trace2", config_sha256=ch)
    result["native_admission"]["binding"]["role"] = "trace2"
    result = reseal_chain(result)
    node = result["capsule_chain"]
    events = [json.loads(line) for line in (leaf / "events.jsonl").read_bytes().splitlines()]
    for event in events:
        if event["event"] != "long_measurement_preserved":
            continue
        name = event["name"]
        p = leaf / (name + ".json")
        if name.startswith("long_token_"):
            payload = d.selected_json(p)
            payload["role"] = "trace2"
            payload["costs"] = add_costs(old, payload["costs"])
            payload = reseal(payload)
        else:
            key = {"long_capsule_publish": "capsule", "long_terminal_publish": "terminal", "long_capsule_promote": "promotion"}[name]
            payload = node[key]
        info = c.write(p, payload)
        descriptor = {"path": str(p), **info, "payload_sha256": d.digest(payload), "durable": True}
        event["publication"] = descriptor
        if not name.startswith("long_token_"):
            node[{"long_capsule_publish": "capsule_publication", "long_terminal_publish": "terminal_publication", "long_capsule_promote": "promotion_publication"}[name]] = descriptor
    result["capsule_chain"] = reseal(node)
    c.write(leaf / "worker_result.json", reseal(result))
    (leaf / "events.jsonl").write_bytes(b"".join((d.canonical(event) + "\n").encode() for event in events))
    owner = d.selected_json(leaf / "supervisor_ownership.json")
    owner["config_sha256"] = ch
    c.write(leaf / "supervisor_ownership.json", owner)
    sup.update(config_sha256=ch, preflight={**scope, "config_path": str(kw["config"]), "config_sha256": ch,
        "source_commit": kw["anchors"]["source_commit"], "worker_path": str(PACKAGE / "long_native_worker.py")})
    c.write(leaf / "SUPERVISOR_RECEIPT.json", sup)
    lease = Path(sup["lease_path"])
    reservation = d.selected_json(lease / "reservation.json")
    reservation["config_sha256"] = ch
    c.write(lease / "reservation.json", reservation)
    inv = d.selected_json(kw["inventory"])
    inv.update(role="trace2", previous_references=refs)
    c.write(kw["inventory"], reseal(inv))
    (retained / "CONFIG_SNAPSHOT.json").write_bytes(kw["config"].read_bytes())
    pres = d.selected_json(retained / "PRESERVATION.json")
    pres.update(config_snapshot=d.bytes_info(kw["config"]), audit_inventory_snapshot=d.bytes_info(kw["inventory"]))
    c.write(retained / "PRESERVATION.json", reseal(pres))
    kw["anchors"].update(role="trace2", config_file_sha256=ch, inventory_file_sha256=sha(kw["inventory"]))
    c.refresh_raw(kw)
    return kw, {"result": rp, "supervisor": sp, "audit": ap}

if __name__=="__main__":unittest.main()
