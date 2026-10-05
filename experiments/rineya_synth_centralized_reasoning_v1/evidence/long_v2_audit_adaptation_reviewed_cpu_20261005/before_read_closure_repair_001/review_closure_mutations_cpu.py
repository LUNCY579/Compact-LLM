"""CPU temp-file mutation controls for lease and prior-role read closure."""
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib.util, json, sys, tempfile, time

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent / "long_v2_audit_adaptation_work"
spec = importlib.util.spec_from_file_location("independent_closure_fixture", STAGE / "test_long_v2_audit_adaptation.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
d, a = t.d, t.a
c = t.FilesystemControls()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

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
        "source_commit": kw["anchors"]["source_commit"], "worker_path": str(t.PACKAGE / "long_native_worker.py")})
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

def main():
    target = HERE / "LEASE_PRIOR_CLOSURE_MUTATION_REVIEW_001.json"
    if target.exists():
        raise ValueError("write-once review already exists")
    names = ("long_native_auditor_v2.py", "long_terminal_driver_v2.py", "test_long_v2_audit_adaptation.py")
    before = {name: sha(STAGE / name) for name in names}
    started = time.monotonic()
    outcomes = []
    for family, name in [("lease", name) for name in ("reservation", "worker", "resume", "terminal")] + [("prior", name) for name in ("result", "supervisor", "audit")]:
        with tempfile.TemporaryDirectory() as root:
            if family == "prior":
                kw, previous_paths = trace2_fixture(root)
                path = previous_paths[name]
            else:
                kw = c.fixture(root)
                sup = d.selected_json(kw["attempt"] / "SUPERVISOR_RECEIPT.json")
                path = Path(sup["lease_path"]) / (name + ".json")
            change = {"path": str(path), "before_sha256": sha(path)}
            def checker(scope):
                path.write_bytes(path.read_bytes() + b" ")
                change["after_sha256"] = sha(path)
                return True
            kw["asset_checker"] = checker
            try:
                report = d.check_filesystem_fixture(**kw)
                outcomes.append({"control": family + "_" + name + "_changed_after_semantics", "outcome": "ACCEPTED_CPU_REPORT_DESPITE_MUTATION", "kind": report["kind"], "status": report["status"], "native_PASS_issued": report["native_AUDIT_KIND_PASS_issued"], **change})
            except Exception as error:
                outcomes.append({"control": family + "_" + name + "_changed_after_semantics", "outcome": "REJECTED", "error_type": type(error).__name__, "error": str(error), **change})
    after = {name: sha(STAGE / name) for name in names}
    forbidden = sorted(name for name in sys.modules if name.split(".", 1)[0] in {"torch", "transformers", "tensorflow", "jax", "flax", "tokenizers", "jinja2", "markupsafe"})
    receipt = {"schema_version": 1, "status": "CPU_MUTATION_REVIEW_ONLY_NO_NATIVE_AUTHORITY", "recorded_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started,
        "source_before": before, "source_after": after, "source_unchanged": before == after, "mutation_controls": outcomes, "forbidden_modules_loaded": forbidden,
        "prior_audit_metadata_CPU_fabricated_for_lineage_validator_control": True, "actual_native_identity_assets_model_binary_encoder_GPU_probes": False,
        "native_AUDIT_KIND_PASS_issued": False, "three_arm_goal_complete": False}
    receipt["sha256"] = d.digest(receipt)
    with target.open("xb") as stream:
        stream.write((d.canonical(receipt) + "\n").encode())
    print(d.canonical(receipt))

if __name__ == "__main__":
    main()
