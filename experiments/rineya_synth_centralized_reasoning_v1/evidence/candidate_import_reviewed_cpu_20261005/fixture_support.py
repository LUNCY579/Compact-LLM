"""Independent, inert CPU fixtures for the staged candidate import contract.

No model, framework, decoder, grader, production receipt, or checkpoint payload
is used. Source files are read only to bind their actual source-byte identities.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
STUDY_KIND = "SYNTH_STUDY_CHUNK_PROTOCOL_V1"
BACKEND_KIND = "SYNTH_STUDY_TORCH_BOUNDARY_V1"
RAW_SCOPE = "CPU_SYNTHETIC_SCORE_IMPORT_FIXTURE"
SOURCE_NAMES = {
    "evaluation_panel.py", "rs_contract.py", "rs_solver.py", "rs_verifier.py",
    "task_compiler.py", "balanced_corpus.py", "EVALUATION_PANEL_POLICY.json",
    "EVALUATION_PANEL_POLICY_002.json",
}
STATISTICS_NAMES = {"dev_statistics.py", "DEV_STATISTICS_POLICY.json"}
PUBLISH_OPERATIONS = (
    "checkpoint_publish", "terminal_costs_publish", "checkpoint_promote",
)


def canonical(value, *, study=False):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=not study, allow_nan=False)


def digest(value, *, study=False):
    return hashlib.sha256(canonical(value, study=study).encode(
        "utf-8" if study else "ascii")).hexdigest()


def seal(value, *, study=False):
    body = {key: deepcopy(item) for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": digest(body, study=study)}


def write_json(path, value, *, study=False):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((canonical(value, study=study) + "\n").encode(
        "utf-8" if study else "ascii"))
    return path


def raw_file_anchor(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw),
            "file_sha256": hashlib.sha256(raw).hexdigest()}


def json_anchor(path, convention="ASCII_SEALED_EXCLUDE_OWN_SHA256"):
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    sealed = convention.endswith("SEALED_EXCLUDE_OWN_SHA256")
    study = convention.startswith("UTF8_STUDY_")
    body = {key: item for key, item in value.items() if key != "sha256"} if sealed else value
    return {**raw_file_anchor(path), "body_sha256": digest(body, study=study),
            "body_hash_convention": convention}


def count(number):
    return {"attempted": number, "returned": number, "failed": 0, "pending": 0}


def rates():
    """Independently express the registered rational warmup/decay schedule."""
    return [float(Fraction(1, 20000)), float(Fraction(1, 10000))] + [
        float(Fraction(27 - ordinal, 250000)) for ordinal in range(3, 28)]


def factor_shapes():
    values = {}
    for layer in range(28):
        prefix = "model.layers.%d.self_attn." % layer
        for projection, factor, shape in (
            ("q_proj", "A", [8, 1024]), ("q_proj", "B", [2048, 8]),
            ("v_proj", "A", [8, 1024]), ("v_proj", "B", [1024, 8]),
        ):
            values[prefix + projection + ".lora_" + factor] = shape
    return values


def factor_inventory():
    return [{"name": name, "shape": shape, "dtype": "float32", "device": "cpu",
             "requires_grad": True, "parameters": shape[0] * shape[1]}
            for name, shape in factor_shapes().items()]


def population_documents():
    parents = []
    for skill in ("R", "S", "RS"):
        for ordinal in range(32):
            name = "%s_%02d" % (skill, ordinal)
            parents.append({"parent_id": "fixture_parent_" + name,
                            "component_id": "component_fixture_" + name,
                            "skill": skill, "source_hash": digest("fixture_source_" + name),
                            "stratum": "fixture_stratum"})
    panel = seal({"selected_parents": parents, "shortages": []})
    prompts, inputs = [], []
    for parent in parents:
        for condition in (0, 1):
            for rendering in ("familiar", "heldout_ledger"):
                identity = digest({"panel": panel["sha256"], "parent": parent["parent_id"],
                                   "condition": condition, "rendering": rendering})
                messages = [{"role": "user", "content": "Synthetic visible fixture %s %d %s" % (
                    parent["parent_id"], condition, rendering)}]
                prompt_hash = digest(messages)
                prompts.append({"row_id": identity, "messages": messages,
                                "prompt_sha256": prompt_hash})
                ids = [17, 101 + condition, 300 if rendering == "familiar" else 301]
                inputs.append({"row_id": identity, "prompt_sha256": prompt_hash,
                               "prompt_ids": ids, "prompt_tokens": len(ids),
                               "attention_mask": [1] * len(ids)})
    prompt_doc = {"panel_sha256": panel["sha256"], "rows": prompts}
    token_doc = seal({"rows": inputs, "source_packet_sha256": digest(prompt_doc),
                      "context_cap": 512, "maximum_generated_tokens": 32,
                      "rows_dropped": 0, "rows_truncated": 0, "overlong_row_ids": []})
    def references(names):
        return {name: {"path": "CPU_SYNTHETIC_NOT_NATIVE_AUTHORITY", "bytes": 1,
                       "sha256": digest("fixture_reference_" + name)} for name in names}
    refs = references(("panel.json", "policy/prompts.json", "prompt_token_inputs_cpu.json"))
    return panel, prompt_doc, token_doc, refs, references(SOURCE_NAMES), references(STATISTICS_NAMES)


def registration_document():
    arms = {}
    for arm in ("M1", "M2"):
        updates, ordered = [], []
        for ordinal, learning_rate in enumerate(rates(), 1):
            ids = [digest("fixture_occurrence_%s_%03d" % (arm, position))
                   for position in range((ordinal - 1) * 20, ordinal * 20)]
            ordered.extend(ids)
            updates.append({"update_ordinal": ordinal, "occurrence_row_ids": ids,
                            "input_tokens": 60 + ordinal, "supervised_tokens": 20,
                            "learning_rate": learning_rate})
        arms[arm] = {"updates": updates, "occurrence_order_sha256": digest(ordered, study=True)}
    return seal({"schema_version": 1, "kind": "CPU_SYNTHETIC_STUDY_SCHEDULE_FIXTURE",
                 "seed": 101, "arms": arms,
                 "source_and_decision_pins": {"fixture/source_and_decision": digest("fixture decision")}},
                study=True)


def recipe_document(registration, arm):
    return seal({"schema_version": 1, "registration_sha256": registration["sha256"],
                 "arm": arm, "seed": 101,
                 "cohort_sha256": registration["arms"][arm]["occurrence_order_sha256"],
                 "source_pins": deepcopy(registration["source_and_decision_pins"]),
                 "execution_scope": "CPU_PROTOCOL_FAKE",
                 "chunk_source_sha256": hashlib.sha256((PACKAGE / "study_native_chunk.py").read_bytes()).hexdigest(),
                 "backend_source_sha256": hashlib.sha256((PACKAGE / "study_torch_backend.py").read_bytes()).hexdigest()},
                study=True)


def binding_document(registration, arm, completed):
    selected = registration["arms"][arm]["updates"][:completed]
    ids = [row_id for update in selected for row_id in update["occurrence_row_ids"]]
    return {"completed_updates": completed, "consumed_occurrences": 20 * completed,
            "input_tokens": sum(update["input_tokens"] for update in selected),
            "supervised_tokens": sum(update["supervised_tokens"] for update in selected),
            "consumed_order_sha256": digest(ids, study=True), "next_update_ordinal": completed + 1,
            "learning_rate": rates()[completed - 1]}


def posted_costs(binding, chunk):
    units = {"gradient_reset": chunk * 9, "forward": chunk * 180,
             "completion_loss": chunk * 180, "backward": chunk * 180,
             "finite_gradient_clip": chunk * 9, "optimizer_step": chunk * 9,
             "checkpoint_save": chunk, "checkpoint_reload": chunk - 1}
    units.update({name: chunk - 1 for name in PUBLISH_OPERATIONS})
    primitives = {"fixture_source_metadata_read": count(chunk * 112)}
    for name in PUBLISH_OPERATIONS:
        for suffix in ("write", "verify"):
            primitives["chunk_" + name + "_" + suffix] = count(chunk - 1)
    return {"schema_version": 1, "scope": STUDY_KIND,
            "protocol": {name: count(number) for name, number in units.items()},
            "positions": {"input_attempted": binding["input_tokens"],
                          "input_returned": binding["input_tokens"],
                          "supervised_attempted": binding["supervised_tokens"],
                          "supervised_returned": binding["supervised_tokens"]},
            "primitive_operations": primitives}


def projected_costs(costs, operation):
    result = deepcopy(costs)
    for field in ("attempted", "returned"):
        result["protocol"][operation][field] += 1
    for suffix in ("write", "verify"):
        name = "chunk_" + operation + "_" + suffix
        item = result["primitive_operations"].setdefault(name, count(0))
        for field in ("attempted", "returned"):
            item[field] += 1
    return result


def publication(path, payload):
    path = write_json(path, payload, study=True)
    anchor = raw_file_anchor(path)
    return {"path": anchor["path"], "bytes": anchor["bytes"], "sha256": anchor["file_sha256"],
            "payload_sha256": digest(payload, study=True), "durable": True}


def refresh_checkpoint(descriptor):
    path = write_json(descriptor["path"], {"kind": "CPU_INERT_CHECKPOINT_METADATA_ONLY",
                                         "metadata": descriptor["metadata"],
                                         "factor_byte_observation": factor_observations(
                                             descriptor["metadata"]["arm"])["archive"]})
    anchor = raw_file_anchor(path)
    descriptor.update({"path": anchor["path"], "bytes": anchor["bytes"],
                       "sha256": anchor["file_sha256"],
                       "metadata_sha256": digest(descriptor["metadata"], study=True)})


def refresh_resume(resume, *, recompute_costs=False):
    envelope = resume["envelope"]
    envelope.update(seal(envelope, study=True))
    resume["publication"] = publication(resume["publication"]["path"], envelope)
    terminal = resume["durable_terminal"]
    terminal["envelope_sha256"] = envelope["sha256"]
    if recompute_costs:
        terminal["costs"] = projected_costs(projected_costs(
            envelope["posted_save_costs"], "checkpoint_publish"), "terminal_costs_publish")
        resume["terminal_costs"] = projected_costs(terminal["costs"], "checkpoint_promote")
    terminal["costs_sha256"] = digest(terminal["costs"], study=True)
    resume["terminal_publication"] = publication(resume["terminal_publication"]["path"], terminal)
    promotion = resume["promotion"]
    promotion.update({"envelope_sha256": envelope["sha256"],
                      "durable_terminal_sha256": digest(terminal, study=True),
                      "terminal_costs": deepcopy(resume["terminal_costs"])})
    resume["promotion_publication"] = publication(resume["promotion_publication"]["path"], promotion)
    resume.update(seal(resume, study=True))
    return resume


def study_chain(root, registration, arm):
    root = Path(root).resolve()
    recipe, chain = recipe_document(registration, arm), []
    for chunk in (1, 2, 3):
        binding = binding_document(registration, arm, chunk * 9)
        metadata = {"schema_version": 1, "kind": BACKEND_KIND, "experiment": "SYNTH-CRS-001",
                    "scope": "CPU_PROTOCOL_FAKE", "arm": arm, "seed": 101,
                    "recipe_sha256": recipe["sha256"], "cohort_sha256": recipe["cohort_sha256"],
                    "backend_source_sha256": recipe["backend_source_sha256"],
                    "fresh_start_attestation_sha256": digest("fixture fresh zero B " + arm, study=True),
                    "adapter_parameter_order": list(factor_shapes()),
                    "adapter_inventory_sha256": digest(factor_inventory(), study=True),
                    "binding": binding, "native_launch_authorized": False, "training_authorized": False,
                    "native_resume_qualified": False, "qualification_STATE1_used": False}
        descriptor = {"schema_version": 1, "kind": BACKEND_KIND,
                      "path": str(root / ("fixture_%s_chunk%d_checkpoint.json" % (arm, chunk))),
                      "metadata": metadata}
        refresh_checkpoint(descriptor)
        posted = posted_costs(binding, chunk)
        envelope = seal({"schema_version": 1, "kind": STUDY_KIND, "arm": arm, "seed": 101,
                         "chunk_ordinal": chunk, "recipe": deepcopy(recipe), "binding": binding,
                         "backend_descriptor": descriptor, "posted_save_costs": posted,
                         "native_resume_qualified": False, "training_authorized": False}, study=True)
        terminal_costs = projected_costs(projected_costs(posted, "checkpoint_publish"), "terminal_costs_publish")
        terminal = {"envelope_sha256": envelope["sha256"], "costs": terminal_costs,
                    "costs_sha256": digest(terminal_costs, study=True)}
        latest = projected_costs(terminal_costs, "checkpoint_promote")
        promotion = {"envelope_sha256": envelope["sha256"],
                     "durable_terminal_sha256": digest(terminal, study=True), "terminal_costs": latest}
        chain.append(seal({"schema_version": 1, "kind": STUDY_KIND,
                           "status": "SUCCESSFUL_STUDY_CHUNK_PROTOCOL_ONLY", "envelope": envelope,
                           "publication": publication(root / ("fixture_%s_chunk%d_envelope.json" % (arm, chunk)), envelope),
                           "durable_terminal": terminal,
                           "terminal_publication": publication(root / ("fixture_%s_chunk%d_terminal.json" % (arm, chunk)), terminal),
                           "promotion": promotion,
                           "promotion_publication": publication(root / ("fixture_%s_chunk%d_promotion.json" % (arm, chunk)), promotion),
                           "terminal_costs": latest}, study=True))
    return chain, deepcopy(chain[-1]["terminal_costs"])


def factor_observations(arm):
    parameters = {name: {"shape": shape, "dtype": "float32", "bytes": 4 * shape[0] * shape[1],
                         "sha256": digest("fixture inert factor metadata " + arm + " " + name)}
                  for name, shape in factor_shapes().items()}
    observation = seal({"parameter_order": list(parameters), "parameters": parameters,
                        "hash_recipe": "IEEE754_FP32_LITTLE_ENDIAN_CONTIGUOUS_ROW_MAJOR_RAW_BYTES_V1",
                        "maximum_hash_read_bytes": 65536,
                        "total_factor_bytes": sum(item["bytes"] for item in parameters.values())})
    return {name: deepcopy(observation) for name in (
        "archive", "resident_before_capture", "resident_after_capture")}


def restore_receipt(arm, descriptor, binding):
    operations = ["final_adapter_checkpoint_bytes_read", "final_adapter_weights_only_load",
                  "final_adapter_archive_validate", "inference_frozen_base_inventory",
                  "inference_CPU_base_snapshot", "inference_FP32_compute_install",
                  "inference_caller_RNG_snapshot", "inference_fresh_zeroB_qv_install",
                  "inference_caller_RNG_restore", "inference_fresh_zeroB_verify",
                  "final_adapter_bitwise_copy_verify", "inference_base_storage_verify_after",
                  "inference_CPU_base_bytes_verify_after"]
    costs = {name: count(1) for name in operations}
    costs["final_adapter_parameter_copy"] = count(112)
    return {"status": "RESTORED_FINAL_ADAPTER_INFERENCE_UNQUALIFIED",
            "scope": "CPU_RANDOM_TINY_FINAL_ADAPTER_INFERENCE_FIXTURE", "arm": arm,
            "inference_allowed": True, "native_launch_authorized": False, "training_authorized": False,
            "study_checkpoint_qualification_established": False, "optimizer_restored": False,
            "training_RNG_restored": False, "optimizer_policy": "archive validated but never instantiated/restored",
            "RNG_policy": "caller CPU/CUDA0 RNG snapshot restored in finally around transient fresh q/v initialization; private archive validation only; training RNG never applied",
            "caller_RNG_preserved": True, "confirmation_eligible": False, "CPU_fixture": True,
            "real_final_study_chain_verified": False, "checkpoint_sha256": descriptor["sha256"],
            "final_binding": deepcopy(binding), "adapter_tensors_copied": 112,
            "base_unchanged": True, "evaluation_mode_preserved": True,
            "no_grad_required_for_all_raw_forwards": True,
            "prompt_boundary": "existing inference_component.validate_row exact five fields; no oracle/labels/teacher",
            "operation_costs": costs}


def raw_outputs(inputs):
    maximum_positions = sum(32 * row["prompt_tokens"] + 496 for row in inputs)
    forwards = positions = 0
    outputs = []
    for ordinal, row in enumerate(inputs, 1):
        generated = [700 + ordinal] * 32 if ordinal == 2 else [700 + ordinal, 151645]
        trace = [{"token": token, "logprob": -0.25,
                  "prefix_length": row["prompt_tokens"] + step} for step, token in enumerate(generated)]
        forwards += len(trace)
        positions += sum(step["prefix_length"] for step in trace)
        eos = generated[-1] in (151643, 151645)
        outputs.append({"schema_version": 1, "implementation": "synth-prompt-only-inference-component-1",
                        "scope": RAW_SCOPE, "status": "RAW_RESPONSE_COLLECTED_UNSCORED",
                        "row_id": row["row_id"], "prompt_sha256": row["prompt_sha256"],
                        "prompt_ids_sha256": digest(row["prompt_ids"]), "input_row_sha256": digest(row),
                        "source_packet_sha256": digest(inputs), "prompt_tokens": row["prompt_tokens"],
                        "raw_generated_ids": generated, "selected_logprobs": [-0.25] * len(trace), "trace": trace,
                        "finish_reason": "eos" if eos else "generation_cap", "truncated": not eos,
                        "eos_token_id": generated[-1] if eos else None, "context_cap": 512,
                        "maximum_generated_tokens": 32, "decoding": "greedy_no_cache_no_grad",
                        "padding": False, "packing": False, "prompt_truncated": False,
                        "oracle_inputs_read": False, "capability_correctness_scored": False,
                        "native_launch_authorized": False, "native_qualification_established": False,
                        "confirmation_eligible": False,
                        "costs": {"operations": {"inference_forward": count(forwards)},
                                  "input_positions": count(positions), "attempted_forwards": forwards,
                                  "maximum_attempted_forwards": 2048, "maximum_input_positions": maximum_positions},
                        "decoded_response": "synthetic response metadata %d" % ordinal,
                        "parameter_preservation": {"parameter_aliases": 423,
                            "objects_pointers_dtypes_flags_gradients_preserved": True,
                            "mutation_counters_preserved": True,
                            "independent_parameter_byte_verification_required": True}})
    return outputs


def batch_refs(root, arm, ordinal, inputs, outputs, identity_hash, restore, factors):
    directory = Path(root) / "batches" / ("batch%d" % ordinal)
    config_path = write_json(directory / ("fixture_%s_config.json" % arm), seal({
        "schema_version": 1, "kind": "CPU_SYNTHETIC_CANDIDATE_IMPORT_FIXTURE", "arm": arm,
        "ordinal": ordinal, "candidate_identity_sha256": identity_hash, "inputs": inputs,
        "source_packet_sha256": digest(inputs), "restore_receipt_sha256": digest(restore)}))
    config = json_anchor(config_path)
    audit_path = write_json(directory / ("fixture_%s_audit.json" % arm), seal({
        "schema_version": 1, "kind": "CPU_SYNTHETIC_CANDIDATE_IMPORT_FIXTURE", "arm": arm,
        "ordinal": ordinal, "candidate_identity_sha256": identity_hash, "outputs": outputs,
        "config_file_sha256": config["file_sha256"], "restore_receipt_sha256": digest(restore),
        "factor_observations_sha256": digest(factors)}))
    return {"config": config, "audit": json_anchor(audit_path)}


class Scenario:
    """Own synthetic documents separately from the contract's private handles."""
    def __init__(self, contract, root, arm="M1"):
        self.contract, self.root, self.arm = contract, Path(root).resolve(), arm
        self.root.mkdir(parents=True, exist_ok=True)
        self.population_docs = population_documents()
        self.inputs = self.population_docs[2]["rows"]
        self.population = contract.fixture_population(*self.population_docs)
        self.registration = registration_document()
        self.chain, self.latest = study_chain(self.root / "study", self.registration, arm)
        self.selection = contract.fixture_selection(self.population, self.registration, arm,
                                                    self.chain, self.latest, root=self.root)
        descriptor = self.chain[-1]["envelope"]["backend_descriptor"]
        binding = self.chain[-1]["envelope"]["binding"]
        self.restore = restore_receipt(arm, descriptor, binding)
        self.factors = factor_observations(arm)
        identity_hash = contract.value(self.selection, "selection")["identity"]["sha256"]
        self.batch_documents = []
        self.batches = []
        for ordinal in range(1, 7):
            rows = deepcopy(self.inputs[(ordinal - 1) * 64:ordinal * 64])
            outputs = raw_outputs(rows)
            refs = batch_refs(self.root, arm, ordinal, rows, outputs, identity_hash, self.restore, self.factors)
            self.batch_documents.append({"inputs": rows, "outputs": outputs, "refs": refs})
            self.batches.append(self.batch(ordinal))
        self.import_selection = contract.fixture_import_selection(self.selection, self.batches, root=self.root)

    def batch(self, ordinal, **overrides):
        document = self.batch_documents[ordinal - 1]
        return self.contract.fixture_batch(self.selection, ordinal,
            overrides.get("inputs", document["inputs"]), overrides.get("outputs", document["outputs"]),
            overrides.get("restore_receipt", self.restore), overrides.get("factor_observations", self.factors),
            overrides.get("refs", document["refs"]), root=self.root)

    def prepare(self, *, batches=None, population=None, selection=None, import_selection=None):
        return self.contract.prepare_candidate_import(
            self.population if population is None else population,
            self.batches if batches is None else batches,
            selection=self.selection if selection is None else selection,
            import_selection=self.import_selection if import_selection is None else import_selection)
