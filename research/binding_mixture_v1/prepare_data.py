"""Create one frozen native+binding mixture from unchanged compiled JSONL rows.

Only data preparation is performed. No tokenizer/model, source weights,
training grant, runtime qualification, or evaluation is invoked.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

BASE = Path("C:/Users/clclc/BusinessProject")
RECORDS = BASE / "rineya-dataset-retention-records"
NATIVE_ROOT = RECORDS / "doseint001-20260919"
NATIVE = NATIVE_ROOT / "prepared-dual899"
BINDING = RECORDS / "binding-signal-prep-20260920/candidate-v2"
COMPILED = BINDING / "compiled-v1"
OUT = RECORDS / "binding-mixture001-20260920/prepared"
PILOT = BASE / "ChatBubble-retention-doselora-resume-20260920"
ORDER_SEED = 17
PINS = {
    NATIVE / "DATA_CONTRACT.json": "9661fa2d1b1a6e1eb8c82cd187a0fed98581ac134c66bc0f29f88f305e43dba6",
    NATIVE_ROOT / "PLAN_DUAL899.json": "adc40e3cea512bbfe7d20fe0d9db823c5f7fcbc7cf781f99f0be22f2ec215fe2",
    NATIVE_ROOT / "BUILD_GATE_DUAL899.json": "3dc082a3b2295f65123a8f89c2d3ee01d4487d71dc89ed2303072c128aeda658",
    BINDING / "BUILD.json": "f2a4e1986ce7b42b7317f3d888a3471addc358a5aabab6659d49685035c62741",
    BINDING / "DATA_ADMISSION.json": "854ff8124cdc2e9411a0cb867633b96ec3a08b1bfceb28ee0d8be90706bf9935",
    COMPILED / "TOKEN_AUDIT.json": "d240838961e4d0fe0be66b1f7248e934c0d6b15db592c83aaaa6c038b5060f34",
    PILOT / "experiments/model_quality_gpu_v1/data.py": "148204b01b9e17e9979f3ef1a92b3effae6c72186cb9d07ce72e0afb611f65ab",
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), "expected metadata object")
    return value


def bind_file(path, expected, bound):
    path = Path(path)
    actual = sha(path)
    require(actual == expected, "frozen input hash differs: " + str(path))
    bound[str(path)] = actual


def original_lines(path, expected, source):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "compiled input changed before consumption")
    entries = []
    for index, line in enumerate(raw.splitlines(keepends=True)):
        require(line.endswith(b"\n") and line.strip(), "input must contain nonempty newline-terminated JSONL rows")
        row = json.loads(line)
        ids, labels, prompt_ids = row["ids"], row["labels"], row["prompt_ids"]
        end = row["answer_end"]
        require(row["source_split"] == "train" and row["mode"] == "direct" and row["truncated"] is False,
                "only unchanged direct TRAIN views are eligible")
        require(len(ids) == len(labels) == row["processed_tokens"] and 0 < len(ids) <= 2048
                and ids[:len(prompt_ids)] == prompt_ids and len(prompt_ids) + 128 <= 2048,
                "compiled input/prompt geometry differs")
        require(labels[0] == -100 and row["answer_start"] == len(prompt_ids)
                and len(prompt_ids) <= end < len(ids) and ids[end] == 151645
                and labels == [token if len(prompt_ids) <= i <= end else -100 for i, token in enumerate(ids)]
                and sum(token != -100 for token in labels[1:]) == row["supervised_tokens"],
                "compiled response/EOS mask differs")
        entries.append({"row": row, "raw": line, "source": source, "source_index": index,
                        "source_row_sha256": hashlib.sha256(line).hexdigest()})
    return entries


def uniform_mixture(native, binding):
    """Preserve native order; use one fixed Random(17) permutation for binding.

    Position p is binding iff floor((p+1)*B/N) exceeds floor(p*B/N).
    Since B<N, position zero is native. No view is sampled out or repeated.
    """
    require(native and binding, "both fixed mixture components are required")
    shuffled = list(binding)
    random.Random(ORDER_SEED).shuffle(shuffled)
    total, bcount = len(native) + len(binding), len(binding)
    mixture, ni, bi = [], 0, 0
    for position in range(total):
        if ((position + 1) * bcount) // total > (position * bcount) // total:
            mixture.append(shuffled[bi]); bi += 1
        else:
            mixture.append(native[ni]); ni += 1
    require(ni == len(native) and bi == len(binding) and mixture[0] is native[0], "mixture inventory/first row differs")
    return mixture


def prepare():
    require(not any(name in sys.modules for name in ("torch", "transformers", "tokenizers")), "data preparation must be model/tokenizer-free")
    require(not OUT.exists() and not OUT.is_symlink(), "prepared outputs must be fresh")
    bound = {}
    for path, expected in PINS.items():
        bind_file(path, expected, bound)
    native_contract = read(NATIVE / "DATA_CONTRACT.json")
    native_plan = read(NATIVE_ROOT / "PLAN_DUAL899.json")
    native_gate = read(NATIVE_ROOT / "BUILD_GATE_DUAL899.json")
    for name, expected in native_contract["files"].items():
        bind_file(NATIVE / name, expected, bound)
    build, admission, token_audit = read(BINDING / "BUILD.json"), read(BINDING / "DATA_ADMISSION.json"), read(COMPILED / "TOKEN_AUDIT.json")
    require(admission["status"] == "ADMITTED_FOR_NAMED_EXPLORATORY_SCOPE"
            and admission["training_permission"] is False and admission["external_exclusions_complete"] is False,
            "binding scoped data admission differs; no model permission is inferred")
    for relative, expected in admission["evidence_sha256"].items():
        path = (BINDING / relative).resolve()
        require(path.is_relative_to(BINDING.resolve()), "admission evidence path escapes candidate")
        bind_file(path, expected, bound)
    require(admission["compiled_files_sha256"] == token_audit["output_sha256"], "admission/compilation outputs differ")
    for name, expected in token_audit["output_sha256"].items():
        bind_file(COMPILED / name, expected, bound)
    require(token_audit["input_sha256_before"] == token_audit["input_sha256_after"], "binding input custody changed during compilation")
    for name, expected in token_audit["input_sha256_before"].items():
        bind_file(BINDING / name, expected, bound)
    require(build["rows"] == {"train": 768, "dev": 384} and build["worlds"] == {"train": 64, "dev": 32}
            and build["entity_overlap"] == build["world_overlap"] == build["template_content_overlap"] == 0,
            "binding world/entity/template partition inventory differs")
    open_audit = read(BINDING / "OPEN_EXCLUSION_AUDIT_V2.json")
    lc_audit = read(BINDING / "SPENT_LC_TEXT_CONTACT_AUDIT.json")
    require(open_audit["status"] == "PASS_OPEN_EXCLUSIONS_BUILD_ONLY"
            and open_audit["candidate_contact_rows"] == {"train": 0, "dev": 0}
            and lc_audit["status"] == "PASS_DOCUMENTED_NATIVE_TEXT_CONTACT_ONLY"
            and all(value["contact_rows"] == 0 for value in lc_audit["checks"].values()),
            "binding named open/LC text-contact scope did not pass")
    native = original_lines(NATIVE / "microfit-train.jsonl", native_contract["files"]["microfit-train.jsonl"], "native_dual899")
    binding = original_lines(COMPILED / "TRAIN_COMPILED.jsonl", token_audit["output_sha256"]["TRAIN_COMPILED.jsonl"], "binding_v2")
    require(len(native) == 2697 and len(binding) == 768, "fixed source view inventory differs")
    require([entry["row"]["id"] for entry in native] == native_plan["row_order_per_exposure"], "original native order changed")
    require(len(native[0]["row"]["ids"]) == 116, "first native qualification row is not the frozen 116-token input")
    # Read only self-generated TRAIN/DEV metadata to verify actual grouped partition custody.
    raw_binding = {split: [json.loads(line) for line in (BINDING / (split.upper() + ".jsonl")).read_text(encoding="utf-8").splitlines()]
                   for split in ("train", "dev")}
    train_by_id = {row["id"]: row for row in raw_binding["train"]}
    require(len(train_by_id) == 768 and set(train_by_id) == {entry["row"]["id"] for entry in binding}, "compiled binding TRAIN membership differs")
    parents, entities, families = {}, {}, {}
    for split, source_rows in raw_binding.items():
        parents[split] = {row["parent_id"] for row in source_rows}
        entities[split] = {value for row in source_rows for value in row["world"]["roles"] + row["world"]["people"]}
        families[split] = {row["template_sha256"] for row in source_rows}
        require(set(Counter(row["parent_id"] for row in source_rows).values()) == {12}, "binding world derivatives incomplete")
    require(len(parents["train"]) == 64 and len(parents["dev"]) == 32
            and not parents["train"] & parents["dev"] and not entities["train"] & entities["dev"]
            and not families["train"] & families["dev"], "binding TRAIN/DEV parent/entity/template overlap")
    for entry in binding:
        row, original = entry["row"], train_by_id[entry["row"]["id"]]
        require(row["parent_id"] == original["parent_id"] and row["category"] == original["category"]
                and row["prompt"] == original["messages"][:2] and row["reference"] == original["reference"],
                "compiled binding row differs from its admitted source")
    mixed = uniform_mixture(native, binding)
    rows = [entry["row"] for entry in mixed]
    ids = [row["id"] for row in rows]
    all_parents = {row["parent_id"] for row in rows}
    require(len(rows) == len(set(ids)) == 3465 and len(all_parents) == 2761, "mixture views/parent-world inventory differs")
    require(len({entry["row"]["parent_id"] for entry in native}) == 2697
            and not {entry["row"]["parent_id"] for entry in native} & parents["train"], "native/binding parent identity overlap")
    require(not all_parents & parents["dev"], "mixture touches binding DEV parents")
    exclusions = []
    for label, bank in native_gate["banks"].items():
        path = Path(bank["path"])
        bind_file(path, bank["sha256"], bound)
        views = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        bank_parents = {row["parent_id"] for row in views}
        require(len(views) == len(bank_parents) == 384 and not all_parents & bank_parents,
                "mixture acquisition-parent exclusion failed")
        exclusions.append({"bank": label, "path": str(path), "sha256": bank["sha256"],
                           "eval_parents": 384, "mixture_parent_overlap": 0,
                           "scope": "all 3465 views / 2761 actual parent worlds"})
    sys.path.insert(0, str(PILOT))
    from experiments.model_quality_gpu_v1.data import group_updates
    require(Path(group_updates.__code__.co_filename).resolve() == (PILOT / "experiments/model_quality_gpu_v1/data.py").resolve(),
            "grouping helper imported from another checkout")
    groups = group_updates(rows, 2048)
    processed = sum(len(row["ids"]) for row in rows)
    supervised = sum(sum(token != -100 for token in row["labels"][1:]) for row in rows)
    require(processed == 577417 and supervised == 39502, "mixture exact token inventory differs")
    require([entry["source_index"] for entry in mixed if entry["source"] == "native_dual899"] == list(range(2697)),
            "mixture changed native relative order")
    require(sorted(entry["source_index"] for entry in mixed if entry["source"] == "binding_v2") == list(range(768)),
            "mixture duplicated or omitted binding views")
    require(all(sha(Path(path)) == expected for path, expected in bound.items()), "source/admission input changed during preparation")
    require(not any(name in sys.modules for name in ("torch", "transformers", "tokenizers")), "unexpected model/tokenizer import")
    OUT.mkdir(parents=True, exist_ok=False)
    with (OUT / "microfit-train.jsonl").open("xb") as stream:
        for entry in mixed:
            stream.write(entry["raw"])
    membership = [{"mixture_index": index, "id": entry["row"]["id"], "parent_id": entry["row"]["parent_id"],
                   "source": entry["source"], "source_index": entry["source_index"],
                   "source_row_sha256_including_newline": entry["source_row_sha256"]}
                  for index, entry in enumerate(mixed)]
    group_record = {"schema": 1, "update_tokens": 2048, "update_groups": groups,
                    "row_order_per_exposure": ids, "row_order_sha256": digest(ids),
                    "processed_tokens": processed, "supervised_tokens": supervised,
                    "training_permission": False}
    for name, value in (("SOURCE_MEMBERSHIP.json", membership), ("UPDATE_GROUPS.json", group_record)):
        with (OUT / name).open("xb") as stream:
            stream.write(canonical(value) + b"\n")
    output_lines = (OUT / "microfit-train.jsonl").read_bytes().splitlines(keepends=True)
    require(len(output_lines) == len(mixed) and all(line == entry["raw"] for line, entry in zip(output_lines, mixed)),
            "written mixture row bytes differ from original compiled inputs")
    build = {"schema": 1, "record": "BINDING_MIXTURE_DATA_BUILD", "status": "PASS_DATA_PREPARATION_ONLY",
        "rows": 3465, "presentations": 3465, "unique_worlds": 2761, "parent_worlds": 2761,
        "exposures_per_view": 1, "processed_tokens": processed, "supervised_tokens": supervised,
        "update_groups_count": len(groups), "update_tokens": 2048,
        "first_row_id": rows[0]["id"], "first_row_tokens": len(rows[0]["ids"]),
        "max_sequence_tokens": max(len(row["ids"]) for row in rows), "truncated_rows": 0,
        "ordering": {"binding_shuffle_seed": ORDER_SEED, "binding_shuffle_algorithm": "Python random.Random(17).shuffle once",
                     "insertion_rule": "at position p, choose binding iff floor((p+1)*768/3465)>floor(p*768/3465)",
                     "native_relative_order_preserved": True, "first_native_row_preserved": True,
                     "row_order_sha256": digest(ids)},
        "source_geometry": {"native_dual899": {"views": 2697, "parent_worlds": 2697, "processed_tokens": 380921, "supervised_tokens": 37090},
                            "binding_v2": {"views": 768, "parent_worlds": 64, "views_per_world": 12,
                                           "processed_tokens": 196496, "supervised_tokens": 2412,
                                           "categories": dict(Counter(entry["row"]["category"] for entry in binding))}},
        "input_files_sha256": bound, "output_files_sha256": {name: sha(OUT / name) for name in
            ("microfit-train.jsonl", "UPDATE_GROUPS.json", "SOURCE_MEMBERSHIP.json")},
        "binding_data_admission_path": str(BINDING / "DATA_ADMISSION.json"),
        "binding_train_dev_separation": {"train_parent_worlds": 64, "dev_parent_worlds": 32,
                                        "parent_overlap": 0, "entity_overlap": 0, "template_overlap": 0},
        "acquisition_parent_exclusions": exclusions, "rows_copied_byte_identically": True,
        "labels_or_teacher_inputs_modified": False, "source_rows_repeated_or_dropped": False,
        "scope_policy": {"native_dual899": "Existing qualified native corpus and its acquisition parent exclusions; binding literal rules are not retroactively applied.",
                         "binding_v2": "Only the named exploratory scope in DATA_ADMISSION; approved literal/open/LC checks stay attached to this component.",
                         "reserved_indices_and_legacy_lc_fingerprints": "UNVERIFIED; no reserved raw content opened; no universal exclusion claim."},
        "builder_sha256": sha(Path(__file__)), "grouping_helper_sha256": PINS[PILOT / "experiments/model_quality_gpu_v1/data.py"],
        "qualification_tokens_included": False, "model_calls": 0, "source_weights_read": False,
        "tokenizer_used": False, "training_permission": False, "evaluation_permission": False,
        "quality_claim": False, "release_permitted": False, "human_data_review": "NOT_PERFORMED"}
    with (OUT / "DATA_BUILD.json").open("xb") as stream:
        stream.write(canonical(build) + b"\n")
    return build


if __name__ == "__main__":
    result = prepare()
    print(json.dumps({key: result[key] for key in ("status", "rows", "unique_worlds", "processed_tokens",
                      "supervised_tokens", "update_groups_count", "first_row_tokens", "max_sequence_tokens")}, indent=2))
