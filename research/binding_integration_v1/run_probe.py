"""Run one frozen diagnostic under the existing watchdog and bind its model bytes.

The runner uses only the standard library. It never imports the scorer or torch,
retries an attempt, removes files, or reports endpoint scores. Every --out must
be new; the legacy probe label must also have no pre-existing result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

BASE = Path("C:/Users/clclc/BusinessProject")
PILOT = BASE / "ChatBubble-binding-integration-pilot-20260920"
PYTHON = BASE / ".venv-rab001-cu126/Scripts/python.exe"
RAB = PILOT / "experiments/architecture_benchmark_001"
WRAPPER = PILOT / "docs/model-quality/audit-2026-09-17/supervised_probe.py"
SUPERVISOR_CODE = PILOT / "experiments/model_quality_gpu_v1/supervisor.py"
SIDECARS = ("config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "vocab.json", "merges.txt")
PREFIXES = {"process": "PROCESS", "retention": "RETENTION_PROBE", "long_context": "LONGCTX"}
BANK_NAMES = {"process-transition-probe-001": "dose", "select-views-001": "select"}
EXPERIMENT = "BINDING-INTEGRATION-001"
DATA_FILES = {"microfit-train.jsonl", "INTEGRATION_GROUPS.json", "SOURCE_MEMBERSHIP.json", "DATA_BUILD.json"}
FINAL_PROGRESS = {"updates": 193, "next_group": 193, "processed_tokens": 577417, "supervised_tokens": 39502}
NATIVE_CONTROL = BASE / "rineya-dataset-retention-records/highdose-lora001-resume-20260920"
ORIGINAL_PILOT = BASE / "ChatBubble-binding-mixture-pilot-20260920"
MIXTURE_RECORD = BASE / "rineya-dataset-retention-records/binding-mixture001-20260920"


def qualified_trainable_shapes() -> dict:
    projections = {"self_attn.q_proj": (1024, 2048), "self_attn.k_proj": (1024, 1024),
                   "self_attn.v_proj": (1024, 1024), "self_attn.o_proj": (2048, 1024),
                   "mlp.gate_proj": (1024, 3072), "mlp.up_proj": (1024, 3072), "mlp.down_proj": (3072, 1024)}
    result = {}
    for layer in range(8, 28):
        for name, (nin, nout) in projections.items():
            result[f"model.layers.{layer}.{name}.lora_A"] = [16, nin]
            result[f"model.layers.{layer}.{name}.lora_B"] = [nout, 16]
        for name in ("input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight"):
            result[f"model.layers.{layer}.{name}"] = [128 if name.startswith("self_attn.") else 1024]
    return result


def valid_sha(value) -> bool:
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_zero_addend_body(qualification: dict, plan: dict) -> None:
    """Validate the exact bounded prefix evidence; this is not a full-run claim."""
    reference = plan['zero_addend_reference']
    require(qualification.get('schema') == 1 and qualification.get('record') == 'BINDING_INTEGRATION_ZERO_ADDEND_QUALIFICATION' and
            qualification.get('experiment') == EXPERIMENT and qualification.get('status') == 'PASS' and
            qualification.get('complete') is True and qualification.get('scope') == 'EXACT_TWO_UPDATE_PREFIX_ONLY',
            'missing successful bounded-prefix zero-addend qualification')
    require(qualification.get('optimizer_updates') == qualification.get('optimizer_step_count') == 2 and
            qualification.get('binding_forward_calls') == 0 and qualification.get('checked_trainable_tensors') == 360 and
            qualification.get('checked_component_tensors') == 1080,
            'zero-addend qualification changed its registered prefix/scope')
    flags = ('all_trainable_parameters_bitwise_equal', 'all_first_moments_bitwise_equal', 'all_second_moments_bitwise_equal',
             'trainable_state_restored_exactly', 'rng_restored_exactly', 'reference_rng_bitwise_equal', 'frozen_base_unchanged',
             'qualification_optimizer_discarded', 'initial_model_training_mode_restored')
    require(all(qualification.get(key) is True for key in flags), 'zero-addend comparison or state restoration failed')
    require(qualification.get('failure') is None and qualification.get('full_193_update_trajectory_equality_measured') is False and
            qualification.get('generation_calls') == 0, 'zero-addend receipt overclaims its scope or records failure/generation')
    for before, after in (('initial_trainable_sha256', 'restored_trainable_sha256'),
                          ('rng_before_sha256', 'rng_after_restore_sha256'),
                          ('reference_rng_sha256', 'actual_prefix_rng_sha256'),
                          ('frozen_base_sha256_before', 'frozen_base_sha256_after')):
        require(valid_sha(qualification.get(before)) and qualification[before] == qualification.get(after),
                'zero-addend equality flags disagree with state/RNG/frozen-base hashes')
    require(qualification.get('reference_manifest_sha256') == reference['manifest_sha256'] and
            qualification.get('reference_tensor_files_sha256') == reference['tensor_files_sha256'], 'zero-addend reference pins changed')
    require(qualification.get('budget_charges') == {'processed_training_tokens_all_attempts': 4032,
            'evaluation_and_qualification_model_calls': 28}, 'zero-addend qualification costs differ from the two-update prefix')
    shapes = qualified_trainable_shapes()
    checks = qualification.get('checks')
    require(isinstance(checks, list) and len(checks) == 360 and {row.get('name') for row in checks} == set(shapes),
            'zero-addend tensor inventory is incomplete or duplicated')
    files = reference['tensor_files_sha256']
    require(len(files) == 360 and all(valid_sha(pin) for pin in files.values()), 'zero-addend reference file inventory differs')
    require({row.get('reference_file') for row in checks} == set(files), 'zero-addend reference files do not match tensor checks')
    for row in checks:
        require(row.get('shape') == shapes[row['name']] and row.get('reference_file_sha256') == files[row['reference_file']],
                'zero-addend tensor shape or file binding differs')
        for component in ('parameter', 'exp_avg', 'exp_avg_sq'):
            comparison = row.get(component, {})
            require(comparison.get('equal') is True and valid_sha(comparison.get('actual_sha256')) and
                    comparison['actual_sha256'] == comparison.get('expected_sha256'),
                    'zero-addend parameter or optimizer moment equality is missing')


def zero_addend_snapshot(receipt: dict, plan: dict) -> dict:
    path = Path(receipt['zero_addend_qualification_path']).resolve()
    checkpoint = Path(receipt['checkpoint']).resolve()
    require(path == checkpoint.parent / 'zero_addend_qualification.json' and
            sha(path) == receipt['zero_addend_qualification_sha256'], 'zero-addend qualification receipt identity differs')
    qualified = read(path)
    validate_zero_addend_body(qualified, plan)
    checkpoint_binding = receipt.get('checkpoint_binding', receipt.get('binding'))
    require(isinstance(checkpoint_binding, dict) and qualified.get('binding') == checkpoint_binding,
            'zero-addend qualification belongs to another training binding')
    require(qualified.get('reference_progress') == {'updates': 2, 'next_group': 2, 'processed_tokens': 4032, 'supervised_tokens': 388} and
            qualified.get('native_denominators') == plan['native_denominators'][:2] and
            qualified.get('applied_native_learning_rates') == plan['applied_native_learning_rates'][:2] and
            qualified.get('native_row_ids') == [plan['row_order_per_exposure'][index]
                for group in plan['native_update_groups'][:2] for index in group], 'zero-addend native prefix data/schedule differs')
    reference = plan['zero_addend_reference']; reference_root = Path(reference['checkpoint_path']).resolve()
    require(reference_root == (NATIVE_CONTROL / 'journal/runs/train-001/checkpoint-00002').resolve(),
            'zero-addend reference is not the registered original control prefix')
    require(Path(qualified['reference_checkpoint_path']).resolve() == reference_root,
            'zero-addend qualified reference checkpoint path differs')
    require(sha(reference_root / 'manifest.json') == reference['manifest_sha256'], 'zero-addend reference manifest changed')
    for name, expected in reference['tensor_files_sha256'].items():
        require(re.fullmatch(r'tensor-\d{4}\.safetensors', name) is not None and sha(reference_root / name) == expected,
                'zero-addend reference optimizer state changed')
    return {'zero_addend_qualification_path': str(path), 'zero_addend_qualification_sha256': sha(path),
            'zero_addend_scope': 'EXACT_TWO_UPDATE_PREFIX_ONLY', 'native_clock_tokens': 380921,
            'zero_addend_frozen_base_sha256': qualified['frozen_base_sha256_before']}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def regular(path: Path) -> Path:
    require(path.is_file() and not path.is_symlink(), f"missing/nonregular file: {path}")
    return path


def sha(path: Path) -> str:
    with regular(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    value = json.loads(regular(path).read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def native_digest(value: object) -> str:
    """MQ native integrity.digest uses unescaped Unicode, unlike RAB's digest."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write(path: Path, value: dict) -> None:
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def copy_new(source: Path, destination: Path) -> str:
    """Copy exact bytes, without overwriting either an old result or a failed copy."""
    before = sha(source)
    with regular(source).open("rb") as src, destination.open("xb") as dst:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            dst.write(chunk)
        dst.flush()
        os.fsync(dst.fileno())
    require(sha(source) == sha(destination) == before, "raw result changed while copying")
    return before


def implementation_digest() -> str:
    """Exactly gates.implementation_digest, without importing its torch dependency."""
    files = list(RAB.rglob("*.py")) + [RAB / name for name in
            ("protocol.json", "source_lock.json", "requirements.txt", "requirements-native.txt")]
    for name in read(RAB / "protocol.json").get("amendments", []):
        path = RAB / name
        require(path.resolve().is_relative_to(RAB.resolve()), "protocol amendment escapes RAB")
        regular(path)
        files.append(path)
    manifest = {p.relative_to(RAB).as_posix(): sha(p)
                for p in sorted(files, key=lambda p: p.relative_to(RAB).as_posix())}
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def row_ids(path: Path) -> set[str]:
    result = set()
    with regular(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get("id")
            require(isinstance(value, str) and value and value not in result,
                    f"missing or duplicate row ID in {path}")
            result.add(value)
    require(bool(result), f"empty instrument: {path}")
    return result


def source_relocation_snapshot(gate: dict) -> dict:
    """Qualify B-to-C code/cwd/input aliases; never alias weights or results."""
    path = Path(gate['source_relocation_path']).resolve()
    require(sha(path) == gate['source_relocation_sha256'], 'source relocation receipt changed')
    receipt = read(path)
    require(receipt.get('schema') == 1 and receipt.get('record') == 'DOSELORA_SOURCE_RELOCATION' and
            receipt.get('status') == 'QUALIFIED' and Path(receipt['original_root']).resolve() == ORIGINAL_PILOT.resolve() and
            Path(receipt['archive_root']).resolve() == PILOT.resolve() and
            Path(receipt['python_path']).resolve() == PYTHON.resolve() and
            receipt.get('git_head') == '5b4c22dae78f587c41575269f419087b0c7997f6' and
            receipt.get('git_branch') == 'codex/dataset-retention-audit-v1', 'unqualified source/code execution relocation')
    pins = receipt['qualification_files_sha256']
    original_receipt = read(MIXTURE_RECORD / 'SOURCE_RELOCATION.json')
    require(Path(original_receipt['archive_root']).resolve() == ORIGINAL_PILOT.resolve() and
            pins == original_receipt['qualification_files_sha256'], 'new relocation changed the previously qualified source inventory')
    for relative, expected in pins.items():
        child = Path(relative)
        require(not child.is_absolute() and '..' not in child.parts, 'qualified code path escapes its root')
        for folder in (ORIGINAL_PILOT, PILOT):
            require(sha(folder / child) == expected, 'qualified source code differs across execution clones')
    digest = gate['implementation_digest']
    expected_files = {
        'docs/model-quality/audit-2026-09-17/supervised_probe.py': 'wrapper',
        **{f'experiments/rab001_exploration/{kind}_probe.py': 'script' for kind in PREFIXES},
        'experiments/architecture_benchmark_001/evidence/retention-design-v1/PROBE_LABEL_AUDIT.json': 'additional_input',
        f'experiments/architecture_benchmark_001/outputs/retention-{digest[:8]}/B28/retention.json': 'additional_input',
        'Rineya_Long_Context_Demo_v1/data/all_answer_key.jsonl': 'additional_input'}
    found = {}; aliases = []
    for item in receipt['files']:
        old, new = Path(item['original_path']).resolve(), Path(item['archive_path']).resolve()
        require(old.is_relative_to(ORIGINAL_PILOT.resolve()) and new.is_relative_to(PILOT.resolve()),
                'relocation mapping escapes qualified execution roots')
        relative = old.relative_to(ORIGINAL_PILOT.resolve()).as_posix()
        require(relative not in found and relative in expected_files and item['role'] == expected_files[relative] and
                new.relative_to(PILOT.resolve()).as_posix() == relative and sha(old) == sha(new) == item['sha256'],
                'relocation tries to substitute an unqualified file or changes bytes')
        found[relative] = item['role']
        if item['role'] == 'additional_input':
            aliases.append({'original_path': str(old), 'archive_path': str(new), 'sha256': item['sha256']})
    require(found == expected_files, 'relocation must contain exactly four code files and three auxiliary inputs')
    return {'source_relocation_path': str(path), 'source_relocation_sha256': sha(path),
            'original_root': str(ORIGINAL_PILOT), 'archive_root': str(PILOT),
            'qualified_additional_input_aliases': aliases, 'weight_or_result_aliases_permitted': False}


def validate_integration_groups(plan: dict, rows: list, membership: list, registered: dict) -> dict:
    """Recount row-wise accumulation; never concatenate a 3114-token update."""
    require(plan.get("experiment") == EXPERIMENT and plan.get("exposures_per_view") == 1,
            "integration experiment/exposure changed")
    require(len(rows) == len(membership) == 3465 and len({row['id'] for row in rows}) == 3465,
            "integration training views are missing or duplicated")
    require(registered.get('schema') == 1 and registered.get('record') == 'BINDING_INTEGRATION_GROUPS',
            "wrong integration group contract")
    groups = plan['update_groups']; native = plan['native_update_groups']; binding = plan['binding_update_groups']
    require(len(groups) == len(native) == len(binding) == 193 and
            registered['combined_update_groups'] == groups and registered['native_update_groups'] == native and
            registered['binding_update_groups'] == binding, "integration groups differ from the frozen plan")
    require([index for group in groups for index in group] == list(range(3465)),
            "combined groups do not visit each prepared row exactly once in execution order")
    require(all(group == a + b for group, a, b in zip(groups, native, binding)),
            "binding rows are not appended after each unchanged native group")
    lengths = [len(row['ids']) for row in rows]
    supervised = [sum(token != -100 for token in row['labels'][1:]) for row in rows]
    native_indices = [index for group in native for index in group]
    binding_indices = [index for group in binding for index in group]
    require(len(native_indices) == 2697 and len(binding_indices) == 768 and
            set(native_indices).isdisjoint(binding_indices), "native/binding source counts or partitions differ")
    parents = {row['parent_id'] for row in rows}
    require(len(parents) == 2761 and len({rows[i]['parent_id'] for i in native_indices}) == 2697 and
            len({rows[i]['parent_id'] for i in binding_indices}) == 64, "views confused with independent parent worlds")
    for index, member in enumerate(membership):
        require(member['execution_index'] == index and member['id'] == rows[index]['id'] and
                member['parent_id'] == rows[index]['parent_id'], "training membership join differs")
    require(all(membership[i]['source'] == 'native_dual899' for i in native_indices) and
            all(membership[i]['source'] == 'binding_v2' for i in binding_indices) and
            [membership[i]['source_index'] for i in native_indices] == list(range(2697)) and
            sorted(membership[i]['source_index'] for i in binding_indices) == list(range(768)),
            "native/binding source membership or one-presentation scope differs")
    previous = [membership[i]['previous_mixture_index'] for i in binding_indices]
    require(previous == sorted(previous) and len(set(previous)) == 768, "existing binding subsequence order changed")
    for index, group in enumerate(groups):
        require(all(membership[i]['optimizer_group_index'] == index for i in group), "optimizer group membership differs")
        require(binding[index] == binding_indices[index * 768 // 193:(index + 1) * 768 // 193],
                "binding partition is not the registered floor schedule")
    # Reconstruct native-only greedy groups independently of the combined updates.
    native_rebuilt, current, used = [], [], 0
    for index in native_indices:
        if current and used + lengths[index] > 2048:
            native_rebuilt.append(current); current, used = [], 0
        current.append(index); used += lengths[index]
    if current:
        native_rebuilt.append(current)
    require(native_rebuilt == native, "original 2048-token native group boundaries changed")
    native_denominators = [sum(supervised[i] for i in group) for group in native]
    binding_denominators = [sum(supervised[i] for i in group) for group in binding]
    cumulative, total = [], 0
    for group in native:
        total += sum(lengths[i] for i in group); cumulative.append(total)
    require(native_denominators == plan['native_denominators'] == registered['native_denominators'] and
            binding_denominators == registered['binding_denominators'] and min(native_denominators) > 0,
            "integration must normalize by the original native denominator")
    require(cumulative == plan['native_processed_cumulative'] == registered['native_processed_cumulative'] and total == 380921,
            "native LR clock differs from the original native-token trajectory")
    learning_rates = plan['applied_native_learning_rates']
    require(learning_rates == registered['applied_native_learning_rates'] and len(learning_rates) == 193 and
            all(type(rate) in (int, float) and math.isfinite(rate) and rate > 0 for rate in learning_rates),
            "applied native LR inventory is invalid")
    original_indices = [[membership[i]['source_index'] for i in group] for group in native]
    require(original_indices == registered['original_native_update_groups'], "original/native group mapping differs")
    require(sum(lengths) == plan.get('processed_tokens') == 577417 and sum(supervised) == plan.get('supervised_tokens') == 39502 and
            plan.get('presentations') == 3465 and plan.get('unique_worlds') == 2761 and
            [row['id'] for row in rows] == plan['row_order_per_exposure'], "registered integration data/token totals changed")
    combined = [sum(lengths[i] for i in group) for group in groups]
    require(max(combined) == 3114 and max(lengths) <= 2048,
            "combined accumulation budget or per-sequence context differs")
    return {'updates': 193, 'processed_tokens': 577417, 'supervised_tokens': 39502,
            'native_clock_tokens': total, 'native_supervised_tokens': sum(native_denominators),
            'binding_supervised_tokens': sum(binding_denominators), 'maximum_combined_update_tokens': max(combined),
            'original_native_update_groups': original_indices, 'applied_native_learning_rates': learning_rates}


def training_snapshot(gate: dict, receipts: tuple[dict, dict]) -> dict:
    """Verify the new accumulation contract and live source/qualification custody."""
    plan_path = Path(gate['plan_path']).resolve()
    require(sha(plan_path) == gate['plan_file_sha256'], 'training plan differs from the frozen build gate')
    plan = read(plan_path); prepared = Path(gate['prepared']).resolve()
    contract_path = prepared / 'DATA_CONTRACT.json'; contract = read(contract_path)
    plan_digest, contract_sha = native_digest(plan), sha(contract_path)
    require(contract.get('experiment') == plan.get('experiment') == EXPERIMENT,
            'integration contract/plan experiment differs')
    files = contract.get('files')
    require(isinstance(files, dict) and set(files) == DATA_FILES, 'integration data contract file inventory differs')
    hashes = {}
    for name, expected in files.items():
        path = prepared / name; hashes[str(path)] = sha(path)
        require(hashes[str(path)] == expected, 'prepared integration artifact changed: ' + name)
    rows, line_hashes = [], []
    with regular(prepared / 'microfit-train.jsonl').open('rb') as stream:
        for line in stream:
            require(0 < len(line) <= 2**24 and line.endswith(b'\n'), 'invalid prepared training source line')
            row = json.loads(line.decode('utf-8'))
            require(row.get('source_split') == 'train' and row.get('mode') == 'direct' and row.get('truncated') is False,
                    'training row is not untruncated direct TRAIN')
            require(isinstance(row.get('id'), str) and isinstance(row.get('parent_id'), str), 'training identity missing')
            tokens, labels = row.get('ids'), row.get('labels')
            require(isinstance(tokens, list) and isinstance(labels, list) and 0 < len(tokens) == len(labels) <= 2048 and
                    all(type(token) is int and 0 <= token < 151936 for token in tokens) and
                    all(type(label) is int and (label == -100 or label == tokens[i]) for i, label in enumerate(labels)),
                    'training IDs/shifted label inventory invalid')
            rows.append(row); line_hashes.append(hashlib.sha256(line).hexdigest())
    membership = json.loads(regular(prepared / 'SOURCE_MEMBERSHIP.json').read_text(encoding='utf-8-sig'))
    require(isinstance(membership, list) and len(membership) == len(rows) and
            all(member['source_row_sha256_including_newline'] == line_hashes[i] for i, member in enumerate(membership)),
            'prepared source row bytes changed')
    groups = read(prepared / 'INTEGRATION_GROUPS.json')
    trajectory = validate_integration_groups(plan, rows, membership, groups)
    require(contract.get('rows') == gate.get('training_rows') == 3465 and gate.get('training_parent_count') == 2761,
            'build gate confuses training views and parent worlds')
    build = read(prepared / 'DATA_BUILD.json')
    require(sha(prepared / 'DATA_BUILD.json') == gate['data_build_sha256'] and
            set(build['output_files_sha256']) == DATA_FILES - {'DATA_BUILD.json'}, 'integration DATA_BUILD identity differs')
    for name, expected in build['output_files_sha256'].items():
        require(sha(prepared / name) == files[name] == expected, 'DATA_BUILD/data-contract output identity differs')
    admissions = []
    for path, expected in build['input_files_sha256'].items():
        require(sha(Path(path)) == expected, 'integration source input bytes changed')
        if Path(path).name == 'DATA_ADMISSION.json':
            admissions.append(expected)
    require(gate['binding_data_admission_sha256'] in admissions and len(set(admissions)) == 1,
            'integration binding source admission differs')
    native_plan_path = NATIVE_CONTROL / 'PLAN.json'; native_steps_path = NATIVE_CONTROL / 'journal/runs/train-001/steps.jsonl'
    for path in (native_plan_path, native_steps_path):
        require(any(Path(name).resolve() == path.resolve() for name in build['input_files_sha256']),
                'DATA_BUILD lacks the original native plan or applied-LR ledger')
    old_mixture_path = MIXTURE_RECORD / 'prepared/microfit-train.jsonl'
    require(any(Path(name).resolve() == old_mixture_path.resolve() for name in build['input_files_sha256']),
            'DATA_BUILD lacks the unchanged original mixture rows')
    old_lines = regular(old_mixture_path).read_bytes().splitlines(keepends=True)
    require(len(old_lines) == 3465 and sorted(member['previous_mixture_index'] for member in membership) == list(range(3465)) and
            all(hashlib.sha256(old_lines[member['previous_mixture_index']]).hexdigest() == member['source_row_sha256_including_newline']
                for member in membership), 'integration changes, drops, or repeats previously qualified source row bytes')
    old_plan = read(native_plan_path)
    steps = [json.loads(line) for line in regular(native_steps_path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    require(groups['original_native_update_groups'] == old_plan['update_groups'] and len(steps) == 193 and
            [row['lr'] for row in steps] == plan['applied_native_learning_rates'] and
            [row['processed_tokens'] for row in steps] == plan['native_processed_cumulative'],
            'native groups/LR schedule do not reproduce the original completed control')
    parents = {row['parent_id'] for row in rows}
    for bank in gate['banks'].values():
        path = Path(bank['path']).resolve(); require(sha(path) == bank['sha256'], 'registered acquisition bank changed')
        bank_parents = {json.loads(line)['parent_id'] for line in regular(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()}
        require(len(bank_parents) == bank['parents'] and len(parents & bank_parents) == bank['training_parent_overlap'] == 0,
                'training/evaluation parent overlap differs from the build gate')
    qualification = None
    for receipt in receipts:
        bound = receipt.get('checkpoint_binding')
        require(isinstance(bound, dict) and native_digest(bound) == receipt.get('checkpoint_binding_sha256') and
                bound.get('plan_sha256') == plan_digest and bound.get('data_contract_sha256') == contract_sha and
                bound.get('training_files_sha256') == files and bound.get('experiment') == EXPERIMENT,
                'arm checkpoint training/data binding differs')
        require(bound.get('training_inventory') == {'views': 3465, 'parent_worlds': 2761, 'exposures_per_view': 1},
                'checkpoint training inventory differs')
        source = bound.get('source', {})
        require(source.get('assets') == gate['source_assets_sha256'] and source.get('sha256') == native_digest(gate['source_assets_sha256']),
                'checkpoint source assets differ')
        require(receipt.get('checkpoint_progress') == FINAL_PROGRESS and receipt.get('native_clock_tokens') == 380921,
                'candidate is not the complete193-update integration/native clock')
        current = zero_addend_snapshot(receipt, plan)
        require(qualification is None or current == qualification, 'arm/materialization zero-addend provenance differs')
        qualification = current
    return {'plan_path': str(plan_path), 'plan_file_sha256': gate['plan_file_sha256'], 'plan_sha256': plan_digest,
            'prepared': str(prepared), 'data_contract_path': str(contract_path), 'data_contract_sha256': contract_sha,
            'data_files_sha256': hashes, **trajectory, **qualification}


def arm_snapshot(arm: Path, gate: dict) -> dict:
    """Compare both arm receipts and live sidecars against the frozen source assets."""
    arm_path, materialization_path = arm / "ARM.json", arm / "MATERIALIZATION.json"
    arm_receipt, materialization = read(arm_path), read(materialization_path)
    require(arm_receipt.get("record") == "BINDING_INTEGRATION_ARM" and
            arm.resolve() == (Path(gate['plan_path']).resolve().parent / 'arms/integrated') and
            all(receipt.get('experiment') == EXPERIMENT and receipt.get('arm_role') == 'integrated'
                for receipt in (arm_receipt, materialization)) and
            all(receipt.get("alpha") == 1.0 and receipt.get("parameterization") == "lora" and
                receipt.get("posthoc_scaling_performed") is False for receipt in (arm_receipt, materialization)),
            "candidate is not the single full-strength unscaled integrated LoRA arm")
    training = training_snapshot(gate, (arm_receipt, materialization))
    expected = gate["source_assets_sha256"]
    require(set(expected) == {"model.safetensors", *SIDECARS}, "incomplete frozen source asset map")
    source = Path(gate["source"]).resolve()
    weight = sha(arm / "model.safetensors")
    require(arm_receipt.get("arm_weight_sha256") == materialization.get("out_weight_sha256") == weight,
            "arm/MATERIALIZATION/live weight identity mismatch")
    for key in ("checkpoint_manifest_sha256", "checkpoint_binding_sha256", "alpha"):
        require(key in arm_receipt and key in materialization and arm_receipt[key] == materialization[key],
                f"arm/MATERIALIZATION provenance differs: {key}")
    for receipt in (arm_receipt, materialization):
        require(Path(receipt.get("source", "")).resolve() == source, "arm source path differs from build gate")
        require(receipt.get("source_assets_sha256") == expected, "arm source assets differ from build gate")
        require(receipt.get("source_weight_sha256") == expected["model.safetensors"], "arm source weight differs")
        require(receipt.get("stored_dtype") == "float32", "expected a registered FP32 arm")
        for key in ("arm_weight_sha256", "out_weight_sha256"):
            if key in receipt:
                require(receipt[key] == weight, f"conflicting arm weight field: {key}")
    sidecars = {name: sha(arm / name) for name in SIDECARS}
    source_sidecars = {name: sha(source / name) for name in SIDECARS}
    require(sidecars == source_sidecars == {name: expected[name] for name in SIDECARS},
            "arm/source sidecar hash mismatch")
    return {"weight_sha256": weight, "arm_receipt_sha256": sha(arm_path),
            "materialization_sha256": sha(materialization_path),
            "sidecar_sha256": sidecars, "source_sidecar_sha256": source_sidecars,
            "training": training}


def inputs(kind: str, instrument: str, gate: dict, digest: str) -> tuple[Path, list[Path]]:
    if kind == "process":
        bank = gate["banks"][BANK_NAMES[instrument]]
        require(type(bank.get("training_parent_overlap")) is int and bank["training_parent_overlap"] == 0,
                "BLOCKED_TRAIN_EVAL_PARENT_OVERLAP: process bank is not disjoint")
        views = Path(bank["path"]).resolve()
        require(sha(views) == bank["sha256"], "process views differ from frozen build gate")
        return views, []
    if kind == "retention":
        return RAB / "data/monitor_probe.jsonl", [
            RAB / "evidence/retention-design-v1/PROBE_LABEL_AUDIT.json",
            RAB / f"outputs/retention-{digest[:8]}/B28/retention.json"]
    suite = PILOT / "Rineya_Long_Context_Demo_v1/data"
    return suite / "all_prompts.jsonl", [suite / "all_answer_key.jsonl"]


def verify_result(raw: dict, kind: str, label: str, arm: Path,
                  views_sha: str, expected_ids: set[str], digest: str, worker: Path,
                  weight_sha: str) -> dict:
    require(raw.get("label") == label, "raw result label differs")
    recorded_dir = raw.get({"process": "model_dir", "retention": "target",
                            "long_context": "model_directory"}[kind])
    require(isinstance(recorded_dir, str) and recorded_dir, "raw result has no arm directory")
    if kind == "retention":
        require(recorded_dir.startswith("hf:"), "retention must use the verified hf: materialization loader")
        recorded_dir = recorded_dir[3:]
        identity = raw.get("hf_candidate_identity")
        require(isinstance(identity, dict) and identity.get("weight_sha256") == weight_sha,
                "retention HF loader weight differs from live before/after arm identity")
    path = Path(recorded_dir)
    require((path if path.is_absolute() else PILOT / path).resolve() == arm, "raw result arm differs")
    field = "reference_digest" if kind == "retention" else "implementation_digest"
    require(raw.get(field) == digest, "raw scorer digest differs from before/after implementation")
    extra = {}
    if kind == "process":
        require(raw.get("views_sha256") == views_sha, "raw process views hash differs")
        require(raw.get("checkpoint") is None, "probe unexpectedly applied a checkpoint")
        predictions = raw.get("predictions")
        require(isinstance(predictions, list), "process predictions missing")
        ids = [r.get("id") for r in predictions]
        require(len(ids) == len(set(ids)) and set(ids) == expected_ids, "incomplete process prediction IDs")
        require(raw.get("views") == len(expected_ids), "process view count differs")
        require(raw.get("max_new_tokens") == 128, "process generation limit differs from registered 128")
    elif kind == "retention":
        require(raw.get("probe_sha256") == views_sha, "raw retention panel hash differs")
        require(raw.get("probe_rows") == len(expected_ids), "retention panel count differs")
        prediction_path = worker / "retention_predictions.json"
        saved = read(prediction_path)
        predictions = saved.get("predictions")
        require(isinstance(predictions, list) and len(predictions) == len(expected_ids),
                "wrapper did not preserve every retention prediction")
        ids = [r.get("id") for r in predictions]
        require(len(ids) == len(set(ids)) and set(ids) == expected_ids, "incomplete retention prediction IDs")
        extra = {"retention_predictions_path": str(prediction_path),
                 "retention_predictions_sha256": sha(prediction_path)}
    else:
        require(raw.get("checkpoint_applied") is None, "long-context probe unexpectedly applied a checkpoint")
        require(set(raw.get("predictions", {})) == expected_ids and
                set(raw.get("per_row_correct", {})) == expected_ids,
                "incomplete long-context prediction IDs")
        require(raw.get("rows_total") == len(expected_ids), "long-context row count differs")
    return extra


def run(args: argparse.Namespace) -> int:
    out = args.out.absolute()
    # No failure output is allowed to alter an earlier attempt.
    require(not out.exists() and not out.is_symlink(), "--out must be a fresh attempt directory")
    out.mkdir(parents=True, exist_ok=False)
    phase, code = "preflight", None
    try:
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.label) is not None,
                "label must be a simple bounded filename component")
        arm, gate_path = args.arm.resolve(), args.build_gate.resolve()
        gate_hash, gate = sha(gate_path), read(gate_path)
        require(str(gate.get("status", "")).startswith("PASS"), "build gate is not PASS")
        relocation = source_relocation_snapshot(gate)
        regular(PYTHON)
        script = PILOT / "experiments/rab001_exploration" / f"{args.kind}_probe.py"
        expected_script = gate["probe_script_sha256"][args.kind]
        require(sha(script) == expected_script, "probe script differs from build gate")
        require(sha(WRAPPER) == gate["wrapper_sha256"], "supervised wrapper differs from build gate")
        if "supervisor_sha256" in gate:
            require(sha(SUPERVISOR_CODE) == gate["supervisor_sha256"], "supervisor differs from build gate")
        digest = implementation_digest()
        if "implementation_digest" in gate:
            require(gate["implementation_digest"] == digest, "scorer differs from build gate")
        views, additional = inputs(args.kind, args.instrument, gate, digest)
        expected_ids = row_ids(views)
        if args.kind == "long_context":
            require(row_ids(additional[0]) == expected_ids, "long-context answer key IDs differ")
        views_hash = sha(views)
        additional_hashes = {str(p): sha(p) for p in additional}
        before = arm_snapshot(arm, gate)
        code_hashes = {str(p): sha(p) for p in (script, WRAPPER, SUPERVISOR_CODE, Path(__file__))}
        legacy = RAB / "evidence/retention-design-v1" / f"{PREFIXES[args.kind]}_{args.label}.json"
        require(not legacy.exists() and not legacy.is_symlink(), "legacy label already has a result")
        worker = out / "worker"
        arguments = ([args.label, "hf:" + str(arm), "cuda"] if args.kind == "retention" else
                     ["--model", str(arm), "--label", args.label, "--device", "cuda"])
        if args.kind == "process":
            require(gate.get("process_max_new_tokens", 128) == 128,
                    "build gate process generation limit differs from registered 128")
            arguments += ["--instrument", args.instrument, "--max-new-tokens", "128"]
        command = [str(PYTHON), str(WRAPPER), "--kind", args.kind, "--out", str(worker),
                   "--timeout", "900", "--expected-script-sha", expected_script, "--", *arguments]
        write(out / "PRECHECK.json", {"schema": 1, "status": "PRECHECK_PASSED",
              "created_utc": datetime.now(timezone.utc).isoformat(), "kind": args.kind,
              "arm_dir": str(arm), "arm_before": before, "build_gate_path": str(gate_path),
              "build_gate_sha256": gate_hash, "implementation_digest": digest,
              "views_path": str(views), "views_sha256": views_hash,
              "additional_input_sha256": additional_hashes, "code_sha256": code_hashes,
              "source_relocation": relocation,
              "command": command, "cwd": str(PILOT), "quality_claim": False})
        phase = "supervised_probe"
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        with (out / "launcher.log").open("xb") as log:
            code = subprocess.run(command, cwd=PILOT, stdout=log, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, check=False, **options).returncode
        phase = "verify_terminal"
        require(code == 0, "supervised wrapper returned nonzero; see retained worker receipts")
        supervisor_path, terminal_path = worker / "supervisor.json", worker / "probe_terminal.json"
        supervisor, terminal = read(supervisor_path), read(terminal_path)
        require(supervisor.get("status") == "EXITED" and type(supervisor.get("returncode")) is int
                and supervisor["returncode"] == 0, "supervisor did not establish successful exit")
        require(terminal.get("status") == "COMPLETED" and terminal.get("complete") is True
                and type(terminal.get("system_exit_code")) is int and terminal["system_exit_code"] == 0
                and terminal.get("failure") is None and terminal.get("script_unchanged") is True
                and terminal.get("script_hash_error") is None
                and terminal.get("script_sha256_after") == expected_script,
                "probe terminal is incomplete, failed, or changed")
        require(Path(terminal.get("result", "")).resolve() == legacy.resolve(), "terminal points to unexpected raw result")
        phase = "verify_identity_after"
        after = arm_snapshot(arm, gate)
        require(after == before, "arm identity changed during probe")
        require(sha(gate_path) == gate_hash, "build gate changed during probe")
        require(source_relocation_snapshot(gate) == relocation, 'qualified source relocation changed during probe')
        require({path: sha(Path(path)) for path in code_hashes} == code_hashes,
                "probe/wrapper/supervisor/runner code changed during execution")
        require(implementation_digest() == digest, "scorer implementation changed during execution")
        require(sha(views) == views_hash, "instrument views changed during probe")
        require({path: sha(Path(path)) for path in additional_hashes} == additional_hashes,
                "additional instrument inputs changed during probe")
        phase = "seal_result"
        copied = out / "result.json"
        result_hash = copy_new(legacy, copied)
        raw = read(copied)
        if args.kind == "process":
            require(raw.get("record") == "PROCESS_TRANSITION_PROBE" and
                    raw.get("instrument") == args.instrument, "raw process record/instrument differs")
        extra = verify_result(raw, args.kind, args.label, arm, views_hash, expected_ids, digest, worker,
                              before["weight_sha256"])
        require(sha(copied) == sha(legacy) == result_hash, "raw result changed before binding")
        receipt = {"schema": 1, "status": "COMPLETE", "complete": True, "returncode": 0,
            "kind": args.kind, "label": args.label, "created_utc": datetime.now(timezone.utc).isoformat(),
            "raw_result_path": str(copied), "raw_result_sha256": result_hash,
            "original_raw_result_path": str(legacy), "original_raw_result_sha256": result_hash,
            "arm_dir": str(arm), "arm_receipt_path": str(arm / "ARM.json"),
            "arm_receipt_sha256": before["arm_receipt_sha256"],
            "materialization_path": str(arm / "MATERIALIZATION.json"),
            "materialization_sha256": before["materialization_sha256"],
            "weight_path": str(arm / "model.safetensors"), "weight_sha256": before["weight_sha256"],
            "weight_sha256_before": before["weight_sha256"], "weight_sha256_after": after["weight_sha256"],
            "views_path": str(views), "views_sha256": views_hash, "implementation_digest": digest,
            "script_path": str(script), "script_sha256": expected_script,
            "wrapper_path": str(WRAPPER), "wrapper_sha256": code_hashes[str(WRAPPER)],
            "supervisor_path": str(supervisor_path), "supervisor_sha256": sha(supervisor_path),
            "terminal_path": str(terminal_path), "terminal_sha256": sha(terminal_path),
            "supervisor_code_path": str(SUPERVISOR_CODE), "supervisor_code_sha256": code_hashes[str(SUPERVISOR_CODE)],
            "sidecar_sha256_before": before["sidecar_sha256"], "sidecar_sha256_after": after["sidecar_sha256"],
            "source_sidecar_sha256_before": before["source_sidecar_sha256"],
            "source_sidecar_sha256_after": after["source_sidecar_sha256"],
            "build_gate_path": str(gate_path), "build_gate_sha256": gate_hash,
            "training_snapshot_before": before["training"], "training_snapshot_after": after["training"],
            "additional_input_sha256_before": additional_hashes, "additional_input_sha256_after": additional_hashes,
            "source_relocation_path": relocation['source_relocation_path'],
            "source_relocation_sha256": relocation['source_relocation_sha256'],
            "quality_claim": False, "release_permitted": False, **extra}
        write(out / "PROBE_WEIGHT_BINDING.json", receipt)
        print(json.dumps({"status": "COMPLETE", "attempt": str(out), "kind": args.kind,
                          "binding": str(out / "PROBE_WEIGHT_BINDING.json")}), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        failure = {"schema": 1, "status": "FAILED", "complete": False,
                   "returncode": code if type(code) is int and code != 0 else 1,
                   "phase": phase, "failure_type": type(exc).__name__, "message": str(exc),
                   "quality_claim": False, "release_permitted": False}
        write(out / "FAILURE.json", failure)
        print(json.dumps({"status": "FAILED", "attempt": str(out), "phase": phase,
                          "message": str(exc)}), file=sys.stderr, flush=True)
        return failure["returncode"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--kind", choices=tuple(PREFIXES), required=True)
    parser.add_argument("--instrument", choices=tuple(BANK_NAMES), default="process-transition-probe-001")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--build-gate", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
