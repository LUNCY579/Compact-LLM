"""S0 on-policy throughput pilot, stage S0-P (COMP-RS-001 Amendment 1, section 16; sections 2, 8.2 and 12).

python -m experiments.maxrl_comp002_pretraining_v1.s0_pilot --inventory A \
    --q1-receipt Q1_RECEIPT.json --q1-peak-rss-key DOTTED.KEY [--out RECORDS\\S0_PILOT_A.json]
python -m experiments.maxrl_comp002_pretraining_v1.s0_pilot --inventory B --a-receipt RECORDS\\S0_PILOT_A.json \
    --q1-receipt Q1_RECEIPT.json --q1-peak-rss-key DOTTED.KEY

The pilot measures cost and memory. It fixes no training recipe, and its weights are discarded: nothing here
saves a model, a delta or an optimizer state.

- Anchor (section 18, Amendment 2: the pilot ladder is re-run on theta0'). --runtime-freeze (optional) names the
  S0 runtime freeze to run under. Its anchor is loaded by s0_runtime.load_anchor (plain Supa when it is None;
  theta0' = Supa + delta_warm with the delta and arm digests asserted), identity['anchor'] must match it, the live
  decoding_sha256 must equal the freeze's, and its host floor is the launch floor (it replaces --q1-receipt, which
  is then refused). The receipt records the freeze and 'anchor' (the freeze's anchor block, None for Supa); a B
  pilot's A receipt must carry the same anchor. With an anchored freeze the registered default receipt is
  S0_PILOT_<inventory>_WARM.json, and a registered --out must carry the _WARM suffix (a Supa pilot may not).
  Without --runtime-freeze the pilot runs on plain Supa exactly as before (anchor None).

- Prompts. 25 atomic_R TRAIN groups: the ids of manifests/atomic_R.json sorted by
  SHA256('COMP-RS-001-S0-PILOT|' + id).hexdigest() (ties broken by id), first 25. The rows come from
  canonical/train.jsonl, streamed once and hash-checked against the dataset manifest (itself checked against
  preflight.MANIFEST). Prompts are rendered with s0_runtime.render_prompt (thinking disabled).
- Sampling (Amendment 1 clarification C10). The G = 8 completions of a group are G sequential calls
  s0_generate.sample_group(model, prompt_ids, 1, seed_k), k = 0..G-1: batch 1, the Q2 sampler (T 1.0, top-p 1.0,
  top-k 0, repetition penalty 1.0), max_new_tokens 128, EOS (151645, 151643), PAD 151643. One sequence at a time
  keeps the FP32 KV cache (224 KiB per token per sequence) to one sequence while the optimizer state is resident.
  seed_k = int(SHA256('COMP-RS-001-S0-PILOT-SAMPLE|%s|%d' % (id, k)).hexdigest()[:16], 16) mod 2^63
  (SAMPLE_SEED_RULE); every seed_k is recorded. The group-level seed
  int(SHA256('COMP-RS-001-S0-PILOT-SEED|' + id).hexdigest()[:16], 16) mod 2^63 (the S0-C form) is still recorded
  with the group; it seeds nothing. decode_steps is the sum over the G calls; the sampler record is the first
  call's, and every call's must be identical to it.
- Reward (clarification C11). corpus.reward(row, response), where response is the STRICT BODY of RESPONSE_RULE
  (s0_competence.RESPONSE_RULE, the rule S0-C and S0-E use; s0_competence.response_text computes it): the
  completion minus its terminal EOS (by position), decoded with skip_special_tokens=False, an unrenderable ID
  shown as '<|unrenderable:ID|>'. A special token inside the body therefore keeps its text and scores 0. Where
  the lenient text decode(completion, skip_special_tokens=True) differs, it is recorded with
  lenient_reward = corpus.reward(row, lenient_response), a diagnostic that never feeds the advantages.
- Advantages. maxrl_verifiable_data_v1.maxrl.advantages(rewards), the inherited estimator (method maxrl,
  epsilon 1e-6; all zero for a uniform group).
- Update (clarification C10). loss = -(1/G) * sum_i A_i * sum_t log pi(y_it), accumulated one sequence at a time:
  for each sample i, log pi_i = s0_generate.teacher_forced_logprobs over the single unpadded row prompt +
  completion_i (padding_side='none'; the LM head applied only at the completion positions and recomputed in
  backward, checkpoint_head), loss_i = -(A_i / G) * sum_t log pi_i, and loss_i.backward() runs before the next
  sample's forward. The reported loss is fsum(loss_i); one optimizer step follows the G backwards. The gradient
  equals the single-batch one up to floating-point summation order. Gradient checkpointing is on
  (non-reentrant) and active in train mode, which only the update uses; requires_grad is set on exactly the
  chosen inventory: A = model.layers.8..27 (314,618,880 parameters), B = model.layers.14..27 (220,233,216). The
  optimizer is torch.optim.AdamW(lr 1e-6, weight_decay 0, foreach False), with capturable=True on CUDA so that
  every state tensor, the step counters included, lives on the model's device (capturable is unsupported on CPU,
  where the device is the CPU anyway). One step per group; a group whose advantages are all zero is recorded and
  not stepped (no backward, no step).
- Per group the receipt records the prompt id, the group seed and the G sample seeds, rewards, advantages,
  completion IDs, generated tokens, stop reasons, the generation (total and per sample) and update walls, the
  per-sample losses, the peak CUDA allocated/reserved over the group and the worker RSS.
  hostbudget.guard runs before every group and every eval decode.
- Eval decode. After the groups, s0_generate.greedy_decode (batch 1, section 8.1) decodes the same prompts once
  each, on the post-update weights; the wall per completion prices the reads. Nothing is scored.
- State. The frozen and trainable fingerprints (s0_runtime digests) are taken before and after. The frozen one
  must be unchanged, and the trainable one must change iff an update was taken (unless an OOM interrupted).
- Decision (decide). The inventory fits iff there is no OOM and max_memory_reserved <= 0.95 x 0.78 x total VRAM,
  measured over a completed pilot with at least one update. A fits: ADOPT_A; A does not fit: RERUN_WITH_B; B fits:
  ADOPT_B; B does not fit: STOP_HOST_INFEASIBLE. STOP_HOST_INFEASIBLE also applies if the peak worker RSS + 1.0 GiB
  exceeds the free physical RAM measured at launch. A pilot that cannot measure the rule (aborted, no update, CPU)
  gets no inventory decision.
- Launch. This module imports no torch, transformers or numpy: the launch snapshot of free RAM (and the launch
  floor, hostbudget.require) is taken before the worker imports torch, so the free RAM it records is what the
  whole worker had (torch's import alone is ~0.47 GiB of RSS). The receipt records whether torch was imported.
  s0_generate and s0_competence are imported on first use, and each must be the file hashed at import.

Registered mode takes no size or path overrides: device cuda, 25 groups, G 8, 128 tokens, the pinned model,
asset and tokenizer receipts, the checkout holding this code (clean), the pinned lease and host budget, and a
launch floor of the S0-Q1 peak worker RSS + 1.0 GiB read from the Q1 receipt. Registered B also needs the
registered A receipt with outcome RERUN_WITH_B. --test-mode (recorded in the receipt, refused with --device cuda)
admits tiny models, sizes and injected lease/host modules for CPU tests. The receipt is create-only, stage 'S0_PILOT_<inventory>'.
Exit status: 0 for an ADOPT outcome, 2 for any other outcome.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

from experiments.maxrl_composable_skills_v2 import contracts, corpus, verifier
from experiments.maxrl_composable_skills_v2.contracts import strict_json
from experiments.maxrl_composable_skills_v2.corpus import reward
from experiments.maxrl_verifiable_data_v1 import maxrl, specs
from experiments.maxrl_verifiable_data_v1.maxrl import advantages

from . import mechanics, preflight
from . import s0_runtime as s0
from .mechanics import parameter_inventory
from .preflight import MANIFEST, digest, require

SCHEMA = 'COMP-RS-001-S0-PILOT-2'   # 2: clarifications C10 (sequential sampling, accumulation) and C11 (strict body)
REGISTRATION = ('COMP_RS_001_AMENDMENT_1.md section 16, on-policy throughput pilot (S0-P), with clarifications '
                'C10 and C11 (section 17)')
REPO = s0.HERE.parents[1]
SELECTION_SALT = 'COMP-RS-001-S0-PILOT|'
SEED_SALT = 'COMP-RS-001-S0-PILOT-SEED|'
SAMPLE_SEED_FORMAT = 'COMP-RS-001-S0-PILOT-SAMPLE|%s|%d'
SEED_LIMIT = 2 ** 63
SELECTION_RULE = ("ids of manifests/atomic_R.json sorted by (sha256('COMP-RS-001-S0-PILOT|' + id).hexdigest(), id); "
                  'the first n')
SEED_RULE = ("group seed (recorded, seeds nothing): int(sha256('COMP-RS-001-S0-PILOT-SEED|' + id).hexdigest()[:16], "
             '16) mod 2**63')
SAMPLE_SEED_RULE = ("sample k = 0..G-1 (C10): int(sha256(('COMP-RS-001-S0-PILOT-SAMPLE|%s|%d' % (id, k)).encode())"
                    '.hexdigest()[:16], 16) mod 2**63, the seed of the k-th sequential '
                    's0_generate.sample_group(model, prompt_ids, 1, seed_k) call')
ATOMIC_R_MANIFEST = 'manifests/atomic_R.json'
TRAIN_FILE = 'canonical/train.jsonl'
MAX_LINE = 131072           # corpus.readrows' strict_json bound
MAX_RECEIPT = 1 << 26
REGISTERED = {'device': 'cuda:0', 'groups': 25, 'group_size': 8, 'max_new_tokens': s0.MAX_NEW_TOKENS}
INVENTORIES = {'A': {'blocks': [8, 27], 'numel': 314618880}, 'B': {'blocks': [14, 27], 'numel': 220233216}}
NUM_LAYERS = 28
EOS_IDS, PAD_ID = s0.EOS_IDS, s0.PAD_ID
VRAM_MARGIN = 0.95      # fits iff max_memory_reserved <= VRAM_MARGIN x (MEMORY_FRACTION x total VRAM)
HOST_MARGIN_GIB = 1.0   # STOP_HOST_INFEASIBLE iff peak worker RSS + 1.0 GiB > free physical RAM at launch
GUARD_GIB = 1.0         # hostbudget.guard floor before every unit (the S0-Q1 guard)
ADAMW = {'lr': 1e-6, 'weight_decay': 0.0, 'foreach': False}
GRADIENT_CHECKPOINTING = {'use_reentrant': False}
ESTIMATOR = {'function': 'experiments.maxrl_verifiable_data_v1.maxrl.advantages', 'method': 'maxrl',
             'epsilon': 1e-6, 'uniform_group': 'all advantages 0.0; no backward, no step'}
LOSS_RULE = ('loss = -(1/G) * sum_i A_i * sum_t log pi(y_it), one unpadded row per micro-batch; gradients '
             'accumulated over the G rows (C10): for each sample i, log pi_i = s0_generate.teacher_forced_logprobs '
             "over the single row prompt + completion_i (padding_side='none', checkpoint_head=True), EOS included; "
             'loss_i = -(A_i / G) * sum_t log pi_i; loss_i.backward() before the next row; reported loss = '
             'fsum(loss_i); float32; one AdamW step per group with nonzero advantages, after all G backwards')
# Clarification C11: the strict body, s0_competence.RESPONSE_RULE verbatim (a test and _competence() assert it).
RESPONSE_RULE = (
    "Strict body. For stop_reason 'eos' the completion must hold exactly one EOS (151645 or 151643), as its last "
    "token, and the body is the completion without it; for 'length' the completion holds no EOS and the body is "
    'the whole completion. response = tokenizer.decode(body, skip_special_tokens=False, '
    'clean_up_tokenization_spaces=False): a special token inside the body keeps its text (e.g. <think>), so '
    "corpus.reward's strict_json fails on it. An ID the tokenizer cannot render (>= len(tokenizer)), which decode "
    "would drop silently, is rendered as '<|unrenderable:ID|>'. The body's runs of renderable IDs are decoded "
    'separately. Malformed extra content is therefore never stripped into a success. lenient_response = '
    'decode(completion, skip_special_tokens=True), the lenient rule superseded by C11, is recorded only where it differs from '
    'response; in S0-C, lenient_reward = corpus.reward(row, lenient_response) is recorded with it. Both are '
    'diagnostic only.')
LENIENT_RULE = ('lenient_reward = corpus.reward(row, lenient_response) where lenient_response is not None (the text '
                'differs from the strict body), else None; diagnostic only, never feeds the advantages')
CODE_FILES = {'s0_pilot.py': Path(__file__).resolve(), 's0_runtime.py': Path(s0.__file__).resolve(),
              's0_generate.py': (s0.HERE / 's0_generate.py').resolve(),
              'arm_build.py': (s0.HERE / 'arm_build.py').resolve(),
              's0_competence.py': (s0.HERE / 's0_competence.py').resolve(),
              'mechanics.py': Path(mechanics.__file__).resolve(), 'preflight.py': Path(preflight.__file__).resolve(),
              'maxrl_composable_skills_v2/corpus.py': Path(corpus.__file__).resolve(),
              'maxrl_composable_skills_v2/verifier.py': Path(verifier.__file__).resolve(),
              'maxrl_composable_skills_v2/contracts.py': Path(contracts.__file__).resolve(),
              'maxrl_verifiable_data_v1/maxrl.py': Path(maxrl.__file__).resolve(),
              'maxrl_verifiable_data_v1/specs.py': Path(specs.__file__).resolve()}
TEST_OPTIONS = ('groups', 'group_size', 'max_new_tokens', 'test_blocks', 'model_dir', 'asset_receipt',
                'host_floor_gib')


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def code_sha256() -> dict:
    return {name: _sha256(path.read_bytes()) for name, path in CODE_FILES.items()}


_IMPORT_CODE = code_sha256()


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _generate():
    """s0_generate, imported on first use (it imports torch); it must be the file hashed at import."""
    from . import s0_generate
    require(Path(s0_generate.__file__).resolve() == CODE_FILES['s0_generate.py'], 's0_generate is not the hashed file')
    return s0_generate


def _competence():
    """s0_competence, imported on first use (C11: its response rule is the pilot's); the hashed file, same rule."""
    from . import s0_competence
    require(Path(s0_competence.__file__).resolve() == CODE_FILES['s0_competence.py'],
            's0_competence is not the hashed file')
    require(s0_competence.RESPONSE_RULE == RESPONSE_RULE, 'the pilot response rule differs from s0_competence')
    return s0_competence


def _receipt_file(path, what: str) -> tuple:
    """(strict JSON value, sha256 of the bytes) of a small receipt file."""
    raw = Path(path).read_bytes()
    require(len(raw) <= MAX_RECEIPT, what + ' receipt too large')
    value = strict_json(raw.decode('utf-8'), MAX_RECEIPT)
    require(type(value) is dict, what + ' receipt is not an object')
    return value, _sha256(raw)


# ------------------------------------------------------------------------------------------ selection
def pilot_order_key(row_id: str) -> str:
    require(type(row_id) is str and row_id, 'row id')
    return hashlib.sha256((SELECTION_SALT + row_id).encode('utf-8')).hexdigest()


def select_pilot_ids(ids, n: int = REGISTERED['groups']) -> list:
    """The first n ids by sorted SHA256('COMP-RS-001-S0-PILOT|' + id); input order is irrelevant."""
    require(type(ids) is list and all(type(i) is str and i for i in ids), 'ids must be a list of strings')
    require(len(set(ids)) == len(ids), 'duplicate id in the selection pool')
    require(type(n) is int and 0 < n <= len(ids), 'selection size outside the pool')
    return sorted(ids, key=lambda i: (pilot_order_key(i), i))[:n]


def pilot_seed(row_id: str) -> int:
    """The group-level seed (the S0-C form under the pilot salt): recorded with the group, seeds nothing."""
    require(type(row_id) is str and row_id, 'row id')
    return int(hashlib.sha256((SEED_SALT + row_id).encode('utf-8')).hexdigest()[:16], 16) % SEED_LIMIT


def sample_seed(row_id: str, k: int) -> int:
    """The seed of sample k of a group (C10): sha256('COMP-RS-001-S0-PILOT-SAMPLE|<id>|<k>')[:16] mod 2^63."""
    require(type(row_id) is str and row_id, 'row id')
    require(type(k) is int and k >= 0, 'sample index')
    return int(hashlib.sha256((SAMPLE_SEED_FORMAT % (row_id, k)).encode('utf-8')).hexdigest()[:16], 16) % SEED_LIMIT


def sample_seeds(row_id: str, group_size: int) -> list:
    require(type(group_size) is int and group_size > 0, 'group size')
    return [sample_seed(row_id, k) for k in range(group_size)]


def _file_entry(manifest: dict, relative: str) -> dict:
    files = manifest.get('files') if type(manifest) is dict else None
    require(type(files) is dict and relative in files, 'data file not in the manifest: ' + relative)
    entry = files[relative]
    require(type(entry) is dict and type(entry.get('bytes')) is int and type(entry.get('sha256')) is str,
            'manifest entry: ' + relative)
    return entry


def _check_row(row: dict) -> None:
    require(type(row) is dict and row.get('role') == 'train' and row.get('tag') == 'core'
            and row.get('skills') == ['R'] and row.get('kind') == 'composition'
            and row.get('data_source') == 'rineya_maxrl_comp_v2', 'not a core atomic-R TRAIN row: '
            + str(row.get('id') if type(row) is dict else row))
    require(row.get('prompt_hash') == contracts.digest(row.get('prompt')), 'prompt_hash mismatch: ' + row['id'])


def load_pilot_rows(data, n: int = REGISTERED['groups']) -> tuple:
    """(rows in selection order, inputs record). Streams canonical/train.jsonl once; keeps only the chosen rows."""
    root = Path(data).resolve(strict=True)
    manifest_raw = (root / 'manifest.json').read_bytes()
    require(_sha256(manifest_raw) == MANIFEST, 'dataset manifest drift')
    manifest = json.loads(manifest_raw.decode('utf-8'))
    plan_entry = _file_entry(manifest, ATOMIC_R_MANIFEST)
    plan_raw = (root / ATOMIC_R_MANIFEST).read_bytes()
    require(len(plan_raw) == plan_entry['bytes'] and _sha256(plan_raw) == plan_entry['sha256'],
            'atomic_R manifest drift')
    plan = strict_json(plan_raw.decode('utf-8'), len(plan_raw))
    require(type(plan) is dict and plan.get('name') == 'atomic_R' and type(plan.get('ids')) is list
            and plan.get('rows') == len(plan['ids']), 'atomic_R manifest schema')
    ids = select_pilot_ids(plan['ids'], n)
    wanted = set(ids)
    entry = _file_entry(manifest, TRAIN_FILE)
    hasher, size, found = hashlib.sha256(), 0, {}
    with open(root / TRAIN_FILE, 'rb') as stream:
        for line in stream:
            hasher.update(line)
            size += len(line)
            require(line.endswith(b'\n'), 'unterminated line in ' + TRAIN_FILE)
            row_id = json.loads(line).get('id')
            if row_id in wanted:
                require(row_id not in found, 'duplicate TRAIN id: ' + row_id)
                found[row_id] = strict_json(line[:-1].decode('utf-8'), MAX_LINE)
    require(size == entry['bytes'] and hasher.hexdigest() == entry['sha256'], TRAIN_FILE + ' drift')
    require(set(found) == wanted, 'selected atomic_R ids missing from TRAIN')
    rows = [found[i] for i in ids]
    for row in rows:
        _check_row(row)
    return rows, {'dataset_manifest_sha256': MANIFEST, 'atomic_R_manifest_sha256': plan_entry['sha256'],
                  'atomic_R_ids': len(plan['ids']), 'train_sha256': entry['sha256'], 'train_bytes': entry['bytes'],
                  'selection_rule': SELECTION_RULE, 'selection_salt': SELECTION_SALT, 'seed_rule': SEED_RULE,
                  'sample_seed_rule': SAMPLE_SEED_RULE, 'selected_ids': ids, 'selection_digest': digest(ids)}


# ------------------------------------------------------------------------------------------ responses
def response_text(tokenizer, completion_ids, stop_reason) -> dict:
    """The scorer's response string (C11): s0_competence.response_text, the strict body of RESPONSE_RULE, with the
    body diagnostics (body_special_ids, body_unknown_ids) and lenient_response (None unless it differs)."""
    return _competence().response_text(tokenizer, completion_ids, stop_reason)


def _completion_record(row, completion, text) -> dict:
    """The strict reward feeds the advantages; lenient_reward is the diagnostic (LENIENT_RULE)."""
    lenient = text['lenient_response']
    return {'completion_ids': completion['completion_ids'], 'stop_reason': completion['stop_reason'],
            'generated_tokens': completion['generated_tokens'],
            'sampled_logprob_sum': math.fsum(completion['raw_step_logprobs']),
            'response': text['response'], 'reward': reward(row, text['response']),
            'body_special_ids': text['body_special_ids'], 'body_unknown_ids': text['body_unknown_ids'],
            'lenient_response': lenient, 'lenient_reward': None if lenient is None else reward(row, lenient)}


def group_outcome(rewards) -> str:
    require(type(rewards) is list and rewards and all(type(r) is int and r in (0, 1) for r in rewards),
            'binary rewards')
    return 'all_fail' if not any(rewards) else 'all_success' if all(rewards) else 'mixed'


# ------------------------------------------------------------------------------------------ inventory
def block_prefixes(first: int, last: int) -> tuple:
    return tuple('model.layers.%d.' % i for i in range(first, last + 1))


def block_numel(model, first: int, last: int) -> int:
    """Unique parameter elements in blocks first..last (works on a meta-device model)."""
    prefixes, seen, total = block_prefixes(first, last), set(), 0
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if name.startswith(prefixes) and id(parameter) not in seen:
            seen.add(id(parameter))
            total += parameter.numel()
    return total


def set_inventory(model, first: int, last: int) -> dict:
    """requires_grad = True on exactly the parameters of model.layers.first..last, False everywhere else."""
    layers = model.config.num_hidden_layers
    require(type(first) is int and type(last) is int and 0 <= first <= last < layers
            and len(model.model.layers) == layers, 'block range outside the model')
    prefixes = block_prefixes(first, last)
    choice = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        wanted = name.startswith(prefixes)
        require(choice.setdefault(id(parameter), wanted) == wanted, 'a shared Parameter straddles the inventory')
    for name, parameter in model.named_parameters(remove_duplicate=False):
        parameter.requires_grad_(choice[id(parameter)])
    inventory = parameter_inventory(model)
    trainable = sorted(name for name, p in model.named_parameters(remove_duplicate=False) if p.requires_grad)
    require(trainable and all(name.startswith(prefixes) for name in trainable), 'trainable scope drift')
    require(inventory['trainable_numel'] == block_numel(model, first, last), 'inventory count drift')
    return {'blocks': [first, last], 'num_hidden_layers': layers, 'trainable_names': trainable,
            'trainable_numel': inventory['trainable_numel'], 'unique_numel': inventory['unique_numel'],
            'inventory_sha256': inventory['inventory_sha256'],
            'frozen_parameters': sum(1 for e in inventory['parameters'] if not e['requires_grad'])}


def enable_checkpointing(model) -> dict:
    """Non-reentrant gradient checkpointing on every decoder layer (transformers applies it in train mode only)."""
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(GRADIENT_CHECKPOINTING))
    layers = list(model.model.layers)
    functions = [getattr(layer, '_gradient_checkpointing_func', None) for layer in layers]
    require(model.is_gradient_checkpointing and layers
            and all(getattr(layer, 'gradient_checkpointing', False) is True for layer in layers)
            and all(getattr(f, 'keywords', None) == GRADIENT_CHECKPOINTING for f in functions),
            'gradient checkpointing is not enabled, non-reentrant, on every decoder layer')
    return dict(GRADIENT_CHECKPOINTING, enabled=True, layers=len(layers),
                active='train mode only (the update); generation and the eval decode run in eval mode')


def _model_device(model):
    devices = {p.device for p in model.parameters()}
    require(len(devices) == 1, 'model parameters span devices')
    return devices.pop()


def _sync(device) -> None:
    if device.type == 'cuda':
        import torch
        torch.cuda.synchronize(device)


def optimizer_kwargs(device_type: str) -> dict:
    """AdamW settings; capturable on CUDA puts the step counters on the GPU (unsupported on CPU)."""
    require(device_type in ('cpu', 'cuda'), 'device type')
    return dict(ADAMW, capturable=device_type == 'cuda')


def build_optimizer(model):
    import torch
    device = _model_device(model)
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    require(params, 'no trainable parameter')
    return torch.optim.AdamW(params, **optimizer_kwargs(device.type))


def optimizer_record(optimizer) -> dict:
    """The optimizer as constructed: class and its actual defaults (betas as a list)."""
    defaults = {k: (list(v) if type(v) is tuple else v) for k, v in sorted(optimizer.defaults.items())}
    params = [p for group in optimizer.param_groups for p in group['params']]
    return {'cls': type(optimizer).__module__ + '.' + type(optimizer).__name__, 'defaults': defaults,
            'param_groups': len(optimizer.param_groups), 'params': len(params),
            'numel': sum(p.numel() for p in params)}


def optimizer_state(optimizer, device) -> dict:
    import torch
    devices, tensors, numel = set(), 0, 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                devices.add(str(value.device))
                tensors += 1
                numel += value.numel()
    return {'tensors': tensors, 'numel': numel, 'devices': sorted(devices),
            'on_model_device': devices <= {str(device)}}


# ------------------------------------------------------------------------------------------ objective
def _check_advantages(completion_ids_list, advantage_list) -> int:
    group = len(completion_ids_list)
    require(group >= 2 and type(advantage_list) in (list, tuple) and len(advantage_list) == group
            and all(type(a) is float and math.isfinite(a) for a in advantage_list), 'group advantages')
    return group


def group_loss(model, prompt_ids, completion_ids_list, advantage_list, pad_id: int = PAD_ID) -> tuple:
    """(loss, per-row logprobs): loss = -(1/G) * sum_i A_i * sum_t log pi(y_it) over ONE left-padded batch of the
    G rows, keeping the autograd graph. The reference objective (tests and back-compat); update_step does not use
    it: since C10 the update accumulates per-row gradients (accumulate_gradients)."""
    import torch
    group = _check_advantages(completion_ids_list, advantage_list)
    logprobs = _generate().teacher_forced_logprobs(model, [list(prompt_ids)] * group,
                                                   [list(c) for c in completion_ids_list], pad_id,
                                                   padding_side='left', checkpoint_head=True)
    weights = torch.tensor(list(advantage_list), dtype=torch.float32, device=logprobs[0].device)
    sums = torch.stack([values.sum() for values in logprobs])
    return -(weights * sums).sum() / group, logprobs


def accumulate_gradients(model, prompt_ids, completion_ids_list, advantage_list, pad_id: int = PAD_ID) -> tuple:
    """(loss_i values, per-row logprob lists), accumulating the gradient of the registered loss into .grad (C10).

    One unpadded row per micro-batch: for each sample i, log pi_i = teacher_forced_logprobs(model, [prompt],
    [completion_i], padding_side='none', checkpoint_head=True), loss_i = -(A_i / G) * sum_t log pi_i, and
    loss_i.backward() runs at once, so one row's graph is alive at a time. sum_i grad(loss_i) is the gradient of
    group_loss up to floating-point summation order. Grad mode and train mode are the caller's (update_step)."""
    group = _check_advantages(completion_ids_list, advantage_list)
    generate = _generate()
    prompt = list(prompt_ids)
    losses, values = [], []
    for completion, advantage in zip(completion_ids_list, advantage_list):
        row = generate.teacher_forced_logprobs(model, [prompt], [list(completion)], pad_id, padding_side='none',
                                               checkpoint_head=True)[0]
        require(row.requires_grad, 'the micro-batch carries no autograd graph (grad mode is off)')
        loss = -(advantage / group) * row.sum()
        loss.backward()
        losses.append(float(loss.detach()))
        values.append(row.detach().tolist())
        del row, loss
    return losses, values


def update_step(model, optimizer, prompt_ids, completion_ids_list, advantage_list, pad_id: int = PAD_ID) -> dict:
    """One MaxRL step on one group in train mode (gradient checkpointing active); back in eval mode after.

    The gradient is accumulated one unpadded row at a time (accumulate_gradients, C10), then one optimizer step.
    The reported loss is fsum(loss_i)."""
    import torch
    _check_advantages(completion_ids_list, advantage_list)
    require(any(a != 0.0 for a in advantage_list), 'all advantages are zero: the step is skipped, not taken')
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    frozen = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    require(all(p.grad is None for _, p in trainable + frozen), 'stale gradient before the update')
    model.train()
    try:
        with torch.enable_grad():
            losses, values = accumulate_gradients(model, prompt_ids, completion_ids_list, advantage_list, pad_id)
        require(all(p.grad is None for _, p in frozen), 'a frozen parameter received a gradient')
        missing = [n for n, p in trainable if p.grad is None]
        require(not missing, 'trainable parameter without a gradient: ' + ', '.join(missing[:3]))
        norms = torch.stack([torch.linalg.vector_norm(p.grad) for _, p in trainable])
        grad_norm = float(torch.linalg.vector_norm(norms))
        value = math.fsum(losses)
        require(math.isfinite(grad_norm) and all(math.isfinite(v) for v in losses), 'nonfinite loss or gradient')
        optimizer.step()
        result = {'loss': value, 'sample_losses': losses, 'grad_norm': grad_norm, 'micro_batches': len(losses),
                  'logprob_sums': [math.fsum(v) for v in values], 'logprobs': values}
    finally:
        optimizer.zero_grad(set_to_none=True)
        model.eval()
    return result


# ------------------------------------------------------------------------------------------ state and memory
def split_fingerprints(model, trainable_names) -> dict:
    """Full, frozen (every other parameter and buffer name) and trainable fingerprints (s0_runtime digests)."""
    digests = s0.state_digests(model)
    names = set(trainable_names)
    require(names and names <= set(digests) and len(names) < len(digests), 'trainable names outside the model state')
    return {'state': s0.combine(digests),
            'frozen': s0.combine({k: v for k, v in digests.items() if k not in names}),
            'trainable': s0.combine({k: v for k, v in digests.items() if k in names}),
            'names': len(digests), 'frozen_names': len(digests) - len(names), 'trainable_names': len(names)}


class Marks:
    """resources() at the end of each interval. On CUDA the peak counters are reset after each reading, so the
    readings cover the whole run without gaps (their maximum is the whole-run peak) and each one is that
    interval's peak."""

    def __init__(self, device: str, host):
        self.device, self.host, self.marks = device, host, []

    def mark(self, label: str) -> dict:
        cuda = self.device != 'cpu'
        if cuda:
            import torch
            torch.cuda.synchronize(0)
        record = dict(s0.resources(self.device, module=self.host), label=label)
        if cuda:
            torch.cuda.reset_peak_memory_stats(0)
        self.marks.append(record)
        return record

    def summary(self) -> dict:
        cuda = [m['cuda'] for m in self.marks if m['cuda'] is not None]
        peaks = [m['process_peak_rss_gib'] for m in self.marks if m['process_peak_rss_gib'] is not None]
        rss = [m['process_rss_gib'] for m in self.marks]
        cap = cuda[-1]['fraction_cap_gib'] if cuda else None
        return {'max_memory_reserved_gib': max((c['max_memory_reserved_gib'] for c in cuda), default=None),
                'max_memory_allocated_gib': max((c['max_memory_allocated_gib'] for c in cuda), default=None),
                'total_vram_gib': cuda[-1]['total_gib'] if cuda else None, 'cap_gib': cap,
                'threshold_gib': None if cap is None else VRAM_MARGIN * cap,
                'peak_worker_rss_gib': max(peaks) if peaks else (max(rss) if rss else None),
                'peak_worker_rss_source': 'peak_wset' if peaks else 'max observed rss',
                'max_worker_rss_gib': max(rss) if rss else None, 'readings': len(self.marks)}


def decide(inventory: str, *, device: str, completed: bool, oom: bool, updates: int, max_reserved_gib,
           cap_gib, peak_rss_gib, free_at_launch_gib) -> dict:
    """The section 16 inventory rule and host check, as a pure function of the measurements."""
    require(inventory in INVENTORIES, 'inventory')
    threshold = None if cap_gib is None else VRAM_MARGIN * cap_gib
    vram_ok = None if max_reserved_gib is None or threshold is None else max_reserved_gib <= threshold
    host_ok = (None if peak_rss_gib is None or free_at_launch_gib is None
               else peak_rss_gib + HOST_MARGIN_GIB <= free_at_launch_gib)
    if oom or vram_ok is False:
        fits = False
    elif completed and updates > 0 and vram_ok is True:
        fits = True
    else:
        fits = None
    fallback = 'RERUN_WITH_B' if inventory == 'A' else 'STOP_HOST_INFEASIBLE'
    if host_ok is False:
        outcome = 'STOP_HOST_INFEASIBLE'
    elif fits is False:
        outcome = fallback
    elif fits is True and host_ok is True:
        outcome = 'ADOPT_' + inventory
    elif not completed:
        outcome = 'INCOMPLETE'
    elif updates == 0:
        outcome = 'UNMEASURED_NO_UPDATE'
    elif device == 'cpu':
        outcome = 'NOT_MEASURED_ON_CPU'
    else:
        outcome = 'UNMEASURED_HOST'
    return {'outcome': outcome, 'inventory': inventory, 'inventory_fits': fits, 'vram_ok': vram_ok,
            'host_ok': host_ok, 'oom': oom, 'completed': completed, 'updates': updates,
            'max_memory_reserved_gib': max_reserved_gib, 'cap_gib': cap_gib, 'threshold_gib': threshold,
            'peak_worker_rss_gib': peak_rss_gib, 'free_phys_gib_at_launch': free_at_launch_gib,
            'rule': ('fits iff no OOM and max_memory_reserved <= 0.95 x cap (cap = 0.78 x total VRAM), over a '
                     'completed pilot with >= 1 update; A fits -> ADOPT_A, else RERUN_WITH_B; B fits -> ADOPT_B, '
                     'else STOP_HOST_INFEASIBLE; STOP_HOST_INFEASIBLE if peak worker RSS + 1.0 GiB > free RAM '
                     'at launch')}


def _mean(values):
    return statistics.fmean(values) if values else None


def _eval_summary(records) -> dict:
    walls = [r['wall_s'] for r in records]
    tokens = sum(r['generated_tokens'] for r in records)
    return {'completions': len(records), 'wall_s_total': math.fsum(walls),
            'wall_s_per_completion_mean': _mean(walls),
            'wall_s_per_completion_median': statistics.median(walls) if walls else None,
            'wall_s_per_completion_max': max(walls) if walls else None, 'generated_tokens': tokens,
            'wall_s_per_generated_token': math.fsum(walls) / tokens if tokens else None,
            'stop_reasons': {k: sum(r['stop_reason'] == k for r in records) for k in ('eos', 'length')}}


def _group_summary(groups) -> dict:
    generation = [g['generation_wall_s'] for g in groups if 'generation_wall_s' in g]
    update = [g['update_wall_s'] for g in groups if g.get('update', {}).get('executed') is True]
    whole = [g['group_wall_s'] for g in groups if 'group_wall_s' in g]
    tokens = sum(sum(g.get('generated_tokens', [])) for g in groups)
    return {'groups': len(groups), 'generation_wall_s_total': math.fsum(generation),
            'generation_wall_s_mean': _mean(generation), 'updates_timed': len(update),
            'update_wall_s_total': math.fsum(update), 'update_wall_s_mean': _mean(update),
            'group_wall_s_total': math.fsum(whole), 'group_wall_s_mean': _mean(whole), 'sampled_tokens': tokens,
            'generation_wall_s_per_sampled_token': math.fsum(generation) / tokens if tokens else None}


# ------------------------------------------------------------------------------------------ CLI
def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog='s0_pilot', description='COMP-RS-001 S0-P on-policy throughput pilot.')
    ap.add_argument('--inventory', choices=sorted(INVENTORIES), required=True)
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--repo', type=Path, default=REPO)
    ap.add_argument('--data', type=Path, default=s0.DATA)
    ap.add_argument('--q1-receipt', type=Path)
    ap.add_argument('--q1-peak-rss-key')
    ap.add_argument('--a-receipt', type=Path, help='the S0_PILOT_A receipt (RERUN_WITH_B); required for registered B')
    ap.add_argument('--runtime-freeze', type=Path,
                    help="an S0 runtime freeze: its anchor (theta0' or Supa), decoding_sha256 and host floor")
    ap.add_argument('--lease-timeout-s', type=float, default=900.0)
    ap.add_argument('--test-mode', action='store_true')
    ap.add_argument('--groups', type=int)
    ap.add_argument('--group-size', type=int)
    ap.add_argument('--max-new-tokens', type=int)
    ap.add_argument('--test-blocks', type=int, nargs=2, metavar=('FIRST', 'LAST'))
    ap.add_argument('--model-dir', type=Path)
    ap.add_argument('--asset-receipt', type=Path)
    ap.add_argument('--host-floor-gib', type=float)
    return ap


def _lookup(value, dotted: str):
    for part in dotted.split('.'):
        require(isinstance(value, dict) and part in value, 'Q1 receipt has no ' + dotted)
        value = value[part]
    return value


def q1_floor(path, key: str) -> dict:
    """The S0 launch floor: the S0-Q1 receipt's measured peak worker RSS + 1.0 GiB."""
    receipt, receipt_sha = _receipt_file(path, 'Q1')
    require(str(receipt.get('stage', '')).startswith('S0_Q1'), 'not an S0-Q1 receipt')
    require(type(key) is str and key, 'Q1 peak-RSS key')
    peak = _lookup(receipt, key)
    require(type(peak) in (int, float) and math.isfinite(peak) and peak > 0, 'Q1 peak worker RSS')
    return {'path': str(Path(path).resolve()), 'sha256': receipt_sha, 'stage': receipt['stage'], 'key': key,
            'peak_worker_rss_gib': float(peak), 'floor_gib': float(peak) + HOST_MARGIN_GIB,
            'source': 'S0-Q1 peak worker RSS + 1.0 GiB'}


def a_pilot(path, registered: bool, anchor=None) -> dict:
    """The A pilot receipt a B pilot runs after: stage S0_PILOT_A, this schema, outcome RERUN_WITH_B, run on the
    same anchor (its 'anchor' field; absent or None = plain Supa)."""
    receipt, receipt_sha = _receipt_file(path, 'S0_PILOT_A')
    require(receipt.get('stage') == 'S0_PILOT_A' and receipt.get('schema') == SCHEMA, 'not an S0_PILOT_A receipt')
    decision = receipt.get('decision')
    require(type(decision) is dict and decision.get('outcome') == 'RERUN_WITH_B',
            'the A pilot did not decide RERUN_WITH_B')
    if registered:
        require(receipt.get('registered_run') is True and receipt.get('test_mode') is False,
                'the A receipt is not a registered run')
    require(receipt.get('anchor') == anchor, 'the A pilot ran on another anchor than this B pilot')
    return {'path': str(Path(path).resolve()), 'sha256': receipt_sha, 'outcome': decision['outcome'],
            'registered_run': receipt.get('registered_run'), 'anchor': receipt.get('anchor')}


def _same_path(a, b) -> bool:
    return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(Path(b).resolve()))


def _config(args, lease_module, host_module) -> dict:
    """Validate the command line and return the effective settings (refuses before any heavy work)."""
    inventory = INVENTORIES[args.inventory]
    if args.runtime_freeze is not None and (args.q1_receipt is not None or args.q1_peak_rss_key is not None):
        raise SystemExit('--runtime-freeze carries the launch floor (the S0-Q1 peak worker RSS + 1.0 GiB): do not '
                         'also give --q1-receipt/--q1-peak-rss-key')
    if not args.test_mode:
        given = [name for name in TEST_OPTIONS if getattr(args, name) is not None]
        if given:
            raise SystemExit('test-mode only option(s): ' + ', '.join('--' + g.replace('_', '-') for g in given))
        if args.device != 'cuda':
            raise SystemExit('the registered pilot runs on cuda')
        if lease_module is not None or host_module is not None:
            raise SystemExit('injected lease/host modules are test-mode only')
        if args.runtime_freeze is None and (args.q1_receipt is None or args.q1_peak_rss_key is None):
            raise SystemExit('the registered launch floor needs --q1-receipt and --q1-peak-rss-key '
                             '(or --runtime-freeze)')
        if not _same_path(args.repo, REPO):
            raise SystemExit('the registered pilot binds the checkout that holds this code: ' + str(REPO))
        if args.inventory == 'B' and args.a_receipt is None:
            raise SystemExit('registered B runs only after the registered A pilot decided RERUN_WITH_B '
                             '(--a-receipt)')
        if args.out is not None and not _same_path(Path(args.out).resolve().parent, s0.RECORDS):
            raise SystemExit('registered receipts are written to ' + str(s0.RECORDS))
        freeze = None if args.runtime_freeze is None else _competence().read_freeze(args.runtime_freeze,
                                                                                    registered=True)
        anchored = freeze is not None and freeze['anchor'] is not None
        out = (args.out if args.out is not None
               else s0.RECORDS / s0.anchored_name('S0_PILOT_%s.json' % args.inventory, anchored))
        if anchored != Path(out).stem.endswith(s0.WARM_SUFFIX):
            raise SystemExit("a theta0' (anchored) pilot writes to a *%s receipt name and a Supa pilot does not: %s"
                             % (s0.WARM_SUFFIX, Path(out).name))
        sizes = dict(groups=REGISTERED['groups'], group_size=REGISTERED['group_size'],
                     max_new_tokens=REGISTERED['max_new_tokens'], blocks=list(inventory['blocks']))
        model_dir, asset_receipt = s0.MODEL_DIR, s0.ASSET_RECEIPT
    else:
        if args.device != 'cpu':
            raise SystemExit('--test-mode is refused with --device cuda: a GPU pilot is always a registered run')
        if args.out is None:
            raise SystemExit('--test-mode needs --out')
        if (args.host_floor_gib is None and args.runtime_freeze is None
                and (args.q1_receipt is None or args.q1_peak_rss_key is None)):
            raise SystemExit('--test-mode needs --host-floor-gib, --q1-receipt with --q1-peak-rss-key, or '
                             '--runtime-freeze')
        out = args.out
        freeze = None if args.runtime_freeze is None else _competence().read_freeze(args.runtime_freeze,
                                                                                    registered=False)

        def pick(value, default):
            return default if value is None else value
        sizes = dict(groups=pick(args.groups, REGISTERED['groups']),
                     group_size=pick(args.group_size, REGISTERED['group_size']),
                     max_new_tokens=pick(args.max_new_tokens, REGISTERED['max_new_tokens']),
                     blocks=list(args.test_blocks) if args.test_blocks else list(inventory['blocks']))
        model_dir = pick(args.model_dir, s0.MODEL_DIR)
        asset_receipt = pick(args.asset_receipt, s0.ASSET_RECEIPT)
    if sizes['groups'] < 1 or sizes['group_size'] < 2 or sizes['max_new_tokens'] < 1:
        raise SystemExit('groups >= 1, group size >= 2 and max_new_tokens >= 1 are required')
    if not (math.isfinite(args.lease_timeout_s) and args.lease_timeout_s > 0):
        raise SystemExit('lease timeout must be > 0')
    target = s0.assert_outside_checkout(out, args.repo)
    if target.exists():
        raise SystemExit('receipt exists (create-only): ' + str(target))
    if args.a_receipt is not None and args.inventory != 'B':
        raise SystemExit('--a-receipt belongs to a B pilot')
    anchor = None if freeze is None else freeze['anchor']
    after = a_pilot(args.a_receipt, not args.test_mode, anchor) if args.a_receipt is not None else None
    return dict(sizes, out=target, model_dir=Path(model_dir), asset_receipt=Path(asset_receipt),
                device=s0.DEVICES[args.device], label='comp-rs-001-s0-pilot-' + args.inventory.lower(),
                stage='S0_PILOT_' + args.inventory, after=after, freeze=freeze, anchor=anchor)


def launch_snapshot(host) -> dict:
    """Free physical RAM (hostbudget.snapshot) and this worker's RSS, and whether torch is imported yet."""
    import psutil
    snapshot = host.snapshot()
    require(type(snapshot) is dict, 'host snapshot')
    return {'free_phys_gib': snapshot.get('free_phys_gib'), 'host': snapshot,
            'process_rss_gib': psutil.Process().memory_info().rss / s0.GIB,
            'torch_imported': 'torch' in sys.modules, 'utc': _now()}


def _prompts(tokenizer, rows, group_size: int) -> list:
    prompts = []
    for row in rows:
        rendered, ids = s0.render_prompt(tokenizer, row['prompt'])
        prompts.append({'id': row['id'], 'row': row, 'prompt_ids': ids, 'seed': pilot_seed(row['id']),
                        'sample_seeds': sample_seeds(row['id'], group_size),
                        'render_sha256': _sha256(rendered.encode('utf-8'))})
    return prompts


def sample_sequential(g, model, prompt_ids, seeds, max_new_tokens: int, device=None) -> dict:
    """The G completions of one group as G sequential single-sample calls (C10), sample k seeded with seeds[k].

    decode_steps is the sum over the calls; the sampler record is the first call's, and every call's record must
    equal it (seed_path aside, the same constant text). A device, if given, is synchronised around every call."""
    require(type(seeds) is list and seeds, 'sample seeds')
    completions, steps, walls, sampler = [], [], [], None
    for seed in seeds:
        if device is not None:
            _sync(device)
        start = time.perf_counter()
        sampled = g.sample_group(model, prompt_ids, 1, seed, max_new_tokens=max_new_tokens)
        if device is not None:
            _sync(device)
        walls.append(time.perf_counter() - start)
        require(sampled['seed'] == seed and sampled['group_size'] == 1 and len(sampled['completions']) == 1,
                'single-sample call shape')
        record = {k: v for k, v in sampled['sampler'].items() if k != 'seed_path'}
        if sampler is None:
            sampler = record
        require(record == sampler, 'the sampler record differs between the sequential sample calls')
        steps.append(sampled['decode_steps'])
        completions.append(sampled['completions'][0])
    return {'seeds': list(seeds), 'completions': completions, 'decode_steps': sum(steps),
            'decode_steps_per_sample': steps, 'wall_s_per_sample': walls, 'sampler': sampler}


def _run_group(g, model, optimizer, tokenizer, item, cfg, device, record) -> bool:
    """Sample, score and (unless uniform) update one group, filling `record`; True iff a step was taken."""
    record['phase'] = 'generate'
    seeds = item['sample_seeds']
    require(seeds == sample_seeds(item['id'], cfg['group_size']), 'sample seeds differ from SAMPLE_SEED_RULE')
    _sync(device)
    start = time.perf_counter()
    sampled = sample_sequential(g, model, item['prompt_ids'], seeds, cfg['max_new_tokens'], device)
    record['generation_wall_s'] = time.perf_counter() - start
    completions = sampled['completions']
    require(sampled['seeds'] == seeds and len(completions) == cfg['group_size'], 'sampled group shape')
    record['generation_wall_s_per_sample'] = sampled['wall_s_per_sample']
    record['decode_steps'] = sampled['decode_steps']
    record['decode_steps_per_sample'] = sampled['decode_steps_per_sample']
    record['sampler'] = sampled['sampler']
    record['phase'] = 'score'
    texts = [response_text(tokenizer, c['completion_ids'], c['stop_reason']) for c in completions]
    record['completions'] = [_completion_record(item['row'], c, t) for c, t in zip(completions, texts)]
    rewards = [c['reward'] for c in record['completions']]
    record.update(rewards=rewards, advantages=advantages(rewards), outcome=group_outcome(rewards),
                  generated_tokens=[c['generated_tokens'] for c in completions],
                  stop_reasons=[c['stop_reason'] for c in completions])
    if not any(a != 0.0 for a in record['advantages']):
        record['update'] = {'executed': False, 'reason': 'all advantages zero (uniform group)', 'wall_s': 0.0}
        record['update_wall_s'] = 0.0
        record['phase'] = 'done'
        return False
    record['phase'] = 'update'
    _sync(device)
    start = time.perf_counter()
    step = update_step(model, optimizer, item['prompt_ids'], [c['completion_ids'] for c in completions],
                       record['advantages'])
    _sync(device)
    wall = time.perf_counter() - start
    gaps = [abs(a - b) for c, lp in zip(completions, step['logprobs'])
            for a, b in zip(c['raw_step_logprobs'], lp)]
    record['update'] = {'executed': True, 'wall_s': wall, 'loss': step['loss'], 'sample_losses': step['sample_losses'],
                        'micro_batches': step['micro_batches'], 'grad_norm': step['grad_norm'],
                        'teacher_forced_logprob_sums': step['logprob_sums'],
                        'sampler_vs_teacher_forced_max_abs': max(gaps)}
    record['update_wall_s'] = wall
    record['phase'] = 'done'
    return True


def _release(model, optimizer, device) -> None:
    """After an OOM (outside the except block, so the traceback no longer pins tensors)."""
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    if model is not None:
        model.eval()
    gc.collect()
    if device.type == 'cuda':
        import torch
        torch.cuda.empty_cache()


def _pilot(args, cfg, host, state) -> dict:
    """Everything that runs under the lease; fills `state` as it goes (for the failure receipt)."""
    import torch
    g = _generate()
    state['settings'] = s0.configure(cfg['device'])
    preimage = s0.decoding_preimage(cfg['device'])
    state['decoding_preimage'], state['decoding_sha256'] = preimage, s0.decoding_sha256(preimage)
    if cfg['freeze'] is not None:
        require(cfg['freeze']['decoding_sha256'] == state['decoding_sha256'],
                'decoding_sha256 differs from the runtime freeze')
    marks = Marks(cfg['device'], host)
    state['marks'] = marks
    state['model_load_started_utc'] = _now()
    competence = _competence()
    model, identity = s0.load_anchor(cfg['model_dir'], cfg['device'], cfg['asset_receipt'],
                                     **competence.anchor_arguments(cfg['anchor']))
    device = _model_device(model)
    require(competence.anchor_matches(identity['anchor'], cfg['anchor']),
            'the loaded anchor differs from the runtime freeze')
    require(identity['attn_implementation'] == preimage['attn_implementation'], 'attention implementation drift')
    state['identity'] = {k: v for k, v in identity.items() if k not in ('stored_tensor_sha256', 'parameter_inventory')}
    first, last = cfg['blocks']
    inventory = set_inventory(model, first, last)
    if not args.test_mode:
        registered = INVENTORIES[args.inventory]
        require(model.config.num_hidden_layers == NUM_LAYERS and [first, last] == registered['blocks']
                and inventory['trainable_numel'] == registered['numel'], 'inventory differs from section 16')
    state['inventory'] = dict({k: v for k, v in inventory.items() if k != 'trainable_names'},
                              name=args.inventory, trainable_names_sha256=digest(inventory['trainable_names']),
                              registered_numel=INVENTORIES[args.inventory]['numel'])
    state['gradient_checkpointing'] = enable_checkpointing(model)
    model.eval()
    optimizer = build_optimizer(model)
    state['optimizer'] = optimizer_record(optimizer)
    before = split_fingerprints(model, inventory['trainable_names'])
    require(before['state'] == identity['state_fingerprint'], 'state changed between load and pilot')
    state['fingerprints'] = {'before': before}
    marks.mark('load')
    tokenizer, prompts, groups = state['tokenizer'], state['prompts'], state['groups']
    low_memory = host.LowMemoryAbort
    updates = 0
    for index, item in enumerate(prompts):
        where = 'group %d' % index
        try:
            s0.host_guard(GUARD_GIB, cfg['label'] + ' ' + where, module=host)
        except low_memory as error:
            state['abort'] = {'type': type(error).__name__, 'message': str(error)[:2000], 'before': where}
            break
        record = {'index': index, 'id': item['id'], 'seed': item['seed'], 'sample_seeds': item['sample_seeds'],
                  'prompt_tokens': len(item['prompt_ids']),
                  'prompt_render_sha256': item['render_sha256'], 'started_utc': _now()}
        groups.append(record)
        start, oom, stepped = time.perf_counter(), None, False
        try:
            stepped = _run_group(g, model, optimizer, tokenizer, item, cfg, device, record)
        except torch.OutOfMemoryError as error:
            oom = {'group': index, 'phase': record['phase'], 'type': type(error).__name__,
                   'message': str(error)[:2000]}
        record['group_wall_s'] = time.perf_counter() - start
        if oom is not None:
            record['oom'] = True
            state['oom'] = oom
            _release(model, optimizer, device)
        updates += bool(stepped)
        state['updates'] = updates
        record['resources'] = marks.mark(where)
        if oom is not None:
            break
        if stepped and state.get('optimizer_state_after_first_step') is None:
            state['optimizer_state_after_first_step'] = optimizer_state(optimizer, device)
            require(state['optimizer_state_after_first_step']['on_model_device'],
                    'optimizer state off the model device')
    evaluation = state['eval_decode']
    if state['oom'] is None and state['abort'] is None:
        for item in prompts:
            where = 'eval ' + item['id']
            try:
                s0.host_guard(GUARD_GIB, cfg['label'] + ' ' + where, module=host)
            except low_memory as error:
                state['abort'] = {'type': type(error).__name__, 'message': str(error)[:2000], 'before': where}
                break
            oom = None
            try:
                _sync(device)
                start = time.perf_counter()
                result = g.greedy_decode(model, item['prompt_ids'], max_new_tokens=cfg['max_new_tokens'])
                _sync(device)
                wall = time.perf_counter() - start
            except torch.OutOfMemoryError as error:
                oom = {'group': None, 'phase': 'eval', 'id': item['id'], 'type': type(error).__name__,
                       'message': str(error)[:2000]}
            if oom is not None:
                state['oom'] = oom
                _release(model, optimizer, device)
                break
            if state.get('eval_decoder') is None:
                state['eval_decoder'] = result['decoder']
            evaluation.append({'id': item['id'], 'prompt_tokens': len(item['prompt_ids']), 'wall_s': wall,
                               'generated_tokens': result['generated_tokens'], 'stop_reason': result['stop_reason'],
                               'completion_sha256': digest(result['completion_ids'])})
        marks.mark('eval decode')
    state['optimizer_state_final'] = optimizer_state(optimizer, device)
    after = split_fingerprints(model, inventory['trainable_names'])
    frozen_unchanged = after['frozen'] == before['frozen']
    trainable_changed = after['trainable'] != before['trainable']
    state['fingerprints'].update(after=after, frozen_unchanged=frozen_unchanged, trainable_changed=trainable_changed)
    require(frozen_unchanged, 'a frozen parameter or buffer changed during the pilot')
    if state['oom'] is None:
        require(trainable_changed == (updates > 0), 'trainable state change does not match the update count')
    marks.mark('final')
    state['completed'] = (state['oom'] is None and state['abort'] is None and len(groups) == cfg['groups']
                          and len(evaluation) == cfg['groups'])
    del model, optimizer
    _release(None, None, device)
    return state


def _receipt(args, cfg, state) -> dict:
    marks = state.get('marks')
    memory = marks.summary() if marks is not None else None
    groups = state['groups']
    counts = {k: sum(1 for g in groups if g.get('outcome') == k) for k in ('mixed', 'all_fail', 'all_success')}
    completions = [c for g in groups for c in g.get('completions', [])]
    decision = decide(args.inventory, device='cpu' if cfg['device'] == 'cpu' else 'cuda',
                      completed=bool(state['completed']), oom=state['oom'] is not None,
                      updates=state['updates'], max_reserved_gib=memory and memory['max_memory_reserved_gib'],
                      cap_gib=memory and memory['cap_gib'], peak_rss_gib=memory and memory['peak_worker_rss_gib'],
                      free_at_launch_gib=state['launch']['free_phys_gib'])
    return {
        'experiment': 'COMP-RS-001', 'stage': cfg['stage'], 'schema': SCHEMA, 'registration': REGISTRATION,
        'test_mode': bool(args.test_mode), 'registered_run': not args.test_mode, 'argv': state['argv'],
        'started_utc': state['started_utc'], 'finished_utc': _now(),
        'wall_s': time.perf_counter() - state['t0'], 'completed': bool(state['completed']),
        'error': state.get('error'), 'oom': state['oom'], 'abort': state['abort'], 'decision': decision,
        'after_a_pilot': cfg['after'], 'anchor': cfg['anchor'], 'runtime_freeze': cfg['freeze'],
        'runtime': state['runtime'], 'settings': state.get('settings'), 'checkout': state['checkout'],
        'code_sha256': _IMPORT_CODE, 'lease': state['lease'], 'launch': state['launch'],
        'decoding_preimage': state.get('decoding_preimage'), 'decoding_sha256': state.get('decoding_sha256'),
        'generation_contract': _generate().generation_contract(cfg['max_new_tokens']),
        'eval_decoder': state.get('eval_decoder'),
        'inputs': dict(state['inputs'], asset_receipt=str(cfg['asset_receipt']),
                       asset_receipt_sha256=preflight.sha(cfg['asset_receipt']), model_dir=str(cfg['model_dir']),
                       tokenizer_receipt_sha256=s0.TOKENIZER_RECEIPT_SHA),
        'identity': state.get('identity'), 'inventory': state.get('inventory'),
        'gradient_checkpointing': state.get('gradient_checkpointing'), 'optimizer': state.get('optimizer'),
        'optimizer_state_after_first_step': state.get('optimizer_state_after_first_step'),
        'optimizer_state_final': state.get('optimizer_state_final'),
        'objective': {'loss': LOSS_RULE, 'estimator': ESTIMATOR, 'lm_head': 'completion positions only'},
        'response_rule': RESPONSE_RULE, 'lenient_rule': LENIENT_RULE, 'sample_seed_rule': SAMPLE_SEED_RULE,
        'sizes': {k: cfg[k] for k in ('groups', 'group_size', 'max_new_tokens', 'blocks')},
        'groups': groups, 'group_outcomes': counts, 'updates': state['updates'],
        'skipped_updates': sum(1 for g in groups if g.get('update', {}).get('executed') is False),
        'walls': _group_summary(groups),
        'completion_diagnostics': {
            'completions': len(completions),
            'body_special': sum(1 for c in completions if c['body_special_ids']),
            'body_unknown': sum(1 for c in completions if c['body_unknown_ids']),
            'lenient_reward_differs': sum(1 for c in completions if c['lenient_reward'] is not None
                                          and c['lenient_reward'] != c['reward']),
            'stop_reasons': {k: sum(c['stop_reason'] == k for c in completions) for k in ('eos', 'length')},
            'generated_tokens': sum(c['generated_tokens'] for c in completions)},
        'eval_decode': {'records': state['eval_decode'], 'summary': _eval_summary(state['eval_decode'])},
        'memory': memory, 'resource_marks': marks.marks if marks is not None else [],
        'fingerprints': state.get('fingerprints'),
        'weights_saved': False, 'checkpoints_written': 0, 'training_recipe_fixed': False,
        'note': 'Throughput/memory pilot: weights discarded, no checkpoint, no scored read.'}


def main(argv=None, *, lease_module=None, host_module=None) -> int:
    t0 = time.perf_counter()
    args = _parser().parse_args(argv)
    cfg = _config(args, lease_module, host_module)
    host = host_module if host_module is not None else s0.hostbudget()
    launch = launch_snapshot(host)   # before anything imports torch (this module imports none)
    if args.q1_receipt is not None:
        floor = q1_floor(args.q1_receipt, args.q1_peak_rss_key)
    elif args.host_floor_gib is not None:
        floor = {'floor_gib': float(args.host_floor_gib), 'source': '--host-floor-gib (test mode)'}
    else:
        floor = {'floor_gib': cfg['freeze']['host_floor_gib'], 'source': 'S0_RUNTIME_FREEZE host_floor_gib',
                 'freeze_sha256': cfg['freeze']['sha256']}
    launch['floor'] = floor
    launch['require'] = s0.host_require(floor['floor_gib'], cfg['label'], module=host)
    if args.test_mode:
        runtime = {'python_executable': sys.executable, 'python': platform.python_version(),
                   'packages': s0.installed_versions(sorted(s0.PINS['packages'])), 'asserted': False}
        checkout = {'checked': False, 'reason': 'test mode'}
    else:
        runtime = dict(s0.assert_runtime(), asserted=True)
        checkout = dict(s0.assert_clean_checkout(args.repo), checked=True)
        require(preflight.sha(s0.ASSET_RECEIPT) == s0.ASSET_RECEIPT_SHA, 'asset receipt drift')
    require(preflight.sha(s0.TOKENIZER_RECEIPT) == s0.TOKENIZER_RECEIPT_SHA, 'tokenizer receipt drift')
    _competence()   # C11: the response rule's module (hashed file, same rule), checked before the lease
    rows, inputs = load_pilot_rows(args.data, cfg['groups'])
    tokenizer = s0.load_tokenizer(s0.MODEL_DIR, s0.TOKENIZER_RECEIPT, asset_receipt=s0.ASSET_RECEIPT)
    prompts = _prompts(tokenizer, rows, cfg['group_size'])
    del rows
    state = {'t0': t0, 'started_utc': _now(), 'argv': [str(a) for a in (sys.argv[1:] if argv is None else argv)],
             'runtime': runtime, 'checkout': checkout, 'launch': launch, 'inputs': inputs, 'tokenizer': tokenizer,
             'prompts': prompts, 'groups': [], 'eval_decode': [], 'updates': 0, 'oom': None, 'abort': None,
             'completed': False, 'lease': None}
    use_lease = cfg['device'] != 'cpu' or lease_module is not None
    try:
        if use_lease:
            with s0.lease(cfg['label'], args.lease_timeout_s, module=lease_module) as token:
                state['lease'] = {'label': cfg['label'], 'held': bool(token),
                                  'module': 'injected' if lease_module is not None else 'pinned gpu_lease.py'}
                _pilot(args, cfg, host, state)
        else:
            state['lease'] = {'label': cfg['label'], 'held': False, 'module': None, 'reason': 'cpu, no GPU used'}
            _pilot(args, cfg, host, state)
    except BaseException as error:
        if 'settings' in state:
            state['error'] = {'type': type(error).__name__, 'message': str(error)[:2000]}
            state['completed'] = False
            _write(args, cfg, state)
        raise
    written = _write(args, cfg, state)
    print(json.dumps({'stage': cfg['stage'], 'test_mode': bool(args.test_mode),
                      'outcome': written['decision']['outcome'], 'groups': len(state['groups']),
                      'updates': state['updates'], 'receipt': str(cfg['out']), 'sha256': written['sha256']}),
          flush=True)
    return 0 if written['decision']['outcome'].startswith('ADOPT_') else 2


def _write(args, cfg, state) -> dict:
    receipt = _receipt(args, cfg, state)
    require(code_sha256() == _IMPORT_CODE, 'pilot source changed during the run')
    sha = s0.write_receipt(cfg['out'], receipt, repo=args.repo)
    return {'decision': receipt['decision'], 'sha256': sha}


if __name__ == '__main__':
    raise SystemExit(main())
