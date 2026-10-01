"""S0 warm-start of the anchor, stage S0_WARMSTART (COMP-RS-001 Amendment 1, section 18 = Amendment 2; sections 2,
8.1, 12 and 16).

python -m experiments.maxrl_comp002_pretraining_v1.s0_warmstart [--runtime-freeze RECORDS\\S0_RUNTIME_FREEZE.json] \
    [--lease-timeout-s 900]

theta0' = Supa 1.7 + delta_warm: one epoch of SFT on the dataset's 600-row WARMSTART bank, then, in the same
process, the descriptive envelope check of section 18 step 2. Nothing here runs S0-C, freezes a runtime or opens
DEV, TEST, CHALLENGE or guard_test.

- Data (section 18 'Data'). All 600 rows of sft/warmstart.messages.jsonl; the ids come from
  sft/warmstart.lineage.jsonl in the same line order. manifest.json must hash to preflight.MANIFEST (86ac8f0d...).
  The three inputs (the two sft files and manifests/warmstart_core.json) must each be listed in manifest['files']
  and reproduce its bytes and sha. The lineage id set must equal manifests/warmstart_core.json's ids (whose count,
  name and sha also match manifest['curricula']['warmstart_core']); each row is exactly [system, user, assistant]
  messages, and sha256(assistant content, UTF-8) must equal its lineage target_sha256.
- Rendering (RENDER_RULE). prompt = s0_runtime.render_prompt(tokenizer, [system, user]) ids (enable_thinking=False,
  the pre-closed think block); target = tokenizer(assistant content, add_special_tokens=False) ids, which must
  decode back to the content and hold no special ID; sequence = prompt + target + [151645]. The chat template's own
  rendering of the three messages must equal the rendered prompt + content + '<|im_end|>\\n', so the target sits
  where the template puts the assistant turn. target + EOS must fit the section 8.1 decode cap (128 tokens).
  Whether a joint tokenization of the whole string gives the same IDs is recorded, not required.
- Labels (LABEL_RULE). mechanics.completion_labels(stop_reason 'eos') builds each row's labels: exactly the target
  and its EOS are labelled, every prompt position is masked, and no EOS may occur inside the target. The loss reads
  the same positions by the one-shift rule of mechanics.selected_logprobs: the hidden states at P-1 .. P+C-2
  predict the tokens at P .. P+C-1 (s0_generate.teacher_forced_logprobs, head at those positions only).
- Order and plan. Rows sorted by SHA256('COMP-RS-001-WARMSTART|' + id).hexdigest() (ties by id), one epoch,
  micro-batch one unpadded row, gradient accumulation over 8 rows: 75 optimizer steps for 600 rows. plan_steps
  refuses a partial last accumulation (600 = 75 x 8, so none exists; asserted).
- Loss (LOSS_RULE). loss_step = (1/8) sum_i (1/C_i) sum_t -log p(y_it): the mean token cross-entropy over each
  row's labelled tokens (target + EOS), averaged over the 8 rows of the step (rows weigh equally, not by token
  count). (loss_i / 8).backward() runs before the next row's forward (C10's one-row-at-a-time accumulation).
- Optimisation. torch.nn.utils.clip_grad_norm_(trainable, 1.0) after the 8 backwards and before each step, its
  return value recorded as the pre-clip norm (CLIP_RULE); torch.optim.AdamW(lr 1e-5, betas (0.9, 0.999), eps 1e-8,
  weight_decay 0, foreach False, capturable on CUDA so every state tensor lives on the model device); no warmup,
  no schedule. FP32, TF32 off, deterministic algorithms, 4 threads and the 0.78 memory cap (s0_runtime.configure
  before CUDA initialises), non-reentrant gradient checkpointing (s0_pilot.enable_checkpointing, active in train
  mode only).
- Trainable inventory. Every parameter under model.layers.14 .. model.layers.27 (inventory B, 220,233,216
  parameters; s0_pilot.set_inventory) and nothing else. The frozen fingerprint (every other parameter and every
  buffer) must be bitwise unchanged by training and the trainable one must change.
- Delta (section 18 'Output'). delta_warm = arm_build.extract_delta(trained FP32 inventory tensor, its stored Supa
  BF16 tensor re-read after re-hashing model.safetensors), tensor by tensor, moved to the CPU and written
  create-only with a streamed safetensors writer to <records>/S0_WARMSTART_DELTA.safetensors. The receipt records its
  data-section digest (the delta_sha256 of the anchor), its state digest (names bound), file sha and bytes, norm
  and counts; the file is re-read to reproduce the digest. Only one inventory tensor is on CPU, with1 MiB
  serialization/statistics chunks; tensors above32 MiB are refused before training. The save-phase guard
  reserves64 MiB above the1 GiB floor. No full CPU delta population or save_file byte copy is materialized.
- Anchor (ANCHOR_RULE: the shared convention s0_runtime.ANCHOR_DIGEST_RULE). theta0' is Supa + delta_warm rebuilt
  exactly as every later stage rebuilds it (s0_runtime.load_anchor): Supa as loaded by the S0 FP32 loader, then the
  delta file as read back by arm_build.read_delta added in FP32 in place (s0_runtime.add_delta). In this process
  the trained inventory is first reset to upcast(stored Supa) - the loader's own copy - so the model holds exactly
  the loader's Supa state (the frozen fingerprint proves the rest), then s0_runtime.add_delta adds the file's delta.
  arm_sha256 = s0_runtime.anchor_digest(model). The trained weights are not theta0' in general (fl32(base +
  fl32(final - base)) != final), so the digest is never taken on them; the residual (elements and tensors that
  differ, max abs, the ratio to the rounding bound 2^-23 (|theta0'| + |delta|)) is recorded, not gated.
- Round trip (ROUND_TRIP_RULE). The trained model is then released (VRAM holds one model at a time, so the rebuild
  does NOT double VRAM and the in-memory model is NOT reused, on CUDA as on the CPU), and s0_runtime.load_anchor(
  model_dir, device, asset receipt, delta_path, delta_sha256, arm_sha256) must accept the anchor: it re-verifies
  Supa (T1), refuses a delta whose data section does not reproduce delta_sha256, and raises ArmRebuildStop unless
  its rebuilt anchor_digest equals arm_sha256. Its identity['anchor'] must equal the receipt's anchor block
  (s0_competence.anchor_matches, the S0-C check).
- Envelope check (section 18 step 2, recorded and not gated). Greedy decode (s0_generate.greedy_decode, batch 1,
  128 tokens) of the 8 S0-Q1 prompts on the rebuilt theta0': the R and S prompts of the first four parents of
  evidence/Q2_SOURCE_SMOKE_SELECTION.json, loaded and rendered by s0_q1_fp32.load_probe_prompts (pinned). Per
  prompt: the strict-body response (s0_competence.response_text, C11), the strict pass (corpus.reward via
  s0_competence.score_completion, which also gives the content pass from content_scoring.score_row) and the
  envelope (ENVELOPE_RULE); the summary gives the fraction with the {"answer": ...} envelope. The rebuilt model's
  anchor_digest after the check must still equal arm_sha256.
- Durability (section12). Before launch and every journal/checkpoint/delta/receipt write, free disk must retain
  the14 GiB floor plus28 bytes per inventory element (oneFP32 delta and two12-byte parameter/Adam checkpoints)
  and64 MiB metadata reserve. A persistent create-only claim binds code, checkout, data/order, recipe,
  runtime, scorer and assets. Per-operation attempted/returned costs and per-step records are fsynced.
  Two rolling safetensors checkpoints bind parameters, Adam moments/scalarsteps, all Python/NumPy/Torch
  CPU/CUDA RNG and the completed-step cursor. --resume needs --resume-reason; identical bindings and a clean
  checkout are required. Restore reloads the last verified boundary and discards later in-memory work;
  repeated attempted compute remains charged. Final applied dose is still75steps/600rows. Corrupt latest
  checkpoints never fall back. After a verified delta and durable completion receipt, checkpoints retire.
- Receipt. S0_WARMSTART.json for completion, create-only, LF, outside the checkout. Interrupted invocations
  retain their own create-only S0_WARMSTART_INTERRUPTION_<session>.json. 'anchor' is exactly s0_runtime.ANCHOR_KEYS
  (base_weights_sha256 = the asset receipt's Supa weights sha as loaded, delta_path = the absolute delta file,
  delta_sha256 = its data-section digest, arm_sha256 = theta0' under ANCHOR_DIGEST_RULE): the block s0_freeze's
  theta0' mode reads (stage S0_WARMSTART, test_mode, registered_run). It is filled only for outcome
  WARMSTART_COMPLETE; any other outcome records None there (s0_freeze does not read the outcome, so an incomplete
  run cannot be frozen) and keeps what was verified under anchor_build / anchor_round_trip. Also: per step the
  rows, row losses, labelled and sequence tokens, the step loss, the pre-clip norm, the clip coefficient, the wall
  and the resources; the peak CUDA allocated/reserved (the first measured update memory) and peak worker RSS
  (s0_pilot.Marks); the recipe (every hyperparameter), data digests, the row order digest, the delta record, the
  envelope check, code hashes and test_mode.
- Launch (s0_competence.prepare_launch). The launch snapshot of free RAM is taken before torch is imported (this
  module imports none); hostbudget.require at the S0 host floor, read from the plain-Supa runtime freeze
  (registered default <records>/S0_RUNTIME_FREEZE.json, host_floor_gib; a freeze with an anchor is refused, the
  warm-start trains Supa); the runtime and a clean checkout are asserted and the scorer bundle must equal the
  freeze's. After configure(), the live decoding_sha256 must equal the freeze's. The shared GPU lease
  'comp-rs-001-s0-warmstart' is held for the whole run, and hostbudget.guard(1.0) runs before every step (and
  before the delta write, the anchor, the round trip and every envelope decode, each with its own transient added).

Registered mode takes no size or path overrides: cuda, all 600 rows (75 steps), inventory B, 128 envelope tokens,
the pinned model, asset and tokenizer receipts, the checkout holding this code (clean), the pinned lease and host
budget, the runtime freeze and s0_runtime.RECORDS. --test-mode (recorded in the receipt, refused with --device
cuda) admits a tiny model (--model-dir, --asset-receipt), --test-blocks, --rows N (the first N rows of the
registered order, N a multiple of 8), --max-new-tokens, --host-floor-gib and injected lease/host modules, for CPU
tests. Exit status: 0 for WARMSTART_COMPLETE, 2 for a host-guard stand-down; any error after configure() writes
the failure receipt and re-raises.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

from experiments.maxrl_composable_skills_v2.contracts import strict_json

from . import content_scoring, mechanics, preflight, scoped_scorer
from . import s0_competence as comp
from . import s0_pilot as pilot
from . import s0_runtime as s0
from . import warmstart_checkpoint as checkpoints
from .preflight import MANIFEST, digest, require

SCHEMA = 'COMP-RS-001-S0-WARMSTART-1'
STAGE = 'S0_WARMSTART'                # s0_freeze.WARMSTART_STAGE (a test compares)
LABEL = 'comp-rs-001-s0-warmstart'
REGISTRATION = ("COMP_RS_001_AMENDMENT_1.md section 18 (Amendment 2, the warm-started anchor theta0'), under the "
                'section 16 runtime')
REPO = s0.HERE.parents[1]
PACKAGE = 'experiments/maxrl_comp002_pretraining_v1/'
RECEIPT_NAME = 'S0_WARMSTART.json'
DELTA_NAME = 'S0_WARMSTART_DELTA.safetensors'
FREEZE_NAME = 'S0_RUNTIME_FREEZE.json'
ANCHOR_KEYS = s0.ANCHOR_KEYS          # the receipt's 'anchor' block, read by s0_freeze's theta0' mode
FILES = {'messages': 'sft/warmstart.messages.jsonl', 'lineage': 'sft/warmstart.lineage.jsonl',
         'core': 'manifests/warmstart_core.json'}
CORE_NAME = 'warmstart_core'
MESSAGE_ROLES = ['system', 'user', 'assistant']
LINEAGE_KEYS = {'id', 'parent_id', 'target_sha256'}
MAX_LINE = 131072
ORDER_SALT = 'COMP-RS-001-WARMSTART|'
ORDER_RULE = "rows sorted by (sha256('COMP-RS-001-WARMSTART|' + id).hexdigest(), id); one epoch"
ACCUMULATION = 8
EPOCHS = 1
REGISTERED = {'device': 'cuda:0', 'rows': 600, 'accumulation': ACCUMULATION, 'steps': 75, 'epochs': EPOCHS,
              'blocks': [14, 27], 'numel': 220233216, 'num_hidden_layers': 28, 'max_new_tokens': s0.MAX_NEW_TOKENS}
EOS_ID = 151645                       # <|im_end|>, the first generation EOS of the tokenizer receipt
PAD_ID = s0.PAD_ID                    # unused by an unpadded row; teacher_forced_logprobs takes it
MAX_TARGET_TOKENS = s0.MAX_NEW_TOKENS  # target + EOS must fit the section 8.1 decode cap
ADAMW = {'lr': 1e-5, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.0, 'amsgrad': False, 'maximize': False,
         'foreach': False}
MAX_GRAD_NORM = 1.0
GRAD_NORM_TYPE = 2.0
GUARD_GIB = 1.0                       # hostbudget.guard floor before every unit (the S0 guard)
ROUNDING_BOUND = 2.0 ** -23           # |fl(a + fl(b - a)) - b| <= 2u (|fl(a + fl(b - a))| + |fl(b - a)|), u = 2^-24
TEST_OPTIONS = ('rows', 'test_blocks', 'model_dir', 'asset_receipt', 'host_floor_gib', 'max_new_tokens')

RENDER_RULE = ("prompt = s0_runtime.render_prompt(tokenizer, [system, user]) ids (add_generation_prompt, "
               "enable_thinking=False, add_special_tokens=False; ends with the pre-closed think block); target = "
               "tokenizer(assistant content, add_special_tokens=False)['input_ids'], decoding back to the content "
               "with no special ID; sequence = prompt + target + [151645]; the chat template's rendering of the three "
               "messages equals rendered prompt + content + '<|im_end|>\\n'; len(target) + 1 <= 128")
LABEL_RULE = ("mechanics.completion_labels(ids, all-true mask, [P], [C], EOS (151645, 151643), ['eos'], 128): labels "
              'exactly positions P .. P+C-1 (target + EOS), masks the P prompt positions, and refuses an EOS inside '
              'the target; the loss uses the one-shift rule of mechanics.selected_logprobs (hidden states P-1 .. '
              "P+C-2 predict tokens P .. P+C-1) through s0_generate.teacher_forced_logprobs(padding_side='none', "
              'checkpoint_head=True)')
LOSS_RULE = ('loss_step = (1/A) * sum_{i in step} (1/C_i) * sum_{t=1..C_i} -log p(y_it | prompt_i, y_i<t), A = 8 rows '
             'per optimizer step, C_i = target tokens + EOS of row i: the mean token cross-entropy over the labelled '
             'tokens of each row, averaged over the rows of the step (rows weigh equally, not by token count). One '
             'unpadded row per micro-batch; (loss_i / A).backward() before the next row, so the accumulated gradient '
             'is that of loss_step up to float32 summation order. Reported: row_losses = [loss_i] and loss = '
             'fsum(row_losses) / A; float32')
CLIP_RULE = ('torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0, norm_type=2.0, error_if_nonfinite=True, '
             'foreach=False) after the A backwards and before optimizer.step(); pre_clip_grad_norm is its return value, '
             'the global L2 norm of the accumulated gradient before clipping; every gradient is then scaled by '
             'min(1, 1.0 / (pre_clip_grad_norm + 1e-6)) (clip_coefficient)')
ANCHOR_RULE = ("theta0' under s0_runtime.ANCHOR_DIGEST_RULE, in this process: every trained inventory tensor is reset "
               'to its stored Supa tensor upcast to FP32 (the S0 FP32 loader\'s copy), then the delta file as read '
               'back by arm_build.read_delta(delta_path, delta_sha256) is added in place by s0_runtime.add_delta; '
               "arm_sha256 = s0_runtime.anchor_digest(model). The frozen fingerprint must equal the loaded Supa's. "
               'The trained weights are never digested as theta0\' (fl32(base + fl32(final - base)) != final in '
               'general); their residual against theta0\' is recorded, not gated')
ROUND_TRIP_RULE = ("after the anchor digest, the trained model is released (one model in VRAM at a time; the "
                   'in-memory model is not reused) and s0_runtime.load_anchor(model_dir, device, asset receipt, '
                   'delta_path, delta_sha256, arm_sha256) must accept it: Supa T1-verified, the delta digest '
                   "reproduced, the rebuilt anchor_digest == arm_sha256 (else ArmRebuildStop), and identity['anchor'] "
                   "== the receipt's anchor block; the envelope check runs on that rebuilt model")
ENVELOPE_RULE = ("envelope iff content_scoring.parse_candidate(response) (composition contract) returns the category "
                 "'answer_object': after at most one outer code fence and one trailing period the response is a JSON "
                 "object with the key 'answer' (the count behind S0-C's 6/3,600). envelope_exact iff the categories "
                 "are exactly ['answer_object'] with no second candidate and no surrounding whitespace: the whole "
                 'response is one JSON object whose only key is answer. Recorded, not gated')

CODE_FILES = comp.code_files(*(PACKAGE + name for name in ('s0_warmstart.py', 's0_pilot.py', 's0_q1_fp32.py',
                                                           'arm_build.py', 'warmstart_checkpoint.py')))
_IMPORT_CODE = comp.code_sha256(CODE_FILES)
for _module in (s0, comp, pilot, mechanics, preflight, scoped_scorer, content_scoring):
    require(Path(_module.__file__).resolve() in {p.resolve() for p in CODE_FILES.values()},
            'module not loaded from its hashed file: ' + _module.__name__)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _lazy(name: str):
    """A package module imported on first use (it imports torch); it must be the file hashed at import."""
    import importlib
    module = importlib.import_module('.' + name, __package__)
    require(Path(module.__file__).resolve() == CODE_FILES[PACKAGE + name + '.py'].resolve(),
            name + ' is not the hashed file')
    return module


def _generate():
    return _lazy('s0_generate')


def _arm():
    return _lazy('arm_build')


def _q1():
    return _lazy('s0_q1_fp32')


# ------------------------------------------------------------------------------------------------ data
def _file_record(files: dict, relative: str, raw: bytes) -> dict:
    """sha/bytes of one input, which must be listed in manifest['files'] with exactly those bytes."""
    entry = files.get(relative)
    require(type(entry) is dict, relative + ' is not listed in the dataset manifest')
    record = {'sha256': _sha256(raw), 'bytes': len(raw)}
    require(entry.get('bytes') == record['bytes'] and entry.get('sha256') == record['sha256'],
            relative + ' drift (manifest)')
    return record


def _jsonl(raw: bytes, what: str) -> list:
    require(raw.endswith(b'\n') and b'\r' not in raw, what + ': LF-terminated lines required')
    return [strict_json(line.decode('utf-8'), MAX_LINE) for line in raw[:-1].split(b'\n')]


def validate_warmstart(messages_rows, lineage_rows, core) -> list:
    """The warm-start rows in file order: ids from the lineage (same line order), prompt, target, lineage fields.

    The lineage id set must equal the core manifest's ids and each target's sha256 its lineage target_sha256."""
    require(type(messages_rows) is list and type(lineage_rows) is list and messages_rows
            and len(messages_rows) == len(lineage_rows), 'messages and lineage must pair line by line')
    require(type(core) is dict and core.get('name') == CORE_NAME and type(core.get('ids')) is list
            and core.get('rows') == len(core['ids']), 'warmstart_core manifest schema')
    ids = []
    for number, entry in enumerate(lineage_rows, start=1):
        require(type(entry) is dict and set(entry) == LINEAGE_KEYS
                and all(type(entry[k]) is str and entry[k] for k in LINEAGE_KEYS),
                'lineage fields at line %d' % number)
        ids.append(entry['id'])
    require(len(set(ids)) == len(ids), 'duplicate lineage id')
    require(len(set(core['ids'])) == len(core['ids']) and set(ids) == set(core['ids']),
            'lineage ids differ from manifests/warmstart_core.json')
    rows = []
    for number, (row, entry) in enumerate(zip(messages_rows, lineage_rows), start=1):
        messages = row.get('messages') if type(row) is dict and set(row) == {'messages'} else None
        require(type(messages) is list and [m.get('role') if type(m) is dict else None for m in messages]
                == MESSAGE_ROLES, 'messages are not [system, user, assistant] at line %d' % number)
        require(all(set(m) == {'role', 'content'} and type(m['content']) is str and m['content'] for m in messages),
                'message fields at line %d' % number)
        target = messages[2]['content']
        require(_sha256(target.encode('utf-8')) == entry['target_sha256'],
                'target sha256 differs from its lineage target_sha256: ' + entry['id'])
        rows.append({'id': entry['id'], 'parent_id': entry['parent_id'], 'line': number,
                     'prompt': [dict(m) for m in messages[:2]], 'messages': messages, 'target': target,
                     'target_sha256': entry['target_sha256']})
    return rows


def load_warmstart(data) -> tuple:
    """(rows in file order, inputs record): the manifest, the three input files and validate_warmstart."""
    root = Path(data).resolve(strict=True)
    manifest_raw = (root / 'manifest.json').read_bytes()
    require(_sha256(manifest_raw) == MANIFEST, 'dataset manifest drift')
    manifest = strict_json(manifest_raw.decode('utf-8'), len(manifest_raw))
    files = manifest.get('files') if type(manifest) is dict else None
    require(type(files) is dict, 'dataset manifest has no file table')
    raws, records = {}, {}
    for key, relative in FILES.items():
        raws[key] = (root / relative).read_bytes()
        records[relative] = _file_record(files, relative, raws[key])
    core = strict_json(raws['core'].decode('utf-8'), len(raws['core']))
    rows = validate_warmstart(_jsonl(raws['messages'], FILES['messages']), _jsonl(raws['lineage'], FILES['lineage']),
                              core)
    curriculum = manifest.get('curricula', {}).get(CORE_NAME)
    require(type(curriculum) is dict and curriculum.get('path') == FILES['core']
            and curriculum.get('rows') == len(core['ids']) == len(rows)
            and curriculum.get('sha256') == records[FILES['core']]['sha256'],
            "manifest['curricula']['warmstart_core'] differs from the core manifest")
    require(manifest.get('counts', {}).get('warmstart') == len(rows), "manifest['counts']['warmstart'] differs")
    ids = [row['id'] for row in rows]
    return rows, {'dataset_manifest_sha256': MANIFEST, 'files': records,
                  'files_rule': "each input is listed in manifest['files'] and reproduces its bytes and sha256",
                  'rows': len(rows), 'lineage_ids_sha256': digest(ids), 'core_ids_sha256': digest(core['ids']),
                  'lineage_order_equals_core_order': ids == core['ids'], 'targets_checked': len(rows),
                  'target_rule': "sha256(assistant content, UTF-8) == lineage target_sha256, line by line"}


# ------------------------------------------------------------------------------------------------ rendering
def row_labels(prompt_ids, completion_ids):
    """(ids, labels) of one unpadded row: mechanics.completion_labels with stop_reason 'eos'."""
    import torch
    prompt, completion = list(prompt_ids), list(completion_ids)
    ids = torch.tensor([prompt + completion], dtype=torch.long)
    labels = mechanics.completion_labels(ids, torch.ones_like(ids, dtype=torch.bool), [len(prompt)], [len(completion)],
                                         list(s0.EOS_IDS), ['eos'], MAX_TARGET_TOKENS)
    return ids, labels


def render_row(tokenizer, row: dict, specials=None) -> dict:
    """One training row: prompt IDs, target + EOS, and the label check (RENDER_RULE, LABEL_RULE)."""
    specials = comp.special_ids(tokenizer) if specials is None else specials
    text, prompt_ids = s0.render_prompt(tokenizer, row['prompt'])
    target = row['target']
    target_ids = tokenizer(target, add_special_tokens=False)['input_ids']
    require(type(target_ids) is list and target_ids and all(type(t) is int for t in target_ids),
            'target encoding: ' + row['id'])
    require(not set(target_ids) & specials, 'special token inside the target: ' + row['id'])
    require(tokenizer.decode(target_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == target,
            'the target does not decode back to its content: ' + row['id'])
    full = tokenizer.apply_chat_template(row['messages'], tokenize=False, enable_thinking=False)
    require(full == text + target + '<|im_end|>\n',
            'the chat template places the assistant turn elsewhere: ' + row['id'])
    completion = target_ids + [EOS_ID]
    require(len(completion) <= MAX_TARGET_TOKENS, 'target + EOS exceeds the decode cap: ' + row['id'])
    ids, labels = row_labels(prompt_ids, completion)
    labelled = (labels[0] != -100).nonzero(as_tuple=True)[0].tolist()
    start = len(prompt_ids)
    require(labelled == list(range(start, start + len(completion)))
            and bool((labels[0, start:] == ids[0, start:]).all()),
            'labels are not exactly the target and its EOS: ' + row['id'])
    joint = tokenizer(text + target + '<|im_end|>', add_special_tokens=False)['input_ids']
    return {'id': row['id'], 'parent_id': row['parent_id'], 'line': row['line'], 'prompt_ids': prompt_ids,
            'completion_ids': completion, 'prompt_tokens': len(prompt_ids), 'target_tokens': len(target_ids),
            'label_positions': [start, start + len(completion)], 'render_sha256': _sha256(text.encode('utf-8')),
            'target_sha256': row['target_sha256'], 'joint_tokenization_equal': joint == prompt_ids + completion}


def render_rows(tokenizer, rows: list) -> tuple:
    require(tokenizer.convert_tokens_to_ids('<|im_end|>') == EOS_ID == s0.EOS_IDS[0], 'EOS 151645 is not <|im_end|>')
    specials = comp.special_ids(tokenizer)
    items = [render_row(tokenizer, row, specials) for row in rows]
    prompts = [item['prompt_tokens'] for item in items]
    completions = [len(item['completion_ids']) for item in items]
    return items, {'render_rule': RENDER_RULE, 'label_rule': LABEL_RULE, 'eos_id': EOS_ID,
                   'max_target_tokens': MAX_TARGET_TOKENS, 'rows_rendered': len(items),
                   'prompt_tokens': {'min': min(prompts), 'max': max(prompts), 'total': sum(prompts)},
                   'labelled_tokens': {'min': min(completions), 'max': max(completions), 'total': sum(completions)},
                   'prompt_render_sha256': digest([item['render_sha256'] for item in items]),
                   'sequence_ids_sha256': digest([item['prompt_ids'] + item['completion_ids'] for item in items]),
                   'joint_tokenization_equal_rows': sum(item['joint_tokenization_equal'] for item in items),
                   'joint_tokenization_rule': ("descriptive: rows whose whole string prompt + target + '<|im_end|>' "
                                               'tokenizes to the same IDs as the registered concatenation')}


# ------------------------------------------------------------------------------------------------ order and plan
def order_key(row_id: str) -> str:
    require(type(row_id) is str and row_id, 'row id')
    return _sha256((ORDER_SALT + row_id).encode('utf-8'))


def order_ids(ids) -> list:
    """The registered one-epoch order: sorted SHA256('COMP-RS-001-WARMSTART|' + id); input order is irrelevant."""
    require(type(ids) is list and ids and all(type(i) is str and i for i in ids), 'ids must be a list of strings')
    require(len(set(ids)) == len(ids), 'duplicate id in the warm-start order')
    return sorted(ids, key=lambda i: (order_key(i), i))


def plan_steps(rows, accumulation: int = ACCUMULATION) -> list:
    """Consecutive groups of `accumulation` rows, one optimizer step each; a partial last group is refused."""
    require(type(rows) is list and rows, 'plan rows')
    require(type(accumulation) is int and accumulation > 0, 'accumulation')
    require(len(rows) % accumulation == 0,
            'a partial last accumulation: %d rows is not a multiple of %d' % (len(rows), accumulation))
    return [rows[start:start + accumulation] for start in range(0, len(rows), accumulation)]


# ------------------------------------------------------------------------------------------------ objective
def accumulate_step(model, items, pad_id: int = PAD_ID, *, operation=None) -> dict:
    """Accumulate the gradient of LOSS_RULE's loss_step over `items` into .grad, one unpadded row at a time."""
    require(type(items) is list and items, 'step rows')
    rows = len(items)
    g = _generate()
    losses, labelled, lengths = [], [], []
    emit = operation or (lambda *args, **kwargs: None)
    for item in items:
        completion = list(item['completion_ids'])
        counts = {'input_positions': len(item['prompt_ids']) + len(completion), 'supervised_positions': len(completion)}
        emit('forward', 'attempted', **counts)
        values = g.teacher_forced_logprobs(model, [list(item['prompt_ids'])], [completion], pad_id,
                                           padding_side='none', checkpoint_head=True)[0]
        emit('forward', 'returned', **counts)
        require(values.requires_grad, 'the micro-batch carries no autograd graph (grad mode is off)')
        require(values.ndim == 1 and values.numel() == len(completion), 'labelled token count')
        row_loss = -values.mean()
        emit('backward', 'attempted')
        (row_loss / rows).backward()
        emit('backward', 'returned')
        losses.append(float(row_loss.detach()))
        labelled.append(len(completion))
        lengths.append(len(item['prompt_ids']) + len(completion))
        del values, row_loss
    require(all(math.isfinite(v) for v in losses), 'nonfinite row loss')
    return {'row_ids': [item['id'] for item in items], 'row_losses': losses, 'row_labelled_tokens': labelled,
            'row_sequence_tokens': lengths, 'rows': rows, 'loss': math.fsum(losses) / rows,
            'labelled_tokens': sum(labelled), 'sequence_tokens': sum(lengths)}


def train_step(model, optimizer, items, pad_id: int = PAD_ID, *, operation=None) -> dict:
    """One optimizer step in train mode: accumulate over the rows, clip (CLIP_RULE), step; back in eval mode after."""
    import torch
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    frozen = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    require(trainable, 'no trainable parameter')
    emit = operation or (lambda *args, **kwargs: None)
    emit('gradient_reset', 'attempted')
    optimizer.zero_grad(set_to_none=True)
    emit('gradient_reset', 'returned')
    require(all(p.grad is None for _, p in trainable + frozen), 'stale gradient before the step')
    model.train()
    try:
        with torch.enable_grad():
            result = accumulate_step(model, items, pad_id, operation=operation)
        require(all(p.grad is None for _, p in frozen), 'a frozen parameter received a gradient')
        missing = [n for n, p in trainable if p.grad is None]
        require(not missing, 'trainable parameter without a gradient: ' + ', '.join(missing[:3]))
        emit('clip', 'attempted')
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], MAX_GRAD_NORM, norm_type=GRAD_NORM_TYPE,
                                              error_if_nonfinite=True, foreach=False)
        emit('clip', 'returned')
        pre = float(norm)
        require(math.isfinite(pre), 'nonfinite gradient norm')
        emit('optimizer_step', 'attempted')
        optimizer.step()
        emit('optimizer_step', 'returned')
    finally:
        emit('gradient_reset', 'attempted')
        optimizer.zero_grad(set_to_none=True)
        emit('gradient_reset', 'returned')
        model.eval()
    coefficient = MAX_GRAD_NORM / (pre + 1e-6)
    return dict(result, pre_clip_grad_norm=pre, clip_coefficient=min(1.0, coefficient), clipped=coefficient < 1.0)


def optimizer_kwargs(device_type: str) -> dict:
    """AdamW settings; capturable on CUDA keeps the step counters on the GPU (unsupported on CPU)."""
    require(device_type in ('cpu', 'cuda'), 'device type')
    return dict(ADAMW, capturable=device_type == 'cuda')


def build_optimizer(model):
    import torch
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    require(params, 'no trainable parameter')
    return torch.optim.AdamW(params, **optimizer_kwargs(_device(model).type))


def _device(model):
    devices = {p.device for p in model.parameters()}
    require(len(devices) == 1, 'model parameters span devices')
    return devices.pop()


def _sync(device) -> None:
    if device.type == 'cuda':
        import torch
        torch.cuda.synchronize(device)


def _release(device) -> None:
    gc.collect()
    if device is not None and device.type == 'cuda':
        import torch
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------ state
def fingerprints(digests: dict, trainable_names) -> dict:
    """Full, frozen (every other parameter and buffer name) and trainable fingerprints (s0_runtime digests)."""
    trainable = set(trainable_names)
    require(trainable and trainable <= set(digests) and len(trainable) < len(digests),
            'names outside the model state')
    return {'state': s0.combine(digests),
            'frozen': s0.combine({k: v for k, v in digests.items() if k not in trainable}),
            'trainable': s0.combine({k: v for k, v in digests.items() if k in trainable}),
            'names': len(digests), 'frozen_names': len(digests) - len(trainable), 'trainable_names': len(trainable)}


def delta_write_gib(numel: int) -> float:
    """Streamed publication; one tensor and bounded analysis/copy buffers, never the full delta population."""
    require(type(numel) is int and numel > 0, 'delta numel')
    return GUARD_GIB + checkpoints.METADATA_RESERVE / s0.GIB


def delta_read_gib(numel: int) -> float:
    """The host floor before a read_delta (the anchor and the round trip): the guard plus one CPU copy."""
    require(type(numel) is int and numel > 0, 'delta numel')
    return GUARD_GIB + 4 * numel / s0.GIB


def write_warm_delta(model, weights, names, path, weights_sha256: str, *, resume=False) -> dict:
    """delta = arm_build.extract_delta(trained FP32 tensor, stored Supa tensor), per tensor, written create-only."""
    import torch
    from safetensors import safe_open
    arm_build = _arm()
    names = sorted(names)
    require(names, 'empty delta inventory')
    weights, path = Path(weights), Path(path)
    require(resume or not os.path.lexists(path), 'delta file exists (create-only): ' + str(path))
    require(preflight.sha(weights) == weights_sha256, 'the Supa weight bytes changed since the load')
    device = _device(model)
    params = dict(model.named_parameters())
    require(set(names) <= set(params) and all(params[n].requires_grad for n in names), 'delta names are not trainable')
    path.parent.mkdir(parents=True, exist_ok=True)
    numel = sum(params[n].numel() for n in names)
    checkpoints.disk_gate(path.parent, numel)
    state_hash, sumsq, nonzero = hashlib.sha256(), [0.0], [0]
    specifications = {n: ['F32', list(params[n].shape)] for n in names}
    def chunks(n):
        checkpoints.disk_gate(path.parent, numel)
        with torch.no_grad(), safe_open(str(weights), framework='pt', device=str(device)) as handle:
            stored = handle.get_tensor(n)
            require(stored.dtype == torch.bfloat16, 'a stored inventory tensor is not BF16: ' + n)
            value = arm_build.extract_delta({n: params[n].detach()}, {n: stored}, [n])[n].to('cpu')
            del stored
            state_hash.update(n.encode('utf8') + b'\0' + arm_build.tensor_digest(value).encode('ascii') + b'\n')
            flat = value.reshape(-1)
            for start in range(0, flat.numel(), checkpoints.CHUNK_BYTES // 4):
                part = flat[start:start + checkpoints.CHUNK_BYTES // 4].double()
                sumsq[0] += float((part * part).sum())
                del part
            nonzero[0] += int(torch.count_nonzero(value))
            yield from checkpoints.tensor_chunks(value)
    if path.exists():
        # Recovery verifies the existing file against the restored final tensors;
        # it does not allocate another full delta on disk or replace any bytes.
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            require(sorted(handle.keys()) == names and all(handle.get_slice(n).get_shape() == specifications[n][1]
                    and handle.get_slice(n).get_dtype() == 'F32' for n in names), 'existing delta schema drift')
        expected = hashlib.sha256()
        for n in names:
            for block in chunks(n):
                expected.update(block)
        require(arm_build.delta_data_digest(path) == expected.hexdigest(), 'existing delta differs from restored final checkpoint')
        saved = {'data_section_sha256': expected.hexdigest()}
    else:
        saved = checkpoints.stream_tensors(path, specifications, chunks)
    data_sha = saved['data_section_sha256']
    record = {'path': str(path), 'data_section_sha256': data_sha, 'state_sha256': state_hash.hexdigest(),
              'global_norm': math.sqrt(sumsq[0]), 'tensors': len(names), 'numel': numel,
              'nonzero': nonzero[0], 'dtype': 'float32',
              'names_sha256': digest(names),
              'rule': ('arm_build.extract_delta(trained FP32 inventory tensor, stored Supa BF16 tensor re-read after '
                       're-hashing model.safetensors) = trained - upcast(Supa), per tensor, moved to the CPU and '
                       'streamed create-only as safetensors; data-section and state digests verified')}
    gc.collect()
    require(arm_build.delta_data_digest(path) == data_sha, 'the written delta does not reproduce its digest')
    record.update(file_sha256=preflight.sha(path), bytes=path.stat().st_size, verified_on_disk=True)
    return record


def materialize_anchor(model, weights, delta_path, delta_sha256: str, names) -> tuple:
    """(arm_sha256, record): theta0' = Supa + delta in the model under ANCHOR_RULE / s0_runtime.ANCHOR_DIGEST_RULE.

    Every trained inventory tensor is reset to its stored Supa tensor upcast to FP32 (the loader's copy), then
    s0_runtime.add_delta adds the delta file as read back by arm_build.read_delta. The residual of theta0' against
    the trained weights is measured on the way (descriptive)."""
    import torch
    from safetensors import safe_open
    arm_build = _arm()
    device = _device(model)
    inventory = sorted(names)
    delta = arm_build.read_delta(delta_path, delta_sha256)
    try:
        require(sorted(delta) == inventory, 'the delta file keys are not the trainable inventory')
        params = dict(model.named_parameters())
        residual = {'elements': 0, 'tensors': 0, 'max_abs': 0.0, 'max_bound_ratio': 0.0}
        with torch.no_grad(), safe_open(str(weights), framework='pt', device=str(device)) as handle:
            for name in inventory:
                stored = handle.get_tensor(name)
                require(stored.dtype == torch.bfloat16, 'a stored inventory tensor is not BF16: ' + name)
                base = stored.to(torch.float32)
                step = delta[name].to(device)
                trained = params[name]
                theta = base + step
                count = int((theta != trained).sum())
                if count:
                    gap = (theta.double() - trained.double()).abs()
                    bound = (theta.double().abs() + step.double().abs()) * ROUNDING_BOUND
                    positive = bound > 0
                    ratio = float((gap[positive] / bound[positive]).max()) if bool(positive.any()) else math.inf
                    if bool((gap[~positive] > 0).any()):
                        ratio = math.inf
                    residual['elements'] += count
                    residual['tensors'] += 1
                    residual['max_abs'] = max(residual['max_abs'], float(gap.max()))
                    residual['max_bound_ratio'] = max(residual['max_bound_ratio'], ratio)
                    del gap, bound, positive
                trained.copy_(base)
                del stored, base, step, theta
        applied = s0.add_delta(model, delta)
    finally:
        delta.clear()
        del delta
    _release(device)
    arm = s0.anchor_digest(model)
    ratio = residual['max_bound_ratio']
    residual.update(within_rounding_bound=ratio <= 1.0, max_bound_ratio=ratio if math.isfinite(ratio) else 'inf',
                    bound="|theta0' - trained| <= 2^-23 * (|theta0'| + |delta|), elementwise (descriptive)")
    return arm, {'arm_sha256': arm, 'anchor_digest_rule': s0.ANCHOR_DIGEST_RULE, 'rule': ANCHOR_RULE,
                 'delta_read': {'path': str(delta_path), 'delta_sha256': delta_sha256}, 'add_delta': applied,
                 'residual': residual}


def round_trip(model_dir, device, asset_receipt, anchor: dict, *, operation=None) -> tuple:
    """(model, record): s0_runtime.load_anchor must rebuild theta0' from Supa + the delta file (ROUND_TRIP_RULE)."""
    require(type(anchor) is dict and set(anchor) == set(ANCHOR_KEYS), 'anchor block keys')
    emit = operation or (lambda *args, **kwargs: None)
    emit('model_load', 'attempted', purpose='anchor_round_trip')
    model, identity = s0.load_anchor(model_dir, device, asset_receipt, **comp.anchor_arguments(anchor))
    emit('model_load', 'returned', purpose='anchor_round_trip')
    try:
        require(comp.anchor_matches(identity['anchor'], anchor), "load_anchor's anchor differs from the receipt's")
        require(identity['state_fingerprint'] == anchor['arm_sha256'], 'load_anchor rebuilt another state')
    except BaseException:
        del model
        raise
    record = {'accepted': True, 'rule': ROUND_TRIP_RULE, 'identity_anchor': identity['anchor'],
              'state_fingerprint': identity['state_fingerprint'], 'stored_names_arm_sha256': identity['arm_sha256'],
              'anchor_delta': identity['anchor_delta'], 'base': identity['base'],
              'weights_sha256': identity['weights_sha256'], 'device': identity['device'],
              'tensors_matched_bitwise': identity['tensors_matched_bitwise'],
              'in_memory_model_reused': False,
              'vram': 'the trained model was released before the rebuild: one model in VRAM at a time'}
    return model, record


# ------------------------------------------------------------------------------------------------ envelope
def load_envelope_prompts(data, tokenizer) -> tuple:
    """The 8 S0-Q1 prompts (s0_q1_fp32.load_probe_prompts, pinned) with their full CALIBRATION rows for scoring."""
    q1 = _q1()
    supa = preflight.read_json(s0.ASSET_RECEIPT)['assets']['model.safetensors']['sha256']
    prompts, inputs = q1.load_probe_prompts(data, q1.Q2_SELECTION, tokenizer, weights_sha256=supa, pinned=True)
    bank = scoped_scorer.load_role(data, 'calibration')
    by_id = {row['id']: row for row in bank}
    for item in prompts:
        row = by_id.get(item['id'])
        require(row is not None and row['prompt_hash'] == item['prompt_hash'] and row['tag'] == 'core'
                and row['skills'] == [item['skill']] and row['parent_id'] == item['parent_id'],
                'envelope prompt differs from its CALIBRATION row: ' + item['id'])
        item['row'] = row
    bank_sha256 = bank.bank_sha256
    require(bank_sha256 == inputs['canonical_calibration_sha256'], 'CALIBRATION bank differs from the probe input')
    del by_id, bank
    return prompts, dict(inputs, calibration_bank_sha256=bank_sha256, source_weight_rule=(
        'the selection receipt is keyed to the Supa weights (evidence/Q1_ASSET_CANDIDATE.json); the prompts do not '
        'depend on the model decoded'))


def envelope_of(response: str) -> dict:
    """ENVELOPE_RULE for one strict-body response."""
    try:
        _, categories, multiple = content_scoring.parse_candidate(response)
    except content_scoring.PARSE_ERRORS:
        return {'envelope': False, 'envelope_exact': False, 'envelope_categories': None}
    return {'envelope': 'answer_object' in categories,
            'envelope_exact': categories == ['answer_object'] and not multiple and response == response.strip(),
            'envelope_categories': categories}


@contextlib.contextmanager
def observe_envelope(model, item, cap, store, attempt_id):
    """Observe the existing decoder: unchanged logits/processors/configuration.

Forward hooks count dispatch and returns; a pass-through streamer sees actual
emissions after token selection. It never changes scores, tokens or stopping.
"""
    observation = {'tokens': [], 'stream_prompt_seen': False, 'stream_ended': False, 'forward_calls': 0}
    if not hasattr(model, 'generate'):
        yield observation  # synthetic callers; returned result or cap bounds remain explicit
        return
    require('generate' not in vars(model), 'generation observer already installed')
    def operation(name, status, **fields):
        store.operation(name, status, row_id=item['id'], attempt_id=attempt_id, **fields)
    def before(module, args, kwargs):
        ids = kwargs.get('input_ids', args[0] if args else None)
        require(ids is not None, 'decoder input positions unavailable')
        observation['forward_calls'] += 1
        operation('decoder_forward', 'attempted', input_positions=int(ids.numel()))
    def after(module, args, kwargs, output):
        ids = kwargs.get('input_ids', args[0] if args else None)
        operation('decoder_forward', 'returned', input_positions=int(ids.numel()))
    class Streamer:
        def put(self, tokens):
            ids = tokens.reshape(-1).tolist()
            if not observation['stream_prompt_seen']:
                require(ids == item['prompt_ids'], 'streamed prompt differs from frozen scope')
                observation['stream_prompt_seen'] = True
                store.append('envelope_stream_prompt', attempt_id=attempt_id, row_id=item['id'],
                             prompt_ids_sha256=item['prompt_ids_sha256'])
                return
            for token in ids:
                require(len(observation['tokens']) < cap, 'streamed generation exceeds registered cap')
                operation('generation_token', 'returned', emitted_tokens=1,
                          token_id=int(token), ordinal=len(observation['tokens']))
                observation['tokens'].append(int(token))
        def end(self):
            store.append('envelope_stream_end', attempt_id=attempt_id, row_id=item['id'],
                         emitted_tokens=len(observation['tokens']))
            observation['stream_ended'] = True
    generate = model.generate
    def observed_generate(*args, **kwargs):
        require('streamer' not in kwargs, 'decoder already has a streamer')
        return generate(*args, **kwargs, streamer=Streamer())
    pre = model.register_forward_pre_hook(before, with_kwargs=True)
    post = model.register_forward_hook(after, with_kwargs=True)
    model.generate = observed_generate
    try:
        yield observation
    finally:
        pre.remove()
        post.remove()
        delattr(model, 'generate')


def envelope_binding(anchor, decoding_sha256, prompts, cap):
    return {'arm_sha256': anchor['arm_sha256'], 'decoding_sha256': decoding_sha256,
            'max_new_tokens': cap, 'rule': ENVELOPE_RULE, 'response_rule': comp.RESPONSE_RULE,
            'prompts': [{'id': item['id'], 'parent_id': item['parent_id'], 'position': item['position'],
                         'prompt_ids_sha256': item['prompt_ids_sha256'], 'prompt_hash': item.get('prompt_hash'),
                         'scoring_row_sha256': digest(item['row'])} for item in prompts]}


def envelope_check(g, model, tokenizer, prompts, max_new_tokens: int, host, label: str, records: list,
                   *, store=None, binding=None) -> None:
    """Durable row reads. Completed rows are reused without generation or scoring."""
    device = _device(model)
    bound = None
    if store is not None:
        require(binding is not None and not records, 'durable envelope starts from its bound journal')
        records.extend(store.envelope_rows(binding))
        bound = digest(binding)
    for item in prompts[len(records):]:
        s0.host_guard(GUARD_GIB, label + ' envelope ' + item['id'], module=host)
        recovered = store.envelope_result(bound, item['id']) if store is not None else None
        if recovered is not None:
            result, wall, attempt_id = recovered['result'], recovered['wall_s'], recovered['attempt_id']
            store.append('envelope_return_reused', row_id=item['id'], attempt_id=attempt_id,
                         binding_sha256=bound, reason='returned decoder result durable; no new generation')
        elif store is None:
            result, wall = comp.timed(device, lambda: g.greedy_decode(model, item['prompt_ids'], max_new_tokens=max_new_tokens))
            attempt_id = None
        else:
            attempt_id = uuid.uuid4().hex
            store.append('envelope_attempt', row_id=item['id'], attempt_id=attempt_id, binding_sha256=bound,
                         max_new_tokens=max_new_tokens,
                         input_positions_upper_bound=max_new_tokens * (len(item['prompt_ids']) + max_new_tokens - 1))
            store.operation('generation', 'attempted', attempt_id=attempt_id, row_id=item['id'])
            try:
                with observe_envelope(model, item, max_new_tokens, store, attempt_id) as observation:
                    result, wall = comp.timed(device, lambda: g.greedy_decode(model, item['prompt_ids'], max_new_tokens=max_new_tokens))
                require(result['generated_tokens'] == len(result['completion_ids']) <= max_new_tokens, 'returned generation count/cap differs')
                if observation['stream_prompt_seen']:
                    require(observation['stream_ended'] and observation['tokens'] == result['completion_ids'], 'streamed emissions differ from returned completion')
                store.append('envelope_returned', row_id=item['id'], attempt_id=attempt_id,
                             binding_sha256=bound, result=result, wall_s=wall,
                             exact_stream_observed=observation['stream_prompt_seen'])
                store.operation('generation', 'returned', attempt_id=attempt_id, row_id=item['id'], generated_tokens=result['generated_tokens'])
            except BaseException as error:
                store.append('envelope_interruption', row_id=item['id'], attempt_id=attempt_id,
                             error={'type': type(error).__name__, 'message': str(error)[:2000]},
                             cost=store.envelope_cost())
                raise
        if store is not None:
            store.operation('envelope_score', 'attempted', attempt_id=attempt_id, row_id=item['id'])
        text = comp.response_text(tokenizer, result['completion_ids'], result['stop_reason'])
        record = dict({'position': item['position'], 'id': item['id'], 'skill': item['skill'],
                             'parent_id': item['parent_id'], 'prompt_tokens': item['prompt_tokens'],
                             'prompt_ids_sha256': item['prompt_ids_sha256'],
                             'completion_ids': result['completion_ids'], 'stop_reason': result['stop_reason'],
                             'generated_tokens': result['generated_tokens'], 'response': text['response'],
                             'body_special_ids': text['body_special_ids'],
                             'body_unknown_ids': text['body_unknown_ids'],
                             'lenient_response': text['lenient_response'], 'wall_s': wall,
                             'decoder': result['decoder']},
                            **comp.score_completion(item['row'], result, text), **envelope_of(text['response']))
        if store is not None:
            store.operation('envelope_score', 'returned', attempt_id=attempt_id, row_id=item['id'])
            store.append('envelope_row', row=record, row_sha256=digest(record), binding_sha256=bound,
                         attempt_id=attempt_id)
        records.append(record)


def envelope_summary(records: list) -> dict:
    n = len(records)

    def counts(mine):
        return {'prompts': len(mine), 'envelope': sum(r['envelope'] for r in mine),
                'envelope_exact': sum(r['envelope_exact'] for r in mine), 'strict_pass': sum(r['reward'] for r in mine),
                'content_pass': sum(r['content_pass'] for r in mine)}
    out = counts(records)
    out.update(envelope_fraction=out['envelope'] / n if n else None,
               envelope_exact_fraction=out['envelope_exact'] / n if n else None,
               by_skill={skill: counts([r for r in records if r['skill'] == skill])
                         for skill in sorted({r['skill'] for r in records})},
               stop_reasons={k: sum(r['stop_reason'] == k for r in records) for k in ('eos', 'length')},
               gated=False)
    return out


# ------------------------------------------------------------------------------------------------ recipe
def recipe(cfg: dict, rows: int, steps: int) -> dict:
    kwargs = optimizer_kwargs('cuda' if cfg['device'] != 'cpu' else 'cpu')
    return {'epochs': EPOCHS, 'rows': rows, 'registered_rows': REGISTERED['rows'], 'accumulation_rows': ACCUMULATION,
            'optimizer_steps': steps, 'registered_steps': REGISTERED['steps'],
            'micro_batch': 'one unpadded row (batch 1, no padding, no attention mask)', 'order_rule': ORDER_RULE,
            'order_salt': ORDER_SALT, 'render_rule': RENDER_RULE, 'label_rule': LABEL_RULE, 'loss_rule': LOSS_RULE,
            'loss_reduction': ('mean token cross-entropy over each row\'s labelled tokens (target + EOS), then the '
                               'mean over the rows of the step'),
            'optimizer': {'cls': 'torch.optim.AdamW', 'kwargs': dict(kwargs, betas=list(kwargs['betas']))},
            'schedule': 'none: constant lr, no warmup', 'gradient_clipping': {
                'max_norm': MAX_GRAD_NORM, 'norm_type': GRAD_NORM_TYPE, 'rule': CLIP_RULE},
            'precision': ('float32 parameters, gradients, Adam moments and activations; TF32 off, deterministic '
                          'algorithms, 4 threads, memory fraction 0.78 on CUDA (s0_runtime.configure)'),
            'gradient_checkpointing': dict(pilot.GRADIENT_CHECKPOINTING, when='train mode (the update) only'),
            'inventory': {'blocks': list(cfg['blocks']), 'registered_blocks': REGISTERED['blocks'],
                          'registered_numel': REGISTERED['numel'],
                          'rule': 'every parameter under model.layers.first .. model.layers.last, nothing else'},
            'eos_id': EOS_ID, 'max_target_tokens': MAX_TARGET_TOKENS, 'anchor_rule': ANCHOR_RULE,
            'round_trip_rule': ROUND_TRIP_RULE, 'envelope_max_new_tokens': cfg['max_new_tokens']}


# ------------------------------------------------------------------------------------------------ CLI
def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog='s0_warmstart', description="COMP-RS-001 section 18 warm-start of theta0'.")
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    ap.add_argument('--records', type=Path, help='records directory (registered: s0_runtime.RECORDS)')
    ap.add_argument('--repo', type=Path, default=REPO)
    ap.add_argument('--data', type=Path, default=s0.DATA)
    ap.add_argument('--runtime-freeze', type=Path, help='S0_RUNTIME_FREEZE.json (registered default: in --records)')
    ap.add_argument('--lease-timeout-s', type=float, default=900.0)
    ap.add_argument('--test-mode', action='store_true')
    ap.add_argument('--rows', type=int, help='test mode: the first N rows of the registered order (N % 8 == 0)')
    ap.add_argument('--test-blocks', type=int, nargs=2, metavar=('FIRST', 'LAST'))
    ap.add_argument('--model-dir', type=Path)
    ap.add_argument('--asset-receipt', type=Path)
    ap.add_argument('--host-floor-gib', type=float)
    ap.add_argument('--max-new-tokens', type=int)
    ap.add_argument('--resume', action='store_true', help='resume the same claimed attempt from its latest verified boundary')
    ap.add_argument('--resume-reason', help='document the interruption/restart; required with --resume')
    return ap


def _same_path(a, b) -> bool:
    return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(Path(b).resolve()))


def read_supa_freeze(path, *, registered: bool) -> dict:
    """The plain-Supa runtime freeze (s0_competence.read_freeze): the warm-start trains Supa, so no anchor."""
    freeze = comp.read_freeze(path, registered=registered)
    if freeze['anchor'] is not None:
        raise SystemExit("the warm-start trains plain Supa: the runtime freeze must carry no anchor (a theta0' "
                         'freeze was given): ' + str(path))
    return freeze


def _config(args, lease_module, host_module) -> dict:
    """Validate the command line and return the effective settings (refuses before any heavy work)."""
    if not args.test_mode:
        given = [name for name in TEST_OPTIONS if getattr(args, name) is not None]
        if given:
            raise SystemExit('test-mode only option(s): ' + ', '.join('--' + g.replace('_', '-') for g in given))
        if args.device != 'cuda':
            raise SystemExit('the registered warm-start runs on cuda')
        if lease_module is not None or host_module is not None:
            raise SystemExit('injected lease/host modules are test-mode only')
        if not _same_path(args.repo, REPO):
            raise SystemExit('the registered warm-start binds the checkout that holds this code: ' + str(REPO))
        records = s0.RECORDS if args.records is None else args.records
        if not _same_path(records, s0.RECORDS):
            raise SystemExit('registered records are written to ' + str(s0.RECORDS))
        if args.runtime_freeze is None:
            args.runtime_freeze = Path(records) / FREEZE_NAME
        rows, blocks, tokens = REGISTERED['rows'], list(REGISTERED['blocks']), REGISTERED['max_new_tokens']
        model_dir, asset_receipt = s0.MODEL_DIR, s0.ASSET_RECEIPT
    else:
        if args.device != 'cpu':
            raise SystemExit('--test-mode is refused with --device cuda: a GPU warm-start is always a registered run')
        if args.records is None:
            raise SystemExit('--test-mode needs --records')
        if args.host_floor_gib is None and args.runtime_freeze is None:
            raise SystemExit('--test-mode needs --host-floor-gib or --runtime-freeze')
        records = args.records
        rows = REGISTERED['rows'] if args.rows is None else args.rows
        blocks = list(REGISTERED['blocks']) if args.test_blocks is None else list(args.test_blocks)
        tokens = REGISTERED['max_new_tokens'] if args.max_new_tokens is None else args.max_new_tokens
        model_dir = s0.MODEL_DIR if args.model_dir is None else args.model_dir
        asset_receipt = s0.ASSET_RECEIPT if args.asset_receipt is None else args.asset_receipt
    if not (0 < rows <= REGISTERED['rows'] and rows % ACCUMULATION == 0):
        raise SystemExit('--rows must be a positive multiple of %d up to %d' % (ACCUMULATION, REGISTERED['rows']))
    if tokens < 1:
        raise SystemExit('max_new_tokens must be >= 1')
    if not (math.isfinite(args.lease_timeout_s) and args.lease_timeout_s > 0):
        raise SystemExit('lease timeout must be > 0')
    if args.host_floor_gib is not None and not (math.isfinite(args.host_floor_gib) and args.host_floor_gib > 0):
        raise SystemExit('--host-floor-gib must be > 0')
    out = s0.assert_outside_checkout(Path(records) / RECEIPT_NAME, args.repo)
    delta = s0.assert_outside_checkout(Path(records) / DELTA_NAME, args.repo)
    if args.resume and not (args.resume_reason and args.resume_reason.strip()):
        raise SystemExit('--resume needs --resume-reason documenting the interruption')
    for target in (out,) if args.resume else (out, delta):
        if os.path.lexists(target):
            raise SystemExit('output exists (create-only): ' + str(target))
    freeze = None
    if args.runtime_freeze is not None:
        freeze = read_supa_freeze(args.runtime_freeze, registered=not args.test_mode)
    return {'records': Path(records).resolve(), 'out': out, 'delta': delta, 'model_dir': Path(model_dir),
            'asset_receipt': Path(asset_receipt), 'weights': Path(model_dir) / 'model.safetensors',
            'device': s0.DEVICES[args.device], 'label': LABEL, 'stage': STAGE, 'test_mode': bool(args.test_mode),
            'rows': rows, 'blocks': blocks, 'max_new_tokens': tokens, 'freeze': freeze, 'resume': args.resume}


def _guarded(host, gib: float, where: str, state: dict) -> bool:
    """hostbudget.guard before a unit; a stand-down is recorded as the abort and returns False."""
    try:
        s0.host_guard(gib, LABEL + ' ' + where, module=host)
    except host.LowMemoryAbort as error:
        state['abort'] = {'type': type(error).__name__, 'message': str(error)[:2000], 'before': where}
        return False
    return True


def _run(args, cfg: dict, host, state: dict) -> None:
    """Everything under the lease; fills `state` as it goes (for the failure receipt)."""
    g = _generate()
    state['settings'] = s0.configure(cfg['device'])
    preimage = s0.decoding_preimage(cfg['device'])
    decoding = s0.decoding_sha256(preimage)
    state['decoding_preimage'], state['decoding_sha256'] = preimage, decoding
    if state['freeze'] is not None:
        require(state['freeze']['decoding_sha256'] == decoding, 'decoding_sha256 differs from the runtime freeze')
    state['generation_contract'] = g.generation_contract(cfg['max_new_tokens'])
    marks = pilot.Marks(cfg['device'], host)
    state['marks'] = marks
    state['store'].operation('model_load', 'attempted', purpose='original_base')
    model, identity = s0.load_model_fp32(cfg['model_dir'], cfg['device'], cfg['asset_receipt'])
    state['store'].operation('model_load', 'returned', purpose='original_base')
    device = _device(model)
    require(identity['attn_implementation'] == preimage['attn_implementation'], 'attention implementation drift')
    state['identity'] = {k: v for k, v in identity.items() if k not in ('stored_tensor_sha256', 'parameter_inventory')}
    first, last = cfg['blocks']
    inventory = pilot.set_inventory(model, first, last)
    names = inventory['trainable_names']
    if not cfg['test_mode']:
        require(model.config.num_hidden_layers == REGISTERED['num_hidden_layers']
                and [first, last] == REGISTERED['blocks'] and inventory['trainable_numel'] == REGISTERED['numel'],
                'inventory differs from section 18 (inventory B)')
    require(set(names) <= set(identity['stored_tensor_sha256']), 'a trainable tensor is not stored in the checkpoint')
    state['inventory'] = dict({k: v for k, v in inventory.items() if k != 'trainable_names'},
                              name='B' if [first, last] == REGISTERED['blocks'] else 'TEST',
                              trainable_names_sha256=digest(names), registered_numel=REGISTERED['numel'])
    state['gradient_checkpointing'] = pilot.enable_checkpointing(model)
    model.eval()
    optimizer = build_optimizer(model)
    state['optimizer'] = pilot.optimizer_record(optimizer)
    before = fingerprints(s0.state_digests(model), names)
    require(before['state'] == identity['state_fingerprint'], 'state changed between load and training')
    state['fingerprints'] = {'before': before}
    marks.mark('load')
    store = state['store']
    store.numel = inventory['trainable_numel']
    require(max(dict(model.named_parameters())[n].numel() * 4 for n in names) <= checkpoints.METADATA_RESERVE // 2,
            'inventory tensor exceeds the streamed save-phase host reserve')
    store.gate()
    if cfg['resume']:
        restored = store.restore(model, optimizer, initial_progress={'steps': [], 'fingerprints_before': before})
        require(restored['progress']['fingerprints_before'] == before, 'resume frozen/base fingerprint drift')
        state['steps'] = restored['progress']['steps']
        state['optimizer_state_after_first_step'] = restored['progress'].get('optimizer_state_after_first_step')
        state['resume_cursor'] = restored['cursor']
        if restored.get('base_restart'):
            store.save(model, optimizer, 0, restored['progress'])
    else:
        state['resume_cursor'] = 0
        store.save(model, optimizer, 0, {'steps': [], 'fingerprints_before': before})
    steps = state['steps']
    for index in range(state['resume_cursor'], len(state['plan'])):
        items = state['plan'][index]
        where = 'step %d' % index
        if not _guarded(host, GUARD_GIB, where, state):
            break
        record = {'index': index, 'started_utc': _now()}
        store.append('step_begin', cursor=index + 1, row_ids=[item['id'] for item in items])
        steps.append(record)
        _sync(device)
        start = time.perf_counter()
        record.update(train_step(model, optimizer, items, operation=store.operation))
        _sync(device)
        record['wall_s'] = time.perf_counter() - start
        record['resources'] = marks.mark(where)
        if index == 0:
            state['optimizer_state_after_first_step'] = pilot.optimizer_state(optimizer, device)
            require(state['optimizer_state_after_first_step']['on_model_device'], 'optimizer state off the model device')
        store.save(model, optimizer, index + 1, {'steps': steps, 'fingerprints_before': before,
                                               'optimizer_state_after_first_step': state.get('optimizer_state_after_first_step')})
        store.append('step_committed', cursor=index + 1, row_ids=record['row_ids'],
                     labelled_tokens=record['labelled_tokens'], sequence_tokens=record['sequence_tokens'], cost=store.cost())
    state['optimizer_state_final'] = pilot.optimizer_state(optimizer, device)
    store.operation('gradient_reset', 'attempted', purpose='final_optimizer_cleanup')
    optimizer.zero_grad(set_to_none=True)
    store.operation('gradient_reset', 'returned', purpose='final_optimizer_cleanup')
    del optimizer
    _release(device)
    after = fingerprints(s0.state_digests(model), names)
    state['fingerprints'].update(after=after, frozen_unchanged=after['frozen'] == before['frozen'],
                                 trainable_changed=after['trainable'] != before['trainable'])
    require(state['fingerprints']['frozen_unchanged'], 'a frozen parameter or buffer changed during training')
    numel = inventory['trainable_numel']
    if state['abort'] is None:
        require(len(steps) == len(state['plan']), 'not every planned step ran')
        require(state['fingerprints']['trainable_changed'], 'training did not change the trainable inventory')
        state['trained'] = True
    if state['abort'] is None and _guarded(host, delta_write_gib(numel), 'delta write', state):
        store.append('delta_begin', cursor=len(steps))
        state['delta'] = write_warm_delta(model, cfg['weights'], names, cfg['delta'], identity['weights_sha256'], resume=cfg['resume'])
        store.append('delta_verified', record=state['delta'])
        marks.mark('delta')
    anchor = None
    if state['abort'] is None and _guarded(host, delta_read_gib(numel), 'anchor', state):
        arm, build = materialize_anchor(model, cfg['weights'], cfg['delta'], state['delta']['data_section_sha256'],
                                        names)
        built = fingerprints(s0.state_digests(model), names)
        require(built['state'] == arm, "anchor_digest is not the model's state fingerprint")
        require(built['frozen'] == before['frozen'], "theta0' changed a frozen parameter or buffer")
        build.update(frozen_equals_supa=True, trained_state_fingerprint=after['state'],
                     trained_equals_theta0=after['state'] == arm)
        state['anchor_build'] = build
        anchor = {'base_weights_sha256': identity['weights_sha256'], 'delta_path': str(cfg['delta']),
                  'delta_sha256': state['delta']['data_section_sha256'], 'arm_sha256': arm}
        marks.mark('anchor')
    del model
    _release(device)
    if anchor is not None and _guarded(host, delta_read_gib(numel), 'round trip', state):
        rebuilt, state['anchor_round_trip'] = round_trip(cfg['model_dir'], cfg['device'], cfg['asset_receipt'],
                                                         anchor, operation=store.operation)
        marks.mark('round trip')
        records = state['envelope_records']
        try:
            envelope_check(g, rebuilt, tokenizer=state['tokenizer'], prompts=state['envelope_prompts'],
                           max_new_tokens=cfg['max_new_tokens'], host=host, label=LABEL, records=records,
                           store=store, binding=envelope_binding(anchor, decoding, state['envelope_prompts'], cfg['max_new_tokens']))
        except host.LowMemoryAbort as error:
            state['abort'] = {'type': type(error).__name__, 'message': str(error)[:2000], 'before': 'envelope'}
        if records:
            state['envelope_decoder'] = comp.distinct_decoders(records, 'decoder', cfg['max_new_tokens'])
        require(s0.anchor_digest(rebuilt) == anchor['arm_sha256'], "the model is not theta0' after the envelope check")
        state['fingerprints']['final_equals_theta0'] = True
        del rebuilt
        _release(device)
    marks.mark('final')
    state['completed'] = state['abort'] is None and len(state['envelope_records']) == len(state['envelope_prompts'])
    if state['completed']:
        state['anchor'] = anchor
        store.append('complete', steps=len(steps), anchor=anchor)


def _envelope_block(cfg: dict, state: dict) -> dict:
    records = [{k: v for k, v in r.items() if k != 'decoder'} for r in state['envelope_records']]
    return {'rule': ENVELOPE_RULE, 'recorded_not_gated': True, 'max_new_tokens': cfg['max_new_tokens'],
            'model': "theta0' as rebuilt by s0_runtime.load_anchor (the round trip), in this process",
            'inputs': state['envelope_inputs'], 'records': records, 'summary': envelope_summary(records),
            'decoder': state.get('envelope_decoder'), 'response_rule': comp.RESPONSE_RULE}


def _training_summary(steps: list) -> dict:
    losses = [s['loss'] for s in steps if 'loss' in s]
    norms = [s['pre_clip_grad_norm'] for s in steps if 'pre_clip_grad_norm' in s]
    reserved = [s['resources']['cuda']['max_memory_reserved_gib'] for s in steps
                if (s.get('resources') or {}).get('cuda')]
    return {'steps': len(losses), 'first_loss': losses[0] if losses else None, 'last_loss': losses[-1] if losses else None,
            'mean_loss': math.fsum(losses) / len(losses) if losses else None,
            'max_pre_clip_grad_norm': max(norms) if norms else None,
            'clipped_steps': sum(1 for s in steps if s.get('clipped')),
            'labelled_tokens': sum(s.get('labelled_tokens', 0) for s in steps),
            'sequence_tokens': sum(s.get('sequence_tokens', 0) for s in steps),
            'wall_s': math.fsum(s['wall_s'] for s in steps if 'wall_s' in s),
            'update_max_memory_reserved_gib': max(reserved) if reserved else None,
            'update_memory_note': 'section 18: the first measured update memory (max over the step intervals)'}


def _outcome(state: dict) -> str:
    if state.get('error') is not None:
        return 'FAILED'
    if state['abort'] is not None:
        return 'ABORTED_HOST_GUARD'
    return 'WARMSTART_COMPLETE' if state['completed'] else 'INCOMPLETE'


def _receipt(args, cfg: dict, state: dict) -> dict:
    marks = state.get('marks')
    selected = [item['id'] for step in state['plan'] for item in step]
    outcome = _outcome(state)
    anchor = state.get('anchor') if outcome == 'WARMSTART_COMPLETE' else None
    return {
        'experiment': 'COMP-RS-001', 'stage': STAGE, 'schema': SCHEMA, 'registration': REGISTRATION,
        'test_mode': cfg['test_mode'], 'registered_run': not cfg['test_mode'], 'argv': state['argv'],
        'session': state['session'], 'started_utc': state['started_utc'], 'finished_utc': _now(),
        'wall_s': time.perf_counter() - state['t0'], 'outcome': outcome, 'completed': outcome == 'WARMSTART_COMPLETE',
        'error': state.get('error'), 'abort': state['abort'],
        'anchor': anchor, 'anchor_digest_rule': s0.ANCHOR_DIGEST_RULE,
        'anchor_rule': ('anchor = exactly s0_runtime.ANCHOR_KEYS, the block s0_freeze reads; filled only for '
                        'WARMSTART_COMPLETE (None otherwise, with anchor_build / anchor_round_trip kept)'),
        'anchor_build': state.get('anchor_build'), 'anchor_round_trip': state.get('anchor_round_trip'),
        'runtime': state['runtime'], 'settings': state.get('settings'), 'checkout': state['checkout'],
        'lease': state.get('lease'), 'launch': state['launch'], 'runtime_freeze': state['freeze'],
        'scorer_bundle': state['bundle'], 'decoding_preimage': state.get('decoding_preimage'),
        'decoding_sha256': state.get('decoding_sha256'), 'generation_contract': state.get('generation_contract'),
        'code_sha256': _IMPORT_CODE,
        'inputs': dict(state['inputs'], rendering=state['rendering'], model_dir=str(cfg['model_dir']),
                       asset_receipt=str(cfg['asset_receipt']),
                       asset_receipt_sha256=preflight.sha(cfg['asset_receipt']),
                       tokenizer_receipt_sha256=s0.TOKENIZER_RECEIPT_SHA),
        'recipe': recipe(cfg, len(selected), len(state['plan'])),
        'plan': {'rows': len(selected), 'steps': len(state['plan']), 'accumulation': ACCUMULATION,
                 'subset': len(selected) != REGISTERED['rows'], 'order_rule': ORDER_RULE,
                 'row_order_sha256': digest(state['order']), 'selected_ids_sha256': digest(selected),
                 'first_ids': selected[:3]},
        'identity': state.get('identity'), 'inventory': state.get('inventory'),
        'gradient_checkpointing': state.get('gradient_checkpointing'), 'optimizer': state.get('optimizer'),
        'optimizer_state_after_first_step': state.get('optimizer_state_after_first_step'),
        'optimizer_state_final': state.get('optimizer_state_final'),
        'steps': state['steps'], 'training': _training_summary(state['steps']),
        'durability': {'claim': str(cfg['records'] / checkpoints.CLAIM), 'journal': str(cfg['records'] / checkpoints.JOURNAL),
                       'checkpoint_directory': str(cfg['records'] / checkpoints.DIRECTORY),
                       'resume_cursor': state.get('resume_cursor'), 'cost': state['store'].cost(),
                       'envelope_cost': state['store'].envelope_cost(),
                       'invocations': state['store'].invocations(),
                       'disk_launch': state['disk_launch'], 'checkpoint_rule': 'two rolling verified safetensors boundaries; rollback discards later state and charges replayed compute'},
        'fingerprints': state.get('fingerprints'), 'delta': state.get('delta'),
        'envelope': _envelope_block(cfg, state),
        'memory': marks.summary() if marks is not None else None,
        'resource_marks': marks.marks if marks is not None else [],
        'weights_saved': 'delta and rolling checkpoints' if state.get('delta') else 'rolling checkpoints',
        'optimizer_state_saved': bool(state['store']._manifests()),
        'note': ("Warm-start of theta0' (section 18): one SFT epoch on WARMSTART, delta_warm written, theta0' "
                 'digested under the shared convention and rebuilt by load_anchor, its envelope recorded (not '
                 "gated). S0-C on theta0' is a separate stage.")}


def _write(args, cfg: dict, state: dict) -> dict:
    receipt = _receipt(args, cfg, state)
    require(comp.code_sha256(CODE_FILES) == _IMPORT_CODE, 'warm-start source changed during the run')
    state['store'].gate()
    target = cfg['out'] if receipt['outcome'] == 'WARMSTART_COMPLETE' else cfg['records'] / ('S0_WARMSTART_INTERRUPTION_' + state['session'] + '.json')
    return {'outcome': receipt['outcome'], 'receipt': str(target), 'sha256': s0.write_receipt(target, receipt, repo=args.repo)}


def main(argv=None, *, lease_module=None, host_module=None) -> int:
    t0 = time.perf_counter()
    args = _parser().parse_args(argv)
    cfg = _config(args, lease_module, host_module)
    disk_launch = checkpoints.disk_gate(cfg['records'], REGISTERED['numel'])
    host = host_module if host_module is not None else s0.hostbudget()
    prepared = comp.prepare_launch(args, cfg, host)   # the launch snapshot precedes the torch import
    rows, inputs = load_warmstart(args.data)
    tokenizer = s0.load_tokenizer(s0.MODEL_DIR, s0.TOKENIZER_RECEIPT, asset_receipt=s0.ASSET_RECEIPT)
    items, rendering = render_rows(tokenizer, rows)
    by_id = {item['id']: item for item in items}
    order = order_ids(list(by_id))
    plan = plan_steps([by_id[i] for i in order[:cfg['rows']]])
    if not cfg['test_mode']:
        require(len(order) == REGISTERED['rows'] and len(plan) == REGISTERED['steps']
                and all(len(step) == ACCUMULATION for step in plan), 'the registered plan is 75 steps of 8 rows')
    prompts, envelope_inputs = load_envelope_prompts(args.data, tokenizer)
    del rows, items
    state = dict(prepared, t0=t0, started_utc=_now(), session=uuid.uuid4().hex,
                 argv=[str(a) for a in (sys.argv[1:] if argv is None else argv)], inputs=inputs, rendering=rendering,
                 order=order, plan=plan, tokenizer=tokenizer, envelope_prompts=prompts,
                 envelope_inputs=envelope_inputs, envelope_records=[], steps=[], abort=None, completed=False,
                 trained=False, lease=None, disk_launch=disk_launch)
    require(comp.code_sha256(CODE_FILES) == _IMPORT_CODE, 'warm-start source changed before claim')
    bindings = {'schema': SCHEMA, 'code': _IMPORT_CODE, 'checkout': prepared['checkout'],
                'runtime': prepared['runtime'], 'freeze': cfg['freeze'], 'inputs': inputs, 'rendering': rendering,
                'host_floor': prepared['launch']['floor'], 'scorer_bundle': prepared['bundle'],
                'envelope_inputs': envelope_inputs,
                'envelope_prompt_scope_sha256': digest([{'id': p['id'], 'prompt_ids_sha256': p['prompt_ids_sha256'],
                                                       'scoring_row_sha256': digest(p['row'])} for p in prompts]),
                'order_sha256': digest(order), 'selected_ids_sha256': digest(order[:cfg['rows']]),
                'recipe': recipe(cfg, cfg['rows'], len(plan)), 'test_mode': cfg['test_mode'],
                'model_dir': str(cfg['model_dir'].resolve()), 'weights_sha256': preflight.sha(cfg['weights']),
                'asset_receipt_sha256': preflight.sha(cfg['asset_receipt'])}
    state['store'] = checkpoints.Store(cfg['records'], REGISTERED['numel'], bindings, resume=cfg['resume'],
                                      reason=args.resume_reason, started=t0)
    use_lease = cfg['device'] != 'cpu' or lease_module is not None
    try:
        if use_lease:
            with s0.lease(cfg['label'], args.lease_timeout_s, module=lease_module) as token:
                state['lease'] = {'label': cfg['label'], 'held': bool(token),
                                  'module': 'injected' if lease_module is not None else 'pinned gpu_lease.py'}
                _run(args, cfg, host, state)
        else:
            state['lease'] = {'label': cfg['label'], 'held': False, 'module': None, 'reason': 'cpu, no GPU used'}
            _run(args, cfg, host, state)
    except BaseException as error:
        state['error'] = {'type': type(error).__name__, 'message': str(error)[:2000]}
        state['completed'] = False
        try:
            state['store'].append('interruption', error=state['error'], cost=state['store'].cost())
            state['store'].append('invocation_end', outcome='FAILED', cost=state['store'].cost())
            if 'settings' in state:
                _write(args, cfg, state)
        finally:
            state['store'].close()
        raise
    try:
        state['store'].append('invocation_end', outcome=_outcome(state), cost=state['store'].cost())
        written = _write(args, cfg, state)
        if written['outcome'] == 'WARMSTART_COMPLETE':
            state['store'].retire(state['delta'])
    finally:
        state['store'].close()
    summary = envelope_summary(state['envelope_records'])
    print(json.dumps({'stage': STAGE, 'test_mode': cfg['test_mode'], 'outcome': written['outcome'],
                      'steps': len(state['steps']), 'theta0_arm_sha256': (state.get('anchor') or {}).get('arm_sha256'),
                      'envelope': [summary['envelope'], summary['prompts']], 'receipt': written['receipt'],
                      'sha256': written['sha256']}), flush=True)
    return 0 if written['outcome'] == 'WARMSTART_COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
