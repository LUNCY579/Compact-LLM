"""S0-Q1 FP32 native probe, stage S0-Q1 (COMP-RS-001 Amendment 1, section 16 'Q1-FP32 probe'; sections 2, 8, 12).

python -m experiments.maxrl_comp002_pretraining_v1.s0_q1_fp32 [--out RECORDS\\S0_Q1_FP32.json]

Launch, in this order (section 16): hostbudget.require(2.5) and guard(1.0); s0_runtime.assert_runtime();
s0_runtime.assert_clean_checkout(); the tokenizer and the eight prompts are prepared on the CPU; then, holding the
shared GPU lease 'comp-rs-001-s0-q1-fp32' for the whole probe: configure() (determinism settings before CUDA
initialises), the decoding_sha256 preimage, load_model_fp32() and checks T1..T7. hostbudget.guard(1.0) runs before
every unit (the load, each prompt, T5, T6, T7).

Prompts. The R and S prompts of the first four parents of evidence/Q2_SOURCE_SMOKE_SELECTION.json, in the receipt's
order (R then S per parent). The receipt, the dataset manifest and both CALIBRATION files are hash-checked; each
canonical row must be the selected core CALIBRATION row (tag, role, skill, parent, prompt_hash = contracts.digest
of its prompt) and each policy row's messages must equal that prompt. Prompts are rendered with
s0_runtime.render_prompt (thinking disabled). No completion is scored: corpus.reward is never called here.

Checks (FP32, TF32 off; tolerances frozen in section 16: TOL_NATS = 1e-4, T2 bitwise):
- T1 identity. load_model_fp32 streams the 310 stored BF16 tensors into a model built on the device and raises
  unless every one round-trips bitwise; T1 then requires tensors_matched_bitwise == stored tensors (310 when
  registered, 28 layers), the tied-head alias (same Parameter, same storage), FP32 on the device, and the
  parameter_inventory record. The whole identity (digests, inventory) goes into the receipt.
- For each prompt: s0_generate.greedy_decode (batch 1, 128 tokens, EOS 151645/151643, raw output_logits), then the
  full sequence prompt+completion is forwarded twice (model(input_ids), all positions, no cache).
- T2 determinism. The two float32 logit tensors of every sequence are bitwise equal (torch.equal).
- T3 generation vs teacher forcing. The raw selected-token logprob at every generated position, EOS included, is
  compared with (a) mechanics.completion_labels + mechanics.selected_logprobs over the full-sequence logits and
  (b) s0_generate.teacher_forced_logprobs (head at completion positions only). Both within 1e-4 nats and the
  teacher-forced argmax equals the generated token at every position; the cross-entropy of the same logits agrees
  with (a) within 1e-4. At least one completion must end in a generated EOS, or EOS inclusion is not exercised.
- T4 cache vs full prefix. s0_generate.cached_logprobs (prefill + one token per forward, KV cache) equals the
  full-prefix selected logprobs of (a) within 1e-4.
- T5 padded vs unpadded. The eight sequences as one left-padded batch (s0_generate.left_pad_batch: bool mask,
  position IDs over real tokens): (a) teacher_forced_logprobs(padding_side='left') equals the individual (a) values
  within 1e-4; (b) at every real position of every row, the selected next-token logprob of the padded decoder
  pass + LM head equals that of the row's individual forward within 1e-4 (logit and full-vocabulary diagnostics
  are recorded, not gated); (c) mechanics.completion_labels on the padded batch labels exactly the completion
  tokens, masks every pad and prompt position, and labels the EOS of every EOS-terminated row; at least one pad
  position and one EOS row must exist, or the check is vacuous.
- T6 backward only. requires_grad is set on exactly the parameters under model.layers.8..27 (candidate inventory
  A, 314,618,880 elements) and cleared everywhere else. loss = mean of the selected completion logprobs of the
  greedy completions of probe positions 1 and 3 (the S prompts of the first two parents; the short prompts, so the
  graph fits under the 0.78 cap without gradient checkpointing), one left-padded batch, one backward. Every
  trainable gradient exists and is finite, every frozen parameter's grad is None, the full state_fingerprint is
  bitwise identical before and after (and equal to the load fingerprint), and no torch.optim.Optimizer object
  exists before or after (gc census).
- T7 resources, read once at the end: peak CUDA allocated/reserved, peak worker RSS (peak working set), free RAM,
  and GPU shared-usage memory from the Windows counters (best effort; recorded as unavailable if the query fails).

A check that raises is recorded as failed with its error, and every later model-dependent check as skipped; T7
always runs. The receipt (stage 'S0_Q1_FP32') is create-only JSON outside the bound checkout. It carries every
check's max_abs / argmax / n and pass flag, the overall verdict, identity digests, the T6 inventory, the
decoding_sha256 and its preimage, resources, walls and counts (forwards, generations, one backward call, zero
optimizer steps), and the later-stage host floor (Q1 peak worker RSS + 1.0 GiB). Exit 0 iff every check passed,
2 otherwise.

Registered mode takes no size or path overrides: device cuda, the pinned model, asset, tokenizer and selection
receipts and dataset, 128 new tokens, inventory A, a clean checkout, the pinned lease and host budget, and the
receipt under s0_runtime.RECORDS. --test-mode (recorded in the receipt, refused with --device cuda) runs on the CPU
against a tiny model dir with injected lease/host/counter modules: the clean-checkout check is skipped unless a
checker is injected.
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
import re
import subprocess
import sys
import time

import torch

from experiments.maxrl_composable_skills_v2 import contracts

from . import mechanics, preflight
from . import s0_generate as g
from . import s0_runtime as s0
from .preflight import MANIFEST, digest, require

SCHEMA = 'COMP-RS-001-S0-Q1-FP32-1'
STAGE = 'S0_Q1_FP32'
REGISTRATION = "COMP_RS_001_AMENDMENT_1.md section 16, 'Q1-FP32 probe (stage S0-Q1)'"
REPO = s0.HERE.parents[1]
LABEL = 'comp-rs-001-s0-q1-fp32'  # s0_runtime.lease admits only comp-rs-001-* labels
DEFAULT_OUT_NAME = 'S0_Q1_FP32.json'
LAUNCH_FLOOR_GIB = 2.5
GUARD_GIB = 1.0
HOST_MARGIN_GIB = 1.0
TOL_NATS = 1e-4
REGISTERED = {'device': 'cuda:0', 'max_new_tokens': s0.MAX_NEW_TOKENS, 'blocks': [8, 27],
              'trainable_numel': 314618880, 'stored_tensors': 310, 'num_hidden_layers': 28}
Q2_SELECTION = s0.HERE / 'evidence' / 'Q2_SOURCE_SMOKE_SELECTION.json'
Q2_SELECTION_SHA = 'ae72fcca600d0def7e7dafc9b4ec8edfc129334189f8a430b9865ad08d327f0c'
PROBE_PARENTS = 4
T6_POSITIONS = (1, 3)
CALIBRATION_FILES = {'canonical': 'canonical/calibration.jsonl', 'policy': 'policy/calibration.jsonl'}
HEAD_CHUNK = 256
CHECKS = ('T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7')
SELECTION_ROW_KEYS = {'id', 'parent_id', 'prompt_hash', 'skill'}
CANONICAL_KEYS = ('id', 'role', 'tag', 'skills', 'parent_id', 'prompt_hash', 'prompt')
TEST_OPTIONS = ('model_dir', 'asset_receipt', 'selection_receipt', 'data', 'max_new_tokens', 'test_blocks', 'repo')

RULES = {
    'prompts': ('the R and S rows of the first 4 parent_worlds of evidence/Q2_SOURCE_SMOKE_SELECTION.json, in the '
                "receipt's row order (R then S per parent); policy/calibration.jsonl messages, rendered by "
                's0_runtime.render_prompt (add_generation_prompt, enable_thinking=False, add_special_tokens=False)'),
    'T2': 'bitwise: torch.equal of the two float32 logit tensors of model(input_ids) over every position',
    'T3': ('raw greedy step logprobs vs (a) mechanics.selected_logprobs of the full-sequence logits and (b) '
           's0_generate.teacher_forced_logprobs; max_abs <= 1e-4 nats and teacher-forced argmax == generated token '
           'at every position; cross-entropy vs (a) <= 1e-4; >= 1 generated EOS'),
    'T4': 's0_generate.cached_logprobs(prefill=True) vs the full-prefix logprobs of T3 (a): max_abs <= 1e-4',
    'T5': ('left-padded batch of the eight sequences: teacher_forced_logprobs(left) vs individual <= 1e-4; selected '
           'next-token logprob at every real position (padded decoder + head vs the individual forward) <= 1e-4; '
           'completion_labels masks every pad and prompt position, labels exactly the completion and every EOS; '
           '>= 1 pad position and >= 1 EOS row'),
    'T6': ('requires_grad on exactly model.layers.{first..last}. (registered: 8..27, inventory A); loss = mean '
           'selected completion logprob of probe positions 1 and 3 (one left-padded batch); one backward; '
           'trainable grads present and finite, frozen grads None, state_fingerprint identical before/after and '
           'equal to the load fingerprint, zero torch.optim.Optimizer objects'),
    'T7': 'resources once at the end: CUDA peaks, peak worker RSS, free RAM, Windows GPU shared-usage counters',
    'host_floor': 'later S0 stages launch with hostbudget.require(Q1 peak worker RSS + 1.0 GiB) (section 16)'}

GPU_COUNTERS = (r'\GPU Process Memory(*)\Shared Usage', r'\GPU Process Memory(*)\Dedicated Usage',
                r'\GPU Adapter Memory(*)\Shared Usage')
_COUNTER_PATH = re.compile(r'\\(gpu process memory|gpu adapter memory)\(([^()]*)\)\\(shared usage|dedicated usage)$')

CODE_FILES = {'s0_q1_fp32.py': Path(__file__).resolve(), 's0_runtime.py': Path(s0.__file__).resolve(),
              's0_generate.py': Path(g.__file__).resolve(), 'mechanics.py': Path(mechanics.__file__).resolve(),
              'preflight.py': Path(preflight.__file__).resolve(),
              'maxrl_composable_skills_v2/contracts.py': Path(contracts.__file__).resolve()}


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def code_sha256() -> dict:
    return {name: _sha256(path.read_bytes()) for name, path in CODE_FILES.items()}


_IMPORT_CODE = code_sha256()


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _error(error: BaseException) -> dict:
    return {'type': type(error).__name__, 'message': str(error)[:2000]}


def _json_safe(value):
    """Receipts are strict JSON: a non-finite float is recorded as its repr ('nan', 'inf', '-inf')."""
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _strict_json(raw: bytes, what: str):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, what + ': duplicate JSON key ' + repr(key))
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(what + ': nonfinite JSON ' + value)
    try:
        return json.loads(raw.decode('utf-8'), object_pairs_hook=unique, parse_constant=invalid)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(what + ': ' + str(error)) from error


# ------------------------------------------------------------------------------------------------ prompts
def probe_rows(selection, n_parents: int = PROBE_PARENTS) -> list:
    """The R and S rows of the first n parents, in the receipt's order (R then S per parent)."""
    require(type(selection) is dict and selection.get('stage') == 'Q2_SOURCE_SMOKE_SELECTION',
            'not a Q2 source-smoke selection receipt')
    parents, rows = selection.get('parent_worlds'), selection.get('rows')
    require(type(parents) is list and type(rows) is list and parents and len(rows) == 2 * len(parents)
            and len(set(parents)) == len(parents), 'selection receipt shape')
    require(all(type(r) is dict and set(r) == SELECTION_ROW_KEYS and all(type(r[k]) is str and r[k] for k in r)
                for r in rows), 'selection row fields')
    require(len({r['id'] for r in rows}) == len(rows), 'selection ids repeat')
    require(selection.get('selection_digest') == digest(rows), 'selection digest mismatch')
    require(type(n_parents) is int and 0 < n_parents <= len(parents), 'probe parent count')
    for k, parent in enumerate(parents):
        pair = rows[2 * k:2 * k + 2]
        require([r['parent_id'] for r in pair] == [parent, parent] and [r['skill'] for r in pair] == ['R', 'S'],
                'selection rows are not (R, S) per parent in parent_worlds order at parent %d' % k)
    return [dict(r, position=i) for i, r in enumerate(rows[:2 * n_parents])]


def _stream_rows(path: Path, wanted: set, keys) -> tuple:
    """One pass: sha256 and size of the whole file, and the wanted rows (only `keys` kept)."""
    hasher, size, found = hashlib.sha256(), 0, {}
    with open(path, 'rb') as stream:
        for line in stream:
            hasher.update(line)
            size += len(line)
            require(line.endswith(b'\n'), 'unterminated line in ' + str(path))
            if not line.strip():
                continue
            row_id = json.loads(line).get('id')
            if row_id in wanted:
                require(row_id not in found, 'duplicate id in %s: %s' % (path, row_id))
                row = _strict_json(line, str(path))
                found[row_id] = {k: row[k] for k in keys if k in row} if keys else row
    return found, size, hasher.hexdigest()


def load_probe_prompts(data, selection_path, tokenizer, *, weights_sha256: str, pinned: bool = True) -> tuple:
    """(prompts, inputs record) for the eight probe prompts; pinned=True binds the registered receipt and corpus."""
    selection_path = Path(selection_path)
    raw = selection_path.read_bytes()
    selection_sha = _sha256(raw)
    if pinned:
        require(selection_path.resolve() == Q2_SELECTION.resolve() and selection_sha == Q2_SELECTION_SHA,
                'Q2 selection receipt drift')
    selection = _strict_json(raw, 'selection receipt')
    require(selection.get('experiment') == 'COMP-RS-001', 'selection receipt is not COMP-RS-001')
    rows = probe_rows(selection)
    require(type(weights_sha256) is str and selection.get('source_weight_sha256') == weights_sha256,
            'the selection receipt was made for other source weights')
    root = Path(data).resolve(strict=True)
    manifest_raw = (root / 'manifest.json').read_bytes()
    manifest_sha = _sha256(manifest_raw)
    if pinned:
        require(manifest_sha == MANIFEST, 'dataset manifest drift')
    require(selection.get('dataset_manifest_sha256') == manifest_sha, 'selection receipt binds another manifest')
    files = _strict_json(manifest_raw, 'dataset manifest').get('files')
    require(type(files) is dict, 'dataset manifest has no file table')
    wanted = {r['id'] for r in rows}
    found, shas = {}, {}
    for kind, relative in CALIBRATION_FILES.items():
        entry = files.get(relative)
        require(type(entry) is dict, 'data file not in the manifest: ' + relative)
        found[kind], size, file_sha = _stream_rows(root / relative, wanted,
                                                   CANONICAL_KEYS if kind == 'canonical' else None)
        require(size == entry['bytes'] and file_sha == entry['sha256'], relative + ' drift (manifest)')
        require(file_sha == selection.get(kind + '_calibration_sha256'), relative + ' drift (selection receipt)')
        require(set(found[kind]) == wanted, 'probe ids missing from ' + relative)
        shas[kind] = file_sha
    prompts = []
    for row in rows:
        canon, policy = found['canonical'][row['id']], found['policy'][row['id']]
        require(canon.get('role') == 'calibration' and canon.get('tag') == 'core'
                and canon.get('skills') == [row['skill']] and canon.get('parent_id') == row['parent_id']
                and canon.get('prompt_hash') == row['prompt_hash'],
                'not the selected core CALIBRATION row: ' + row['id'])
        require(contracts.digest(canon.get('prompt')) == row['prompt_hash'], 'prompt_hash mismatch: ' + row['id'])
        require(set(policy) == {'id', 'messages'} and policy['messages'] == canon['prompt'],
                'policy messages differ from the canonical prompt: ' + row['id'])
        text, ids = s0.render_prompt(tokenizer, policy['messages'])
        prompts.append({'position': row['position'], 'id': row['id'], 'parent_id': row['parent_id'],
                        'skill': row['skill'], 'prompt_hash': row['prompt_hash'], 'prompt_ids': ids,
                        'prompt_tokens': len(ids), 'render_sha256': _sha256(text.encode('utf-8')),
                        'prompt_ids_sha256': digest(ids)})
    inputs = {'selection_receipt': str(selection_path.resolve()), 'selection_receipt_sha256': selection_sha,
              'selection_digest': selection['selection_digest'], 'dataset_manifest_sha256': manifest_sha,
              'canonical_calibration_sha256': shas['canonical'], 'policy_calibration_sha256': shas['policy'],
              'source_weight_sha256': weights_sha256, 'probe_parents': PROBE_PARENTS,
              'probe_ids': [p['id'] for p in prompts], 'probe_ids_digest': digest([p['id'] for p in prompts]),
              'pinned': bool(pinned), 'rule': RULES['prompts']}
    return prompts, inputs


# ---------------------------------------------------------------------------------------------- model math
def _model_device(model) -> torch.device:
    devices = {p.device for p in model.parameters()}
    require(len(devices) == 1, 'model parameters span devices')
    return devices.pop()


def _sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def full_logits(model, ids) -> torch.Tensor:
    """model(input_ids) for one unpadded sequence: float32 logits at every position, shape (L, V)."""
    device = _model_device(model)
    out = model(input_ids=torch.tensor([list(ids)], dtype=torch.long, device=device), use_cache=False)
    logits = out.logits[0]
    require(logits.dtype == torch.float32 and logits.shape == (len(ids), model.config.vocab_size),
            'full-sequence logits shape/dtype')
    return logits


def _max_abs(a: torch.Tensor, b: torch.Tensor, rows: int = HEAD_CHUNK) -> float:
    """max |a - b| over equal-shaped 2-D tensors, in row chunks (bounded transient)."""
    require(a.shape == b.shape and a.ndim == 2, 'max_abs shapes')
    worst = 0.0
    for start in range(0, a.shape[0], rows):
        worst = max(worst, float((a[start:start + rows] - b[start:start + rows]).abs().max()))
    return worst


def _parity(a, b, tol: float = TOL_NATS) -> dict:
    record = g.parity(a, b)
    record['within_tolerance'] = record['max_abs'] <= tol
    return record


def sequence_checks(model, prompt_ids, completion_ids, stop_reason, raw_logprobs, max_new_tokens,
                    tol: float = TOL_NATS) -> dict:
    """T2, T3 and T4 for one prompt + greedy completion; returns their records and the individual selected values."""
    prompt, completion = list(prompt_ids), list(completion_ids)
    ids = prompt + completion
    P, C = len(prompt), len(completion)
    device = _model_device(model)
    start = time.perf_counter()
    a = full_logits(model, ids)
    b = full_logits(model, ids)
    _sync(device)
    bitwise = bool(torch.equal(a, b))
    t2 = {'n_positions': len(ids), 'n_logits': int(a.numel()), 'bitwise_equal': bitwise,
          'max_abs': 0.0 if bitwise else _max_abs(a, b), 'argmax_equal': bool(torch.equal(a.argmax(-1), b.argmax(-1))),
          'wall_s': time.perf_counter() - start}
    t2['pass'] = bitwise
    del b
    start = time.perf_counter()
    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
    labels = mechanics.completion_labels(ids_t, torch.ones_like(ids_t, dtype=torch.bool), [P], [C],
                                         list(s0.EOS_IDS), [stop_reason], max_new_tokens)
    selected = mechanics.selected_logprobs(a[None], ids_t, labels)
    require(selected.shape == (C,), 'selected logprob count != completion length')
    window = a[P - 1:P + C - 1]
    full_argmax = window.argmax(-1).tolist()
    ce = torch.nn.functional.cross_entropy(window, ids_t[0, P:P + C], reduction='none')
    del a, window
    (head,), (head_argmax,) = g.teacher_forced_logprobs(model, [prompt], [completion], s0.PAD_ID,
                                                        padding_side='none', return_argmax=True)
    full = dict(_parity(raw_logprobs, selected, tol), argmax_equal=full_argmax == completion)
    head_only = dict(_parity(raw_logprobs, head, tol), argmax_equal=head_argmax.tolist() == completion)
    ce_check = _parity(-ce, selected, tol)
    eos = stop_reason == 'eos'
    t3 = {'n': C, 'stop_reason': stop_reason, 'eos_positions': int(eos),
          'eos_max_abs': abs(float(raw_logprobs[-1]) - float(selected[-1])) if eos else None,
          'full_forward_mechanics': full, 'teacher_forced_head_only': head_only, 'cross_entropy_vs_selected': ce_check,
          'max_abs': max(full['max_abs'], head_only['max_abs']),
          'argmax_equal': full['argmax_equal'] and head_only['argmax_equal']}
    t3['pass'] = (full['within_tolerance'] and head_only['within_tolerance'] and ce_check['within_tolerance']
                  and t3['argmax_equal'])
    t3['wall_s'] = time.perf_counter() - start
    start = time.perf_counter()
    cached, cached_argmax = g.cached_logprobs(model, prompt, completion, prefill=True, return_argmax=True)
    t4 = dict(_parity(cached, selected, tol), argmax_equal=cached_argmax.tolist() == completion,
              wall_s=time.perf_counter() - start)
    t4['pass'] = t4['within_tolerance']
    return {'T2': t2, 'T3': t3, 'T4': t4, 'selected': selected.tolist()}


def label_check(labels, batch, completions, stop_reasons) -> dict:
    """completion_labels on a left-padded batch: exactly the completion tokens, every pad masked, every EOS kept."""
    attention, ids = batch['attention_mask'], batch['input_ids']
    pads = ~attention
    pad_positions = int(pads.sum())
    pads_masked = bool((labels[pads] == -100).all()) if pad_positions else True
    rows, eos_rows, exact = [], 0, True
    for i, (offset, p, c) in enumerate(zip(batch['offsets'], batch['prompt_lengths'], batch['completion_lengths'])):
        start = offset + p
        expected = torch.full_like(labels[i], -100)
        expected[start:start + c] = ids[i, start:start + c]
        row_exact = bool(torch.equal(labels[i], expected)) and ids[i, start:start + c].tolist() == list(completions[i])
        eos_labelled = None
        if stop_reasons[i] == 'eos':
            last = int(labels[i, start + c - 1])
            eos_labelled = last == completions[i][-1] and last in s0.EOS_IDS
            eos_rows += bool(eos_labelled)
        exact &= row_exact
        rows.append({'row': i, 'offset': offset, 'labelled': int((labels[i] != -100).sum()), 'completion': c,
                     'exact': row_exact, 'stop_reason': stop_reasons[i], 'eos_labelled': eos_labelled})
    eos_expected = sum(1 for r in stop_reasons if r == 'eos')
    record = {'pad_positions': pad_positions, 'pads_masked': pads_masked, 'rows_exact': exact,
              'eos_rows': eos_rows, 'eos_rows_expected': eos_expected, 'pad_id': s0.PAD_ID,
              'pad_id_is_an_eos_id': s0.PAD_ID in s0.EOS_IDS, 'rows': rows}
    record['pass'] = (pads_masked and exact and eos_rows == eos_expected and eos_rows >= 1 and pad_positions >= 1)
    return record


def padded_batch_check(model, prompts, completions, stop_reasons, individual_selected, max_new_tokens,
                       tol: float = TOL_NATS) -> dict:
    """T5: the left-padded batch of every sequence against the individual forwards."""
    device = _model_device(model)
    vocab = model.config.vocab_size
    prompts, completions = [list(p) for p in prompts], [list(c) for c in completions]
    batch = g.left_pad_batch(prompts, completions, s0.PAD_ID, device=device, vocab=vocab)
    left = g.teacher_forced_logprobs(model, prompts, completions, s0.PAD_ID, padding_side='left')
    completion_parity = _parity(left, individual_selected, tol)
    del left
    hidden = model.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'].long(),
                         position_ids=batch['position_ids'], use_cache=False).last_hidden_state
    head = model.get_output_embeddings()
    selected_max = logits_max = logprob_max = 0.0
    selected_n = real_positions = argmax_mismatch = 0
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        ids = prompt + completion
        L, offset = len(ids), batch['offsets'][i]
        individual = full_logits(model, ids)
        following = torch.tensor(ids[1:], dtype=torch.long, device=device)
        for start in range(0, L, HEAD_CHUNK):
            end = min(start + HEAD_CHUNK, L)
            padded = head(hidden[i, offset + start:offset + end]).float()
            alone = individual[start:end]
            logits_max = max(logits_max, float((padded - alone).abs().max()))
            argmax_mismatch += int((padded.argmax(-1) != alone.argmax(-1)).sum())
            padded, alone = padded.log_softmax(-1), alone.log_softmax(-1)
            logprob_max = max(logprob_max, float((padded - alone).abs().max()))
            k = min(end, L - 1) - start  # positions start .. start+k-1 predict ids[start+1 .. start+k]
            if k > 0:
                target = following[start:start + k, None]
                gap = (padded[:k].gather(-1, target) - alone[:k].gather(-1, target)).abs()
                require(bool(torch.isfinite(gap).all()), 'nonfinite real-position logprob')
                selected_max = max(selected_max, float(gap.max()))
                selected_n += k
            real_positions += end - start
            del padded, alone
        del individual
    del hidden
    try:
        labels = mechanics.completion_labels(batch['input_ids'], batch['attention_mask'], batch['prompt_lengths'],
                                             batch['completion_lengths'], list(s0.EOS_IDS), list(stop_reasons),
                                             max_new_tokens)
        labels_record = label_check(labels, batch, completions, stop_reasons)
    except ValueError as error:  # completion_labels refused the batch: recorded, the parities stay visible
        labels_record = {'pass': False, 'error': _error(error)}
    real = {'n': selected_n, 'max_abs': selected_max, 'within_tolerance': selected_max <= tol,
            'real_positions': real_positions, 'diagnostic_logits_max_abs': logits_max,
            'diagnostic_full_vocab_logprob_max_abs': logprob_max, 'diagnostic_argmax_mismatches': argmax_mismatch}
    record = {'rows': len(prompts), 'width': int(batch['input_ids'].shape[1]), 'offsets': batch['offsets'],
              'teacher_forced_left_vs_individual': completion_parity, 'real_positions_selected': real,
              'labels': labels_record, 'n': completion_parity['n'],
              'max_abs': max(completion_parity['max_abs'], selected_max)}
    record['pass'] = completion_parity['within_tolerance'] and real['within_tolerance'] and labels_record['pass']
    return record


# ---------------------------------------------------------------------------------------------- T6 backward
def block_prefixes(first: int, last: int) -> tuple:
    return tuple('model.layers.%d.' % i for i in range(first, last + 1))


def block_numel(model, first: int, last: int) -> int:
    """Unique parameter elements under model.layers.first..last (works on a meta-device model)."""
    prefixes, seen, total = block_prefixes(first, last), set(), 0
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if name.startswith(prefixes) and id(parameter) not in seen:
            seen.add(id(parameter))
            total += parameter.numel()
    return total


def inventory_scope(model, first: int, last: int) -> dict:
    """requires_grad = True on exactly the parameters under model.layers.first..last, False everywhere else."""
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
    inventory = mechanics.parameter_inventory(model)
    names = [name for name, _ in model.named_parameters(remove_duplicate=False)]
    trainable = sorted(n for n in names if n.startswith(prefixes))
    frozen = sorted(n for n in names if not n.startswith(prefixes))
    numel = block_numel(model, first, last)
    require(inventory['trainable_numel'] == numel, 'inventory count drift')
    return {'blocks': [first, last], 'rule': 'every parameter under model.layers.{%d..%d}.' % (first, last),
            'num_hidden_layers': layers, 'trainable_names': len(trainable),
            'trainable_names_sha256': digest(trainable), 'frozen_names': len(frozen),
            'frozen_names_sha256': digest(frozen), 'trainable_numel': numel,
            'unique_numel': inventory['unique_numel'], 'inventory_sha256': inventory['inventory_sha256'],
            'status': inventory['status']}


def optimizer_census() -> int:
    """How many torch.optim.Optimizer objects exist in this process (gc census; type(), never __class__)."""
    return sum(1 for obj in gc.get_objects() if issubclass(type(obj), torch.optim.Optimizer))


def t6_loss(values) -> torch.Tensor:
    """The registered T6 loss: the mean selected completion logprob over both sequences' tokens."""
    return torch.cat(list(values)).mean()


def backward_check(model, prompts, completions, blocks, load_fingerprint, *, registered_numel=None) -> dict:
    """T6: one backward on the inventory; gradients scoped and finite, state bitwise unchanged, no optimizer."""
    first, last = blocks
    optimizers_before = optimizer_census()
    scope = inventory_scope(model, first, last)
    prefixes = block_prefixes(first, last)
    named = list(model.named_parameters(remove_duplicate=False))
    for _, parameter in named:
        parameter.grad = None
    before = s0.state_fingerprint(model)
    backward_calls = 0
    try:
        with torch.enable_grad():
            values = g.teacher_forced_logprobs(model, [list(p) for p in prompts], [list(c) for c in completions],
                                               s0.PAD_ID, padding_side='left')
            loss = t6_loss(values)
            loss.backward()
            backward_calls += 1
        missing, nonfinite, flag_drift, frozen_with_grad, zero = [], [], [], [], 0
        squares, seen = 0.0, set()
        for name, parameter in named:
            if name.startswith(prefixes):
                if not parameter.requires_grad:
                    flag_drift.append(name)
                if parameter.grad is None:
                    missing.append(name)
                elif id(parameter) not in seen:
                    seen.add(id(parameter))
                    finite = bool(torch.isfinite(parameter.grad).all())
                    if not finite:
                        nonfinite.append(name)
                    norm = float(torch.linalg.vector_norm(parameter.grad.double()))
                    squares += norm * norm
                    zero += not bool(parameter.grad.any())
            else:
                if parameter.requires_grad:
                    flag_drift.append(name)
                if parameter.grad is not None:
                    frozen_with_grad.append(name)
        after = s0.state_fingerprint(model)
        optimizers_after = optimizer_census()
        record = {'inventory': scope, 'prompt_positions': list(T6_POSITIONS), 'loss': float(loss.detach()),
                  'loss_tokens': int(sum(v.numel() for v in values)), 'backward_calls': backward_calls,
                  'trainable_parameters': len(seen), 'trainable_with_grad': len(seen) - len(nonfinite),
                  'missing_grads': missing[:20], 'nonfinite_grads': nonfinite[:20],
                  'frozen_with_grad': frozen_with_grad[:20], 'requires_grad_drift': flag_drift[:20],
                  'all_zero_grads': zero, 'grad_norm': math.sqrt(squares),
                  'fingerprint_before': before, 'fingerprint_after': after,
                  'fingerprint_unchanged': before == after, 'unchanged_since_load': before == load_fingerprint,
                  'optimizers_before': optimizers_before, 'optimizers_after': optimizers_after,
                  'optimizer_steps': 0, 'registered_numel': registered_numel,
                  'registered_numel_ok': None if registered_numel is None else scope['trainable_numel'] == registered_numel}
        record['pass'] = (not missing and not nonfinite and not frozen_with_grad and not flag_drift
                          and record['fingerprint_unchanged'] and record['unchanged_since_load']
                          and optimizers_before == 0 and optimizers_after == 0 and len(seen) > 0
                          and record['registered_numel_ok'] is not False)
        return record
    finally:
        for _, parameter in named:
            parameter.grad = None
            parameter.requires_grad_(False)


# --------------------------------------------------------------------------------------------- T7 resources
def gpu_shared_usage(pid=None, *, runner=None, timeout_s: float = 120.0) -> dict:
    """Windows GPU memory counters, best effort: this pid's GPU Process Memory shared/dedicated usage and the
    adapter-wide GPU Adapter Memory shared usage, in bytes. Any failure is recorded, never raised."""
    pid = os.getpid() if pid is None else pid
    paths = ', '.join("'" + path + "'" for path in GPU_COUNTERS)
    script = ("$ErrorActionPreference = 'Stop'; "
              '$samples = (Get-Counter -Counter @(' + paths + ') -MaxSamples 1).CounterSamples; '
              'ConvertTo-Json -Compress -InputObject @($samples | ForEach-Object { [pscustomobject]@{ '
              'path = $_.Path; instance = $_.InstanceName; value = [double]$_.CookedValue } })')
    record = {'available': False, 'method': 'powershell Get-Counter', 'counters': list(GPU_COUNTERS), 'pid': pid,
              'utc': _now(), 'process_shared_usage_bytes': None, 'process_dedicated_usage_bytes': None,
              'process_instances': {}, 'adapter_shared_usage_bytes': {}, 'error': None}
    try:
        run = (runner or subprocess.run)(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script],
                                         capture_output=True, text=True, encoding='utf-8', errors='replace',
                                         timeout=timeout_s)
        require(run.returncode == 0, 'Get-Counter failed: ' + (run.stderr or '').strip()[:500])
        samples = json.loads(run.stdout)
        samples = [samples] if isinstance(samples, dict) else samples
        require(isinstance(samples, list) and samples, 'no counter samples')
        prefix = 'pid_%d_' % pid
        shared = dedicated = None
        for sample in samples:
            match = _COUNTER_PATH.search(str(sample.get('path', '')).strip().lower())
            if match is None:
                continue
            counter_set, instance, counter = match.groups()
            value = float(sample['value'])
            if counter_set == 'gpu adapter memory':
                if counter == 'shared usage':
                    record['adapter_shared_usage_bytes'][instance] = value
            elif instance.startswith(prefix):
                key = 'shared' if counter == 'shared usage' else 'dedicated'
                record['process_instances'].setdefault(instance, {})[key] = value
                if key == 'shared':
                    shared = (shared or 0.0) + value
                else:
                    dedicated = (dedicated or 0.0) + value
        record.update(available=True, process_shared_usage_bytes=shared, process_dedicated_usage_bytes=dedicated)
    except Exception as error:  # best effort by registration: recorded as unavailable
        record['error'] = _error(error)
    return record


def resources_check(device: str, host, gpu_counter) -> dict:
    """T7: CUDA peaks, peak worker RSS, free RAM and GPU shared usage, read once."""
    if device != 'cpu':
        torch.cuda.synchronize(0)
    resources = s0.resources(device, module=host)
    gpu = gpu_counter()
    cuda = resources['cuda']
    peak = resources['process_peak_rss_gib']
    record = {'read': 'once, at the end of the probe', 'device': device,
              'peak_cuda_allocated_gib': None if cuda is None else cuda['max_memory_allocated_gib'],
              'peak_cuda_reserved_gib': None if cuda is None else cuda['max_memory_reserved_gib'],
              'cuda_cap_gib': None if cuda is None else cuda['fraction_cap_gib'],
              'total_vram_gib': None if cuda is None else cuda['total_gib'],
              'peak_worker_rss_gib': peak, 'worker_rss_gib': resources['process_rss_gib'],
              'free_phys_gib': resources['free_phys_gib'], 'gpu_shared_usage': gpu, 'resources': resources}
    record['pass'] = (type(peak) is float and peak > 0 and (device == 'cpu' or cuda is not None)
                      and isinstance(gpu, dict) and 'available' in gpu)
    return record


class ForwardCounter:
    """Counts top-level model forwards and decoder (model.model) forwards with pre-hooks; values are untouched."""

    def __init__(self, model):
        self.counts = {'lm_forwards': 0, 'decoder_forwards': 0}
        self._handles = [model.register_forward_pre_hook(self._bump('lm_forwards')),
                         model.model.register_forward_pre_hook(self._bump('decoder_forwards'))]

    def _bump(self, key):
        def hook(module, args):
            self.counts[key] += 1
        return hook

    def close(self) -> dict:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        return dict(self.counts)


def t1_record(model, identity: dict, device: str, registered: bool) -> dict:
    tied = identity['tied']
    inventory = identity['parameter_inventory']
    unique = sum(p.numel() for p in model.parameters())
    checks = {'tensors_matched_bitwise': identity['stored_tensors'] > 0
              and identity['tensors_matched_bitwise'] == identity['stored_tensors'],
              'tied_alias': tied == {'alias': 'lm_head.weight', 'of': 'model.embed_tokens.weight',
                                     'same_parameter': True, 'same_storage': True},
              'fp32_on_device': all(p.dtype == torch.float32 and p.device == torch.device(device)
                                    for p in model.parameters()),
              'parameter_inventory': inventory is not None and inventory['unique_numel'] == unique}
    if registered:
        checks['registered_tensor_count'] = identity['stored_tensors'] == REGISTERED['stored_tensors']
        checks['registered_layers'] = model.config.num_hidden_layers == REGISTERED['num_hidden_layers']
    return {'pass': all(checks.values()), 'checks': checks, 'n': identity['stored_tensors'],
            'stored_tensors': identity['stored_tensors'], 'tensors_matched_bitwise': identity['tensors_matched_bitwise'],
            'weights_sha256': identity['weights_sha256'], 'stored_sha256': identity['stored_sha256'],
            'arm_sha256': identity['arm_sha256'], 'state_fingerprint': identity['state_fingerprint'],
            'tied': tied, 'unique_numel': unique,
            'inventory_sha256': None if inventory is None else inventory['inventory_sha256']}


# ---------------------------------------------------------------------------------------------------- probe
def _aggregate(name: str, records: list, tol: float = TOL_NATS) -> dict:
    out = {'sequences': records, 'n_sequences': len(records)}
    if name == 'T2':
        out.update(n=sum(r['n_positions'] for r in records), max_abs=max(r['max_abs'] for r in records),
                   bitwise_equal=all(r['bitwise_equal'] for r in records),
                   argmax_equal=all(r['argmax_equal'] for r in records))
        out['pass'] = all(r['pass'] for r in records)
    elif name == 'T3':
        eos = sum(r['eos_positions'] for r in records)
        out.update(n=sum(r['n'] for r in records), max_abs=max(r['max_abs'] for r in records),
                   argmax_equal=all(r['argmax_equal'] for r in records), eos_positions=eos,
                   eos_exercised=eos >= 1, tolerance_nats=tol)
        out['pass'] = all(r['pass'] for r in records) and eos >= 1
    else:
        out.update(n=sum(r['n'] for r in records), max_abs=max(r['max_abs'] for r in records),
                   argmax_equal=all(r['argmax_equal'] for r in records), tolerance_nats=tol)
        out['pass'] = all(r['pass'] for r in records)
    return out


def _failed(error: BaseException, **extra) -> dict:
    return dict(extra, **{'pass': False, 'skipped': False, 'error': _error(error)})


def _skipped(reason: str) -> dict:
    return {'pass': False, 'skipped': True, 'reason': reason}


def _probe(cfg: dict, state: dict, host, gpu_counter) -> None:
    """Everything that runs under the lease; fills state['checks'] (never raises an Exception)."""
    checks, walls = state['checks'], state['walls']
    device = cfg['device']
    guard = lambda what: s0.host_guard(GUARD_GIB, LABEL + ' ' + what, module=host)  # noqa: E731
    blocked = None
    start = time.perf_counter()
    try:
        state['settings'] = s0.configure(device)
        state['decoding_preimage'] = s0.decoding_preimage(device)
        state['decoding_sha256'] = s0.decoding_sha256(state['decoding_preimage'])
    except Exception as error:
        state['setup_error'] = _error(error)
        blocked = 'setup'
    walls['setup_s'] = time.perf_counter() - start
    model, counter = None, None
    try:
        # T1 identity
        if blocked is None:
            start = time.perf_counter()
            try:
                guard('T1')
                model, identity = s0.load_model_fp32(cfg['model_dir'], device, cfg['asset_receipt'])
                state['counts']['model_loads'] += 1
                state['identity'] = identity
                checks['T1'] = t1_record(model, identity, device, not cfg['test_mode'])
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                counter = ForwardCounter(model)
            except Exception as error:
                checks['T1'] = _failed(error)
                blocked = 'T1'
            walls['T1_s'] = time.perf_counter() - start
        else:
            checks['T1'] = _skipped('not run: ' + blocked + ' failed')
        # greedy generation + T2/T3/T4 per sequence
        records = {'T2': [], 'T3': [], 'T4': []}
        if blocked is None:
            start = time.perf_counter()
            try:
                for item in cfg['prompts']:
                    guard('prompt %d' % item['position'])
                    with torch.inference_mode():
                        _sync(model.device)
                        begin = time.perf_counter()
                        greedy = g.greedy_decode(model, item['prompt_ids'], max_new_tokens=cfg['max_new_tokens'])
                        _sync(model.device)
                        wall = time.perf_counter() - begin
                        state['counts']['generations'] += 1
                        state['completions'].append({
                            'position': item['position'], 'id': item['id'], 'skill': item['skill'],
                            'completion_ids': greedy['completion_ids'], 'stop_reason': greedy['stop_reason'],
                            'generated_tokens': greedy['generated_tokens'],
                            'raw_step_logprobs': greedy['raw_step_logprobs'], 'greedy_wall_s': wall,
                            'decoder': greedy['decoder']})
                        result = sequence_checks(model, item['prompt_ids'], greedy['completion_ids'],
                                                 greedy['stop_reason'], greedy['raw_step_logprobs'],
                                                 cfg['max_new_tokens'])
                    for name in records:
                        records[name].append(dict(result[name], position=item['position'], id=item['id']))
                    state['individual_selected'].append(result['selected'])
                for name in records:
                    checks[name] = _aggregate(name, records[name])
            except Exception as error:
                for name in records:
                    checks[name] = _failed(error, sequences=records[name], phase='generation/T2-T4')
                blocked = 'generation/T2-T4'
            walls['sequences_s'] = time.perf_counter() - start
        else:
            for name in records:
                checks[name] = _skipped('not run: ' + blocked + ' failed')
        completions = state['completions']
        # T5 padded vs unpadded
        if blocked is None:
            start = time.perf_counter()
            try:
                guard('T5')
                with torch.inference_mode():
                    checks['T5'] = padded_batch_check(
                        model, [p['prompt_ids'] for p in cfg['prompts']], [c['completion_ids'] for c in completions],
                        [c['stop_reason'] for c in completions], state['individual_selected'], cfg['max_new_tokens'])
            except Exception as error:
                checks['T5'] = _failed(error)
                blocked = 'T5'
            walls['T5_s'] = time.perf_counter() - start
        else:
            checks['T5'] = _skipped('not run: ' + blocked + ' failed')
        # T6 backward only
        if blocked is None:
            start = time.perf_counter()
            try:
                guard('T6')
                chosen = [cfg['prompts'][k] for k in T6_POSITIONS]
                checks['T6'] = backward_check(
                    model, [p['prompt_ids'] for p in chosen], [completions[k]['completion_ids'] for k in T6_POSITIONS],
                    cfg['blocks'], state['identity']['state_fingerprint'],
                    registered_numel=None if cfg['test_mode'] else REGISTERED['trainable_numel'])
                state['counts']['backward_calls'] += checks['T6']['backward_calls']
                state['inventory'] = checks['T6']['inventory']
            except Exception as error:
                checks['T6'] = _failed(error)
                blocked = 'T6'
            walls['T6_s'] = time.perf_counter() - start
        else:
            checks['T6'] = _skipped('not run: ' + blocked + ' failed')
        state['counts']['optimizer_steps'] = 0
        state['counts']['optimizers_at_end'] = optimizer_census()
    finally:
        if counter is not None:
            state['counts'].update(counter.close())
    # T7 resources, once, with the model still resident
    start = time.perf_counter()
    try:
        guard('T7')
        checks['T7'] = resources_check(device, host, gpu_counter)
    except Exception as error:
        checks['T7'] = _failed(error)
    walls['T7_s'] = time.perf_counter() - start
    state['blocked'] = blocked
    del model


# ------------------------------------------------------------------------------------------------------ CLI
def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog='s0_q1_fp32', description='COMP-RS-001 S0-Q1 FP32 native probe.')
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--lease-timeout-s', type=float, default=900.0)
    ap.add_argument('--test-mode', action='store_true')
    ap.add_argument('--repo', type=Path)
    ap.add_argument('--model-dir', type=Path)
    ap.add_argument('--asset-receipt', type=Path)
    ap.add_argument('--selection-receipt', type=Path)
    ap.add_argument('--data', type=Path)
    ap.add_argument('--max-new-tokens', type=int)
    ap.add_argument('--test-blocks', type=int, nargs=2, metavar=('FIRST', 'LAST'))
    return ap


def _config(args, injected: dict) -> dict:
    """Validate the command line; refuses (SystemExit/ValueError) before any host, lease or model work."""
    if args.test_mode:
        if args.device != 'cpu':
            raise SystemExit('--test-mode is refused with --device cuda: the registered probe is never a test run')
        if injected['lease_module'] is None:
            raise SystemExit('--test-mode needs an injected lease module (a CPU test never takes the GPU lease)')
        if args.out is None or args.test_blocks is None:
            raise SystemExit('--test-mode needs --out and --test-blocks FIRST LAST')
        cfg = {'out': args.out, 'repo': args.repo if args.repo is not None else REPO,
               'model_dir': args.model_dir if args.model_dir is not None else s0.MODEL_DIR,
               'asset_receipt': args.asset_receipt if args.asset_receipt is not None else s0.ASSET_RECEIPT,
               'selection': args.selection_receipt if args.selection_receipt is not None else Q2_SELECTION,
               'data': args.data if args.data is not None else s0.DATA,
               'max_new_tokens': args.max_new_tokens if args.max_new_tokens is not None else s0.MAX_NEW_TOKENS,
               'blocks': list(args.test_blocks)}
    else:
        given = [name for name in TEST_OPTIONS if getattr(args, name) is not None]
        if given:
            raise SystemExit('test-mode only option(s): ' + ', '.join('--' + n.replace('_', '-') for n in given))
        if args.device != 'cuda':
            raise SystemExit('the registered probe runs on cuda: the model is never built on the CPU')
        if any(value is not None for value in injected.values()):
            raise SystemExit('injected lease/host/counter/checkout modules are test-mode only')
        out = args.out if args.out is not None else s0.RECORDS / DEFAULT_OUT_NAME
        if Path(out).resolve().parent != s0.RECORDS.resolve():
            raise SystemExit('registered receipts are written to ' + str(s0.RECORDS))
        cfg = {'out': out, 'repo': REPO, 'model_dir': s0.MODEL_DIR, 'asset_receipt': s0.ASSET_RECEIPT,
               'selection': Q2_SELECTION, 'data': s0.DATA, 'max_new_tokens': REGISTERED['max_new_tokens'],
               'blocks': list(REGISTERED['blocks'])}
    if cfg['max_new_tokens'] < 1:
        raise SystemExit('max_new_tokens must be >= 1')
    if not (math.isfinite(args.lease_timeout_s) and args.lease_timeout_s > 0):
        raise SystemExit('lease timeout must be > 0')
    target = s0.assert_outside_checkout(cfg['out'], cfg['repo'])
    if target.exists():
        raise SystemExit('receipt exists (create-only): ' + str(target))
    return dict(cfg, out=target, device=s0.DEVICES[args.device], test_mode=bool(args.test_mode),
                model_dir=Path(cfg['model_dir']), asset_receipt=Path(cfg['asset_receipt']),
                selection=Path(cfg['selection']), data=Path(cfg['data']))


def _receipt(cfg: dict, state: dict) -> dict:
    checks = {name: state['checks'].get(name, _skipped('not reached')) for name in CHECKS}
    failed = [name for name in CHECKS if checks[name].get('pass') is not True]
    t7 = checks['T7']
    peak = t7.get('peak_worker_rss_gib') if t7.get('pass') else None
    counts = dict(state['counts'])
    counts['generated_tokens'] = sum(c['generated_tokens'] for c in state['completions'])
    return {
        'experiment': 'COMP-RS-001', 'stage': STAGE, 'schema': SCHEMA, 'registration': REGISTRATION,
        'test_mode': cfg['test_mode'], 'registered_run': not cfg['test_mode'], 'argv': state['argv'],
        'started_utc': state['started_utc'], 'finished_utc': _now(), 'wall_s': time.perf_counter() - state['t0'],
        'passed': not failed, 'failed_checks': failed,
        'completed': all(not checks[name].get('skipped') and 'error' not in checks[name] for name in CHECKS),
        'checks': checks,
        'summary': {name: {k: checks[name].get(k) for k in ('pass', 'n', 'max_abs', 'argmax_equal', 'skipped')}
                    for name in CHECKS},
        'tolerances': {'nats': TOL_NATS, 'T2': 'bitwise', 'frozen_in': 'section 16 (before any native output)'},
        'rules': RULES, 'error': state.get('setup_error'), 'interrupted': state.get('interrupted'),
        'runtime': state['runtime'], 'settings': state.get('settings'), 'checkout': state['checkout'],
        'lease': state['lease'], 'launch': state['launch'], 'code_sha256': _IMPORT_CODE,
        'decoding_preimage': state.get('decoding_preimage'), 'decoding_sha256': state.get('decoding_sha256'),
        'generation_contract': g.generation_contract(cfg['max_new_tokens']),
        'inputs': dict(state['inputs'], model_dir=str(cfg['model_dir']), asset_receipt=str(cfg['asset_receipt']),
                       asset_receipt_sha256=preflight.sha(cfg['asset_receipt']),
                       tokenizer_receipt_sha256=preflight.sha(s0.TOKENIZER_RECEIPT)),
        'prompts': [{k: v for k, v in p.items() if k != 'prompt_ids'} for p in cfg['prompts']],
        'completions': state['completions'], 'identity': state.get('identity'), 'inventory': state.get('inventory'),
        'resources': {k: t7.get(k) for k in ('peak_cuda_allocated_gib', 'peak_cuda_reserved_gib', 'cuda_cap_gib',
                                             'total_vram_gib', 'peak_worker_rss_gib', 'worker_rss_gib',
                                             'free_phys_gib', 'gpu_shared_usage')},
        'host_floor': {'peak_worker_rss_gib': peak, 'margin_gib': HOST_MARGIN_GIB,
                       'later_stages_floor_gib': None if peak is None else peak + HOST_MARGIN_GIB,
                       'rule': RULES['host_floor']},
        'walls': state['walls'], 'counts': counts,
        'sizes': {'prompts': len(cfg['prompts']), 'probe_parents': PROBE_PARENTS,
                  'max_new_tokens': cfg['max_new_tokens'], 'blocks': cfg['blocks'],
                  't6_positions': list(T6_POSITIONS), 'head_chunk': HEAD_CHUNK},
        'native_outputs_scored': False, 'weights_saved': False, 'optimizer_steps': 0,
        'note': 'S0-Q1 mechanical probe: nothing is scored, trained or saved; no optimizer is constructed.'}


def _write(cfg: dict, state: dict) -> dict:
    receipt = _json_safe(_receipt(cfg, state))
    require(code_sha256() == _IMPORT_CODE, 'probe source changed during the run')
    return {'passed': receipt['passed'], 'failed_checks': receipt['failed_checks'],
            'sha256': s0.write_receipt(cfg['out'], receipt, repo=cfg['repo'])}


def main(argv=None, *, lease_module=None, host_module=None, gpu_counter=None, checkout_check=None) -> int:
    t0, started = time.perf_counter(), _now()
    args = _parser().parse_args(argv)
    cfg = _config(args, {'lease_module': lease_module, 'host_module': host_module, 'gpu_counter': gpu_counter,
                         'checkout_check': checkout_check})
    host = host_module if host_module is not None else s0.hostbudget()
    launch = {'floor_gib': LAUNCH_FLOOR_GIB, 'guard_gib': GUARD_GIB, 'utc': _now()}
    launch['require'] = s0.host_require(LAUNCH_FLOOR_GIB, LABEL, module=host)
    s0.host_guard(GUARD_GIB, LABEL + ' launch', module=host)
    launch['guard'] = 'passed'
    runtime = s0.assert_runtime()
    if cfg['test_mode']:
        checkout = ({'checked': False, 'reason': 'test mode: clean-checkout check skipped'} if checkout_check is None
                    else dict(checkout_check(cfg['repo']), checked=True, injected=True))
    else:
        checkout = dict(s0.assert_clean_checkout(REPO), checked=True, injected=False)
        require(preflight.sha(s0.ASSET_RECEIPT) == s0.ASSET_RECEIPT_SHA, 'asset receipt drift')
    require(preflight.sha(s0.TOKENIZER_RECEIPT) == s0.TOKENIZER_RECEIPT_SHA, 'tokenizer receipt drift')
    asset = preflight.read_json(cfg['asset_receipt'])
    require(asset.get('stage') == 'Q1_ASSET_CANDIDATE', 'not a Q1_ASSET_CANDIDATE receipt')
    tokenizer = s0.load_tokenizer(s0.MODEL_DIR, s0.TOKENIZER_RECEIPT, asset_receipt=s0.ASSET_RECEIPT)
    prompts, inputs = load_probe_prompts(cfg['data'], cfg['selection'], tokenizer,
                                         weights_sha256=asset['assets']['model.safetensors']['sha256'],
                                         pinned=not cfg['test_mode'])
    del tokenizer
    require(len(prompts) == 2 * PROBE_PARENTS, 'probe prompt count')
    cfg['prompts'] = prompts
    state = {'t0': t0, 'started_utc': started, 'argv': [str(a) for a in (sys.argv[1:] if argv is None else argv)],
             'runtime': runtime, 'checkout': checkout, 'launch': launch, 'inputs': inputs, 'checks': {},
             'walls': {'prelaunch_s': time.perf_counter() - t0}, 'completions': [], 'individual_selected': [],
             'counts': {'model_loads': 0, 'generations': 0, 'backward_calls': 0, 'optimizer_steps': 0},
             'lease': None}
    counter_reader = gpu_counter if gpu_counter is not None else gpu_shared_usage
    started_probe = False
    try:
        with s0.lease(LABEL, args.lease_timeout_s, module=lease_module) as token:
            state['lease'] = {'label': LABEL, 'held': bool(token), 'lock': s0.LOCK_NAME,
                              'module': 'injected (test mode)' if lease_module is not None else 'pinned gpu_lease.py'}
            started_probe = True
            _probe(cfg, state, host, counter_reader)
    except BaseException as error:
        if started_probe:
            state['interrupted'] = _error(error)
            _write(cfg, state)
        raise
    written = _write(cfg, state)
    print(json.dumps({'stage': STAGE, 'test_mode': cfg['test_mode'], 'passed': written['passed'],
                      'failed_checks': written['failed_checks'], 'receipt': str(cfg['out']),
                      'sha256': written['sha256']}), flush=True)
    return 0 if written['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
