"""Pure-stdlib, unapproved S1 planning/accounting prototype. No trainer or execution grant.

Inputs are proposal-2 metadata and atomic manifest identities. Numeric wall and
prefill ceilings require separately supplied measured prices/tokenizer census.
Native parameter/Adam/RNG checkpoints and gradient parity are not implemented.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import uuid

SCHEMA = 'COMP-RS-001-S1-RECIPE-PROPOSAL-2'
ORDER_SALT = 'COMP-RS-001-S1-ORDER-v1|'
SAMPLE_SALT = 'COMP-RS-001-S1-SAMPLE-v1|'
ARMS = {'R': ('R',), 'S': ('S',), 'P': ('P',), 'MT': ('R', 'S'),
        'R_to_S': ('S',), 'S_to_R': ('R',), 'R_prime': ('R',), 'S_prime': ('S',)}
B, G, CAP, RESUME_CAP, CADENCE = 2, 8, 128, 8, 10
RHO = Fraction(1, 4)

class AdmissionRejected(ValueError): pass
class CeilingRejected(ValueError): pass
class UnresolvedBudget(ValueError): pass

def require(condition, message):
    if not condition: raise ValueError(message)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')

def digest(value): return hashlib.sha256(canonical(value)).hexdigest()

def file_record(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}

def load_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf8'), object_pairs_hook=unique,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError('nonfinite JSON')))

def validate_packet(packet):
    require(packet['schema'] == SCHEMA and packet['version'] == 2, 'proposal-2 schema required')
    require(packet['recommended_recipe']['choice_id'] == 'S1_MAXRL_FIXED_EXPOSURE_B2_G8_V1', 'recipe identity drift')
    batch = packet['recommended_recipe']['prompt_batch']
    require(batch['prompts_per_optimizer_boundary'] == B and batch['rollouts_per_prompt'] == G
            and batch['rollouts_per_boundary'] == B * G, 'B2/G8 shape drift')
    require(packet['recommended_recipe']['sampler']['max_new_tokens'] == CAP, 'token cap drift')
    require(packet['recommended_recipe']['objective']['epsilon'] == 1e-6, 'MaxRL epsilon drift')
    unit = packet['proposed_unit_and_panel']
    require(unit['prompts_per_skill_per_unit'] == 3000 and unit['nominal_boundaries_per_unit'] == 1500
            and unit['nominal_rollouts_per_unit'] == 24000, 'one-unit constants drift')
    require(unit['arms']['MT']['units'] == 2 and unit['arms']['MT']['boundaries'] == 3000
            and unit['arms']['MT']['prompts_R'] == unit['arms']['MT']['prompts_S'] == 3000, 'MT two-unit constants drift')
    policy = packet['proposed_checkpoint_and_rollback_policy']
    require(policy['proposed_resume_count_cap'] == RESUME_CAP and policy['cadence_completed_boundaries'] == CADENCE
            and Fraction(str(policy['proposed_rollback_work_allowance_fraction'])) == RHO, 'durability constants drift')
    return packet

def execution_admission(packet, prerequisites=None, framework_loader=None):
    """Always reject this prototype. No loader/import is reached, even with forged booleans."""
    if packet.get('approved') is not True or packet.get('training_authorized') is not True:
        raise AdmissionRejected('unapproved proposal is data only; execution denied before framework import')
    raise AdmissionRejected('prototype has no qualified production runner, approved grant or native integration')

def validate_manifest(manifest, skill, expected_count):
    require(manifest.get('name') == 'atomic_' + skill and manifest.get('schema_version') == '2.0.0', 'atomic manifest schema')
    ids = manifest.get('ids')
    require(type(ids) is list and type(manifest.get('rows')) is int and manifest['rows'] == expected_count
            and len(ids) == expected_count, 'missing/extra declared atomic rows')
    require(all(type(value) is str and value.startswith('MC2-') and len(value) == 68
                and all(ch in '0123456789abcdef' for ch in value[4:]) for value in ids), 'row identity schema')
    require(len(set(ids)) == len(ids), 'duplicate atomic rows')
    return tuple(ids)

def sample_seed(run_id, seed, epoch, boundary, row_id, sample_k):
    require(type(sample_k) is int and 0 <= sample_k < G, 'sample index')
    text = SAMPLE_SALT + '|'.join(map(str, (run_id, seed, epoch, boundary, row_id, sample_k)))
    return int(hashlib.sha256(text.encode('ascii')).hexdigest()[:16], 16) % (2 ** 63)

@dataclass(frozen=True)
class Prompt:
    skill: str
    row_id: str
    sample_seeds: tuple

@dataclass(frozen=True)
class Boundary:
    index: int
    prompts: tuple

@dataclass(frozen=True)
class Plan:
    run_id: str
    seed: int
    synthetic: bool
    boundaries: tuple
    sha256: str
    manifests_sha256: dict
    proposal_sha256: str

    @property
    def units(self): return 2 if self.run_id == 'MT' else 1
    @property
    def prompts(self): return B * len(self.boundaries)
    @property
    def nominal_rollouts(self): return self.prompts * G
    def trajectory(self, boundary, prompt, sample_k):
        require(type(boundary) is int and 0 <= boundary < len(self.boundaries)
                and type(prompt) is int and 0 <= prompt < B and type(sample_k) is int and 0 <= sample_k < G, 'logical trajectory index')
        item = self.boundaries[boundary].prompts[prompt]
        body = {'plan_sha256': self.sha256, 'run_id': self.run_id, 'epoch': 0,
                'logical_boundary': boundary, 'row_id': item.row_id, 'skill': item.skill,
                'sample_k': sample_k, 'sample_seed': item.sample_seeds[sample_k]}
        return dict(body, logical_id=digest(body))
    def export(self):
        return {'run_id': self.run_id, 'seed': self.seed, 'epoch': 0, 'units': self.units,
                'synthetic': self.synthetic, 'execution_authorized': False, 'plan_sha256': self.sha256,
                'proposal_sha256': self.proposal_sha256, 'manifests_sha256': self.manifests_sha256,
                'boundaries_count': len(self.boundaries), 'prompts_count': self.prompts,
                'nominal_rollouts': self.nominal_rollouts, 'fixed_loss_denominator': B * G,
                'order_salt': ORDER_SALT, 'sample_salt': SAMPLE_SALT,
                'boundaries': [{'logical_boundary': b.index,
                    'prompts': [{'skill': p.skill, 'row_id': p.row_id, 'sample_seeds': list(p.sample_seeds)} for p in b.prompts]}
                    for b in self.boundaries]}

def build_plan(packet, manifests, run_id, *, synthetic_count=None):
    validate_packet(packet)
    require(run_id in ARMS, 'unknown run namespace')
    count = 3000 if synthetic_count is None else synthetic_count
    require(type(count) is int and count > 0 and count % B == 0, 'positive even stratum count')
    ordered, source_hashes = {}, {}
    seed = packet['proposed_seed_and_order_policy']['seed_assignment'][run_id]
    require(type(seed) is int and seed >= 0, 'seed identity')
    for skill in ARMS[run_id]:
        require(skill in manifests, 'missing atomic manifest')
        ids = validate_manifest(manifests[skill], skill, count)
        order = sorted(ids, key=lambda row: (hashlib.sha256((ORDER_SALT + run_id + '|' + str(seed) + '|' + row).encode('ascii')).hexdigest(), row))
        ordered[skill], source_hashes[skill] = order, digest(manifests[skill])
    if run_id == 'MT':
        require(not (set(ordered['R']) & set(ordered['S'])), 'duplicate row identity across MT strata')
        pairs = [(('R', r), ('S', s)) for r, s in zip(ordered['R'], ordered['S'])]
    else:
        skill = ARMS[run_id][0]
        pairs = [((skill, ordered[skill][i]), (skill, ordered[skill][i + 1])) for i in range(0, count, B)]
    boundaries = tuple(Boundary(i, tuple(Prompt(skill, row, tuple(sample_seed(run_id, seed, 0, i, row, k)
                      for k in range(G))) for skill, row in pair)) for i, pair in enumerate(pairs))
    body = {'run_id': run_id, 'seed': seed, 'epoch': 0, 'proposal_sha256': digest(packet),
            'manifests_sha256': source_hashes, 'pairs': pairs,
            'sample_seeds': [[list(p.sample_seeds) for p in b.prompts] for b in boundaries],
            'synthetic': synthetic_count is not None}
    return Plan(run_id, seed, synthetic_count is not None, boundaries, digest(body), source_hashes, digest(packet))

def maxrl_advantages(rewards):
    require(type(rewards) in (list, tuple) and len(rewards) == G
            and all(type(r) is int and r in (0, 1) for r in rewards), 'G8 strict binary rewards')
    mean = Fraction(sum(rewards), G)
    if mean in (0, 1): return (Fraction(0),) * G
    return tuple((r - mean) / (mean + Fraction(1, 1000000)) for r in rewards)

def boundary_loss(reward_groups, trajectory_logprob_sums):
    require(len(reward_groups) == len(trajectory_logprob_sums) == B, 'B2 fixed denominator')
    total = Fraction(0)
    for rewards, sums in zip(reward_groups, trajectory_logprob_sums):
        require(len(sums) == G, 'eight trajectory sums required')
        for advantage, value in zip(maxrl_advantages(rewards), sums):
            value = Fraction(str(value)) if not isinstance(value, Fraction) else value
            require(value <= 0, 'logprob sum must be nonpositive')
            total += advantage * value
    return -total / (B * G)

MIN_ACTIVE_METADATA = 64 * 1024 ** 2
METADATA_CLASSES = frozenset(('physical_ledger','checkpoint_json','checkpoint_journal','runner_receipt'))


def derive_caps(plan, prompt_lengths=None, prices=None, inventory_numel=None, *, unit_io_budget=None,metadata_census=None):
    """Derive finite compute and active-storage limits from bound unit evidence.

    W, the cumulative unit write ceiling, is already a rigorous upper for every
    owned serialized metadata byte: physical-ledger/header writes, attempts and
    results, and checkpoint journals/manifests are all charged to that ceiling.
    Missing/failed returns do not refund their admitted write reservations.
    Retained bytes cannot exceed bytes ever written. W also includes retired
    tensor-payload rewrites, so this safe bound can be impractically large.

    An optional separately bound metadata census can tighten this bound. Its
    per-class prospective write limits are enforced by the production durable
    ledger and every child/result writer. Reservations survive failed writes,
    cleanup and rollback. No numeric measurement or training grant is invented.
    """
    D, N, I = plan.prompts, len(plan.boundaries), 1 + RESUME_CAP
    trajectories = math.ceil((1 + RHO) * D * G)
    checkpoints = I * (1 + math.ceil(Fraction(N, CADENCE)) + 1 + 1 + 1)
    result = {'plan_sha256': plan.sha256, 'proposal_sha256': plan.proposal_sha256,
              'frozen_prompt_lengths': None, 'loss_forward_started': trajectories, 'loss_forward_returned': trajectories,
              'selected_loss_tokens_charged': CAP * trajectories, 'rollouts_started': trajectories, 'rollouts_returned': trajectories,
              'generated_tokens_charged': CAP * trajectories,
              'backward_started': trajectories, 'backward_returned': trajectories,
              'selected_backward_tokens_charged': CAP * trajectories,
              'optimizer_started': math.ceil((1 + RHO) * N),
              'optimizer_returned': math.ceil((1 + RHO) * N),
              'checkpoint_started': checkpoints, 'checkpoint_returned': checkpoints,
              'invocations_started': I, 'resumes': RESUME_CAP,
              'prefill_tokens_charged': None, 'wall_charged_s': None,
              'final_C': N, 'final_U_upper': N, 'nominal_rollouts': D * G,
              'canonical_generated_tokens_upper': CAP * D * G,
              'active_metadata_reserve_bytes': None, 'active_storage_reserve_bytes': None,
              'active_metadata_reserve_basis': None,'metadata_write_envelope':None, 'unresolved': []}
    if prompt_lengths is not None:
        ids = [p.row_id for b in plan.boundaries for p in b.prompts]
        require(set(prompt_lengths) == set(ids) and len(prompt_lengths) == D, 'missing/extra census rows')
        require(all(type(n) is int and n > 0 for n in prompt_lengths.values()), 'positive prompt census')
        extra = math.floor(RHO * D * G)
        result['prefill_tokens_charged'] = G * sum(prompt_lengths.values()) + extra * max(prompt_lengths.values())
        result['prompt_census_sha256'] = digest(prompt_lengths)
        result['frozen_prompt_lengths'] = dict(prompt_lengths)
    else: result['unresolved'].append('exact tokenizer prompt-length census')
    if prices is not None:
        names = ('t_gen95', 't_update95', 't_ckpt_upper', 't_init_upper', 't_restore_upper', 't_cleanup_upper', 't_finalize_upper')
        require(set(prices) == set(names), 'complete measured price fields required')
        values = {k: Fraction(str(v)) for k, v in prices.items()}
        require(all(v >= 0 for v in values.values()), 'nonnegative measured prices')
        proxy = B * values['t_gen95'] + B * values['t_update95']
        amount = Fraction(3, 2) * ((1 + RHO) * N * proxy + checkpoints * values['t_ckpt_upper']
                 + I * sum(values[k] for k in ('t_init_upper', 't_restore_upper', 't_cleanup_upper', 't_finalize_upper'))) + I * 900
        result['wall_charged_s'] = math.ceil(amount)
        result['prices_sha256'] = digest(prices)
    else: result['unresolved'].append('applicable measured timing upper prices')
    if unit_io_budget is not None:
        require(type(unit_io_budget) is dict and unit_io_budget.get('schema') == 'S1_IO_BUDGET-1'
                and unit_io_budget.get('measured') is True
                and unit_io_budget.get('plan_sha256') == plan.sha256,
                'bound measured/conservative unit IO census required')
        io_caps = unit_io_budget.get('caps')
        require(type(io_caps) is dict and type(io_caps.get('chunk_io_write_bytes')) is int
                and io_caps['chunk_io_write_bytes'] > 0, 'positive cumulative unit IO write ceiling')
        W = io_caps['chunk_io_write_bytes']
        result['active_metadata_reserve_bytes'] = max(MIN_ACTIVE_METADATA, W)
        result['active_metadata_reserve_basis'] = {
            'rule': 'max(64MiB, cumulative_unit_chunk_io_write_bytes)',
            'unit_io_budget_preimage_sha256': digest(unit_io_budget),
            'cumulative_unit_chunk_io_write_bytes': W,
            'scope': 'owned serialized metadata; conservative bound also includes retired tensor-payload writes',
            'smaller_metadata_only_reserve_status': 'UNQUALIFIED_WRITER_CENSUS_AND_ENFORCEMENT_REQUIRED'}
        if metadata_census is not None:
            require(type(metadata_census) is dict and metadata_census.get('schema')=='S1_METADATA_CENSUS-1'
                    and metadata_census.get('plan_sha256')==plan.sha256
                    and metadata_census.get('io_preimage_sha256')==digest({k:v for k,v in unit_io_budget.items() if k!='metadata_census'}),
                    'metadata census schema/plan/IO preimage drift')
            F,classes=metadata_census.get('fixed_bytes_upper'),metadata_census.get('classes')
            require(type(F) is int and F>=0 and type(classes) is dict and set(classes)==METADATA_CLASSES,
                    'metadata fixed allowance/class population')
            require(all(type(value) is dict and set(value)=={'writes_upper','serialized_bytes_upper'}
                        and all(type(value[key]) is int and value[key]>0 for key in value) for value in classes.values()),
                    'metadata class finite count/serialized bound')
            class_bytes=sum(value['writes_upper']*value['serialized_bytes_upper'] for value in classes.values())
            tighter=max(MIN_ACTIVE_METADATA,F+class_bytes)
            result['active_metadata_reserve_bytes']=min(result['active_metadata_reserve_bytes'],tighter)
            result['metadata_write_envelope']={'fixed_bytes_upper':F,'classes':json.loads(canonical(classes))}
            result['active_metadata_reserve_basis'].update(rule='min(W fallback, max(64MiB,fixed+sum(writes*bytes)))',
                metadata_census_preimage_sha256=digest(metadata_census),fixed_bytes_upper=F,class_bytes_upper=class_bytes,
                scope='owned metadata; all prospective writes classified and permanently charged before physical append',
                smaller_metadata_only_reserve_status='ENFORCED_METADATA_CENSUS')
    else:
        result['unresolved'].append('bound cumulative unit IO write ceiling for active metadata')
    if inventory_numel is not None:
        require(type(inventory_numel) is int and inventory_numel > 0, 'adopted inventory size')
        if unit_io_budget is not None:
            require((unit_io_budget.get('inventory') or {}).get('numel') == inventory_numel,
                    'unit IO census inventory size drift')
            result['active_storage_reserve_bytes'] = 28 * inventory_numel + result['active_metadata_reserve_bytes']
    else: result['unresolved'].append('qualified adopted inventory')
    return result

class Ledger:
    """In-memory physical-cost model. No persistence or native execution is claimed."""
    def __init__(self, plan, caps):
        require(caps.get('plan_sha256') == plan.sha256, 'cost ceiling belongs to another plan')
        self.plan, self.caps = plan, dict(caps)
        self.counters = {k: 0 for k in caps if k.endswith(('started', 'returned', 'charged', '_s')) or k == 'resumes'}
        self.pending_rollouts, self.rollouts, self.pending_ops, self.operations = {}, {}, {}, {}
        self.events, self.applied_updates, self.applied_rollouts = [], set(), set()
        self.wall_unknown = False
        self.generation = 0
        self.symbolic_snapshot = self.snapshot_identity(0, 0, [])
    def snapshot_identity(self, C, U, history, restore_digest=None):
        return digest({'plan_sha256': self.plan.sha256, 'generation': self.generation,
                       'C': C, 'U': U, 'history': history, 'restore_digest': restore_digest})
    def fence_after_restore(self, C, U, history, restore_digest):
        self.generation += 1  # never restored from an old checkpoint
        self.symbolic_snapshot = self.snapshot_identity(C, U, history, restore_digest)
    def _add(self, changes):
        if self.wall_unknown: raise UnresolvedBudget('unknown interrupted wall prevents certification of remaining allowance')
        updated = dict(self.counters)
        for key, amount in changes.items():
            require(key in updated and type(amount) in (int, float) and math.isfinite(amount), 'cost field/type')
            cap = self.caps[key]
            if cap is None: raise UnresolvedBudget('unresolved hard cap: ' + key)
            require(amount >= 0 or key == 'generated_tokens_charged', 'physical cumulative cost cannot be decremented')
            updated[key] += amount
            require(updated[key] >= 0, 'negative cumulative charge')
            if updated[key] > cap: raise CeilingRejected(key + ' cumulative hard ceiling reached')
        self.counters = updated
        self.events.append(dict(changes))
    def _budget_ready(self):
        if self.caps['prefill_tokens_charged'] is None or self.caps['wall_charged_s'] is None:
            raise UnresolvedBudget('unresolved prefill/wall caps; no physical-work allowance can be certified')
    def invocation(self, *, resume=False):
        self._budget_ready()
        require(resume or self.counters['invocations_started'] == 0, 'initial invocation already started')
        require(not resume or self.counters['invocations_started'] > 0, 'resume needs initial invocation')
        self._add(dict(invocations_started=1, **({'resumes': 1} if resume else {})))
    def charge_wall(self, seconds):
        require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0, 'finite invocation wall charge')
        self._add({'wall_charged_s': seconds})
    def unresolved_wall(self): self.wall_unknown = True
    def rollout_start(self, trajectory, prompt_tokens):
        require(self.counters['invocations_started'] > 0, 'invocation not started')
        self._budget_ready()
        boundary = trajectory['logical_boundary']
        require(type(boundary) is int and 0 <= boundary < len(self.plan.boundaries), 'logical boundary index')
        item = next((i for i, p in enumerate(self.plan.boundaries[boundary].prompts) if p.row_id == trajectory['row_id']), None)
        require(item is not None and trajectory == self.plan.trajectory(boundary, item, trajectory['sample_k']), 'trajectory/seed identity drift')
        require(type(prompt_tokens) is int and prompt_tokens > 0, 'declared prefill')
        require(self.caps['frozen_prompt_lengths'][trajectory['row_id']] == prompt_tokens, 'prefill differs from frozen census')
        self._add({'rollouts_started': 1, 'prefill_tokens_charged': prompt_tokens, 'generated_tokens_charged': CAP})
        attempt = uuid.uuid4().hex
        self.pending_rollouts[attempt] = dict(trajectory, prompt_tokens=prompt_tokens, generation=self.generation, symbolic_snapshot=self.symbolic_snapshot)
        return attempt
    def rollout_return(self, attempt, generated_tokens):
        require(attempt in self.pending_rollouts and type(generated_tokens) is int and 1 <= generated_tokens <= CAP, 'rollout return identity/count')
        self._add({'rollouts_returned': 1, 'generated_tokens_charged': generated_tokens - CAP})
        self.rollouts[attempt] = dict(self.pending_rollouts.pop(attempt), generated_tokens=generated_tokens)
    def op_start(self, operation, boundary=None, selected_tokens=0, full_sequence_tokens=0):
        self._budget_ready()
        require(self.counters['invocations_started'] > 0, 'invocation not started')
        require(operation in ('optimizer', 'loss_forward', 'backward', 'checkpoint'), 'operation kind')
        changes = {operation + '_started': 1}
        if operation in ('backward', 'loss_forward'):
            require(type(selected_tokens) is int and 1 <= selected_tokens <= CAP and type(full_sequence_tokens) is int
                    and full_sequence_tokens >= selected_tokens, 'backward token accounting')
            changes['selected_backward_tokens_charged' if operation == 'backward' else 'selected_loss_tokens_charged'] = selected_tokens
        if operation == 'optimizer': require(type(boundary) is int and 0 <= boundary < len(self.plan.boundaries), 'optimizer boundary')
        self._add(changes)
        attempt = uuid.uuid4().hex
        self.pending_ops[attempt] = {'operation': operation, 'boundary': boundary,
                                     'selected_tokens': selected_tokens, 'full_sequence_tokens': full_sequence_tokens,
                                     'generation': self.generation, 'symbolic_snapshot': self.symbolic_snapshot}
        return attempt
    def op_return(self, attempt):
        require(attempt in self.pending_ops, 'missing/duplicate operation return')
        operation = self.pending_ops[attempt]['operation']
        self._add({operation + '_returned': 1})
        self.operations[attempt] = self.pending_ops.pop(attempt)
    def report(self):
        return {'generation': self.generation, 'symbolic_snapshot': self.symbolic_snapshot,
                'snapshot_scope': 'logical metadata only; no native weight/Adam/RNG identity claimed',
                'cumulative': dict(self.counters), 'unsealed_rollouts': len(self.pending_rollouts),
                'unsealed_generation_charge_rule': 'each pending attempt retains full128-token reservation and declared prefill',
                'observed_generated_tokens': sum(r['generated_tokens'] for r in self.rollouts.values()),
                'unknown_generated_tokens_upper': CAP * len(self.pending_rollouts),
                'full_backward_sequence_tokens_recorded': sum(r['full_sequence_tokens'] for r in list(self.operations.values()) + list(self.pending_ops.values()) if r['operation'] == 'backward'),
                'unknown_wall': self.wall_unknown, 'execution_authorized': False}

class Progress:
    """Logical C/U prototype; fresh work is fenced after every restored snapshot.

    The symbolic snapshot binds counter/history state, not native parameters.
    A production adapter still has to bind actual weights, Adam and RNG.
    """
    CHECKPOINT_FIELDS = {'plan_sha256', 'C', 'U', 'history', 'Adam_initialized',
                         'Adam_scalar_steps', 'native_state_saved', 'physical_counters_rewound'}
    def __init__(self, plan, ledger):
        self.plan, self.ledger = plan, ledger
        self.C, self.U, self.history = 0, 0, []
    def commit(self, reward_groups, rollout_attempts, optimizer_attempt=None):
        require(self.C < len(self.plan.boundaries), 'plan already complete')
        require(len(reward_groups) == B and len(rollout_attempts) == B * G
                and len(set(rollout_attempts)) == B * G, 'missing/duplicate boundary trajectories')
        expected = {self.plan.trajectory(self.C, prompt, k)['logical_id'] for prompt in range(B) for k in range(G)}
        require(not (set(rollout_attempts) & self.ledger.applied_rollouts),
                'rolled-back samples must be reattempted; physical costs retained')
        returned = [self.ledger.rollouts.get(attempt) for attempt in rollout_attempts]
        require(all(record is not None for record in returned), 'missing/foreign logical samples at boundary')
        require(all(record['generation'] == self.ledger.generation
                    and record['symbolic_snapshot'] == self.ledger.symbolic_snapshot for record in returned),
                'stale generation/snapshot; samples must be reattempted after restore')
        require({record['logical_id'] for record in returned} == expected, 'missing/foreign logical samples at boundary')
        signal = any(any(maxrl_advantages(group)) for group in reward_groups)
        if signal:
            record = self.ledger.operations.get(optimizer_attempt)
            require(record is not None and record['generation'] == self.ledger.generation
                    and record['symbolic_snapshot'] == self.ledger.symbolic_snapshot,
                    'stale optimizer generation/snapshot; a fresh weight update is required')
            require(optimizer_attempt not in self.ledger.applied_updates
                    and record['operation'] == 'optimizer' and record['boundary'] == self.C
                    and record['selected_tokens'] == record['full_sequence_tokens'] == 0,
                    'one fresh returned optimizer call required')
            self.ledger.applied_updates.add(optimizer_attempt)
            self.U += 1
        else:
            require(optimizer_attempt is None, 'all-uniform boundary must skip Adam')
        self.ledger.applied_rollouts.update(rollout_attempts)
        self.history.append({'boundary': self.C,
                             'row_ids': [prompt.row_id for prompt in self.plan.boundaries[self.C].prompts],
                             'updated': signal})
        self.C += 1
        require(0 <= self.U <= self.C, 'C/U invariant')
        self.ledger.symbolic_snapshot = self.ledger.snapshot_identity(self.C, self.U, self.history)
    def _validate_checkpoint_body(self, body):
        require(type(body) is dict and set(body) == self.CHECKPOINT_FIELDS, 'checkpoint metadata schema')
        require(body['plan_sha256'] == self.plan.sha256, 'checkpoint plan identity')
        C, U, history = body['C'], body['U'], body['history']
        require(type(C) is int and type(U) is int and 0 <= U <= C <= self.C
                and C <= len(self.plan.boundaries), 'checkpoint C/U ranges/types')
        require(type(history) is list and len(history) == C, 'checkpoint history length')
        for index, entry in enumerate(history):
            require(type(entry) is dict and set(entry) == {'boundary', 'row_ids', 'updated'}, 'checkpoint history entry schema')
            require(type(entry['boundary']) is int and entry['boundary'] == index
                    and type(entry['row_ids']) is list
                    and entry['row_ids'] == [prompt.row_id for prompt in self.plan.boundaries[index].prompts],
                    'checkpoint history differs from exact plan/order')
            require(type(entry['updated']) is bool, 'checkpoint history updated must be a strict boolean')
        require(U == sum(entry['updated'] for entry in history), 'checkpoint U differs from updated history count')
        require(type(body['Adam_initialized']) is bool and body['Adam_initialized'] == (U > 0)
                and type(body['Adam_scalar_steps']) is int and body['Adam_scalar_steps'] == U,
                'checkpoint Adam metadata differs from U')
        require(body['native_state_saved'] is False and body['physical_counters_rewound'] is False,
                'prototype cannot assert native state or rewind physical costs')
        require(history == self.history[:C], 'checkpoint history is not an ancestor of the applied logical history')
    def checkpoint(self):
        body = {'plan_sha256': self.plan.sha256, 'C': self.C, 'U': self.U,
                'history': json.loads(canonical(self.history)), 'Adam_initialized': self.U > 0,
                'Adam_scalar_steps': self.U, 'native_state_saved': False, 'physical_counters_rewound': False}
        self._validate_checkpoint_body(body)
        return dict(body, checkpoint_sha256=digest(body))
    def rollback(self, checkpoint):
        require(type(checkpoint) is dict and 'checkpoint_sha256' in checkpoint, 'checkpoint schema')
        body = {key: value for key, value in checkpoint.items() if key != 'checkpoint_sha256'}
        require(digest(body) == checkpoint['checkpoint_sha256'], 'checkpoint binding/digest')
        self._validate_checkpoint_body(body)  # all checks precede any live mutation/fence
        self.C, self.U = body['C'], body['U']
        self.history = json.loads(canonical(body['history']))
        self.ledger.fence_after_restore(self.C, self.U, self.history, checkpoint['checkpoint_sha256'])
        self.ledger.events.append({'rollback': True, 'restored_C': self.C, 'restored_U': self.U,
                                   'new_generation': self.ledger.generation,
                                   'new_symbolic_snapshot': self.ledger.symbolic_snapshot,
                                   'retained_physical_counters': dict(self.ledger.counters),
                                   'retained_records': 'pending and returned records remain cost evidence; older generation/snapshot is inadmissible'})
    def complete(self):
        self.checkpoint()  # validates history/C/U/Adam metadata, not native state
        return self.C == len(self.plan.boundaries) and not self.ledger.wall_unknown
