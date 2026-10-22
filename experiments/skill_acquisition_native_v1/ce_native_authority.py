"""Source-bound native CE capabilities. There is no registered issuer today.

This module imports no model/framework, performs no IO, and grants no scope.
A reviewed source change must register the actual controller verifier before
it can issue a permit for separately authorized qualification or training.
Request flags, callback injection and fixture objects cannot do that.
"""
from __future__ import annotations

import copy
import hashlib
import json

SCHEMA = 'CE8-NATIVE-EXECUTION-PERMIT-1'
CONTEXT = 'CE8-OUTER-ADMISSION-CONTEXT-1'
STAGES = frozenset(('backend_construct', 'module_import', 'runtime_anchor',
    'ce_boundary', 'checkpoint_construct', 'checkpoint_save', 'checkpoint_restore',
    'refresh_policy', 'donor_publish_verify', 'delta_roundtrip', 'cleanup',
    'journal_construct', 'journal_operation', 'journal_restore', 'completion_publication',
    'ingestion_tokenizer', 'longest_masks', 'likelihood_gradient_parity',
    'adam_clipping', 'checkpoint_resume', 'resource_io_measurement'))
_REGISTERED_NATIVE_AUTHORITY_VERIFIERS = frozenset()
_SEAL = object()
_EXTERNAL_CONTEXT_FIELDS = frozenset(('native_execution_permit', 'native_services'))


class NativeAdmissionError(ValueError): pass


def require(condition, message):
    if not condition: raise NativeAdmissionError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def _hex(value): return type(value) is str and len(value) == 64 and not set(value) - set('0123456789abcdef')


def context_preimage(context):
    """Freeze all JSON admission evidence; bind service objects separately."""
    require(type(context) is dict, 'explicit native admission context required')
    return {key: value for key, value in context.items() if key not in _EXTERNAL_CONTEXT_FIELDS}


def _context(context):
    require(type(context) is dict and context.get('schema') == CONTEXT
            and context.get('mode') == 'registered_scope'
            and context.get('admission_evidence_verified') is True
            and context.get('phase') in ('qualification', 'training')
            and context.get('arm') in ('canonical', 'permuted'), 'registered phase/arm evidence required')
    require(all(_hex(context.get(key)) for key in
            ('request_sha256', 'scope_sha256', 'static_recipe_sha256')), 'exact request/scope/recipe identity required')
    bindings = context.get('bindings')
    require(type(bindings) is dict and all(type(bindings.get(key)) is dict for key in
            ('scope', 'recipe', 'source', 'runtime', 'anchors')), 'complete native binding evidence required')
    recipe, scope = bindings['recipe'].get('recipe'), bindings['scope']
    require(type(recipe) is dict and recipe.get('synthetic_cpu') is False
            and digest(recipe) == context['static_recipe_sha256'] == scope.get('static_recipe_sha256'),
            'native static recipe binding drift')
    code = bindings['source'].get('code_sha256')
    require(type(code) is dict and code and all(type(k) is str and k and _hex(v) for k, v in code.items()),
            'verified native implementation source map required')
    runtime = bindings['runtime'].get('decoding_preimage')
    require(type(runtime) is dict and runtime.get('device') == 'cuda:0'
            and runtime.get('dtype') == 'float32'
            and digest(runtime) == bindings['runtime'].get('decoding_sha256'), 'native runtime binding drift')
    anchors = bindings['anchors']
    require(anchors.get('common_anchor_sha256') == anchors.get('own_start_sha256')
            and all(_hex(anchors.get(key)) for key in
            ('common_anchor_sha256', 'own_start_sha256', 'base_weights_sha256', 'warm_delta_sha256', 'tokenizer_receipt_sha256')),
            'native independent common-warm anchor binding')
    stages = scope.get('native_execution_stages')
    require(type(stages) is list and stages and all(type(stage) is str and stage in STAGES for stage in stages)
            and len(set(stages)) == len(stages), 'explicit native stage allowlist required')
    if context['phase'] == 'qualification':
        require(scope.get('role') == 'qualification_only' and scope.get('qualification_authorized') is True
                and scope.get('production_training_authorized') is False
                and context.get('production_training_authorized') is False, 'qualification is not production training')
    else:
        require(scope.get('role') == 'production_training' and scope.get('production_training_authorized') is True
                and context.get('production_training_authorized') is True
                and context.get('qualifiedreceipt_verified') is True, 'production scope plus verified qualification required')
    canonical(context_preimage(context))
    return context


class NativeExecutionPermit:
    """Opaque, mutation-detecting capability for one admitted native context.

    The issuer and verifier are registered controller objects, not caller
    choices. Every use re-verifies their source-owned identity and exact frozen
    evidence. The current empty registry rejects even plausible forged inputs.
    """
    __slots__ = ('_seal', '_context', '_context_sha256', '_issuer', '_verifier', '_operation', '_gate')

    def __init__(self, context, issuer, verify_authority, *, operation, gate):
        # Deny before traversing request/service objects or any deferred import.
        require(any(verify_authority is value for value in _REGISTERED_NATIVE_AUTHORITY_VERIFIERS),
                'no registered native CE authority verifier; no native permit can be issued')
        require(issuer is not None and not isinstance(issuer, (dict, list, str, int, float, bool)),
                'opaque registered controller issuer required')
        require(callable(operation) and callable(gate), 'bound native observer/resource gate required')
        _context(context)
        preimage = copy.deepcopy(context_preimage(context))
        identity = digest(preimage)
        require(verify_authority(issuer, identity, context['request_sha256'], context['scope_sha256'],
                context['phase'], context['arm']) is True, 'registered native controller authority failed')
        for name, value in (('_seal', _SEAL), ('_context', preimage), ('_context_sha256', identity),
                ('_issuer', issuer), ('_verifier', verify_authority), ('_operation', operation), ('_gate', gate)):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise NativeAdmissionError('native CE permit fields are immutable')

    def _verify(self):
        require(getattr(self, '_seal', None) is _SEAL
                and any(getattr(self, '_verifier', None) is value for value in _REGISTERED_NATIVE_AUTHORITY_VERIFIERS),
                'native CE permit was not issued by a registered controller')
        _context(self._context)
        require(digest(self._context) == self._context_sha256, 'native CE permit evidence mutated')
        require(self._verifier(self._issuer, self._context_sha256, self._context['request_sha256'],
                self._context['scope_sha256'], self._context['phase'], self._context['arm']) is True,
                'native CE authority no longer valid')

    @property
    def context(self):
        self._verify()
        return copy.deepcopy(self._context)

    def require(self, stage, *, static_recipe_sha256, operation=None, gate=None):
        self._verify()
        require(type(stage) is str and stage in self._context['bindings']['scope']['native_execution_stages'],
                'stage outside admitted native CE scope')
        require(static_recipe_sha256 == self._context['static_recipe_sha256'], 'native static recipe changed')
        require(operation is None or operation is self._operation, 'native accounting callback changed')
        require(gate is None or gate is self._gate, 'native resource callback changed')
        return {'schema': SCHEMA, 'phase': self._context['phase'], 'arm': self._context['arm'],
            'request_sha256': self._context['request_sha256'], 'scope_sha256': self._context['scope_sha256'],
            'static_recipe_sha256': static_recipe_sha256, 'context_sha256': self._context_sha256}


def require_execution_permit(permit, context, stage, *, recipe_sha256=None, operation=None, gate=None):
    require(type(permit) is NativeExecutionPermit, 'opaque source-issued native CE permit required before work')
    permit._verify()
    _context(context)
    require(digest(context_preimage(context)) == permit._context_sha256, 'native backend/admission context drift')
    if recipe_sha256 is None: recipe_sha256 = context['static_recipe_sha256']
    return permit.require(stage, static_recipe_sha256=recipe_sha256, operation=operation, gate=gate)
