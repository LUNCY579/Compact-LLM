"""Permanent, bounded metadata prepayment before the service ledger exists.

The existing ControllerServicePermit admits the first metadata reservation.
This source body grants no model, launch, or experiment authority. Registries
remain closed until the exact source and separate scope have been reviewed.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import time

from . import ce_native_services as services

SCHEMA = 'CE8-SERVICE-CONSTRUCTOR-PREPAYMENT-1'
_SEAL = object()
require = services.require


class ServiceConstructorPrepayment:
    def __init__(self, request, spec, permit):
        started = time.monotonic()
        services._admit(permit, request, spec)  # precedes paths and serialization
        require(any(type(self) is cls for cls in services._REGISTERED_BOOTSTRAP_PREPAYMENT_TYPES),
                'service constructor prepayment type is not registered')
        policy = spec.get('constructor_prepayment')
        require(type(policy) is dict and set(policy) == {'schema', 'owned_directory', 'bounds', 'source_verifier_bounds'}
                and policy['schema'] == SCHEMA, 'explicit constructor metadata prepayment policy')
        verification = policy['source_verifier_bounds']
        require(type(verification) is dict and set(verification) == {'hash_attempts', 'hash_bytes'}
                and type(verification['hash_attempts']) is int and verification['hash_attempts'] >= 5
                and type(verification['hash_bytes']) is int and verification['hash_bytes'] >= 32 * 1024 ** 2,
                'explicit fixed source verifier/configuration hash envelope')
        bounds = policy['bounds']
        require(type(bounds) is dict and set(bounds) == set(services.COUNTERS)
                and all(type(bounds[k]) is int and bounds[k] > 0 for k in services.COUNTERS if k != 'wall_seconds')
                and type(bounds['wall_seconds']) in (int, float) and 0 < bounds['wall_seconds'] < float('inf'),
                'finite complete constructor prepayment bounds')
        directory = Path(policy['owned_directory'])
        require(directory.is_absolute() and directory == Path(spec['bootstrap']['owned_directory']),
                'constructor reservation must own the exact service ledger directory')
        services._profile(spec['bootstrap']['setup_bounds'])
        setup = spec['bootstrap']['setup_bounds']
        needed = {'read_bytes': setup['read_bytes_upper'], 'write_bytes': setup['write_bytes_upper'],
                  'hash_attempts': setup['identity_hash_attempts_upper'],
                  'hash_bytes': setup['identity_hash_bytes_upper'], 'wall_seconds': setup['wall_s_upper']}
        require(all(bounds[k] >= v for k, v in needed.items()), 'constructor prepayment below dependent setup work')
        self._seal, self.request, self.spec, self.permit = _SEAL, copy.deepcopy(request), copy.deepcopy(spec), permit
        request_preimage = services.controller.request_preimage(request)
        self.request_sha256 = services.digest(request_preimage)
        self.service_spec_sha256, self.bootstrap_spec_sha256 = services.digest(spec), services.digest(spec['bootstrap'])
        self.bounds, self.directory = copy.deepcopy(bounds), directory
        self.path, self.started, self.consumed = directory / 'CONSTRUCTOR_PREPAYMENT.json', started, False
        self.validation_calls = 2  # factory and constructor service checks
        record = {'schema': SCHEMA, 'request_sha256': self.request_sha256,
                  'service_spec_sha256': self.service_spec_sha256, 'bootstrap_spec_sha256': self.bootstrap_spec_sha256,
                  'bounds': self.bounds, 'dependent_setup_bounds': copy.deepcopy(setup),
                  'status': 'PERMANENTLY_RESERVED', 'metadata_only': True,
                  'native_qualified': False, 'execution_verified': False}
        raw = services.canonical(record) + b'\n'
        setup_body = {'schema': services.BOOTSTRAP, 'seq': 0, 'previous': None,
                      'request_sha256': self.request_sha256, 'kind': 'setup',
                      'payload': {'setup_bounds': setup}}
        setup_preimage_bytes = len(services.canonical(setup_body))
        setup_row_bytes = len(services.canonical(dict(setup_body, sha256='0' * 64))) + 1
        request_bytes = len(services.canonical(request_preimage))
        spec_bytes, bootstrap_bytes = len(services.canonical(spec)), len(services.canonical(spec['bootstrap']))
        # Each bounded service validation hashes request+spec. Also include
        # the three constructor digests, later mutation check, saved record,
        # and the dependent ledger's actual setup event above its setup price.
        validation_hash_bytes = bounds['operation_attempts'] * (request_bytes + spec_bytes + verification['hash_bytes'])
        own_hash_bytes = request_bytes + spec_bytes + bootstrap_bytes + len(raw) + bounds['operation_attempts'] * spec_bytes
        # The entire write/hash/receipt envelope is reserved before mkdir/open.
        require(bounds['write_bytes'] >= setup['write_bytes_upper'] + setup_row_bytes + len(raw)
                and bounds['metadata_bytes'] >= setup_row_bytes + len(raw)
                and bounds['hash_bytes'] >= setup['identity_hash_bytes_upper'] + setup_preimage_bytes
                    + own_hash_bytes + validation_hash_bytes
                and bounds['hash_attempts'] >= setup['identity_hash_attempts_upper'] + 5
                    + (3 + verification['hash_attempts']) * bounds['operation_attempts']
                and bounds['operation_attempts'] >= 4, 'constructor reservation does not cover its own metadata')
        require(not directory.is_symlink() and not os.path.lexists(self.path), 'regular create-only constructor reservation')
        directory.mkdir(parents=True, exist_ok=True)
        require(not directory.is_symlink(), 'constructor owner changed during creation')
        with self.path.open('xb') as stream:
            require(stream.write(raw) == len(raw), 'short constructor prepayment write')
            stream.flush(); os.fsync(stream.fileno())
        self.reference = {'path': str(self.path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        require(time.monotonic() - self.started <= bounds['wall_seconds'], 'constructor reservation wall ceiling')

    def require_constructor(self, bootstrap_spec_sha256, request_sha256):
        require(type(self) is ServiceConstructorPrepayment and self._seal is _SEAL
                and any(type(self) is cls for cls in services._REGISTERED_BOOTSTRAP_PREPAYMENT_TYPES),
                'source-owned service constructor prepayment required')
        require(not self.consumed and bootstrap_spec_sha256 == self.bootstrap_spec_sha256
                and request_sha256 == self.request_sha256, 'one exact permanently reserved service constructor')
        self._validate_service()
        require(services.digest(self.spec) == self.service_spec_sha256,
                'one exact permanently reserved service constructor')
        require(time.monotonic() - self.started <= self.bounds['wall_seconds'], 'constructor prepayment expired')
        self.consumed = True  # dependent failure never makes the reservation reusable
        return copy.deepcopy(self.reference)

    def _validate_service(self):
        require(self.validation_calls < self.bounds['operation_attempts'], 'constructor validation reservation exhausted')
        self.validation_calls += 1  # failed checks retain their full charge
        services._admit(self.permit, self.request, self.spec)

    @property
    def evidence(self):
        require(type(self) is ServiceConstructorPrepayment and self._seal is _SEAL,
                'exact constructor reservation owner')
        self._validate_service()
        return {'schema': SCHEMA, 'reservation_ref': copy.deepcopy(self.reference),
                'bounds': copy.deepcopy(self.bounds), 'request_sha256': self.request_sha256,
                'service_spec_sha256': self.service_spec_sha256,
                'bootstrap_spec_sha256': self.bootstrap_spec_sha256, 'consumed': self.consumed,
                'metadata_only': True, 'native_qualified': False, 'execution_verified': False}


def bootstrap_constructor_prepayment(request, spec, permit):
    services._admit(permit, request, spec)
    descriptor = spec.get('role_descriptors', {}).get('bootstrap_constructor_prepayment')
    require(services.resolve_role(descriptor, spec.get('approved_code_sha256', {})) is bootstrap_constructor_prepayment,
            'exact source constructor prepayment factory required')
    return ServiceConstructorPrepayment(request, spec, permit)
