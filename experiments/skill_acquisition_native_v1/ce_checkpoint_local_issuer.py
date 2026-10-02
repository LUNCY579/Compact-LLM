"""Bounded child-side RPC and concrete post-launch scope derivation.

No native entry provider or authority is installed. The CLI refuses before
arguments, environment, files or OS work. A future source entry provider must
own prepayment and an actual current WinWorkerHandle before this body reads the
immutable request. JSON contains evidence only, never permits or callables.
"""
from __future__ import annotations

from contextlib import redirect_stdout

import copy
import hashlib
import os
from pathlib import Path
import queue
import sys
import threading
import time
import uuid

from . import ce_checkpoint_worker as child, ce_process_launch as launch
from . import ce_native_authority as authority, ce_worker_evidence as worker
from . import ce_qualification_artifacts as raw

SPEC = 'CE8-CHECKPOINT-LOCAL-RPC-SPEC-1'
FIXTURE = 'CE8-CHECKPOINT-LOCAL-RPC-FIXTURE-1'
# Source constants, deliberately closed. There is no JSON-selected provider.
_SOURCE_ENTRY_PROVIDER = None
_AUTHORITY_VERIFIER = None
_BOOTSTRAP_ISSUER = None
_BOOTSTRAP_VERIFIER = None
_PREDECESSOR_VERIFIER = None
_PUBLICATION_ISSUER = None
require, digest, canonical = authority.require, raw.digest, raw.canonical


def validate_spec(spec):
    require(type(spec) is dict and spec.get('schema') == SPEC, 'explicit source-owned child RPC policy')
    for key in ('deadline_seconds', 'rpc_seconds'):
        require(type(spec.get(key)) in (int, float) and 0 < spec[key] < float('inf'), 'finite child RPC ' + key)
    for key in ('max_message_bytes', 'max_messages', 'max_request_bytes'):
        require(type(spec.get(key)) is int and spec[key] > 0, 'finite child RPC ' + key)
    require(spec['max_message_bytes'] <= 1024 ** 2 and spec['max_request_bytes'] <= raw.MAX_JSON
            and spec['max_messages'] >= 5, 'bounded request/RPC ceilings and full ready/run/finish budget')
    bounds = spec.get('operation_bounds')
    require(type(bounds) is dict and set(bounds) == {'read_bytes_upper', 'write_bytes_upper',
            'identity_hash_attempts_upper', 'identity_hash_bytes_upper', 'wall_s_upper'}
            and all(type(value) in (int, float) and 0 < value < float('inf') for value in bounds.values())
            and all(type(bounds[key]) is int for key in bounds if key != 'wall_s_upper'), 'finite complete child RPC prepayment')
    require(bounds['read_bytes_upper'] >= max(spec['max_message_bytes'], spec['max_request_bytes']) + 1
            and bounds['write_bytes_upper'] >= spec['max_message_bytes']
            and bounds['identity_hash_attempts_upper'] >= 32
            and bounds['identity_hash_bytes_upper'] >= 64 * spec['max_request_bytes'],
            'RPC/source serialization, parsing and full context hash envelope must be prepaid')
    return spec


def _paid(operation, bounds, kind, work):
    identifier, start = uuid.uuid4().hex, time.monotonic()
    operation(kind, 'started', operation_id=identifier, **bounds)
    try: result = work()
    except BaseException as error:
        try: operation(kind, 'failed', operation_id=identifier, wall_s=time.monotonic() - start,
                       error={'type': type(error).__name__, 'message': str(error)[:1000]})
        except BaseException as failure:
            raw._accounting(error, failure); raise error from failure
        raise
    operation(kind, 'returned', operation_id=identifier, wall_s=time.monotonic() - start)
    return result


def entry_lifetime(payment, body, cleanup=None):
    """Retain every entry failure, including reads before ready and finish IO."""
    try:
        try:
            payment.begin()
            result = body()
        except BaseException as error:
            if cleanup is not None:
                try: cleanup()
                except BaseException as failure:
                    error.entry_cleanup_failure = {'type': type(failure).__name__, 'message': str(failure)[:1000]}
                    raise error from failure
            raise
        if cleanup is not None: cleanup()
        payment.complete()
    except BaseException as error:
        try: payment.fail(error)
        except BaseException as failure:
            raw._accounting(error, failure); raise error from failure
        raise
    return result


def read_json(reference, owned_root, operation, bounds, maximum):
    """Bounded exact-byte read/hash/parse, with reservation before metadata IO."""
    require(type(reference) is dict and set(reference) == {'path', 'sha256', 'bytes'}
            and type(reference['path']) is str and Path(reference['path']).is_absolute()
            and raw._hex(reference['sha256']) and raw._int(reference['bytes'], 1)
            and reference['bytes'] <= maximum <= raw.MAX_JSON, 'exact bounded local source reference')
    require(bounds['read_bytes_upper'] >= reference['bytes'] + 1
            and bounds['identity_hash_attempts_upper'] >= 1
            and bounds['identity_hash_bytes_upper'] >= reference['bytes'], 'prepaid complete local source read/hash')
    def read():
        root, path = Path(owned_root), Path(reference['path'])
        require(root.is_absolute() and not root.is_symlink() and not path.is_symlink()
                and path.resolve().is_relative_to(root.resolve()) and path.is_file()
                and path.stat().st_size == reference['bytes'], 'owned regular local source path/size')
        with path.open('rb') as stream: content = stream.read(reference['bytes'] + 1)
        require(len(content) == reference['bytes'] and hashlib.sha256(content).hexdigest() == reference['sha256'],
                'local source bytes/hash drift')
        value = raw._json(content)
        require(type(value) is dict, 'local source JSON object')
        return value
    return _paid(operation, bounds, 'input_io', read)


def read_delegation(path, context, operation, bounds, maximum):
    """Observed file has no prelaunch hash; derive its exact ref while prepaid."""
    branch = context['checkpoint_child_declaration']['branch']
    expected = Path(context['bindings']['scope']['checkpoint_qualification']['owned_directory']) / 'children' / branch / 'LEASE_DELEGATION.json'
    require(type(path) is str and Path(path) == expected, 'exact source-derived child delegation path')
    require(bounds['read_bytes_upper'] >= maximum + 1 and bounds['identity_hash_bytes_upper'] >= maximum
            and bounds['identity_hash_attempts_upper'] >= 1, 'full delegation read/hash prepayment')
    def read():
        root = Path(context['bindings']['scope']['native_backend']['output_directory'])
        require(root.is_absolute() and not root.is_symlink() and not expected.is_symlink()
                and expected.resolve().is_relative_to(root.resolve()) and expected.is_file(), 'owned observed delegation file')
        with expected.open('rb') as stream: content = stream.read(maximum + 1)
        require(0 < len(content) <= maximum and expected.stat().st_size == len(content), 'bounded stable delegation bytes')
        record = raw._json(content); require(type(record) is dict, 'observed delegation JSON object')
        return record, {'path': str(expected), 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
    return _paid(operation, bounds, 'input_io', read)


def derive_context(common, request, reference, proof, delegation, delegation_ref):
    """Pure evidence linkage. This returns JSON and issues no capability."""
    from . import ce_qualification_accounting as accounting, ce_bootstrap_accounting as boot
    from . import ce_checkpoint_publication as publication, ce_checkpoint_lease as lease
    context = copy.deepcopy(authority.context_preimage(common))
    forbidden = {'checkpoint_child_declaration', 'checkpoint_child_observed_lineage',
        'checkpoint_child_accounting', 'checkpoint_child_bootstrap', 'checkpoint_child_publication',
        'checkpoint_child_lease_delegation', 'checkpoint_child_delegation_ref', 'checkpoint_child_inputs'}
    require(not forbidden & set(context), 'common study context cannot predefine observed child scope')
    declaration = request.get('lineage_declaration')
    context['checkpoint_child_declaration'] = copy.deepcopy(declaration)
    child.validate_request(request, context)
    inputs = request.get('initializer_inputs')
    require(type(inputs) is dict and set(inputs) == ({'parent_binding', 'prefix_store'} if request['branch'] == 'resumed' else {'parent_binding'})
            and type(inputs.get('parent_binding')) is dict, 'exact retained parent initializer inputs')
    parent = declaration.get('parent_proof')
    require(type(parent) is dict and type(parent.get('os_identity')) is dict
            and type(proof) is dict and proof.get('schema') == worker.PROOF
            and proof.get('sources', {}).get('request_ref') == reference
            and proof.get('request_sha256') == common['request_sha256'] and proof.get('scope_sha256') == common['scope_sha256']
            and type(proof.get('os_identity')) is dict
            and proof['os_identity']['pid'] != parent['os_identity']['pid']
            and proof['os_identity']['boot_identity'] == parent['os_identity']['boot_identity']
            and {name: ref['sha256'] for name, ref in proof['sources']['source_files'].items()} == common['bindings']['source']['code_sha256']
            and digest(worker.worker_identity_preimage(proof['os_identity'], proof['sources'],
                  common['request_sha256'], common['scope_sha256'])) == proof.get('worker_identity_sha256'),
            'independently observed current child/source/raw request identity')
    expected_path = Path(common['bindings']['scope']['checkpoint_qualification']['owned_directory']) / 'children' / request['branch'] / 'LEASE_DELEGATION.json'
    require(request.get('lease_delegation_path') == str(expected_path) and delegation_ref.get('path') == str(expected_path)
            and delegation_ref.get('sha256') == hashlib.sha256(canonical(delegation) + b'\n').hexdigest()
            and delegation_ref.get('bytes') == len(canonical(delegation)) + 1, 'canonical observed delegation reference')
    require(type(delegation) is dict and delegation.get('schema') == lease.DELEGATION
            and delegation.get('branch') == request['branch'] and delegation.get('request_sha256') == common['request_sha256']
            and delegation.get('scope_sha256') == common['scope_sha256']
            and delegation.get('child_worker_identity_sha256') == proof['worker_identity_sha256']
            and delegation.get('child_os_identity') == proof['os_identity']
            and delegation.get('child_request_ref') == reference
            and delegation.get('parent_worker_identity_sha256') == parent.get('worker_identity_sha256')
            and delegation.get('parent_os_identity') == parent['os_identity']
            and declaration.get('parent_worker_identity_sha256') == parent.get('worker_identity_sha256')
            and delegation.get('lock_path') == common['bindings']['scope']['checkpoint_lease']['lock_path'],
            'observed original-supervisor/current-child delegation linkage')
    observed = dict(copy.deepcopy(declaration), schema=accounting.CHILD_PREFIX,
        worker_identity_sha256=proof['worker_identity_sha256'], child_request_sha256=reference['sha256'],
        child_request_ref=copy.deepcopy(reference), child_proof=copy.deepcopy(proof), declaration_sha256=digest(declaration))
    context.update(checkpoint_child_observed_lineage=observed,
        checkpoint_child_inputs=dict(copy.deepcopy(inputs), parent_context=copy.deepcopy(common)),
        checkpoint_child_lease_delegation=copy.deepcopy(delegation), checkpoint_child_delegation_ref=copy.deepcopy(delegation_ref),
        worker_proof_sha256=digest(proof), worker_identity_sha256=proof['worker_identity_sha256'])
    root = expected_path.parent; scope = context['bindings']['scope']
    spec = dict(accounting.accounting_specification(common), owned_directory=str(root / 'ledger'))
    context['checkpoint_child_accounting'] = {'specification': spec, 'owner': {
        'accounting_spec_sha256': digest(spec), 'prepared_receipt_sha256': spec['ingestion_receipt_sha256']}}
    context['checkpoint_child_bootstrap'] = dict(scope['qualifier_bootstrap'], owned_directory=str(root / 'bootstrap'))
    context['checkpoint_child_publication'] = dict(scope['checkpoint_branch_publication'], owned_directory=str(root / 'published'))
    accounting.accounting_specification(context); boot.bootstrap_specification(context); publication.publication_specification(context)
    child.validate_observed_lineage(request, context, proof)
    return context


def validate_parent_view(actual, retained):
    """Same OS/source identity permits different informational observer views."""
    require(type(actual) is dict and type(retained) is dict
            and all(actual.get(key) == retained.get(key) for key in
                    ('worker_identity_sha256', 'request_sha256', 'scope_sha256')), 'actual retained parent worker bindings')
    left = worker.worker_identity_preimage(actual['os_identity'], actual['sources'], actual['request_sha256'], actual['scope_sha256'])
    right = worker.worker_identity_preimage(retained['os_identity'], retained['sources'], retained['request_sha256'], retained['scope_sha256'])
    require(left == right and digest(left) == actual['worker_identity_sha256'], 'actual retained parent OS/source identity drift')
    return actual


class BoundedRPC:
    """Actual bounded pipe protocol; fixtures use streams without processes."""
    def __init__(self, spec, binding, pid, input_stream, output_stream, operation):
        validate_spec(spec)
        require(type(binding) is dict and set(binding) == {'session_id', 'request_sha256', 'scope_sha256'}
                and type(binding['session_id']) is str and binding['session_id']
                and all(raw._hex(binding[key]) for key in ('request_sha256', 'scope_sha256'))
                and type(pid) is int and pid > 0 and callable(operation), 'exact owned child RPC binding/observer')
        self.spec, self.binding, self.pid = copy.deepcopy(spec), copy.deepcopy(binding), pid
        self.input, self.output, self.operation = input_stream, output_stream, operation
        self.deadline, self.messages = time.monotonic() + spec['deadline_seconds'], 0

    def _call(self, work):
        remaining = min(self.spec['rpc_seconds'], self.deadline - time.monotonic())
        require(remaining > 0, 'child RPC physical deadline exhausted')
        result = queue.Queue(maxsize=1)
        def run():
            try: result.put((True, work()))
            except BaseException as error: result.put((False, error))
        threading.Thread(target=run, daemon=True).start()
        try: success, value = result.get(timeout=remaining)
        except queue.Empty: raise child.MissingCheckpointWorkerService('child RPC deadline expired')
        if not success: raise value
        return value

    def receive(self, sequence):
        self.messages += 1; require(self.messages <= self.spec['max_messages'], 'child RPC message cap')
        def read():
            encoded = self._call(lambda: self.input.readline(self.spec['max_message_bytes'] + 1))
            require(type(encoded) is bytes and 0 < len(encoded) <= self.spec['max_message_bytes'] and encoded.endswith(b'\n'),
                    'child bounded complete RPC control required')
            return launch.validate_message(raw._json(encoded), self.binding, expected_kind='control',
                    expected_sequence=sequence, expected_pid=self.pid)
        return _paid(self.operation, self.spec['operation_bounds'], 'input_io', read)

    def emit(self, kind, sequence, payload):
        self.messages += 1; require(self.messages <= self.spec['max_messages'], 'child RPC message cap')
        encoded = canonical({'schema': launch.RPC, **self.binding, 'kind': kind,
            'sequence': sequence, 'worker_pid': self.pid, 'payload': payload}) + b'\n'
        require(len(encoded) <= self.spec['max_message_bytes'], 'child bounded RPC response')
        def write():
            offset = 0
            while offset < len(encoded):
                amount = self._call(lambda: self.output.write(encoded[offset:]))
                require(type(amount) is int and 0 < amount <= len(encoded) - offset, 'child RPC short/failed write')
                offset += amount
            self._call(self.output.flush)
        _paid(self.operation, self.spec['operation_bounds'], 'input_io', write)

    def serve(self, request, body):
        self.emit('ready', 0, {'branch': request['branch'], 'plan_sha256': request['plan_sha256'], 'native_qualified': False})
        payload = self.receive(1)
        require(set(payload) == {'command', 'data'} and payload['command'] == 'checkpoint_qualification',
                'branch must complete before finish; exact checkpoint control required')
        command = child.validate_command(payload['data'], request)
        with redirect_stdout(sys.stderr):
            result = body(command)
        require(type(result) is dict and type(result.get('raw_artifact')) is dict
                and result.get('body_completed') is True and result.get('native_qualified') is False,
                'only completed retained branch artifact can be returned')
        reference = result['raw_artifact']
        require(set(reference) == {'path', 'sha256', 'bytes', 'role', 'binding_sha256'}
                and type(reference['path']) is str and Path(reference['path']).is_absolute()
                and raw._hex(reference['sha256']) and raw._hex(reference['binding_sha256'])
                and raw._int(reference['bytes'], 1) and reference['bytes'] <= raw.MAX_JSON
                and reference['role'] == 'checkpoint_branch', 'bounded retained checkpoint artifact ref only')
        # No capability/large branch body is serialized into the response.
        response = {'raw_artifact': result['raw_artifact'], 'body_completed': True,
                    'os_terminal_verified': False, 'native_qualified': False}
        self.emit('response', 1, response)
        payload = self.receive(2)
        require(payload == {'command': 'finish', 'data': {}}, 'explicit finish only after retained branch response')
        self.emit('response', 2, {'finished': True, 'native_qualified': False})
        return 0


class LocalCheckpointIssuer:
    """Source singleton entry. It is intentionally absent from all registries."""
    def __init__(self):
        self._active_entry = self._active_context = self._active_proof = None
        self._active_request = self._predecessor_readback = None
    def run_frozen_request(self, request_path, handler_type, command_validator):
        require(any(self is value for value in child._REGISTERED_LOCAL_ISSUERS)
                and _SOURCE_ENTRY_PROVIDER is not None, 'source-owned checkpoint entry provider remains closed')
        require(handler_type is child.CheckpointChildHandler and command_validator is child.validate_command,
                'local entry can dispatch only the exact source checkpoint handler/command validator')
        # Existing source registries must all qualify before dependent IO.
        from . import ce_bootstrap_accounting as boot, ce_qualification_accounting as accounting
        from . import ce_publication as publication
        boot.require_registered_verifier(_BOOTSTRAP_VERIFIER)
        require(any(_AUTHORITY_VERIFIER is value for value in authority._REGISTERED_NATIVE_AUTHORITY_VERIFIERS)
                and any(_BOOTSTRAP_ISSUER is value for value in boot._REGISTERED_NATIVE_ISSUERS)
                and any(_PREDECESSOR_VERIFIER is value for value in accounting._REGISTERED_CHILD_PREFIX_VERIFIERS)
                and any(_PUBLICATION_ISSUER is value for value in publication._REGISTERED_PUBLICATION_ISSUERS),
                'checkpoint entry native authority/bootstrap/predecessor/publication remains unqualified')
        from . import ce_checkpoint_initializer as initializer, ce_checkpoint_publication as branch_publication
        from . import ce_checkpoint_lease as lease
        require(any(initializer.CheckpointChildInitializer is value for value in child._REGISTERED_CHILD_INITIALIZERS)
                and any(branch_publication.CheckpointBranchPublisher is value for value in child._REGISTERED_CHILD_PUBLICATION_SERVICES)
                and any(lease.DelegatedGpuLease is value for value in child._REGISTERED_CHILD_LEASE_SERVICES)
                and bool(worker._REGISTERED_METADATA_VERIFIERS),
                'complete exact child source constructors/OS observation must qualify before entry IO')
        entry = _SOURCE_ENTRY_PROVIDER.open_prepaid_entry(request_path)
        return self._consume_entry(entry, request_path, handler_type, command_validator)

    def _consume_entry(self, entry, request_path, handler_type, command_validator):
        require(type(entry) is dict, 'actual source-owned current OS observation entry')
        from .ce_checkpoint_entry_provider import MetadataEntryPayment
        def close_entry():
            failure = None
            for current in (entry.get('prefix_handle'), entry.get('parent_handle'), entry.get('worker_handle')):
                if current is not None:
                    try:
                        require(type(current) is worker.WinWorkerHandle, 'only retained entry OS handles may close')
                        current.close()
                    except BaseException as error:
                        if failure is None: failure = error
                        else: raw._accounting(failure, error)
            if failure is not None: raise failure
        payment = entry.get('entry_prepayment')
        if type(payment) is not MetadataEntryPayment:
            error = child.MissingCheckpointWorkerService('exact source-owned metadata entry payment required')
            try: close_entry()
            except BaseException as failure:
                error.entry_cleanup_failure = {'type': type(failure).__name__, 'message': str(failure)[:1000]}
                raise error from failure
            raise error
        def validate_and_run():
            require(type(entry.get('worker_handle')) is worker.WinWorkerHandle
                    and type(entry.get('observation_permit')) is worker.WorkerObservationPermit,
                    'actual source-owned current OS observation entry')
            operation, handle = entry['operation'], entry['worker_handle']
            entry['observation_permit'].require(handle.spec, operation)
            proof = worker.verify_worker_handle(handle, entry['expected_worker'], require_native=True)
            require(proof['os_identity']['pid'] == os.getpid() and entry['request_ref'] == proof['sources']['request_ref']
                    and Path(request_path) == Path(entry['request_ref']['path']), 'entry is exact current raw child request')
            spec = validate_spec(entry['context']['bindings']['scope']['checkpoint_qualification']['local_issuer'])
            payment.verify(entry['context'], proof,
                {'accounting_spec_sha256': digest(spec), 'prepared_receipt_sha256':
                 entry['context']['bindings']['scope']['qualifier_accounting']['ingestion_receipt_sha256']})
            return self._run_entry(entry, spec, proof, handler_type, command_validator)
        result = entry_lifetime(payment, validate_and_run, close_entry)
        from .ce_checkpoint_publication import retain_child_final_cost_clock
        retain_child_final_cost_clock(entry)
        return result

    def _run_entry(self, entry, spec, proof, handler_type, command_validator):
        operation, gate, handle = entry['operation'], entry['gate'], entry['worker_handle']
        # Entry and final RPC remain on their outer bootstrap owner. The CE
        # dispatcher attaches Q accounting during the branch and Q closes
        # before its raw artifact is returned; those costs must not migrate.
        bootstrap_operation = entry['bootstrap_operation']
        request = read_json(entry['request_ref'], entry['owned_root'], bootstrap_operation, spec['operation_bounds'], spec['max_request_bytes'])
        require(request.get('common_context_ref') == entry['common_context_ref'], 'raw request cannot replace source-pinned common scope')
        common = read_json(entry['common_context_ref'], entry['owned_root'], bootstrap_operation, spec['operation_bounds'], spec['max_request_bytes'])
        require(common == authority.context_preimage(entry['context']), 'retained frozen common scope/recipe/source drift')
        _paid(bootstrap_operation, spec['operation_bounds'], 'input_io', lambda:
            child.validate_request(request, dict(common, checkpoint_child_declaration=request['lineage_declaration'])))
        binding = raw._json(os.environ['CE8_RPC_BINDING'])
        require(binding.get('request_sha256') == proof['request_sha256'] and binding.get('scope_sha256') == proof['scope_sha256'],
                'RPC environment cannot replace observed logical request/scope')
        pipe = BoundedRPC(spec, binding, os.getpid(), sys.stdin.buffer, sys.stdout.buffer, bootstrap_operation)
        def run(command):
            command_validator(command, request)
            refreshed = worker.verify_worker_handle(handle, entry['expected_worker'], require_native=True)
            require(refreshed == proof, 'current child OS/source proof changed while awaiting command')
            parent_proof = request['lineage_declaration']['parent_proof']
            actual_parent = worker.verify_worker_handle(entry['parent_handle'], {key: parent_proof[key]
                for key in ('request_sha256', 'scope_sha256', 'worker_identity_sha256')}, require_native=True)
            _paid(bootstrap_operation, spec['operation_bounds'], 'input_io', lambda: validate_parent_view(actual_parent, parent_proof))
            provisional = dict(common, checkpoint_child_declaration=request['lineage_declaration'])
            delegation, reference = read_delegation(request['lease_delegation_path'], provisional,
                bootstrap_operation, spec['operation_bounds'], spec['max_request_bytes'])
            context = _paid(bootstrap_operation, spec['operation_bounds'], 'input_io', lambda:
                derive_context(common, request, entry['request_ref'], proof, delegation, reference))
            if request['branch'] == 'resumed':
                from .ce_prefix_handle_transfer import prefix_terminal_handle
                require(entry.get('prefix_factory') is prefix_terminal_handle, 'exact reviewed prefix adoption source factory')
                entry['prefix_handle'] = prefix_terminal_handle(common, request, entry['metadata_service_permit'], operation,
                    controller_request=entry['controller_request'], service_spec=entry['service_spec'], current_handle=handle,
                    observation_permit=entry['observation_permit'], delegation=delegation)
            # Current lock bytes are independently checked BEFORE any permit.
            from . import ce_checkpoint_lease as lease
            def check_lock():
                current, lockref = lease._read_lock(Path(delegation['lock_path']), context['bindings']['scope']['checkpoint_lease']['lock_bytes_upper'])
                require(lockref == delegation['current_lock_ref'] and lease._owner(current) == delegation['owner_marker']
                        and current['pid'] == delegation['parent_os_identity']['pid']
                        and current['worker_pids'] == [proof['os_identity']['pid']], 'current supervisor raw lock/delegation changed')
            require(spec['operation_bounds']['read_bytes_upper'] >= context['bindings']['scope']['checkpoint_lease']['lock_bytes_upper'] + 1
                    and spec['operation_bounds']['identity_hash_bytes_upper'] >= context['bindings']['scope']['checkpoint_lease']['lock_bytes_upper'],
                    'full current supervisor lock verification prepayment')
            _paid(bootstrap_operation, spec['operation_bounds'], 'input_io', check_lock)
            require(self._active_context is None, 'one source-owned active checkpoint child issuance')
            self._active_entry, self._active_context = entry, context
            self._active_proof, self._active_request = copy.deepcopy(proof), copy.deepcopy(request)
            context['native_services'] = {'local_checkpoint_issuer': self, 'operation': operation, 'gate': gate}
            permit = _paid(bootstrap_operation, spec['operation_bounds'], 'input_io', lambda:
                authority.NativeExecutionPermit(context, self, _AUTHORITY_VERIFIER, operation=operation, gate=gate))
            services = _paid(bootstrap_operation, spec['operation_bounds'], 'input_io', lambda:
                self._services(entry, context, permit, proof, request))
            handler = handler_type(context, permit, entry['observation_permit'], handle, entry['prepared'], services)
            result = handler.run(request, command)
            entry['child_result'] = result
            entry['child_final_context'], entry['child_final_proof'] = context, copy.deepcopy(proof)
            entry['child_final_request'] = copy.deepcopy(request)
            return result
        try: return pipe.serve(request, run)
        finally:
            self._active_entry = self._active_context = self._active_proof = None
            self._active_request = self._predecessor_readback = None

    def _services(self, entry, context, permit, proof, request):
        from . import ce_bootstrap_accounting as boot, ce_qualification_accounting as accounting
        from . import ce_checkpoint_initializer as initializer, ce_checkpoint_publication as publication
        from . import ce_checkpoint_lease as lease
        outer = boot.BootstrapAccounting(context, permit, proof, native_issuer=_BOOTSTRAP_ISSUER)
        entry['child_scope_bootstrap'] = outer
        receipt = entry['prepared'].receipt; spec = accounting.accounting_specification(context)
        owner = context['checkpoint_child_accounting']['owner']
        setup_bounds = accounting._price(spec['setup_bounds'])
        setup_bounds['metadata_bytes_upper'] = 2 * accounting.MAX_LINE
        payment = outer.reserve_setup(owner, setup_bounds)
        cap = accounting.CheckpointChildAdmission(context, permit, proof,
                context['checkpoint_child_observed_lineage'], verifier=_PREDECESSOR_VERIFIER)
        reader = raw.RawEvidenceReader(entry['operation'], context['bindings']['scope']['checkpoint_qualification']['child_initializer']['reader_bounds'],
            context['bindings']['scope']['native_backend']['output_directory'])
        init = initializer.CheckpointChildInitializer(context, permit, proof, operation=entry['operation'], gate=entry['gate'],
            reader=reader, bootstrap_handle=payment, bootstrap_verifier=_BOOTSTRAP_VERIFIER,
            admission_handle=cap, predecessor_verifier=_PREDECESSOR_VERIFIER, prefix_handle=entry.get('prefix_handle'),
            parent_handle=entry['parent_handle'], child_handle=entry['worker_handle'])
        pubspec = publication.publication_specification(context)
        pubpay = outer.reserve_setup({'accounting_spec_sha256': digest(pubspec), 'prepared_receipt_sha256': receipt['receipt_sha256']}, pubspec['bounds'])
        publisher = publication.CheckpointBranchPublisher(context, permit, proof, receipt,
            bootstrap_handle=pubpay, bootstrap_verifier=_BOOTSTRAP_VERIFIER, native_issuer=_PUBLICATION_ISSUER)
        delegated = lease.DelegatedGpuLease(context, permit, entry['worker_handle'], entry['operation'], entry['gate'])
        return {'operation': entry['operation'], 'gate': entry['gate'], 'observer': entry['observer'],
                'initializer': init, 'publisher': publisher, 'lease': delegated}


def serve_fixture(context, request, binding, input_stream, output_stream, operation, body):
    require(type(context) is dict and context == {'schema': FIXTURE, 'native_qualified': False, 'execution_verified': False},
            'explicit non-native streams-only RPC fixture')
    return BoundedRPC(request['rpc_spec'], binding, 41, input_stream, output_stream, operation).serve(request, body)
