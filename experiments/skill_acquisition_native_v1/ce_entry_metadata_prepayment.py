"""Fixed paid child preparation window; no model/worker authority.

Native expiry reuses the service watchdog's current-process exit policy. It
never accepts a PID or termination callback. Fixtures never create a Timer.
An interrupted watchdog leaves the original paid operation pending, with its
full reservation retained by the original service bootstrap.
"""
from __future__ import annotations

import math
import time
import uuid

from . import ce_native_services as services

SECONDS = 1800
require = services.require


def _profile(bootstrap, configuration):
    require(type(bootstrap) is services.BootstrapLedger and not bootstrap.fatal and not bootstrap.sealed
            and bootstrap.spec == configuration['service_spec']['bootstrap']
            and bootstrap.request_sha256 == configuration['launch_spec']['request_sha256'],
            'exact live original child preparation bootstrap')
    profile = services._profile(bootstrap.spec['operation_profiles'].get('entry_preparation'))
    require(type(profile['wall_s_upper']) in (int, float) and profile['wall_s_upper'] == SECONDS,
            'fixed1800-second child preparation profile')
    return profile


def _error(primary, accounting):
    primary.preparation_accounting_failure = {'type': type(accounting).__name__, 'message': str(accounting)[:1000]}


def _run(bootstrap, configuration, body, *, clock, native):
    profile = _profile(bootstrap, configuration)
    require(callable(body) and callable(clock), 'fixed source preparation body/clock')
    started = clock()
    require(type(started) in (int, float) and math.isfinite(started) and started >= 0,
            'finite original preparation clock')
    identifier = uuid.uuid4().hex
    # The existing metadata-entry lifetime pays admission/constructor work.
    # This independent full profile is retained before Timer or body work.
    bootstrap.operation('entry_preparation', 'started', identifier,
        preparation_physical_seconds_upper=SECONDS, metadata_only=True,
        scope_sha256=configuration['launch_spec']['scope_sha256'], **profile)
    timer = None
    returned = False
    try:
        now = clock()
        require(type(now) in (int, float) and math.isfinite(now) and started <= now
                and now - started < SECONDS, 'child preparation deadline before body')
        if native:
            # Same mechanism as NativeControllerServices.watchdog: only this
            # current owned worker exits, no PID lookup or foreign kill.
            timer = services.threading.Timer(SECONDS - (now - started), lambda: services.os._exit(124))
            timer.daemon = True
            timer.start()
        value = body()
        finished = clock()
        require(type(finished) in (int, float) and math.isfinite(finished) and finished >= started
                and finished - started <= SECONDS, 'child preparation physical deadline exceeded')
        bootstrap.operation('entry_preparation', 'returned', identifier,
            wall_s=finished - started, metadata_only=True,
            preparation_physical_seconds_upper=SECONDS)
        returned = True
        # Encoding, append and fsync are part of this same paid window.
        # A committed return cannot be followed by another terminal phase.
        appended = clock()
        require(type(appended) in (int, float) and math.isfinite(appended) and appended >= finished
                and appended - started <= SECONDS, 'child preparation physical deadline after returned append exceeded')
        return value
    except BaseException as primary:
        if returned:
            bootstrap.fatal = True
            raise
        try:
            finished = clock()
            require(type(finished) in (int, float) and math.isfinite(finished) and finished >= started,
                    'unknown/regressed child preparation failure clock')
            bootstrap.operation('entry_preparation', 'failed', identifier,
                wall_s=finished - started, metadata_only=True,
                error={'type': type(primary).__name__, 'message': str(primary)[:1000]},
                preparation_physical_seconds_upper=SECONDS)
        except BaseException as accounting:
            _error(primary, accounting)
            raise primary from accounting
        raise
    finally:
        if timer is not None: timer.cancel()


def run_preparation(runtime, configuration, body):
    """Source provider's one pre-ready metadata/full3000-row preparation body."""
    # Admission before configuration, paths, Timer or supplied body traversal.
    require(type(runtime) is services.NativeControllerServices,
            'actual native child preparation services owner')
    runtime._admit()
    require(runtime.bootstrap._native is True and runtime.spec == configuration['service_spec'],
            'original native preparation service/spec binding')
    return _run(runtime.bootstrap, configuration, body, clock=time.monotonic, native=True)


def fixture_preparation(bootstrap, configuration, body, *, clock):
    """Explicit pure traps/clock controls; cannot issue native authority."""
    require(type(bootstrap) is services.BootstrapLedger and bootstrap._native is False,
            'explicit synthetic preparation bootstrap only')
    return _run(bootstrap, configuration, body, clock=clock, native=False)
