"""External CPU-only 45-second watchdog. No model/GPU calls or CLI overrides."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from . import diagnose_windows_job as probe
from .windows_job_cpu import Kernel, TracedJob, QUERY, SYNC, TERMINATE, WAIT_TIMEOUT, require, validate_owned_identity, cleanup_owned


def main():
    require(len(sys.argv) == 1 and os.name == 'nt' and not probe.forbidden_modules(), 'CPU guard has no overrides')
    cfg = probe.read(probe.CONFIG)
    require(hashlib.sha256(probe.CONFIG.read_bytes()).hexdigest() == probe.CONFIG_SHA256, 'frozen CPU config changed')
    require(Path(sys.executable).resolve() == Path(cfg['executable']).resolve()
            and hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest() == cfg['executable_sha256'], 'CPU executable changed')
    require(probe.git('status', '--porcelain') == '', 'commit clean CPU source before launch')
    directory = probe.OUTPUT.with_name(probe.OUTPUT.name + '_outer')
    require(not directory.exists() and not probe.OUTPUT.exists(), 'CPU diagnostic attempt consumed')
    directory.mkdir(parents=True)
    kernel = Kernel(probe.logger(directory / 'OUTER_API.jsonl'))
    source_sha, frozen = probe.git('rev-parse', 'HEAD'), probe.sources()
    children, token, peak = {}, uuid.uuid4().hex, 0
    process = job = root_identity = None
    error, started = None, time.monotonic()

    def retain(pid):
        if pid not in children:
            handle = kernel.open(pid, QUERY | SYNC | TERMINATE | 0x0100)
            try:
                identity = kernel.identity(handle, pid)
                lineage = kernel.lineage(pid, os.getpid())
                validate_owned_identity(identity, identity, lineage, os.getpid(), process.pid)
                require(identity['created_filetime'] >= root_identity['created_filetime'], 'CPU child predates owned launcher')
                children[pid] = {'handle': handle, 'identity': identity, 'lineage': lineage}
            except BaseException:
                kernel.close(handle)
                raise
        return children[pid]

    def guard():
        nonlocal peak
        require(time.monotonic() - started < 45 and not kernel.logging_errors, 'CPU outer time/evidence ceiling')
        require(kernel.available() >= 1.25, 'CPU host reserve floor')
        rss = 0
        if process is not None and process.poll() is None:
            rss += kernel.rss(int(process._handle))
            for pid, parent in kernel.parent_table().items():
                if parent == process.pid:
                    retain(pid)
        if job is not None:
            pids = kernel.job_pids(job.handle)
            require(len(pids) <= 4, 'CPU simultaneous process ceiling')
            for pid in pids:
                if pid != process.pid:
                    retain(pid)
        for item in children.values():
            if kernel.call('WaitForSingleObject', item['handle'], 0)[0] == WAIT_TIMEOUT:
                rss += kernel.rss(item['handle'])
        peak = max(peak, rss)
        require(rss <= .5 * 2**30, 'CPU full-tree RSS ceiling')
        for path in directory.glob('*'):
            if path.is_file():
                require(path.stat().st_size <= probe.MAX_BYTES, 'CPU outer file ceiling')

    try:
        require(kernel.available() >= 3.3125, 'CPU preworker reserve blocks launch')
        command = [cfg['executable'], '-B', '-m', 'experiments.post_training_mimo_centralized_v1.scripts.diagnose_windows_job',
                   '--supervise', '--outer-pid', str(os.getpid()), '--outer-token', token]
        probe.write(directory / 'OUTER_LAUNCH.json', {'source_sha': source_sha, 'sources': frozen, 'command': command,
                    'limits': cfg['limits'], 'started_monotonic_ns': time.monotonic_ns(), 'model_operations': 0, 'gpu_operations': 0})
        with (directory / 'stdout.log').open('xb') as stdout, (directory / 'stderr.log').open('xb') as stderr:
            process = subprocess.Popen(command, cwd=str(probe.ROOT), stdout=stdout, stderr=stderr,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1'))
            job = TracedJob(kernel, process)
            root_identity = kernel.identity(int(process._handle), process.pid)
            ready = probe.wait_file(directory / 'BOOT_READY.json', min(started + 45, time.monotonic() + 5), guard)
            require(ready['token'] == token and ready['sources'] == frozen, 'CPU bootstrap source/token mismatch')
            guard()
            actual = {'handle': int(process._handle), 'identity': root_identity} if ready['identity']['pid'] == process.pid else retain(ready['identity']['pid'])
            validate_owned_identity(ready['identity'], actual['identity'], ready['lineage'], os.getpid(), process.pid)
            before = kernel.membership(actual['handle'], job.handle, 'CPU_controller_before_GO')
            require(before['api_success'], 'CPU outer membership API failed')
            if not before['member']:
                result, code = kernel.call('AssignProcessToJobObject', job.handle, actual['handle'], context={'verified_cpu_controller': actual['identity']})
                require(result, f'CPU-only outer enrollment failed: {code}')
            after = kernel.membership(actual['handle'], job.handle, 'CPU_controller_enrolled_before_GO')
            require(after['api_success'] and after['member'] and not kernel.logging_errors, 'CPU external containment unavailable')
            probe.write(directory / 'BOOT_GO.json', {'token': token, 'identity': actual['identity'], 'sources': frozen, 'job_handle': int(job.handle)})
            while process.poll() is None:
                guard()
                time.sleep(.1)
            require(process.returncode == 0 and time.monotonic() - started < 45, 'CPU controller nonzero/late')
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    finally:
        cleanup = cleanup_owned(kernel, job, children, process, os.getpid())
        probe.write(directory / 'OUTER_CLEANUP.json', cleanup)
    passed = error is None and cleanup['status'] == 'OWNED_CPU_PROCESSES_EXITED' and not kernel.logging_errors
    result = {'status': 'PASS_CPU_ONLY_EXTERNAL_CONTAINMENT' if passed else 'BLOCKED_CPU_OUTER_GUARD',
              'error': error, 'elapsed_seconds': time.monotonic() - started, 'source_sha': source_sha,
              'peak_observed_process_tree_rss_bytes': peak, 'cleanup': cleanup,
              'model_operations': 0, 'gpu_operations': 0, 'native_capture_authorized': False, 'forbidden_modules': probe.forbidden_modules()}
    probe.write(directory / 'OUTER_RESULT.json', result)
    print(probe.json.dumps(result, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == '__main__':
    sys.exit(main())
