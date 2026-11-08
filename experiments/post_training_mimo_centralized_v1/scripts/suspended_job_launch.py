"""CPU-qualified public-API launcher: specific Job membership precedes resume."""
from __future__ import annotations
import ctypes as c
from ctypes import wintypes as w
import os
from pathlib import Path
import subprocess
import time

from .windows_job_cpu import TracedJob, cleanup_owned, require, validate_owned_identity

CREATE_SUSPENDED = 0x00000004
CREATE_NEW_PROCESS_GROUP = 0x00000200
THREAD_RIGHTS = 0x0802


def launch_suspended(kernel, command, cwd, stdout, stderr, deadline, guard=lambda: None,
                     popen_factory=None, job_factory=None):
    """Return owned Popen/Job only after verified assignment and one successful resume."""
    process = job = None
    thread = None
    popen_factory = popen_factory or subprocess.Popen
    job_factory = job_factory or TracedJob
    try:
        guard()
        require(time.monotonic() < deadline and not kernel.logging_errors, 'suspended launch admission deadline/evidence')
        process = popen_factory(command, cwd=str(cwd), stdout=stdout, stderr=stderr,
                                creationflags=CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1'))
        identity = kernel.identity(int(process._handle), process.pid)
        require(os.path.normcase(identity['image_path']) == os.path.normcase(str(Path(command[0]).resolve())),
                'suspended launcher executable identity changed')
        lineage = kernel.lineage(process.pid, os.getpid())
        validate_owned_identity(identity, identity, lineage, os.getpid(), process.pid)
        job = job_factory(kernel, process)
        member = kernel.membership(int(process._handle), job.handle, 'suspended_root_assigned_before_resume')
        require(member['api_success'] and member['member'], 'suspended root specific membership unavailable')
        tids = kernel.thread_ids(process.pid)
        require(len(tids) == 1, 'suspended launcher must have exactly one owned thread')
        thread, error = kernel.call('OpenThread', THREAD_RIGHTS, False, tids[0],
                                    context={'owned_pid': process.pid, 'owned_tid': tids[0]})
        require(thread, f'owned suspended thread open failed: {error}')
        owner, error = kernel.call('GetProcessIdOfThread', thread)
        require(owner == process.pid, f'owned suspended thread owner mismatch: {error}')
        creation, exit_time, kernel_time, user_time = [w.FILETIME() for _ in range(4)]
        result, error = kernel.call('GetThreadTimes', thread, c.byref(creation), c.byref(exit_time),
                                    c.byref(kernel_time), c.byref(user_time))
        thread_created = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        require(result and thread_created >= identity['created_filetime'], f'owned thread creation unavailable/stale: {error}')
        validate_owned_identity(identity, kernel.identity(int(process._handle), process.pid),
                                lineage, os.getpid(), process.pid)
        member = kernel.membership(int(process._handle), job.handle, 'suspended_root_immediately_before_resume')
        guard()
        require(member['api_success'] and member['member'] and not kernel.logging_errors
                and time.monotonic() < deadline, 'suspended resume membership/deadline/evidence guard')
        previous, error = kernel.call('ResumeThread', thread, context={'owned_pid': process.pid, 'owned_tid': tids[0]})
        require(previous == 1, f'owned thread previous suspend count must equal one: {previous}/{error}')
        proof = {'root_identity': identity, 'root_lineage': lineage, 'thread_id': tids[0],
                 'thread_created_filetime': thread_created, 'specific_membership_before_resume': member,
                 'previous_suspend_count': int(previous), 'creationflags': CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP}
        kernel.close(thread)
        thread = None
        require(not kernel.logging_errors, 'suspended launch evidence lost')
        kernel.record({'event': 'suspended_launch_verified', **proof})
        require(not kernel.logging_errors, 'suspended launch proof lost')
        return process, job, proof
    except BaseException:
        if thread is not None:
            try:
                kernel.close(thread)
            except Exception as error:
                kernel.record({'event': 'owned_thread_close_failure', 'error': str(error)})
        result = cleanup_owned(kernel, job, {}, process, os.getpid())
        kernel.record({'event': 'suspended_launch_failure_cleanup', 'cleanup': result})
        raise
