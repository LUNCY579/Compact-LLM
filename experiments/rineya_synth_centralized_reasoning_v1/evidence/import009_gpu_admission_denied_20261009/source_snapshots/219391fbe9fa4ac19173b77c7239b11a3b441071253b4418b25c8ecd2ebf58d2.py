"""NVML ABI, read-only contract and deadlines with fake DLLs; no GPU calls."""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import native_nvml as nv

UUID = "GPU-bf962d49-2b03-1983-ef14-df4bc7fbb8b9"
GIB = 1024 ** 3


def fake_api():
    api = mock.Mock()
    api.nvmlInit_v2.return_value = 0
    api.nvmlShutdown.return_value = 0
    def handle(index, pointer):
        if index != 0:
            raise AssertionError("only GPU0 admitted")
        nv.ctypes.cast(pointer, nv.ctypes.POINTER(nv.ctypes.c_void_p)).contents.value = 123
        return 0
    def memory(handle, pointer):
        value = nv.ctypes.cast(pointer, nv.ctypes.POINTER(nv._MemoryInfo)).contents
        value.total, value.free, value.used = 8 * GIB, 6 * GIB, 2 * GIB
        return 0
    def identifier(handle, buffer, length):
        buffer.value = UUID.encode("ascii")
        return 0
    api.nvmlDeviceGetHandleByIndex_v2.side_effect = handle
    api.nvmlDeviceGetMemoryInfo.side_effect = memory
    api.nvmlDeviceGetUUID.side_effect = identifier
    return api


class NVMLTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in (("_CLIENT", None), ("_ACTIVE", None), ("_FAULT", None), ("_STATE_LOCK", threading.Lock())):
            self.stack.enter_context(mock.patch.object(nv, name, value))

    def test_framework_free_import(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_binding_hash_does_not_load_DLL(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nvml.dll"
            path.write_bytes(b"fake NVML bytes")
            with mock.patch.object(nv, "library_path", return_value=path), \
                    mock.patch.object(nv.ctypes, "WinDLL", create=True) as load:
                binding = nv.library_binding()
            self.assertEqual(str(path), binding["path"])
            self.assertEqual(nv.hashlib.sha256(b"fake NVML bytes").hexdigest(), binding["sha256"])
            load.assert_not_called()

    def test_library_path_uses_actual_system_directory_and_requires_regular_DLL(self):
        with tempfile.TemporaryDirectory() as directory:
            system = Path(directory) / "System32"
            system.mkdir()
            candidate = system / "nvml.dll"
            candidate.write_bytes(b"fake NVML")
            kernel = mock.Mock()
            def directory_query(buffer, length):
                buffer.value = str(system)
                return len(str(system))
            kernel.GetSystemDirectoryW.side_effect = directory_query
            with mock.patch.object(nv.os, "name", "nt"), mock.patch.object(nv.ctypes, "WinDLL", return_value=kernel, create=True):
                self.assertEqual(candidate.resolve(), nv.library_path())
                candidate.unlink()
                with self.assertRaisesRegex(nv.NVMLError, "regular System32 DLL"):
                    nv.library_path()

    def test_absolute_DLL_readonly_memory_UUID_ABI_and_shutdown(self):
        api = fake_api()
        path = Path("C:/Windows/System32/nvml.dll")
        with mock.patch.object(nv, "library_path", return_value=path), \
                mock.patch.object(nv.ctypes, "WinDLL", return_value=api, create=True) as load:
            client = nv._Client()
            result = client.query()
            client.close()
            client.close()
        self.assertEqual({"GPU_index": 0, "gpu_uuid": UUID, "gpu_total_bytes": 8 * GIB,
                          "gpu_free_bytes": 6 * GIB}, result)
        load.assert_called_once_with(str(path), use_last_error=True, winmode=0x100 | 0x800)
        api.nvmlInit_v2.assert_called_once_with()
        api.nvmlShutdown.assert_called_once_with()
        self.assertEqual({"nvmlInit_v2", "nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetMemoryInfo",
                          "nvmlDeviceGetUUID", "nvmlShutdown"}, {call[0] for call in api.mock_calls})
        self.assertEqual(24, nv.ctypes.sizeof(nv._MemoryInfo))

    def test_status_and_malformed_memory_fail_closed(self):
        for kind in ("error", "memory", "uuid"):
            with self.subTest(kind=kind):
                api = fake_api()
                if kind == "error":
                    api.nvmlDeviceGetMemoryInfo.side_effect = None
                    api.nvmlDeviceGetMemoryInfo.return_value = 999
                elif kind == "memory":
                    def bad(handle, pointer):
                        value = nv.ctypes.cast(pointer, nv.ctypes.POINTER(nv._MemoryInfo)).contents
                        value.total, value.free = 1, 2
                        return 0
                    api.nvmlDeviceGetMemoryInfo.side_effect = bad
                else:
                    api.nvmlDeviceGetUUID.side_effect = lambda handle, buffer, length: 0
                with mock.patch.object(nv, "library_path", return_value=Path("fake/nvml.dll")), \
                        mock.patch.object(nv.ctypes, "WinDLL", return_value=api, create=True):
                    client = nv._Client()
                    with self.assertRaises(nv.NVMLError):
                        client.query()
                    client.close()

    def test_two_snapshots_one_lazy_init_and_same_schema(self):
        client = mock.Mock()
        client.query.return_value = {"GPU_index": 0, "gpu_uuid": UUID, "gpu_total_bytes": 8 * GIB,
                                     "gpu_free_bytes": 6 * GIB}
        with mock.patch.object(nv, "_Client", return_value=client) as create:
            first, second = nv.gpu_snapshot(), nv.gpu_snapshot()
            nv.shutdown()
        self.assertEqual({"GPU_index", "gpu_uuid", "gpu_total_bytes", "gpu_free_bytes",
                          "gpu_sample_started_monotonic", "gpu_sample_completed_monotonic", "gpu_observation_id"}, set(first))
        self.assertLessEqual(first["gpu_sample_started_monotonic"], first["gpu_sample_completed_monotonic"])
        self.assertNotEqual(first["gpu_observation_id"], second["gpu_observation_id"])
        create.assert_called_once_with()
        self.assertEqual(2, client.query.call_count)
        client.close.assert_called_once_with()
        self.assertEqual([], nv.pending_query_threads())

    def test_unreturned_native_call_fixed_deadline_no_retry_and_explicit_thread(self):
        threads = []
        class BlockedThread:
            def __init__(self, *, target, daemon, name):
                self.target, self.name, self.ident = target, name, 123
                self.timeout = None
                threads.append(self)
            def start(self):
                pass
            def join(self, timeout):
                self.timeout = timeout
            def is_alive(self):
                return True
        with mock.patch.object(nv.threading, "Thread", BlockedThread), mock.patch.object(nv.time, "monotonic", return_value=100.0):
            with self.assertRaisesRegex(TimeoutError, "fixed 2s"):
                nv.gpu_snapshot()
            pending = nv.pending_query_threads()
            with self.assertRaisesRegex(nv.NVMLError, "has not returned"):
                nv.gpu_snapshot()
            with self.assertRaisesRegex(nv.NVMLError, "cannot verify NVML cleanup"):
                nv.shutdown()
        self.assertEqual(1, len(threads))
        self.assertEqual(2.0, threads[0].timeout)
        self.assertEqual(100.0, pending[0]["started_monotonic"])
        self.assertEqual(102.0, pending[0]["deadline_monotonic"])
        self.assertEqual(123, pending[0]["ident"])

    def test_late_return_is_not_a_fresh_observation_and_cannot_retry(self):
        now = [100.0]
        client = mock.Mock()
        def query():
            now[0] = 102.001
            return {"GPU_index": 0, "gpu_uuid": UUID, "gpu_total_bytes": 8 * GIB, "gpu_free_bytes": 6 * GIB}
        client.query.side_effect = query
        class ImmediateThread:
            def __init__(self, *, target, daemon, name):
                self.target, self.name, self.ident = target, name, 123
            def start(self):
                self.target()
            def join(self, timeout):
                pass
            def is_alive(self):
                return False
        with mock.patch.object(nv, "_Client", return_value=client), \
                mock.patch.object(nv.threading, "Thread", ImmediateThread), \
                mock.patch.object(nv.time, "monotonic", side_effect=lambda: now[0]):
            with self.assertRaisesRegex(TimeoutError, "fixed 2s"):
                nv.gpu_snapshot()
            with self.assertRaisesRegex(nv.NVMLError, "no automatic query retry"):
                nv.gpu_snapshot()
        self.assertEqual(1, client.query.call_count)


if __name__ == "__main__":
    unittest.main()
