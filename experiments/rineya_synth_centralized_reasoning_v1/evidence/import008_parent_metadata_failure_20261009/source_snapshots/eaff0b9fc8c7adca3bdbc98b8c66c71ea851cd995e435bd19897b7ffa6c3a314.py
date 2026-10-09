"""Mock Win32 handle/terminality tests; no framework or model process."""
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import native_process as np


def api_fixture():
    api = mock.Mock()
    api.OpenProcess.return_value = 123
    api.WaitForSingleObject.return_value = np.WAIT_TIMEOUT
    api.CloseHandle.return_value = True
    def times(handle, born, exited, kernel, user):
        value = np.ctypes.cast(born, np.ctypes.POINTER(np.FILETIME)).contents
        ticks = np.EPOCH_FILETIME + 10000000
        value.low, value.high = ticks & 0xFFFFFFFF, ticks >> 32
        return True
    def memory(handle, pointer, size):
        value = np.ctypes.cast(pointer, np.ctypes.POINTER(np.MEMORY_COUNTERS_EX)).contents
        value.WorkingSetSize = 3072
        return True
    def inside(handle, job, pointer):
        np.ctypes.cast(pointer, np.ctypes.POINTER(np.ctypes.c_int32)).contents.value = 1
        return True
    api.GetProcessTimes.side_effect = times
    api.K32GetProcessMemoryInfo.side_effect = memory
    api.IsProcessInJob.side_effect = inside
    def available(pointer):
        value = np.ctypes.cast(pointer, np.ctypes.POINTER(np.MEMORY_STATUS_EX)).contents
        value.total_phys, value.available_phys = 16 * 1024 ** 3, 4 * 1024 ** 3
        return True
    api.GlobalMemoryStatusEx.side_effect = available
    return api


class ProcessTests(unittest.TestCase):
    def build(self, api):
        with mock.patch.object(np.os, "name", "nt"), mock.patch.object(np, "library_path", return_value=Path("C:/Windows/System32/kernel32.dll")), \
                mock.patch.object(np.ctypes, "WinDLL", return_value=api, create=True):
            sampler = np.Sampler()
        sampler._error = lambda operation: PermissionError(operation)
        return sampler

    def test_same_fresh_handle_birth_RSS_membership_and_immediate_close(self):
        api = api_fixture()
        value = self.build(api).process(42, job_handle=456)
        api.OpenProcess.assert_called_once_with(0x101000, False, 42)
        self.assertEqual(0, 0x101000 & 0x10)  # no unnecessary PROCESS_VM_READ
        self.assertEqual(123, api.GetProcessTimes.call_args.args[0])
        self.assertEqual(123, api.K32GetProcessMemoryInfo.call_args.args[0])
        api.CloseHandle.assert_called_once_with(123)
        self.assertEqual(np.EPOCH_FILETIME + 10000000, value["creation_filetime_100ns"])
        self.assertEqual(3072, value["rss_bytes"])
        self.assertFalse(value["terminal_confirmed"])
        self.assertIn("K32GetProcessMemoryInfo_seconds", value["api_latency_seconds"])
        self.assertEqual(8, np.ctypes.sizeof(np.FILETIME))
        self.assertEqual(80, np.ctypes.sizeof(np.MEMORY_COUNTERS_EX))

    def test_skip_only_handle_signaled_terminal(self):
        api = api_fixture()
        api.WaitForSingleObject.return_value = 0
        value = self.build(api).process(42)
        self.assertTrue(value["terminal_confirmed"])
        api.GetProcessTimes.assert_not_called()
        api.K32GetProcessMemoryInfo.assert_not_called()
        api.CloseHandle.assert_called_once_with(123)

    def test_failed_memory_on_live_handle_raises_not_zero_RSS(self):
        api = api_fixture()
        api.K32GetProcessMemoryInfo.side_effect = None
        api.K32GetProcessMemoryInfo.return_value = False
        with self.assertRaisesRegex(PermissionError, "K32GetProcessMemoryInfo"):
            self.build(api).process(42)
        api.CloseHandle.assert_called_once_with(123)

    def test_failed_memory_only_skips_if_same_handle_became_terminal(self):
        api = api_fixture()
        api.K32GetProcessMemoryInfo.side_effect = None
        api.K32GetProcessMemoryInfo.return_value = False
        api.WaitForSingleObject.side_effect = [258, 0]
        value = self.build(api).process(42)
        self.assertTrue(value["terminal_confirmed"])
        api.CloseHandle.assert_called_once_with(123)

    def test_open_failure_does_not_guess_terminal_or_retry(self):
        api = api_fixture()
        api.OpenProcess.return_value = None
        with self.assertRaisesRegex(PermissionError, "OpenProcess"):
            self.build(api).process(42)
        api.OpenProcess.assert_called_once()
        api.WaitForSingleObject.assert_not_called()

    def test_membership_and_wait_failure_fail_closed(self):
        api = api_fixture()
        api.IsProcessInJob.side_effect = lambda handle, job, inside: True
        with self.assertRaisesRegex(RuntimeError, "outside owned Job"):
            self.build(api).process(42, job_handle=456)
        api.CloseHandle.assert_called_once_with(123)
        api = api_fixture()
        api.WaitForSingleObject.return_value = 0xFFFFFFFF
        with self.assertRaisesRegex(PermissionError, "WaitForSingleObject"):
            self.build(api).process(42)
        api.CloseHandle.assert_called_once_with(123)

    def test_raw_root_birth_exact_and_legacy_float_unchanged(self):
        sampler = mock.Mock()
        sampler.process.return_value = {"pid": 42, "terminal_confirmed": False, "creation_filetime_100ns": 12345,
                                        "create_time": 1.0000001, "rss_bytes": 1024}
        root = {"pid": 42, "creation_filetime_100ns": 12345, "create_time": 1.0000002}
        result = np.owned_snapshot([42], root, 456, sampler)
        self.assertEqual(1.0000002, result["identities"][0]["create_time"])
        root["creation_filetime_100ns"] += 1
        with self.assertRaisesRegex(RuntimeError, "raw birth FILETIME changed"):
            np.owned_snapshot([42], root, 456, sampler)

    def test_raw_alive_ignores_rounded_float_but_rejects_PID_reuse(self):
        api = api_fixture()
        sampler = self.build(api)
        identity = {"pid": 42, "creation_filetime_100ns": np.EPOCH_FILETIME + 10000000,
                    "create_time": 1.0000002}
        self.assertTrue(sampler.is_alive(identity))
        identity["creation_filetime_100ns"] += 1
        self.assertFalse(sampler.is_alive(identity))
        api.WaitForSingleObject.return_value = 0
        self.assertFalse(sampler.is_alive(identity))

    def test_current_available_physical_memory_same_metric_and_failure_closed(self):
        api = api_fixture()
        sampler = self.build(api)
        self.assertEqual({"total_bytes": 16 * 1024 ** 3, "available_bytes": 4 * 1024 ** 3}, sampler.memory_status())
        self.assertEqual(64, np.ctypes.sizeof(np.MEMORY_STATUS_EX))
        api.GlobalMemoryStatusEx.side_effect = None
        api.GlobalMemoryStatusEx.return_value = False
        with self.assertRaisesRegex(PermissionError, "GlobalMemoryStatusEx"):
            sampler.memory_status()

    def test_import_requires_no_model_framework(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)


if __name__ == "__main__":
    unittest.main()
