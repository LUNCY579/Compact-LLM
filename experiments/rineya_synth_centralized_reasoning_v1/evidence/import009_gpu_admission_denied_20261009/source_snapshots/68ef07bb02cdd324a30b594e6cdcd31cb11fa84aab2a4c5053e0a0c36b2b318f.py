"""Injected CPU API fixtures: no framework, device, model, or resource probe."""
from copy import deepcopy
import weakref


class PrimaryFixtureError(RuntimeError):
    """An owned CPU error used to test primary-exception preservation."""


class Recorder:
    def __init__(self, faults=None):
        self.events = []
        self.guards = []
        self.faults = {} if faults is None else dict(faults)
        self.counts = {}
        self.arguments = {}

    def record(self, name):
        self.events.append(name)
        self.counts[name] = self.counts.get(name, 0) + 1

    def result(self, name, default):
        self.record(name)
        key = name + ":" + str(self.counts[name])
        value = self.faults.get(key, self.faults.get(name, default))
        if isinstance(value, BaseException):
            raise value
        return value

    def guard(self, stage, torch_view=None):
        self.guards.append(stage)
        if torch_view is not None and torch_view.cuda.is_initialized():
            torch_view.cuda.mem_get_info()
        value = self.faults.get("guard:" + stage)
        if isinstance(value, BaseException):
            raise value


class FakeDevice:
    def __init__(self, recorder, role, value):
        self.recorder, self.role, self.value = recorder, role, value

    def __str__(self):
        return self.recorder.result(self.role + ".device.str", self.value)


class FakeTensor:
    def __init__(self, recorder, role, rows, dtype, device):
        self.recorder, self.role = recorder, role
        self._rows, self._dtype, self._device = deepcopy(rows), dtype, device

    @property
    def shape(self):
        return self.recorder.result(self.role + ".shape", (len(self._rows), len(self._rows[0])))

    @property
    def dtype(self):
        return self.recorder.result(self.role + ".dtype", self._dtype)

    @property
    def device(self):
        value = self.recorder.result(self.role + ".device", self._device)
        return FakeDevice(self.recorder, self.role, value)

    def tolist(self):
        return self.recorder.result(self.role + ".tolist", deepcopy(self._rows))

    def __del__(self):
        # Observe fixture reference release; this is not device memory telemetry.
        self.recorder.record(self.role + ".released")


class FakeCuda:
    def __init__(self, recorder):
        self.recorder = recorder
        self._initialized = False

    def init(self):
        result = self.recorder.result("cuda.init", None)
        self._initialized = True
        return result

    def mem_get_info(self, *args, **kwargs):
        return self.recorder.result("cuda.mem_get_info", (16 * 1024**3, 24 * 1024**3))

    def manual_seed(self, *args, **kwargs):
        return self.recorder.result("cuda.manual_seed", None)

    def manual_seed_all(self, *args, **kwargs):
        self.recorder.arguments.setdefault("cuda.manual_seed_all", []).append((args, kwargs))
        return self.recorder.result("cuda.manual_seed_all", None)

    def reset_peak_memory_stats(self, *args, **kwargs):
        return self.recorder.result("cuda.reset_peak_memory_stats", None)

    def is_initialized(self):
        return self.recorder.result("cuda.is_initialized", self._initialized)

    def max_memory_allocated(self, *args, **kwargs):
        return self.recorder.result("cuda.max_memory_allocated", 0)

    def max_memory_reserved(self, *args, **kwargs):
        return self.recorder.result("cuda.max_memory_reserved", 0)


class FakeTorch:
    long = "CPU_FAKE_INT64"

    def __init__(self, recorder):
        self.recorder, self.cuda = recorder, FakeCuda(recorder)
        self.created = []
        self.tensor_arguments = []

    def manual_seed(self, *args, **kwargs):
        self.recorder.arguments.setdefault("torch.manual_seed", []).append((args, kwargs))
        return self.recorder.result("torch.manual_seed", None)

    def tensor(self, rows, *, dtype, device):
        self.recorder.result("torch.tensor", None)
        self.tensor_arguments.append({"rows": deepcopy(rows), "dtype": dtype, "device": device})
        result = FakeTensor(self.recorder, "ids", rows, dtype, device)
        self.created.append(weakref.ref(result))
        return result

    def ones_like(self, tensor):
        self.recorder.result("torch.ones_like", None)
        # Internal fixture storage access deliberately adds no source property
        # reads to the expression-under-test's shape/dtype/device/toList trace.
        rows = [[1 for _ in row] for row in tensor._rows]
        result = FakeTensor(self.recorder, "mask", rows, tensor._dtype, tensor._device)
        self.created.append(weakref.ref(result))
        return result

    def is_tensor(self, value):
        role = value.role if type(value) is FakeTensor else "unknown"
        return self.recorder.result(role + ".is_tensor", type(value) is FakeTensor)


class InjectedAPI(dict):
    """Named callback map whose calls/property reads are independently visible."""
    def __init__(self, faults=None):
        super().__init__()
        self.recorder = Recorder(faults)
        self.torch = FakeTorch(self.recorder)
        self["runtime_version"] = "CPU_FAKE_NO_FRAMEWORK"
        self.update({name: self.callback(name, value) for name, value in {
            "config_preflight": {"scope": "CPU_INJECTED_PREFIX_INPUT_ONLY", "identity": "CPU_FIXTURE_CONFIG"},
            "ownership_verify": True,
            "disk_free": 64 * 1024**3, "torch_import": self.torch,
            "torch_version_read": "CPU_FAKE_NO_FRAMEWORK",
            "growth_observation": {"available_bytes": 8 * 1024**3, "worker_RSS_bytes": 512 * 1024**2},
            "configure": None, "python_seed": None,
        }.items()})

    def callback(self, name, default):
        def invoke(*args, **kwargs):
            self.recorder.arguments.setdefault("api." + name, []).append((args, kwargs))
            return self.recorder.result("api." + name, default)
        return invoke

    def guard(self, stage, torch_view=None):
        return self.recorder.guard(stage, torch_view)


EXPECTED_CHECK_READS = [
    "ids.is_tensor", "ids.shape", "ids.dtype", "ids.device", "ids.device.str", "ids.tolist",
    "mask.is_tensor", "mask.shape", "ids.shape", "mask.dtype", "mask.device", "mask.device.str", "mask.tolist",
]


def check_reads(recorder):
    return [event for event in recorder.events if event.startswith(("ids.", "mask."))
            and not event.endswith(".released")]
