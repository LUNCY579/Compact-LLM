"""Mock-only reserve-pressure checks; no CUDA workload or model is loaded."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sbhm import protocol


GIB = 2**30


@pytest.fixture(autouse=True)
def isolate_events():
    protocol.CACHE_RELEASE_EVENTS.clear()
    yield
    protocol.CACHE_RELEASE_EVENTS.clear()


def mock_resources(monkeypatch, free, reserved=5*GIB, allocated=2*GIB,
                   available_ram=16*GIB, rss=GIB):
    monkeypatch.setattr(protocol.psutil, 'virtual_memory',
                        lambda: SimpleNamespace(available=available_ram))
    monkeypatch.setattr(protocol.psutil, 'Process',
                        lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=rss)))
    gpu = SimpleNamespace(
        available=Mock(return_value=True),
        info=Mock(side_effect=[(value, 8*GIB) for value in free]),
        reserved=Mock(side_effect=reserved if isinstance(reserved, list) else None,
                      return_value=reserved if not isinstance(reserved, list) else None),
        allocated=Mock(side_effect=allocated if isinstance(allocated, list) else None,
                       return_value=allocated if not isinstance(allocated, list) else None),
        empty=Mock(),
    )
    monkeypatch.setattr(protocol.torch.cuda, 'is_available', gpu.available)
    monkeypatch.setattr(protocol.torch.cuda, 'mem_get_info', gpu.info)
    monkeypatch.setattr(protocol.torch.cuda, 'memory_reserved', gpu.reserved)
    monkeypatch.setattr(protocol.torch.cuda, 'memory_allocated', gpu.allocated)
    monkeypatch.setattr(protocol.torch.cuda, 'empty_cache', gpu.empty)
    return gpu


def test_runtime_releases_unused_cache_then_passes_original_reserve(monkeypatch):
    gpu = mock_resources(monkeypatch, [GIB//2, 3*GIB], reserved=[5*GIB, 2*GIB],
                         allocated=[2*GIB, 2*GIB])
    result = protocol.resources('cuda:0', runtime=True)
    gpu.empty.assert_called_once_with()
    assert gpu.info.call_count == 2
    assert result['free_vram_bytes'] == 3*GIB
    event, = protocol.CACHE_RELEASE_EVENTS
    assert datetime.fromisoformat(event['created_utc']).utcoffset().total_seconds() == 0
    assert {key: value for key, value in event.items() if key != 'created_utc'} == {
        'device': 'cuda:0', 'free_vram_before_bytes': GIB//2,
        'reserved_before_bytes': 5*GIB, 'allocated_before_bytes': 2*GIB,
        'free_vram_after_bytes': 3*GIB, 'reserved_after_bytes': 2*GIB,
        'allocated_after_bytes': 2*GIB, 'recovered': True}


def test_runtime_reclaim_does_not_lower_reserve_if_free_stays_low(monkeypatch):
    gpu = mock_resources(monkeypatch, [GIB//2, 3*GIB//4], reserved=[5*GIB, 4*GIB])
    with pytest.raises(protocol.Blocked, match='less than 1 GiB free VRAM safety reserve'):
        protocol.resources('cuda:0', runtime=True)
    gpu.empty.assert_called_once_with()
    assert gpu.info.call_count == 2
    event, = protocol.CACHE_RELEASE_EVENTS
    assert event['free_vram_after_bytes'] == 3*GIB//4
    assert event['recovered'] is False


def test_runtime_live_pressure_without_unused_cache_fails_without_cleanup(monkeypatch):
    gpu = mock_resources(monkeypatch, [GIB//2], reserved=2*GIB, allocated=2*GIB)
    with pytest.raises(protocol.Blocked, match='less than 1 GiB free VRAM safety reserve'):
        protocol.resources('cuda:0', runtime=True)
    gpu.empty.assert_not_called()
    assert gpu.info.call_count == 1
    assert protocol.CACHE_RELEASE_EVENTS == []


@pytest.mark.parametrize('runtime', [False, True])
@pytest.mark.parametrize('free', [GIB, 3*GIB])
def test_healthy_checks_never_release_or_query_allocator_cache(monkeypatch, runtime, free):
    gpu = mock_resources(monkeypatch, [free])
    assert protocol.resources('cuda:0', runtime=runtime)['free_vram_bytes'] == free
    gpu.empty.assert_not_called()
    gpu.reserved.assert_not_called()
    gpu.allocated.assert_not_called()
    assert protocol.CACHE_RELEASE_EVENTS == []


def test_admission_does_not_reclaim_cache_to_bypass_reserve(monkeypatch):
    gpu = mock_resources(monkeypatch, [GIB//2])
    with pytest.raises(protocol.Blocked, match='less than 1 GiB free VRAM safety reserve'):
        protocol.resources('cuda:0', runtime=False)
    gpu.empty.assert_not_called()
    gpu.reserved.assert_not_called()
    gpu.allocated.assert_not_called()
    assert protocol.CACHE_RELEASE_EVENTS == []


@pytest.mark.parametrize('runtime,ram', [(False, int(7.4*GIB)-1), (True, 6*GIB-1)])
def test_low_ram_blocks_before_any_gpu_cleanup(monkeypatch, runtime, ram):
    gpu = mock_resources(monkeypatch, [GIB//2], available_ram=ram)
    with pytest.raises(protocol.Blocked, match='free RAM below'):
        protocol.resources('cuda:0', runtime=runtime)
    gpu.info.assert_not_called()
    gpu.empty.assert_not_called()
    assert protocol.CACHE_RELEASE_EVENTS == []


def test_rss_ceiling_blocks_before_any_gpu_cleanup(monkeypatch):
    gpu = mock_resources(monkeypatch, [GIB//2], rss=4*GIB+1)
    with pytest.raises(protocol.Blocked, match='worker RSS exceeds 4 GiB guard'):
        protocol.resources('cuda:0', runtime=True)
    gpu.info.assert_not_called()
    gpu.empty.assert_not_called()
    assert protocol.CACHE_RELEASE_EVENTS == []


@pytest.mark.parametrize('device,available', [('cpu', True), ('cuda:1', True), ('cuda:0', False)])
def test_cleanup_never_enables_device_fallback(monkeypatch, device, available):
    gpu = mock_resources(monkeypatch, [GIB//2])
    gpu.available.return_value = available
    with pytest.raises(protocol.Blocked, match='explicit cuda:0; no CPU fallback'):
        protocol.resources(device, runtime=True)
    gpu.info.assert_not_called()
    gpu.empty.assert_not_called()
    assert protocol.CACHE_RELEASE_EVENTS == []
