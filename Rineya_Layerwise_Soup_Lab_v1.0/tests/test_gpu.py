from rineya_soup_lab.gpu import (
    competing_python_gpu_rows,
    model_from_pretrained_kwargs,
    resolve_gpu_memory_gib,
    should_wait_for_gpu,
)


def test_model_kwargs_offload_cap(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("RINEYA_MAX_MEMORY_GPU_GIB", raising=False)
    monkeypatch.delenv("RINEYA_MAX_MEMORY_CPU_GIB", raising=False)
    monkeypatch.setattr("rineya_soup_lab.gpu.gpu_memory_gib", lambda: (None, None))
    kwargs = model_from_pretrained_kwargs(
        {"max_memory_gpu_gib": 6, "max_memory_cpu_gib": 30, "device_map": "auto"},
        offload_root=tmp_path,
    )
    assert kwargs["device_map"] == "auto"
    assert kwargs["max_memory"][0] == "6.0GiB"
    assert kwargs["max_memory"]["cpu"] == "30.0GiB"
    assert kwargs["low_cpu_mem_usage"] is True
    assert kwargs["offload_folder"].endswith("_hf_offload")


def test_env_gpu_cap_overrides_yaml_and_clamps_to_free(monkeypatch) -> None:
    monkeypatch.setenv("RINEYA_MAX_MEMORY_GPU_GIB", "3.5")
    monkeypatch.setattr("rineya_soup_lab.gpu.gpu_memory_gib", lambda: (3.0, 8.0))
    # free=5.0, headroom=0.75 -> clamp 3.5 stays 3.5
    assert resolve_gpu_memory_gib({"max_memory_gpu_gib": 6}) == 3.5
    monkeypatch.setattr("rineya_soup_lab.gpu.gpu_memory_gib", lambda: (6.5, 8.0))
    # free=1.5, headroom=0.75 -> clamp to 0.75
    assert abs(resolve_gpu_memory_gib({"max_memory_gpu_gib": 6}) - 0.75) < 1e-9


def test_wait_flag_env(monkeypatch) -> None:
    monkeypatch.delenv("RINEYA_WAIT_FOR_GPU", raising=False)
    assert should_wait_for_gpu({"wait_for_gpu": True}) is True
    monkeypatch.setenv("RINEYA_WAIT_FOR_GPU", "0")
    assert should_wait_for_gpu({"wait_for_gpu": True}) is False


def test_competing_python_gpu_filters_desktop(monkeypatch) -> None:
    class Fake:
        returncode = 0
        stdout = (
            "123, python.exe, 2048 MiB\n"
            "999, NVIDIA Overlay.exe, 50 MiB\n"
            "1, python.exe, 100 MiB\n"
        )

    monkeypatch.setattr("rineya_soup_lab.gpu.subprocess.run", lambda *args, **kwargs: Fake())
    rows = competing_python_gpu_rows(ours="1")
    assert len(rows) == 1
    assert "123" in rows[0]


def test_wait_requires_consecutive_clears(monkeypatch) -> None:
    calls = {"n": 0}

    def fake_rows() -> list[str]:
        calls["n"] += 1
        return []

    sleeps: list[int] = []
    monkeypatch.setattr("rineya_soup_lab.gpu.competing_python_gpu_rows", fake_rows)
    monkeypatch.setattr("rineya_soup_lab.gpu.time.sleep", lambda seconds: sleeps.append(seconds))
    from rineya_soup_lab.gpu import wait_for_python_gpu

    wait_for_python_gpu(poll_seconds=60, clear_polls=2, settle_seconds=15)
    assert calls["n"] == 2
    assert sleeps == [15]
