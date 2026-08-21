from rineya_soup_lab.gpu import competing_python_gpu_rows, model_from_pretrained_kwargs


def test_model_kwargs_offload_cap(tmp_path) -> None:
    kwargs = model_from_pretrained_kwargs(
        {"max_memory_gpu_gib": 6, "max_memory_cpu_gib": 30, "device_map": "auto"},
        offload_root=tmp_path,
    )
    assert kwargs["device_map"] == "auto"
    assert kwargs["max_memory"][0] == "6.0GiB"
    assert kwargs["max_memory"]["cpu"] == "30.0GiB"
    assert kwargs["low_cpu_mem_usage"] is True
    assert kwargs["offload_folder"].endswith("_hf_offload")


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
