# Direct CPU process fixture, 2026-10-02

Observer `/root`. Receipt
`evidence/LEARNING_TOKENIZATION_PROCESS_FIXTURE_001.json` SHA
72c28c0841fdd1c35ff771b75d294d84d786a295a0435365b802bf50823a230d
records a bounded helper using the direct base CPython executable (hash28d2965d)
and pinned psutil7.2.2. This is an actual Windows API fixture, not token instrument
qualification, OS containment, unseen-side-effect attestation or a model run.

The helper command used `-I -S -B -u -c` and this fixed body:

```python
import sys, time, json
print(json.dumps({
    "executable": sys.executable,
    "version": sys.version.split()[0],
    "forbidden_modules_loaded": [m for m in
        ("torch", "transformers", "tokenizers", "safetensors") if m in sys.modules]
}), flush=True)
time.sleep(0.25)
```

Parent observations verified the exact child executable, PID/creation time,
no sampled descendants, four RSS observations and exit0. Observed peak RSS was
10.796875MiB; measured parent wall time0.4433259s. The child reported CPython3.11.15
and none of the listed modules loaded. There were no tokenizer encode calls,
model loads/forwards, training updates, canonical host-DB access or quarantined
helper use. The five-second fixture deadline was not exercised as a failure test
and this success does not qualify another launcher's timeout implementation.

The receipt's `sampling_seconds: 0.01` is the requested inter-loop sleep, not a
measured maximum observation gap. Timestamped gaps and complete peak telemetry
were not captured here; sampled RSS is not an absolute peak. A separate read-only
actual psutil query confirmed that Windows memory_info exposes `peak_wset`.
The eventual instrument must record actual sample times/gaps and a final live
kernel peak counter before child exit, enforce its unchanged resource ceiling,
and independently test deadline/cleanup behavior. No source/measurement approval
is inferred from this small fixture; fresh non-builder source reviews remain.
