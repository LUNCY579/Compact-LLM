"""Runner launched through WMI (outside every tool process tree). Opens its own log, runs the
target unbuffered, and writes rc.json on exit. `--probe N` just sleeps N seconds (launcher test)."""
import json, subprocess, sys, time
from pathlib import Path

import os
PY = os.environ.get("RUNNER_PY") or r"C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe"
args = sys.argv[1:]
if args[0] == "--probe":
    out = Path(args[2]); out.mkdir(parents=True, exist_ok=True)
    (out / "probe_started.json").write_text(json.dumps({"t": time.time()}))
    time.sleep(int(args[1]))
    (out / "rc.json").write_text(json.dumps({"rc": 0, "probe": True, "t": time.time()}))
    sys.exit(0)

target, workdir, log_path, rc_path = Path(args[0]), Path(args[1]), Path(args[2]), Path(args[3])
if log_path.exists() or rc_path.exists():
    sys.exit(9)
t0 = time.time()
with open(log_path, "xb") as log:
    log.write(f"[wmi_runner] start {time.strftime('%Y-%m-%d %H:%M:%S')} target={target}\n".encode()); log.flush()
    try:
        rc = subprocess.call([PY, "-u", str(target)], cwd=str(workdir), stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL)
    except Exception as e:                      # record, never swallow
        log.write(f"[wmi_runner] launch error {e!r}\n".encode()); rc = -1
    log.write(f"[wmi_runner] exit rc={rc} after {time.time() - t0:.0f}s\n".encode())
rc_path.write_text(json.dumps({"rc": rc, "seconds": round(time.time() - t0), "end": time.strftime("%Y-%m-%d %H:%M:%S")}))
