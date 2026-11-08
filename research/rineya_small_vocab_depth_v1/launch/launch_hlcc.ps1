# Launch HLCC-001 through WMI (parent WmiPrvSE, outside every Claude tool's process tree) using the
# COMMITTED launcher: launch/smallvd_runner.py runs launch/hlcc_run.sh, which refuses unless run_hlcc.py,
# hlcc_verdict.py, linear_compensation.py, the panel (8eec83da) and the all-axis map (b5137cde) hash to
# the registered values (HLCC001_PREREG.md §9-10). Run ONLY after CLOCKFIX-001's "ended".
# Run dir and log go to the session scratchpad (never the records tree). Prints the runner pid.
$py = "C:\Users\clclc\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe"
$sp = "C:\Users\clclc\AppData\Local\Temp\claude\C--Users-clclc-BusinessProject-ChatBubble-dose-param\0896c09e-a683-438c-88e0-80305acf12ed\scratchpad"
$scratchFwd = $sp -replace '\\', '/'
$pkg = "C:\Users\clclc\BusinessProject\ChatBubble-small-vocab-depth\research\rineya_small_vocab_depth_v1"
$pkgFwd = $pkg -replace '\\', '/'
$ts = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$rd = "$sp\hlcc_run_$ts"
$logFwd = "$scratchFwd/hlcc_run_$ts.log"
$cl = "`"$py`" -u `"$pkg\launch\smallvd_runner.py`" `"$rd`" `"$scratchFwd`" `"$pkg`" `"C:\Program Files\Git\bin\bash.exe`" `"$pkgFwd/launch/hlcc_run.sh`" `"$logFwd`""
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cl; CurrentDirectory=$pkg; ProcessStartupInformation=$si}
"ReturnValue=$($r.ReturnValue) RunnerPid=$($r.ProcessId) RunDir=$rd Log=$logFwd"
