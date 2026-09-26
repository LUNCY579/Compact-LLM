# Launch ABST-001 through WMI (parent WmiPrvSE, outside every Claude tool's process tree) using the
# COMMITTED launcher: launch/smallvd_runner.py runs launch/abst_run.sh, which refuses unless every file in
# launch/abst_hashes.txt hashes to its registered value (ABST001_PREREG.md §9-10). Run ONLY in the agreed
# queue slot. Run dir and log go to the session scratchpad (never the records tree). Prints the runner pid.
$py = "C:\Users\clclc\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe"
$sp = "C:\Users\clclc\AppData\Local\Temp\claude\C--Users-clclc-BusinessProject-ChatBubble-dose-param\0896c09e-a683-438c-88e0-80305acf12ed\scratchpad"
$scratchFwd = $sp -replace '\\', '/'
$pkg = "C:\Users\clclc\BusinessProject\ChatBubble-small-vocab-depth\research\rineya_small_vocab_depth_v1"
$pkgFwd = $pkg -replace '\\', '/'
$expectedRunSh = "1d091677df8ed35edbfee3be5bd702534f67fd23af7d13b6e5d711290b9119b4"
$gotRunSh = (Get-FileHash -Algorithm SHA256 "$pkg\launch\abst_run.sh").Hash.ToLower()
if ($gotRunSh -ne $expectedRunSh) { "REFUSE: launch/abst_run.sh is not the registered script ($gotRunSh)"; exit 3 }
$ts = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$rd = "$sp\abst_run_$ts"
$logFwd = "$scratchFwd/abst_run_$ts.log"
$cl = "`"$py`" -u `"$pkg\launch\smallvd_runner.py`" `"$rd`" `"$scratchFwd`" `"$pkg`" `"C:\Program Files\Git\bin\bash.exe`" `"$pkgFwd/launch/abst_run.sh`" `"$logFwd`""
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cl; CurrentDirectory=$pkg; ProcessStartupInformation=$si}
"ReturnValue=$($r.ReturnValue) RunnerPid=$($r.ProcessId) RunDir=$rd Log=$logFwd"
