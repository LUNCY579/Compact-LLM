$p = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
       Where-Object { $_.CommandLine -match 'SupaExperiment' })
$os = Get-CimInstance Win32_OperatingSystem
"{0} {1}" -f $p.Count, [math]::Round($os.FreePhysicalMemory / 1MB, 2)
