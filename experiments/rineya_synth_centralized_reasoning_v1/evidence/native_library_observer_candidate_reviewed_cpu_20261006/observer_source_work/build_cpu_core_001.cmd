@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > cpu_build_001\VCVARS_OUTPUT.txt 2>&1
if errorlevel 1 exit /b 1
cl /nologo /TC /O2 /W4 /WX /GS- /DPLO_CORE_ONLY /Fecpu_build_001\test_prefix_library_observer_core.exe /Focpu_build_001\ prefix_library_observer.c test_prefix_library_observer_core.c > cpu_build_001\COMPILE_OUTPUT.txt 2>&1
if errorlevel 1 exit /b 1
cpu_build_001\test_prefix_library_observer_core.exe > cpu_build_001\TEST_OUTPUT.txt 2>&1
exit /b %errorlevel%
