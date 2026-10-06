@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > fixture_build_001\VCVARS_OUTPUT.txt 2>&1
if errorlevel 1 exit /b %errorlevel%
cl.exe /nologo /c /TC /O2 /W4 /WX /GS- /Zl observer_only_fixture.c /Fofixture_build_001\observer_only_fixture.obj > fixture_build_001\COMPILE_OUTPUT.txt 2>&1
if errorlevel 1 exit /b %errorlevel%
link.exe /NOLOGO /DLL /NOENTRY /NODEFAULTLIB /MACHINE:X64 /EXPORT:plfixture_ping /OUT:fixture_build_001\observer_only_fixture.dll fixture_build_001\observer_only_fixture.obj > fixture_build_001\LINK_OUTPUT.txt 2>&1
if errorlevel 1 exit /b %errorlevel%
dumpbin.exe /imports fixture_build_001\observer_only_fixture.dll > fixture_build_001\IMPORTS.txt
if errorlevel 1 exit /b %errorlevel%
dumpbin.exe /exports fixture_build_001\observer_only_fixture.dll > fixture_build_001\EXPORTS.txt
if errorlevel 1 exit /b %errorlevel%
endlocal
