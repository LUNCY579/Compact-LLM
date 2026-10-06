@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > native_build_001\VCVARS_OUTPUT.txt 2>&1
if errorlevel 1 exit /b 1
cl > native_build_001\COMPILER_VERSION.txt 2>&1
link > native_build_001\LINKER_VERSION.txt 2>&1
cl /nologo /c /TC /O2 /W4 /WX /GS- /Zl /Fonative_build_001\prefix_library_observer.obj prefix_library_observer.c > native_build_001\COMPILE_OUTPUT.txt 2>&1
if errorlevel 1 exit /b 1
link /nologo /DLL /NOENTRY /NODEFAULTLIB /MACHINE:X64 /EXPORT:plobs_abi_version /EXPORT:plobs_start /EXPORT:plobs_snapshot /EXPORT:plobs_copy_rows /EXPORT:plobs_stop /OUT:native_build_001\native_library_observer.dll native_build_001\prefix_library_observer.obj > native_build_001\LINK_OUTPUT.txt 2>&1
if errorlevel 1 exit /b 1
dumpbin /imports native_build_001\native_library_observer.dll > native_build_001\IMPORTS.txt 2>&1
if errorlevel 1 exit /b 1
dumpbin /exports native_build_001\native_library_observer.dll > native_build_001\EXPORTS.txt 2>&1
if errorlevel 1 exit /b 1
dumpbin /symbols native_build_001\prefix_library_observer.obj > native_build_001\SYMBOLS.txt 2>&1
if errorlevel 1 exit /b 1
dumpbin /disasm native_build_001\prefix_library_observer.obj > native_build_001\OBJECT_DISASSEMBLY.txt 2>&1
if errorlevel 1 exit /b 1
dumpbin /disasm native_build_001\native_library_observer.dll > native_build_001\DLL_DISASSEMBLY.txt 2>&1
exit /b %errorlevel%
