# Concrete Windows Torch import obligations

The installed pinned Torch __init__ source invokes Windows _load_dll_libraries
before importing torch._C. It selects torch/lib *.dll, attempts VC runtime names,
can add optional environment-dependent DLL directories, and has a fallback PATH
route when LoadLibraryEx fails. These are source obligations, not observed loads.

TORCH_STATIC_EFFECT_OBLIGATIONS001.json parses the actual import and delay-import
tables of 37 selected Torch DLLs and torch/_C.cp311-win_amd64.pyd after checking
their current registered hashes. It reads 3,763,222,184 bytes in 3.718 seconds,
using bounded 1 MiB reads, size sentinel and opened/path metadata joins. The
existing passive PE parser is compiled alone; no native module is imported.

There are 244 literal PE edges. Thirty-four ordinary-import names have a unique
selected candidate; 24 candidate paths are outside observer009's 35-library
baseline. This cannot determine when Windows loads them or which guard owns each
addition. torch._C imports torch_python.dll, KERNEL32.dll, VCRUNTIME140.dll and
the API-set CRT runtime name. Remaining unresolved edges are 107 API-set names,
eight edges for unselected MSVCP140.dll/nvcuda.dll names, and two ambiguous names
asmjit.dll/fbgemm.dll. nvcuda.dll appears as a delay import. These counts classify
edges, not independent missing files.

The minimal missing static byte facts are exact selected MSVCP140.dll and
nvcuda.dll candidate identities and a reviewed choice between duplicate
asmjit/fbgemm locations. API-set resolution, optional-directory selection,
delay-load reachability, extension import order and precise per-boundary library
effects remain dynamic obligations. They are not filled with guessed zero
effects or an all-DLL union. No actual tensor-program baseline is established.

TWO_WINDOWS_DLL_CANDIDATES001.json now supplies passive identities for exactly
C:/Windows/System32/msvcp140.dll (643,512 bytes) and nvcuda.dll (3,807,352 bytes).
ADDITIVE_WINDOWS_RUNTIME_CANDIDATE001.json retains the tensor candidate's 8,714
descriptors and adds these two, for 8,716 files. This count is a different map
from the historical observer superset. The Torch source's explicit absolute
torch/lib DLL loop selects the lib copies of asmjit/fbgemm; bin copies remain
byte-rostered. Neither that source choice nor selected System32 candidates
proves Windows imported-name resolution or exact boundary effects. No automatic
recursive System32 scan occurred.

A reviewable future measurement could be one owned source-pinned -I -S -B
Torch-import-only qualification stage using the intended registered loader,
observer, source/API journal, Job/resource/death checks and selected exact DLL
maps. It would stop before user CUDA context initialization and the two input
tensors, perform no model load/forward/backward/optimizer/decoding/scoring, and
produce actual named extension/frame effects for registration. Torch import may
itself load CUDA-dependent libraries and reach driver interfaces; that scope
needs explicit owner permission and must be stated honestly. It is not covered
by the existing NoModel observer12-operation grant. Alternatively the intended
two-tensor diagnostic could include the import observation with exact preregistered
falsifiable effects and stop on any mismatch; uncertain effects will then cause
an honest failure. Root decides the concrete proposal; no launch occurred here.
