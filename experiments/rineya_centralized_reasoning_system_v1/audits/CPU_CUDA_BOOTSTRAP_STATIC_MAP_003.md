# CUDA bootstrap003: bounded static source map

Date: 2026-10-05. Reviewer `/root/exception_ops`.
**ADVISORY_ONLY_NOT_SOURCE_OR_IMPORT_ACCEPTANCE.** This maps only the immediate
pinned CUDA definition bootstrap implicated by CPU2-OPS-001, plus the specifically
named memory/viz source distinction. No Torch/framework import, execution,
falsifier/test, asset/model/GPU/database action or source edit occurred. Only this
report was written; checks stop here. Full CPU import/fit remains unmeasured.

All paths below are relative to the exact registered directory
`C:/Users/clclc/BusinessProject/.venv-mq001/Lib/site-packages/torch/cuda/`.
Selected source bytes were read/hashed and parsed as AST, never executed.

| File | SHA256 | Immediate definition classes |
| --- | --- | --- |
| __init__.py | `674c693e77b8355458200b6851D5778427e68ee061da0a916c582933173979e9` | Main module body |
| _device_limits.py | `36c91fbe3a09c9645f1b72d74cc00a0154f7449a47bbe072920f4225a30bb02a` | GPULimits |
| gds.py | `c4576634e0367b54ab6844a5e4d54557eb98affcee482f596884fd37c4cdc6e8` | GdsFile |
| _utils.py | `a09393e526033163339aa04d67d460ad0b25f91af801cfb95178b319f8302ec7` | _CudaModule, _CudaKernel |
| graphs.py | `65b7411c148f4393f7d61ca9c43809c8ae1090a044dcdf3d750b790addb83214` | CUDAGraph, graph |
| green_contexts.py | `7d07da19bc3a6695e35a9c6e7f71c6fedaab4475b3947d08a633f745c380c9a1` | GreenContext |
| streams.py | `be5e604030c3903ecbea3fadf3e65ef5f641f66985be328ebbd6514c6de5d88c` | Stream, ExternalStream, Event |
| memory.py | `025a9a345f5b036fd3dd15fe04705a8054cbecbbe9879330c0c63a0ef57fdb96` | Later import, not initial None bootstrap |
| _memory_viz.py | `9c8ce244f8f9d998faa311c226a1e63cfe8cf17b8a639e12bcae8547311ec270` | Bytes; later import, not initial None bootstrap |

The main module imports the six immediate sibling files at29–39, then assigns
`_initialized=False` at50. Nested sibling module/class bodies can therefore enter
and return while the main attribute is absent/None. These six have no further
relative CUDA-submodule imports in their top-level AST; graphs imports the
existing non-CUDA torch._utils helper. This is a finite immediate map, not a
recursive audit of every transitive import or compiled binary.

## Minimal None allowance

During the registered import phase only, permit None entry/return for `<module>`
in exactly torch.cuda and those six exact sibling modules, and the listed class
definition code objects at their frozen AST locations. Require matching pinned
filename/module/name/location and definition code flags; do not treat a method,
constructor call or arbitrary same-file function as a class definition. Bind the
entry/return decision to the same permitted definition identity. None means
initialization metadata not yet assigned, never a fabricated initialized=False.
Unknown files/events, initialized=True and all post-import None cases refuse.

Normal functions remain outside that allowance. The existing `_is_compiled`,
`is_initialized` and narrowly queued `_lazy_call` decisions still require verified
False state and their exact main-file source/import phase. `_lazy_call` can invoke
its callback when initialized, so None/True must not admit it or any callback body.
Availability/init/device/RNG/sync/CUDA-allocation and C `_cuda_*` actions stay fatal.
Nested calls from an allowed definition are checked independently; permitting
the definition event is not permission for a device body inside it. Hidden C
effects remain UNMEASURED; this is not exhaustive zero-GPU proof.

## Important limits already visible in the bounded map

memory.py is imported by the main module at1513, after False is assigned; it imports
_memory_viz at27. Neither needs a None exception merely because it is CUDA-named.
The viz module's legend at233 constructs/formats `Bytes(PAGE_SIZE)`; Bytes.__init__
assigns a Python value and its repr calls the size-format helper. Those are
ordinary function calls, not definition events, so this map does not authorize
them. If the final source requires them, their exact pinned metadata-only bodies
and restricted context need explicit treatment within the registered scope;
module/class None permission alone does not establish complete import viability.

Likewise gds defines conditional `_dummy_fn` registrations when compiled GDS
symbols are absent. The inspected helper returns a Python function that raises
later; it does not itself register/use a GPU buffer. Whether that conditional
branch runs in the installed binary was not evaluated. No blanket gds helper or
device-body permission follows. Preserve that uncertainty rather than claim an
import path was executed successfully.

The map supports only the known initial bootstrap correction under final scope003.
Fresh frozen source/fixture/SCI/OPS review must adjudicate actual rule fidelity
and remaining viability. No model attempt, new policy/cap, extra round or existing
GPU/probe/native qualification is granted.
