# Resource policy

30 September 2026. The user's revision authorizes a low-host-RAM laptop route.
The inherited 7.4/6 GiB admission floors budgeted a GPU model plus a CPU reference;
they were not measured requirements of this Titan checkpoint. The current loader
constructs BF16 weights directly on GPU and copies through a fixed 2 MiB buffer,
without a complete CPU model or checkpoint memory map.

| Named scope | Worker process-tree RSS cap | Available system RAM reserve | Before-worker budget including supervisor | Evidence |
|---|---|---|---|---|
| LAPTOP_GPU_RESIDENT_V1 | 2.5 GiB | 2 GiB | Dynamic growth budget | Exact load and frozen-forward pilots; historical profile |
| LAPTOP_STREAMED_V1 / CE512 | 1.75 GiB | 1.25 GiB | 3.0625 GiB | Complete short adapter mechanics passed; sustained training unmeasured |
| LAPTOP_CAPTURE_RESIDENT_V1 | 2 GiB | 1.25 GiB | 3.3125 GiB | Capture memory checks and held-hidden diagnostic passed; capture likelihood remains failed |

Admission after imports requires reserve + max(0, RSS cap - current worker RSS).
The before-worker budget reserves the full allowed worker growth, plus 64 MiB
for the supervisor. These bounds are prospective limits, not measured demand.
GPU admission/runtime free-memory floors are 4/0.5 GiB; disk free floor is 4 GiB.
External 100 ms process-tree monitoring and Windows containment remain active.
Internal stage samples and external peaks are recorded separately; either can
observe a peak missed by the other.

The complete CE512 fixture observed 1.748646 GiB peak process RAM and 3.098094 GiB
peak CUDA allocation. Only 1.387 MiB of RSS margin remained. A future sustained
training successor must budget reader/driver/accumulation overhead and measure
that exact envelope; a proposed 2 GiB cap is not presently a qualified training
profile. All prepared TRAIN rows have maximum length 398, allowing context 512
without changing the data dose or truncating targets.

Use one resident GPU worker, microbatch 1, streamed prepared sequences and
adapter-only checkpoints. Keep the frozen base and adapter optimizer on GPU;
run decoding and evaluation sequentially. Do not create a CPU reference or add
CPU optimizer offload to the first low-RAM route. For a measured VRAM failure,
register and qualify activation checkpointing before changing precision or
quantizing the base. Short-context fit does not qualify 2048/4096 preservation.

Native short capture/diagnostics have a 600-second ceiling. CE qualification
retains 1800 seconds and its registered update cap. Future E1 needs current
source/input/runtime/budgets and its own native executor; historical per-seed
24-hour ceilings are not launch authority. Resource success grants no numerical
or efficacy qualification. Preserve fixed numerical tolerances and all failed
work. A controller admission timeout is terminal; retry only after a real state
change or an independently registered successor, with durable new receipts.

## Historical policy

Original configs retain PRLC minima: available host RAM 7.4 GiB before load,
6 GiB throughout, GPU 4/0.5 GiB, disk 4 GiB. They and old receipts remain unchanged.
Named laptop profiles in Amendments 009-016 supersede them only for their explicit
scopes. No user application is terminated; no cloud rental or paid teacher calls
are authorized. CHECK/FINAL remain closed.


AMENDMENT_018 adds a source-reviewed bounded resource wait after framework imports
for the causal diagnostic only. It retains2 GiB workerRSS,1.25 GiB systemreserve,
GPU4/0.5 GiB,disk4 GiB and600seconds; no model call precedes two current full-budget
observations and a strictfresh preload check. The maximum wait is60seconds with
2-second polls. Timeout or source/runtime-floor failure stops without model retry.
The prior admitted worker failed before any model call; the successor's first
preworker window remained below3.3125 GiB. It is not native-qualified or E1-authorized.
