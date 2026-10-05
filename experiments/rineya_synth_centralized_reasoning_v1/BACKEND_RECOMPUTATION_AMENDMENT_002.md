# Full decoder recomputation qualification amendment

The first uncheckpointed phase A failed the unchanged GPU free-memory floor
before any loss/backward/update. Its source/config/receipts remain immutable.
This successor changes only decoder recomputation and its source/cost binding;
it is a new controlled scope, not an unchanged retry or study warmstart.

Keep the pinned base, BF16 storage/FP32 compute, evaluation mode, zero dropout,
112 rank-8 q/v factors, 1,146,880 trainables, source rows/masks, optimizer/global
learning rates, 131,072-token study ceiling, numerical tolerances and every
resource/deadline floor unchanged. Nonreentrant checkpointing completes the full
decoder function, preserves RNG, prohibits cache mutation and bypasses no-grad.
Exact parameter objects/storage/flags and original forward methods are checked.

Phase A still has 202 whole-model forwards / 48,112 input positions, 200 native
backwards, ten updates and 16 full-base checks. Additional decoder work is
explicit: 56 no-grad regions, 5,600 initial regions and 5,600 full recomputations.
Phase B still has 42 whole-model forwards, one native update, 20 reference
gradients and nine base checks, with 56/1,120/1,120 decoder regions. Each region
has a paired identity, layer, outcome and duration; missing, duplicated, failed
or pending work forbids success. Region files are owned, byte/hash bound and
verified directly before B. Parameter/witness equality cannot replace that
execution evidence. All cleanup paths preserve partial work and terminal costs.

The reviewed candidate and current-source tiny CPU proof are pinned. The runtime
closure extends the original 64 pins to 82, including actual checkpoint Python
dependencies and the interpreter DLL. Seven focused amendment controls pass;
the complete integrated source set passes 571 CPU tests without skips or source
changes. CPU storage measurements and in-memory continuation do not establish
native VRAM, disk-resume or sustained fit.

Any native attempt requires fresh host admission, clean committed sources, a new
write-once explicit qualification-only config and independent terminal audit.
No paid services, protected banks, study training, DEV scoring, full resumed-nine,
full-27 or transfer/retention conclusion is opened by this amendment.
