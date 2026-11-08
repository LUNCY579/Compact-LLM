# Amendment 017: bounded E1 reader, accounting and accumulation components

30 September 2026. The low-RAM route has qualified short CE mechanics, but no
native production trainer is connected. Host headroom remains below the GPU
launch budget; the latest automatic agent turn separately failed with a model
service-capacity error. Preserve both conditions without attributing the app's
stalled flag to a measured Titan hardware requirement.

Register three new internal components and their CPU controls. Leave all old
E0/E1 configs, source, plan, data and failed receipts unchanged. The reader binds
the byte-frozen historical seed-101 PLAN/DATA_BINDING and actual TRAIN source-join
receipt, retains offsets and metadata rather than token populations, and yields
one complete array in immutable schedule order. Bound lines at64 KiB, individual
metadata files at8 MiB, combined metadata at16 MiB and retained measured metadata
objects at32 MiB. The latter excludes interpreter/allocator overhead; separately
measure OS process high-water RSS. Validate every consumed line hash, file IDs
and Windows ChangeTime, with mandatory final whole-file hashes. These are practical
drift checks, not adversarial filesystem immutability. Prior language replies
remain masked and final supervised EOS remains included.

The existing E0 journal permits only five disposable optimizer calls. A separate
E1 component journal has immutable caps for256 optimizer and5120 forward/backward
calls,256 resets/clips, five checkpoint saves and one reload. Preserve attempt,
returned, failed and pending work; failed publication cannot turn returned work
into a failed model call. Cost restoration keeps used ceilings. Pending work
requires explicit reconciliation and cannot resume automatically. A saved cursor
alone does not prove optimizer, tensors, RNG, source or checkpoint byte integrity.

The internal engine keeps256 updates,20 sequences per update,16 acquisition plus
four replay, exact seed-101 order,13 warmup steps and final LR0. Backpropagate each
sequence's mean completion loss divided by20, reset once, clip once and call the
optimizer once. Save only at0/64/128/192/256 complete boundaries. Component restore
requires detached parameters, optimizer, training RNG, exact cursor and cumulative
post-save costs. Current support is a controlled stop exactly at a saved boundary;
refuse rollback after later work instead of erasing its cost. Cross-process native
reloading and campaign rollback are not qualified.

Analytic synthetic controls test sequence-weighting versus pooled-token weighting,
full order/dose, LR0 optimizer-state advancement, state/RNG restoration and durable
failure accounting. They do not run Rineya or prove native CE/backward/accumulation
fit. The prospective actual-reader audit uses Python3.11.15 to match the historical
shuffle contract,128 MiB process high-water RSS and600 seconds; no model/framework
or tokenizer import, updates, generation, paid work or closed-bank payload access.
Resource checks are cooperative, not hard CPU process containment.

Every component and audit grant remains false. Current native capture, language
calibration, long-context preservation, current source/runtime launch validators,
a connected native backend and sustained data/accumulation/RSS measurement remain
required before E1. The original one-Titan verified capability/retention objective
and unchanged best are retained. These modules are concrete infrastructure toward
that objective, not a promotion or a redefinition of success.
