# Amendment 011: fixed-buffer checkpoint loading for smaller available RAM

29 September 2026, America/Chicago. The original laptop load pilot measured a
1.928 GiB process-tree RSS peak and a 1.349 GiB RSS snapshot after short inference.
The full run's peak was 1.951 GiB. Later available system RAM fell to 3.159 GiB
before imports; the diagnostic retry correctly failed the previously registered
admission budget, with zero model operations. All these runs remain immutable.

The user's request calls for a route that accommodates ongoing desktop/system use.
Register LAPTOP_STREAMED_V1 as a distinct successor. Replace CPU memory-mapped
tensor access with strict safetensors-header validation and fixed <=2 MiB reads
copied directly into the existing BF16 GPU tensors. Never allocate a complete CPU
tensor or CPU model. Hash source bytes incrementally and verify every loaded GPU
tensor with the existing bounded digest. Preserve exact values, dtype, constructor
and adapter RNG, numerical math, masks, optimizer and thresholds.

The source file remains bound by its full SHA256 before loading. Validate bounded
header size, duplicate keys, BF16 dtype, tensor shapes, byte counts, offsets and
payload coverage. Journal one composite tensor-copy span per serialized tensor;
retain the total chunk count as loader diagnostics. Copy chunks synchronously so
the CPU buffer cannot be reused before transfer completes. Bind the loader module
in source receipts. This is a runtime/loading change, not a new learned method.

For this bounded measurement, retain 1.25 GiB available system RAM and cap the
worker process tree at 1.75 GiB RSS. Admission reserves remaining worker growth:
required available RAM = 1.25 + max(0, 1.75 - current RSS) GiB. Before launching a
worker this budgets 3 GiB total available RAM; after imports the already occupied
worker RSS is deducted. The system reserve is about 8% of this machine's measured
15.218 GiB usable physical memory. External 100 ms monitoring and containment remain
mandatory; stop on floor/cap violation. These are prospectively registered test
bounds, not promised fit or a hard OS allocation guarantee. Do not kill user apps.

The initial 1.75 GiB cap is motivated by measured 1.349 GiB steady short-inference
RSS plus a smaller fixed-buffer loader, leaving approximately 0.401 GiB for other
CPU overhead. Backward and optimizer-framework import peaks remain unmeasured; an
excess must produce a failure, not an automatic cap increase. GPU admission/runtime
4/0.5 GiB and disk 4 GiB remain fixed. No simultaneous native GPU jobs or heavy CPU
preparation accompany these probes.

Register the same three explicitly distinct scopes: <=600 s load_forward with
zero optimizer/generation/parity budgets; <=1800 s full E0 with original checks;
and <=1800 s backward_resume with the limited scope and RNG disclosure in
AMENDMENT_010. Every resource-only status remains unable to authorize E1. The
unresolved padded likelihood gate retains its exact tolerances. Preserve the
earlier LAPTOP_GPU_RESIDENT_V1 configs, original E0 config and all prior evidence.
