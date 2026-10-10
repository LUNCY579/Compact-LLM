# CPU causal replay002: qualification passes, complete-task forecast refuses

The corrected authorized attempt executed29 qualification APIs:27 full-forward
attempts,26 complete, three state adjoints,4764 completed prefix positions, one
expected partial suffix exception and two precompute cache rejections. Eight
noextra/unit/scaled value pairs are exactly equal. Both scalar CE checks and the
accepted-EOS-mass adjoint pass the fixed FD tolerance. No unexpected numerical or
resource failure is recorded. This is qualification evidence only.

I independently parsed the three saved FP32 G/d2 NPZ arrays, verified file/array
hashes and shapes, and recomputed dot products in FP64. Differences from native
FP32 sums are4.14e-11,1.73e-7 and5.28e-10. Recomputed FD discrepancies are
7.16e-7,2.74e-4 and6.59e-5, all within .001+.03*abs(adjoint). The tiny numerical
EOS-mass versus single-EOT CE difference is FP32 reduction rounding, not a new
semantic result. Source/current96-file closure and qualification-state hashes match.

Actual maximum CPU whole-forward+adjoint1.7388157s and value.9941931s give
632*(1.7388157+2*.9941931)*1.5=3533.387401s remaining. Adding64.910390s elapsed
and180s internal reserve yields3778.297791s, exceeding the authorized3600s by
178.297791s. The worker refused before any task row. Targets/features/outcomes
are empty: all192 cases/632 target positions remain unmeasured. Primary answer,
EOS and semantic comparisons stay null; no subset was substituted.

Loaded state/custom head are b23/b77. Fresh qualification records unchanged
before/after state/assets/source and gradNone. The failing terminal receipt does
not serialize final state/source/parameter success proof, so the latest verified
checkpoint is qualification, not a fully completed preservation receipt. Frozen
reader reports INCOMPLETE and null primaries without raising or fabricating data.

The one3600s grant with SHA5397a0b5... is consumed. Native charge66.109s brings
ledger usage1448.515s, remaining20151.485; parent wall73.234s and worker body
wall64.911443s are separate clocks. Recorded lifetime and sampled RSS stay below
6GiB. No model-quality, long-run stability or production-speed inference follows.

A prospective5400s parent allocation plus1800s global reserve (7200 total) fits
remaining budget and exceeds this observed forecast by1621.702209s. It is not
currently authorized: require another exact reviewed one-attempt allowance,
anti-replay/source/resource/qualification/complete-forecast gates and versioned
deadline proof handling. Preserve this attempt, frozen3600 reader/plan and all
192 required cases; do not lift a deadline, drop cases or retry automatically.
No models, optimizer, author code or endpoints changed in this independent audit.
