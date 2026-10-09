# Corrected no-site supervisor entry

The corrected import supervisor qualifies its own dependency resolution before
the original lease and resource lifecycle. It adds only the selected physical
PACKAGE/SITE roots within a scoped finder, checks selected Python/cache/extension
bytes, and restores the original path and finder containers on return or error.
The child still starts with `-I -S -B`. Resource thresholds, ownership checks,
monitor behavior and native supervisor source are unchanged.

The monitor's import symbol remains in compilation and resolves to the exact
selected supervisor module; this preserves original method code while avoiding
the monitor's separate path insertion. Historical preload records are joined to
their selected runtime descriptors without altering the observed maps. Absent
Windows `pwd` and parent `org` imports raise `ModuleNotFoundError` after all three
standard finders confirm absence, preserving installed dependency fallbacks and
preventing delegation to an unselected later finder. Available/preloaded
substitutions reject. The child receives only the narrow `pwd` correction.

Thirteen authored/Root CPU controls pass using inert dependency and resource
fixtures. They cover original lease-birth resolution, low-resource rejection,
Job-before-resume ordering, monitor code identity, cleanup and descriptor/fallback
negative cases. Independent review and the earlier foreign-finder counterexample
are [preserved](evidence/import_supervisor_entry_reviewed_cpu_20261009/ROOT_PRESERVATION.json),
along with the corrected external check and all intermediate/failing versions.
CPU fixtures establish no native import or host fit.

Update the two dispatcher supervisor descriptors, commit the implementation,
refresh the source/runtime registration and current byte review, then instantiate
and run one separately bound successor under the owner's standing authorization.
The failed006 attempt remains retained. The M0/M1/M2 transfer result is still open.
