# Prospective library-observation amendment 002

This CPU source amendment follows partial DLL milestone001 at HEAD 91769fd7b0b27317b5ff2fa84395a3fd0386af74.
It changes future qualification design, not the completed original native runs
or their source/65-entry maps, numerical/resource checks, failures or receipts.
Native launch gates remain closed. Actual study M1/M2 learning is still unmeasured.

## Honest managed scope and executed-library observations

V1 requires a complete DLL-directory registration trace from process creation.
That requirement was introduced by the staged prefix provider; it is absent from
the original native contract's finite runtime hash/requalification checks.
Windows AddDllDirectory returns a cookie for a caller's registration and does
not provide a documented enumeration of foreign registrations. A managed-cookie
list therefore cannot support a claim that all registrations have been observed.

A prospective V2 boundary will report its selected managed directories, observed
legacy directory, actual setup/cleanup outcomes and foreign registrations as
UNOBSERVED. It will not claim complete search-directory coverage. V1 sources,
schema and successful CPU observations remain historical and are preserved.
No V1 native authority is transferred by renaming a boundary.

The executed-library gate remains: selected complete initial process module
identities, qualified source/runtime bytes, separately qualified lossless ordered
load/unload observations across the guarded scope, exact prospective dependency
and before/after path/handle joins, missing/unexpected/transient-library refusal,
source drift and terminal cleanup checks. Unknown directory state is not accepted
as proof of execution completeness. This observes execution and does not prevent
all unregistered native execution or constitute an OS sandbox.

A loader notification callback cannot run Python here: Microsoft documents that
it must not call any other module. A future observer needs a bounded native
collector or independently qualified external event route, exact provenance,
overflow/missing-event refusal, source/binary identity and scoped cleanup.
This amendment supplies no such native result or launch grant.

## Preserve entry identity

The current selector accepts reconstruction of selected code with forged Python
globals and closure state. The successor main route will retain exact selected
worker/contract/loader instances, original context and an installation handle
validated by that selected installer. Private execution will revalidate the
original context before helpers/framework; public arbitrary callback entry and
sourcefilename/meta-path rediscovery will be removed. CPU capabilities and
installation mode remain explicit and cannot be promoted by documentary flags.
This is registered control flow; hostile arbitrary same-process mutation of
trusted code, globals or closures remains outside its assurance scope.

## Finite Windows identities

A separately selected optional System32 DLL identity map permits additional
observed Windows dependencies. It must preserve original65 and stdlib779 pins,
join exact descriptors to runtime files and the selected library map, enforce
physical regular leaves, exact System32 root/suffix/role and bounded population,
and reject omitted/substituted/resealed entries. It does not enumerate or qualify
current host dependencies. No wildcard allowance or resource check changes.

## Primary documentation checked

- [AddDllDirectory](https://learn.microsoft.com/en-us/windows/win32/api/libloaderapi/nf-libloaderapi-adddlldirectory): registration cookie and unspecified ordering.
- [DLL search order](https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order): dependencies and LOAD_LIBRARY_SEARCH flags.
- [LdrRegisterDllNotification](https://learn.microsoft.com/en-us/windows/win32/devnotes/ldrregisterdllnotification): before-link notification registration.
- [LdrDllNotification](https://learn.microsoft.com/en-us/windows/win32/devnotes/ldrdllnotification): load/unload data and cross-module callback prohibition.

Checked2026-10-05. Sources describe API contracts, not observations on this host.
No model/driver/GPU/binary was loaded or job launched for the source amendment.
