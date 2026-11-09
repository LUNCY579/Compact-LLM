# Engineering failure before model imports

The worker rejected its parent ownership check. Windows virtual-environment Python
uses a redirector, so the real interpreter need not be the supervisor's immediate
child. No Torch import, model loading, training or generation occurred. Preserve
the failed supervisor/log and exact source snapshot. Pilot002 corrects ownership
verification to require the live supervisor PID and process creation time in the
worker's ancestry plus its exact shared lease token. It does not weaken resource,
data, model or outcome checks. No scientific result was observed or used to retune.
