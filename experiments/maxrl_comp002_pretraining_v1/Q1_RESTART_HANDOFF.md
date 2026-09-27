# Q1 native-loader waiter after a PC or Codex restart

The active waiter is a local Python process running
`experiments.maxrl_comp002_pretraining_v1.q1_native_wait`. Its log is
`evidence/Q1_NATIVE_RUN.log`; a normal terminal outcome writes
`evidence/Q1_NATIVE_STATUS.json`, and a successful native probe also writes
`evidence/Q1_NATIVE_LOADER.json`. A log line or GPU lock alone is not proof
that the process is alive or finished.

On resumption, inspect the status/result files **and** the actual process
table or persistent terminal handle. If the waiter is still live, keep that
handle and do not launch another. If status says `Q1_NATIVE_LOADER_PASS`,
validate the native receipt before more calls. If it records a failure, read
the log and repair the cause; do not silently repeat a model load.

If a restart killed the waiter with **no status or result**, inspect the log.
When it contains no `[wait] launch` line, no native model load occurred. After
verifying the old waiter PID is absent, move the old log to a uniquely named
interruption log, then launch a new waiter with the unchanged plan. If the log
does contain `[wait] launch`, inspect the child process, GPU use and lease
before deciding whether the one-load Q1 budget was consumed. Do not infer
terminal state from an observation timeout or remove another job's GPU lease.

If a stale lease remains after a reboot, verify its recorded holder and workers
are dead and that the card has no foreign heavy Python process before any
manual recovery. Neither this handoff nor Q0 authorizes training.
