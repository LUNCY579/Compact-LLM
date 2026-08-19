# Recursive-State Forgetting

## Definition

Recursive-state forgetting is the repeated transformation:

```text
minimal sufficient current state
+
new observation/evidence
→
new minimal sufficient current state
```

It is not merely transcript summarization. It explicitly distinguishes:

- pinned invariants;
- mutable current state;
- unresolved future dependencies;
- causally absorbed history;
- recoverable archived evidence;
- truly purgeable content.

## Five-way retention rule

1. **Preserve global invariants.** Goals, rules, safety constraints, and stable task parameters remain pinned.
2. **Replace current sufficient state.** New state supersedes old state when the future can be computed from the replacement.
3. **Preserve exceptional/path-dependent information.** Keep facts that alter future policy, such as irreversible choices or unresolved conflicts.
4. **Forget causally absorbed trajectory.** Once a completed step is fully represented by current state, evict its narrative from active context.
5. **Archive recoverable inactive information.** Prefer reversible eviction/archive to irreversible purge.

## Correctness condition

Let `G` be global invariants, `S_t` current state, `O_t` new observation, and `H_t` historical trajectory. A deletion is valid when:

```text
P(next_action | G, S_t, O_t, H_t)
≈
P(next_action | G, S_{t+1})
```

for the task-relevant future policy, while provenance and recovery remain externally available.

This is an empirical condition. The evaluation suite measures premature deletion and future-utility loss rather than assuming every compression is safe.

## Physical context barrier

After a deletion commit:

1. the active-state query changes;
2. the context hash changes;
3. stale dynamic prompt state must not be reused;
4. the next model call receives a freshly assembled dynamic context.

The stable system prefix may still be cached by a compatible runtime.
