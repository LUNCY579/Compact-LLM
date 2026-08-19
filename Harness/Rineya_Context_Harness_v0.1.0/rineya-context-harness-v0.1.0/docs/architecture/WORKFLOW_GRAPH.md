# Workflow Graph and Termination

## Nodes

```text
INTAKE
ROUTE
SEARCH
FETCH
EXTRACT
RECONSTRUCT
COMPACT
VERIFY
ANSWER
CLARIFY
ABSTAIN
DONE
ERROR
```

## Termination predicates

The workflow stops when one of the following is true:

- a direct answer has been generated;
- required evidence is sufficient and the grounded answer validates;
- the user must clarify;
- evidence remains insufficient and no search budget remains;
- a critical conflict remains;
- a budget is exceeded;
- a technical failure invalidates the run;
- cancellation is requested by a future runtime adapter.

## Retry policy

Search rounds are not blind retries. A repeated round must target an unresolved gap and should not repeat an identical normalized query. Fetch failures are isolated per source. State commits do not retry automatically after revision conflicts; callers must reload current state and construct a new patch.
