# Recursive-State-Forgetting Evaluation Protocol

## Conditions

Run the exact same task set under:

1. full history;
2. sliding window;
3. one-shot summary;
4. structured state without deletion;
5. deterministic RSF;
6. learned RSF;
7. learned RSF plus exception memory;
8. oracle RSF.

## Task families

- route/checkpoint progression;
- Tower of Hanoi and other fully observable state tasks;
- scheduling with changed constraints;
- shopping/packing checklists;
- code debugging with superseded hypotheses;
- search with distractor pages;
- conflict resolution;
- procedural evidence tasks;
- hidden future dependency traps;
- backtracking and restoration.

## Freeze protocol

- Freeze task generators before comparing policies.
- Keep DEV and TEST separated.
- Tune only on training/DEV.
- Freeze one policy/configuration before TEST.
- Preserve negative results and failed traces.

## Metrics

The built-in evaluator computes:

- task success rate;
- obsolete-eviction precision/recall;
- future-utility recall;
- deletion regret;
- invariant retention;
- active-item reduction;
- token reduction.

Product evaluation should additionally measure correctness, answer support, latency, model calls, and human naturalness.
