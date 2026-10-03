# Real TRAIN examples: annotated preview

These are actual generated references, not model predictions. The snippets below are deliberately abbreviated: they are not standalone tasks. Full model-visible prompts and all intermediate checkpoints are in the corresponding JSONL files. This preview is an editorial illustration, not a human-review receipt.

## ledger_replay

**ID:** `RRFR2-0439d508c00ae22d2ad1668d`  
**Input:** 514 words; native token count not measured.

**Query and representative reference-solution checkpoints**

```json
{"account":"D","cutoff":14}
```

```text
1. Identify the requested account and cutoff. Initialize every account, sort events by tick, and ignore void or future events. Replay signed transfers literally; negative balances are allowed.
2. Checkpoint: {"after":{"A":-16,"B":-28,"C":-17,"D":15},"before":{"A":-16,"B":-20,"C":-17,"D":15},"operation":"add","tick":1}.
3. Checkpoint: {"after":{"A":-16,"B":-28,"C":-34,"D":15},"before":{"A":-16,"B":-28,"C":-17,"D":15},"operation":"scale","tick":2}.
4. Checkpoint: {"after":{"A":-16,"B":-28,"C":-34,"D":15},"before":{"A":-16,"B":-28,"C":-34,"D":15},"operation":"transfer","tick":3}.
5. Checkpoint: {"after":{"A":-16,"B":-28,"C":-40,"D":15},"before":{"A":-16,"B":-28,"C":-34,"D":15},"operation":"add","tick":4}.
[Middle checkpoints omitted only in this preview.]
16. The replay or witness satisfies the task. The exact solver checks the optimization objective where applicable. Emit only the required fields.
17. Final result: {"value":27}.
```

**Complete public answer**

```json
{"value":27}
```

## register_machine

**ID:** `RRFR2-a1c6e2d50c32637852a914fc`  
**Input:** 525 words; native token count not measured.

**Query and representative reference-solution checkpoints**

```json
{"cutoff":12,"modulus":97,"register":"X"}
```

```text
1. Initialize X, Y, and Z. Process the numbered instructions in order. Read conditions and both source registers from the pre-step state; use a nonnegative residue modulo 97.
2. Checkpoint: {"after":{"X":76,"Y":30,"Z":65},"before":{"X":76,"Y":28,"Z":65},"operation":"add","tick":1}.
3. Checkpoint: {"after":{"X":76,"Y":30,"Z":17},"before":{"X":76,"Y":30,"Z":65},"operation":"mix","tick":2}.
4. Checkpoint: {"after":{"X":76,"Y":92,"Z":17},"before":{"X":76,"Y":30,"Z":17},"operation":"mix","tick":3}.
5. Checkpoint: {"after":{"X":76,"Y":21,"Z":17},"before":{"X":76,"Y":92,"Z":17},"operation":"mix","tick":4}.
[Middle checkpoints omitted only in this preview.]
14. The replay or witness satisfies the task. The exact solver checks the optimization objective where applicable. Emit only the required fields.
15. Final result: {"value":21}.
```

**Complete public answer**

```json
{"value":21}
```

## temporal_route

**ID:** `RRFR2-fdceb26846411e903e137ca8`  
**Input:** 568 words; native token count not measured.

**Query and representative reference-solution checkpoints**

```json
{"cutoff":22,"goal":"N3","nodes":["N0","N1","N2","N3","N4","N5","N6"],"start":"N6","waypoints":[]}
```

```text
1. For each directed edge, select the latest approved version at the cutoff and discard closed edges. Search states consisting of location and completed ordered waypoints; minimize total positive cost.
2. Checkpoint: {"cumulative_cost":9,"edge_cost":9,"from":"N6","to":"N2"}.
3. Checkpoint: {"cumulative_cost":12,"edge_cost":3,"from":"N2","to":"N0"}.
4. Checkpoint: {"cumulative_cost":19,"edge_cost":7,"from":"N0","to":"N3"}.
5. The replay or witness satisfies the task. The exact solver checks the optimization objective where applicable. Emit only the required fields.
6. Final result: {"cost":19,"path":["N6","N2","N0","N3"]}.
```

**Complete public answer**

```json
{"cost":19,"path":["N6","N2","N0","N3"]}
```

## prerequisite_plan

**ID:** `RRFR2-a30cc914a182a6a7eb485ca4`  
**Input:** 517 words; native token count not measured.

**Query and representative reference-solution checkpoints**

```json
{"bits":7,"goal_off":0,"goal_on":68,"initial":43}
```

```text
1. Separate the goal flags from action preconditions. Search finite flag states using only applicable actions, applying deletes before adds. Minimize total positive action cost, then verify the goal.
2. Checkpoint: {"action":"A3","after":107,"before":43,"cumulative_cost":1}.
3. Checkpoint: {"action":"A12","after":110,"before":107,"cumulative_cost":3}.
4. The replay or witness satisfies the task. The exact solver checks the optimization objective where applicable. Emit only the required fields.
5. Final result: {"actions":["A3","A12"],"cost":3}.
```

**Complete public answer**

```json
{"actions":["A3","A12"],"cost":3}
```

## deadline_schedule

**ID:** `RRFR2-5aca5e1469bd9eeb64f668e5`  
**Input:** 527 words; native token count not measured.

**Query and representative reference-solution checkpoints**

```json
{"start_time":3}
```

```text
1. List all jobs and precedence constraints. Search feasible subsets on the single machine, start each selected job at the earliest time allowed by its release, reject late completions, and minimize final finish.
2. Checkpoint: {"after":[],"deadline":24,"finish":11,"job":"J1","start":9}.
3. Checkpoint: {"after":[],"deadline":28,"finish":12,"job":"J0","start":11}.
4. Checkpoint: {"after":["J1"],"deadline":33,"finish":17,"job":"J2","start":12}.
5. Checkpoint: {"after":["J0"],"deadline":23,"finish":19,"job":"J3","start":17}.
[Middle checkpoints omitted only in this preview.]
8. The replay or witness satisfies the task. The exact solver checks the optimization objective where applicable. Emit only the required fields.
9. Final result: {"finish":23,"order":["J1","J0","J2","J3","J4","J5"]}.
```

**Complete public answer**

```json
{"finish":23,"order":["J1","J0","J2","J3","J4","J5"]}
```

## budget_selection

**ID:** `RRFR2-63e525d0f1b424d500f63da4`  
**Input:** 516 words; native token count not measured.

**Query and representative reference-solution checkpoints**

```json
{"cutoff":5}
```

```text
1. Resolve the latest approved budget and capacity. Compare item subsets satisfying both limits, all dependencies, and all exclusions. Maximize total value, then minimize cost.
2. Checkpoint: {"cumulative_cost":7,"cumulative_value":8,"cumulative_weight":4,"item":"I1"}.
3. Checkpoint: {"cumulative_cost":10,"cumulative_value":28,"cumulative_weight":5,"item":"I4"}.
4. Checkpoint: {"cumulative_cost":13,"cumulative_value":42,"cumulative_weight":13,"item":"I6"}.
5. The replay or witness satisfies the task. The exact solver checks the optimization objective where applicable. Emit only the required fields.
6. Final result: {"cost":13,"items":["I1","I4","I6"],"value":42}.
```

**Complete public answer**

```json
{"cost":13,"items":["I1","I4","I6"],"value":42}
```

## evidence_chain

**ID:** `RRFR2-e5ebe5063f54d027a8764ec7`  
**Input:** 725 words; native token count not measured.

**Query and representative reference-solution checkpoints**

```json
{"cutoff":36,"path":["owner","owner","owner","next","site"],"start":"E4"}
```

```text
1. Resolve only approved fields at the cutoff. Follow the requested relations in order; stop with UNKNOWN when a required field has no approved evidence. Drafts are not facts.
2. Checkpoint: {"entity":"E4","field":"owner","tick":26,"value":"E3"}.
3. Checkpoint: {"entity":"E3","field":"owner","tick":20,"value":"E4"}.
4. Checkpoint: {"entity":"E4","field":"owner","tick":26,"value":"E3"}.
5. Checkpoint: {"entity":"E3","field":"next","tick":22,"value":"E0"}.
6. Checkpoint: {"entity":"E0","field":"site","tick":3,"value":"S4"}.
7. The replay or witness satisfies the task. The exact solver checks the optimization objective where applicable. Emit only the required fields.
8. Final result: {"answer":"S4"}.
```

**Complete public answer**

```json
{"answer":"S4"}
```

## continuous_replan

**ID:** `RRFR2-80d42ae13a4cd869573622fd`  
**Input:** 655 words; native token count not measured.

**Query and representative reference-solution checkpoints**

```json
{"cutoff":28,"goal":"N1","nodes":["N0","N1","N2","N3","N4","N5","N6"],"start":"N0","waypoints":[]}
```

```text
1. Rebuild the current approved graph from the causal history. Earlier plans are not binding. Plan from the reported current location with the currently known costs, not unobserved future updates.
2. Checkpoint: {"cumulative_cost":7,"edge_cost":7,"from":"N0","to":"N2"}.
3. Checkpoint: {"cumulative_cost":9,"edge_cost":2,"from":"N2","to":"N5"}.
4. Checkpoint: {"cumulative_cost":11,"edge_cost":2,"from":"N5","to":"N1"}.
5. The replay or witness satisfies the task. The exact solver checks the optimization objective where applicable. Emit only the required fields.
6. Final result: {"cost":11,"path":["N0","N2","N5","N1"]}.
```

**Complete public answer**

```json
{"cost":11,"path":["N0","N2","N5","N1"]}
```
