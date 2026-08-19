# Paired Dataset Examples

Each example shows one canonical full-context record and the unordered transition records derived from it.

## science — biology

### Original problem

A cell population starts at 30 and doubles once per interval. How many cells are present after 6 intervals?

### Full-context target

```text
Reasoning:
1. Each interval multiplies the population by 2, so use 30 × 2^6.
2. Compute 2^6 = 64.
3. Multiply: 30 × 64 = 1920.
4. Therefore, the answer is 1920 cells.
```

### Derived compact-state transitions

**Step 1 / 4**

Current state: `No derived facts yet.`

Target: `Each interval multiplies the population by 2, so use 30 × 2^6.`

**Step 2 / 4**

Current state: `Growth model: 30 × 2^6.`

Target: `Compute 2^6 = 64.`

**Step 3 / 4**

Current state: `Doubling factor after 6 intervals: 64.`

Target: `Multiply: 30 × 64 = 1920.`

**Step 4 / 4**

Current state: `Computed population: 1920.`

Target: `Therefore, the answer is 1920 cells.`

## logic_and_symbolic — boolean_logic

### Original problem

Given V = True, S = False, and W = True, evaluate NOT (V OR S) AND W.

### Full-context target

```text
Reasoning:
1. Evaluate V OR S: True OR False = True.
2. Negate it: NOT True = False.
3. Combine with W using AND: False AND True = False.
4. Therefore, the answer is False.
```

### Derived compact-state transitions

**Step 1 / 4**

Current state: `No derived facts yet.`

Target: `Evaluate V OR S: True OR False = True.`

**Step 2 / 4**

Current state: `V OR S = True.`

Target: `Negate it: NOT True = False.`

**Step 3 / 4**

Current state: `NOT (V OR S) = False.`

Target: `Combine with W using AND: False AND True = False.`

**Step 4 / 4**

Current state: `Expression value: False.`

Target: `Therefore, the answer is False.`

## coding — code_trace

### Original problem

Trace the following Python-like code and give the printed output:
```python
total = -3
for i in range(1, 6):
    total = total + 4 * i
print(total)
```

### Full-context target

```text
Reasoning:
1. Initialize total = -3.
2. At i = 1, update total = -3 + 4×1 = 1.
3. At i = 2, update total = 1 + 4×2 = 9.
4. At i = 3, update total = 9 + 4×3 = 21.
5. At i = 4, update total = 21 + 4×4 = 37.
6. At i = 5, update total = 37 + 4×5 = 57.
7. The loop is complete, so print 57.
8. Therefore, the answer is 57.
```

### Derived compact-state transitions

**Step 1 / 8**

Current state: `Program execution has not started.`

Target: `Initialize total = -3.`

**Step 2 / 8**

Current state: `Program state: total = -3; next loop index i = 1.`

Target: `At i = 1, update total = -3 + 4×1 = 1.`

**Step 3 / 8**

Current state: `Program state: total = 1; next loop index i = 2.`

Target: `At i = 2, update total = 1 + 4×2 = 9.`

**Step 4 / 8**

Current state: `Program state: total = 9; next loop index i = 3.`

Target: `At i = 3, update total = 9 + 4×3 = 21.`

**Step 5 / 8**

Current state: `Program state: total = 21; next loop index i = 4.`

Target: `At i = 4, update total = 21 + 4×4 = 37.`

**Step 6 / 8**

Current state: `Program state: total = 37; next loop index i = 5.`

Target: `At i = 5, update total = 37 + 4×5 = 57.`

**Step 7 / 8**

Current state: `Program state: total = 57; loop complete; print total.`

Target: `The loop is complete, so print 57.`

**Step 8 / 8**

Current state: `Printed output: 57.`

Target: `Therefore, the answer is 57.`

## daily_life — cooking

### Original problem

Scale a recipe from 2 to 3 servings. It originally uses 3/2 cups of milk. Find the new amount.

### Full-context target

```text
Reasoning:
1. Compute the serving scale factor: 3/2 = 3/2.
2. Multiply the ingredient amount by the scale factor: 3/2 × 3/2 = 9/4.
3. Therefore, the answer is 9/4 cups of milk.
```

### Derived compact-state transitions

**Step 1 / 3**

Current state: `No derived facts yet.`

Target: `Compute the serving scale factor: 3/2 = 3/2.`

**Step 2 / 3**

Current state: `Scale factor: 3/2.`

Target: `Multiply the ingredient amount by the scale factor: 3/2 × 3/2 = 9/4.`

**Step 3 / 3**

Current state: `Scaled milk amount: 9/4 cups.`

Target: `Therefore, the answer is 9/4 cups of milk.`

## data_reasoning — data_table

### Original problem

Use the table below. What is the total across all groups in Period 2?

| Group | Period 1 | Period 2 |
|---|---:|---:|
| Beacon | 157 | 160 |
| Grove | 67 | 97 |
| Cedar | 73 | 79 |

### Full-context target

```text
Reasoning:
1. Read the Period 2 values: 160, 97, and 79.
2. Add them: 160 + 97 + 79 = 336.
3. Therefore, the answer is 336.
```

### Derived compact-state transitions

**Step 1 / 3**

Current state: `No derived facts yet.`

Target: `Read the Period 2 values: 160, 97, and 79.`

**Step 2 / 3**

Current state: `Selected Period 2 values: 160, 97, 79.`

Target: `Add them: 160 + 97 + 79 = 336.`

**Step 3 / 3**

Current state: `Period 2 total: 336.`

Target: `Therefore, the answer is 336.`

## mathematics — decimals

### Original problem

Start with 35.11 units, add 8.93, then subtract 3.44. What is the result?

### Full-context target

```text
Reasoning:
1. Add the first two quantities: 35.11 + 8.93 = 44.04.
2. Subtract the released amount: 44.04 - 3.44 = 40.60.
3. Therefore, the answer is 40.60 liters.
```

### Derived compact-state transitions

**Step 1 / 3**

Current state: `No derived facts yet.`

Target: `Add the first two quantities: 35.11 + 8.93 = 44.04.`

**Step 2 / 3**

Current state: `Current quantity: 44.04. Remaining operation: subtract 3.44.`

Target: `Subtract the released amount: 44.04 - 3.44 = 40.60.`

**Step 3 / 3**

Current state: `Final quantity computed: 40.60.`

Target: `Therefore, the answer is 40.60 liters.`

## time_and_planning — elapsed_time

### Original problem

A process starts at 18:13 and runs for 4 hours 10 minutes. At what time does it finish?

### Full-context target

```text
Reasoning:
1. Convert the start time to minutes after midnight: 18 × 60 + 13 = 1093 minutes.
2. Convert the duration to minutes: 4 × 60 + 10 = 250 minutes.
3. Add them: 1093 + 250 = 1343 total minutes from the starting midnight.
4. Convert 1343 minutes back to clock time: 22:23.
5. Therefore, the answer is 22:23.
```

### Derived compact-state transitions

**Step 1 / 5**

Current state: `No derived facts yet.`

Target: `Convert the start time to minutes after midnight: 18 × 60 + 13 = 1093 minutes.`

**Step 2 / 5**

Current state: `Start time represented as 1093 minutes after midnight.`

Target: `Convert the duration to minutes: 4 × 60 + 10 = 250 minutes.`

**Step 3 / 5**

Current state: `Duration represented as 250 minutes.`

Target: `Add them: 1093 + 250 = 1343 total minutes from the starting midnight.`

**Step 4 / 5**

Current state: `Unwrapped finish time: 1343 minutes.`

Target: `Convert 1343 minutes back to clock time: 22:23.`

**Step 5 / 5**

Current state: `Clock finish time: 22:23.`

Target: `Therefore, the answer is 22:23.`

## language — language_rules

### Original problem

Apply these pluralization rules in order: (1) a word ending in consonant + y changes y to ies; (2) a word ending in s, x, z, ch, or sh adds es; (3) every other word adds s. Pluralize these words in the given order: lady, box, planet.

### Full-context target

```text
Reasoning:
1. Apply the consonant + y rule to 'lady', producing 'ladies'.
2. Apply the sibilant-ending rule to 'box', producing 'boxes'.
3. Apply the default rule to 'planet', producing 'planets'.
4. Therefore, the answer is ladies, boxes, planets.
```

### Derived compact-state transitions

**Step 1 / 4**

Current state: `No derived facts yet.`

Target: `Apply the consonant + y rule to 'lady', producing 'ladies'.`

**Step 2 / 4**

Current state: `Plural forms completed so far: ladies.`

Target: `Apply the sibilant-ending rule to 'box', producing 'boxes'.`

**Step 3 / 4**

Current state: `Plural forms completed so far: ladies, boxes.`

Target: `Apply the default rule to 'planet', producing 'planets'.`

**Step 4 / 4**

Current state: `Plural forms completed so far: ladies, boxes, planets.`

Target: `Therefore, the answer is ladies, boxes, planets.`

## state_reasoning — recursive_state_updates

### Original problem

A robot starts at (2, -5) facing W. Apply these commands in order: ['F1', 'F4', 'L', 'L', 'F1', 'F3']. L means turn left, R means turn right, and Fn means move forward n grid steps. What is the final state?

### Full-context target

```text
Reasoning:
1. Command 1 is F1: move 1 step(s) while facing W, from (2, -5) to (1, -5).
2. Command 2 is F4: move 4 step(s) while facing W, from (1, -5) to (-3, -5).
3. Command 3 is L: turn left, changing direction from W to S; position stays (-3, -5).
4. Command 4 is L: turn left, changing direction from S to E; position stays (-3, -5).
5. Command 5 is F1: move 1 step(s) while facing E, from (-3, -5) to (-2, -5).
6. Command 6 is F3: move 3 step(s) while facing E, from (-2, -5) to (1, -5).
7. Therefore, the answer is position (1, -5), facing E.
```

### Derived compact-state transitions

**Step 1 / 7**

Current state: `Current state: position (2, -5), facing W; no commands applied yet.`

Target: `Command 1 is F1: move 1 step(s) while facing W, from (2, -5) to (1, -5).`

**Step 2 / 7**

Current state: `Current state: position (1, -5), facing W; remaining commands: ['F4', 'L', 'L', 'F1', 'F3'].`

Target: `Command 2 is F4: move 4 step(s) while facing W, from (1, -5) to (-3, -5).`

**Step 3 / 7**

Current state: `Current state: position (-3, -5), facing W; remaining commands: ['L', 'L', 'F1', 'F3'].`

Target: `Command 3 is L: turn left, changing direction from W to S; position stays (-3, -5).`

**Step 4 / 7**

Current state: `Current state: position (-3, -5), facing S; remaining commands: ['L', 'F1', 'F3'].`

Target: `Command 4 is L: turn left, changing direction from S to E; position stays (-3, -5).`

**Step 5 / 7**

Current state: `Current state: position (-3, -5), facing E; remaining commands: ['F1', 'F3'].`

Target: `Command 5 is F1: move 1 step(s) while facing E, from (-3, -5) to (-2, -5).`

**Step 6 / 7**

Current state: `Current state: position (-2, -5), facing E; remaining commands: ['F3'].`

Target: `Command 6 is F3: move 3 step(s) while facing E, from (-2, -5) to (1, -5).`

**Step 7 / 7**

Current state: `Current state: position (1, -5), facing E; remaining commands: none.`

Target: `Therefore, the answer is position (1, -5), facing E.`

## coding_and_symbolic — string_processing

### Original problem

Start with the string 'poge93'. In order: replace every occurrence of 'g' (matching the current case) with 'i'; then convert to uppercase; then append '_v2'; then reverse the entire string. What is the final string?

### Full-context target

```text
Reasoning:
1. Replace every 'g' with 'i': 'poge93' → 'poie93'.
2. Convert the string to uppercase: 'poie93' → 'POIE93'.
3. Append '_v2': 'POIE93' → 'POIE93_v2'.
4. Reverse the string: 'POIE93_v2' → '2v_39EIOP'.
5. Therefore, the answer is 2v_39EIOP.
```

### Derived compact-state transitions

**Step 1 / 5**

Current state: `Current string: 'poge93'; no operations completed.`

Target: `Replace every 'g' with 'i': 'poge93' → 'poie93'.`

**Step 2 / 5**

Current state: `Current string: 'poie93'; completed operation: replace.`

Target: `Convert the string to uppercase: 'poie93' → 'POIE93'.`

**Step 3 / 5**

Current state: `Current string: 'POIE93'; completed operation: uppercase.`

Target: `Append '_v2': 'POIE93' → 'POIE93_v2'.`

**Step 4 / 5**

Current state: `Current string: 'POIE93_v2'; completed operation: append.`

Target: `Reverse the string: 'POIE93_v2' → '2v_39EIOP'.`

**Step 5 / 5**

Current state: `Current string: '2v_39EIOP'; completed operation: reverse.`

Target: `Therefore, the answer is 2v_39EIOP.`

## measurement — unit_conversion

### Original problem

Convert 11 meters to centimeters, then add 33 centimeters. What is the total in centimeters?

### Full-context target

```text
Reasoning:
1. Use 1 m = 100 cm.
2. Convert 11 m: 11 × 100 = 1100 cm.
3. Add the remaining 33 cm: 1100 + 33 = 1133 cm.
4. Therefore, the answer is 1133 cm.
```

### Derived compact-state transitions

**Step 1 / 4**

Current state: `No derived facts yet.`

Target: `Use 1 m = 100 cm.`

**Step 2 / 4**

Current state: `Conversion factor: 1 m = 100 cm.`

Target: `Convert 11 m: 11 × 100 = 1100 cm.`

**Step 3 / 4**

Current state: `Converted first amount: 1100 cm.`

Target: `Add the remaining 33 cm: 1100 + 33 = 1133 cm.`

**Step 4 / 4**

Current state: `Combined total: 1133 cm.`

Target: `Therefore, the answer is 1133 cm.`
