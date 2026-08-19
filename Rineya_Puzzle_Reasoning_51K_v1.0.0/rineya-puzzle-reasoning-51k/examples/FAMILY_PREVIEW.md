# Family Preview

One procedurally generated training example from each reasoning family. The complete JSONL files contain 51,000 rows.

## Epistemic Hats

**Difficulty:** `medium`  
**ID:** `rineya-pr51k-hat-tra-01393`

**Problem**

A line-of-hats puzzle uses 1 white and 4 orange. Exactly 4 hats are worn; unused hats may remain. The line from front to back is Rei (front), Esme (position 2), Gia (position 3), and Mo (back). Each person sees every hat ahead, sees no hat behind, and cannot see their own hat. Everyone knows the complete available stock. Starting from the back, Mo says, "I cannot determine my hat color." Gia says, "I know my hat color." Esme says, "I know my hat color." What color must Rei's hat be? Give only the color.

**Verified reasoning**

1. The inventory permits 5 distinct hat assignments.
2. Mo's statement that they do not know removes inconsistent worlds; possibilities: 5 → 2.
3. Gia's statement that they know is compatible with every current world, so 2 remain.
4. Esme's statement that they know is compatible with every current world, so 2 remain.
5. Every remaining world gives Rei an orange hat.

**Short final answer:** `Orange.`

## Knights Knaves

**Difficulty:** `medium`  
**ID:** `rineya-pr51k-knk-tra-00337`

**Problem**

A puzzle tribunal contains only knights, who speak truth, and knaves, who speak falsely. Gus says, "Gus and Asha are both knaves." Quin says, "Gus is a knight." Mina says, "Gus and Asha are both knights." Is Gus a knight or a knave? Give only the type.

**Verified reasoning**

1. There are 16 possible type assignments.
2. Requiring Gus's type to match the truth value of “Gus and Asha are both knaves”; assignments: 16 → 4.
3. Requiring Quin's type to match the truth value of “Gus is a knight”; assignments: 4 → 2.
4. Requiring Mina's type to match the truth value of “Gus and Asha are both knights”; assignments: 2 → 1.
5. The only consistent assignment is Gus=knave, Asha=knight, Mina=knave, Quin=knave.

**Short final answer:** `Knave.`

## Prisoner Alibis

**Difficulty:** `easy`  
**ID:** `rineya-pr51k-alb-tra-00206`

**Problem**

Only one prisoner hid the key. Ian says, "Ian did it." Vic says, "Neither Ian nor Vic is the culprit." Gia says, "Neither Ian nor Vic is the culprit." Exactly 1 of these 3 statements is true. Who is responsible? Give only the name.

**Verified reasoning**

1. If Ian were responsible, 1 statement would be true.
2. If Vic were responsible, 0 statements would be true.
3. If Gia were responsible, 2 statements would be true.
4. Only Ian produces exactly 1 true statement.

**Short final answer:** `Ian.`

## Mislabeled Boxes

**Difficulty:** `medium`  
**ID:** `rineya-pr51k-box-tra-02079`

**Problem**

Each sealed box contains one distinct item. Box A is labeled “marble”; Box B is labeled “gem”; Box C is labeled “star”; Box D is labeled “plum”. Every label is wrong. Inspection shows Box C contains the gem; A scan shows Box B does not contain the plum. What is in Box A? Give only the item.

**Verified reasoning**

1. The label rule permits 9 content assignments.
2. Applying “Box C contains gem”; assignments: 9 → 3.
3. Applying “Box B does not contain plum”; assignments: 3 → 2.
4. Every remaining assignment puts the plum in Box A.

**Short final answer:** `Plum.`

## Logic Grid

**Difficulty:** `easy`  
**ID:** `rineya-pr51k-grid-tra-01040`

**Problem**

A logic grid pairs each resident with a distinct animal and hometown. People: Dina, Nora, and Wade. Pets: rabbit, fox, and gecko. Cities: Frost, Birch, and Cedar. Clues: 1. The gecko owner does not live in Birch. 2. The fox owner does not live in Birch. 3. The fox owner does not live in Frost. 4. Nora does not own the fox. 5. Nora does not own the rabbit. 6. Dina does not own the fox. What pet does Dina own? Give only the answer.

**Verified reasoning**

1. There are 36 possible person–pet–city matchings.
2. Applying “the gecko owner does not live in Birch”; matchings: 36 → 24.
3. Applying “the fox owner does not live in Birch”; matchings: 24 → 12.
4. Applying “the fox owner does not live in Frost”; matchings: 12 → 6.
5. Applying “Nora does not own the fox”; matchings: 6 → 4.
6. Applying “Nora does not own the rabbit”; matchings: 4 → 2.
7. Applying “Dina does not own the fox”; matchings: 2 → 1.
8. The unique grid is Dina=rabbit/Birch; Nora=gecko/Frost; Wade=fox/Cedar.

**Short final answer:** `Rabbit.`

## Linear Order

**Difficulty:** `medium`  
**ID:** `rineya-pr51k-ord-tra-02318`

**Problem**

A set of presentations is scheduled in distinct positions. Participants: Esme, Ada, Fia, Niko, Vito, and Hana. Clues: 1. Hana is immediately before Fia. 2. Vito is exactly 4 positions before Niko. 3. Vito is exactly 3 positions before Ada. 4. Vito is exactly 5 positions before Esme. Who is last? Give only the answer.

**Verified reasoning**

1. There are 720 possible total orders.
2. Applying “Hana is immediately before Fia”; orders: 720 → 120.
3. Applying “Vito is exactly 4 positions before Niko”; orders: 120 → 8.
4. Applying “Vito is exactly 3 positions before Ada”; orders: 8 → 2.
5. Applying “Vito is exactly 5 positions before Esme”; orders: 2 → 1.
6. The unique order is Vito → Hana → Fia → Ada → Niko → Esme.

**Short final answer:** `Esme.`

## Rule Deduction

**Difficulty:** `medium`  
**ID:** `rineya-pr51k-rule-tra-00939`

**Problem**

A small rule system follows; do not assume unstated facts. Facts: Orla is tidy; Ari is curious; Ari is tidy; Noel is alert; Sara is not curious; Yara is alert. Rules: If something is curious and tidy, then it is steady; If something is curious and careful, then it is warm; If something is tidy and warm, then it is quiet; If something is steady, then it is alert; If something is tidy, then it is not alert; If something is ready, then it is warm; If something is curious and tidy, then it is steady; If something is alert, then it is warm. Claim: Ari is alert. Answer Yes if the claim is forced, No if its negation is forced, otherwise Unknown.

**Verified reasoning**

1. Begin with the explicit facts and repeatedly apply rules whose premises are known.
2. At inference depth 1, derive Ari is steady.
3. At inference depth 2, derive Ari is alert.
4. The claim “Ari is alert” is derived, so the answer is Yes.

**Short final answer:** `Yes.`

## Object Tracking

**Difficulty:** `easy`  
**ID:** `rineya-pr51k-track-tra-02629`

**Problem**

A card row undergoes deterministic moves. Initial row: 1: red disk; 2: violet card; 3: amber token; 4: blue cube; 5: copper coin. Operations: 1. Rotate the whole row left by 4. 2. Move the object in position 1 to position 5, shifting intervening objects. 3. Move the object in position 2 to position 5, shifting intervening objects. 4. Rotate the whole row left by 4. Which object is finally in position 1? Give only the answer.

**Verified reasoning**

1. Start: red disk | violet card | amber token | blue cube | copper coin.
2. After step 1 (rotate the whole row left by 4): copper coin | red disk | violet card | amber token | blue cube.
3. After step 2 (move the object in position 1 to position 5, shifting intervening objects): red disk | violet card | amber token | blue cube | copper coin.
4. After step 3 (move the object in position 2 to position 5, shifting intervening objects): red disk | amber token | blue cube | copper coin | violet card.
5. After step 4 (rotate the whole row left by 4): violet card | red disk | amber token | blue cube | copper coin.
6. Read the requested value from the final state.

**Short final answer:** `Violet card.`

## Code Breaking

**Difficulty:** `easy`  
**ID:** `rineya-pr51k-code-tra-01247`

**Problem**

Crack a secret digit code. It has 3 digits chosen from 0, 1, 2, 3, 4, and 5. No digit repeats. For each clue, 'exact-position' and 'wrong-position' counts are both exact. Guess 013: 0 exact-position and 3 correct-digit-wrong-position. Guess 402: 1 exact-position and 0 correct-digit-wrong-position. What is the code? Give only the code.

**Verified reasoning**

1. Initially, 120 codes satisfy the length and repetition rule.
2. Feedback for 013 (0 exact, 3 misplaced); codes: 120 → 2.
3. Feedback for 402 (1 exact, 0 misplaced); codes: 2 → 1.
4. Only 301 remains.

**Short final answer:** `3-0-1.`

## Calendar Reasoning

**Difficulty:** `medium`  
**ID:** `rineya-pr51k-time-tra-01738`

**Problem**

Use the exact day offsets in this monthly plan. Day 1 is Wednesday. Clues: The inspection is on day 14; the rehearsal is 7 days before the inspection; the workshop is 3 days before the rehearsal; the briefing is 4 days after the workshop; the delivery is 5 days after the briefing; the inspection is 7 days after the rehearsal. On which numbered day does the delivery occur? Give only the answer.

**Verified reasoning**

1. Anchor inspection at day 14.
2. From inspection to rehearsal apply -7 days, giving day 7.
3. From rehearsal to workshop apply -3 days, giving day 4.
4. From workshop to briefing apply +4 days, giving day 8.
5. From briefing to delivery apply +5 days, giving day 13.
6. Therefore the delivery is on day 13.

**Short final answer:** `Day 13.`

## Family Relations

**Difficulty:** `easy`  
**ID:** `rineya-pr51k-kin-tra-01488`

**Problem**

A family-tree puzzle gives these parent facts. Facts: Usha is the mother of Dax; Nia is the mother of Lena; Lena is the mother of Tara; Leo is the father of Vic; Vic is the father of Dax; Leo is the father of Lena; Evan is the father of Tara; Nia is the mother of Vic. Gender note: Usha is female. How is Usha related to Dax? Give only the relationship.

**Verified reasoning**

1. Usha is listed directly as a parent of Dax.
2. Using Usha's recorded gender, the precise relationship is mother.

**Short final answer:** `Mother.`

## Shortest Path

**Difficulty:** `medium`  
**ID:** `rineya-pr51k-path-tra-00956`

**Problem**

A courier network has the following travel costs. Edges: Brook — Alder costs 9; Brook — Harbor costs 5; Alder — Cairn costs 1; Brook — Grove costs 6; Brook — Delta costs 5; Alder — Iris costs 2; Iris — Brook costs 6; Iris — Cairn costs 12; Grove — Alder costs 12; Delta — Iris costs 3; Delta — Cairn costs 3; Cairn — Grove costs 6. Which location should be visited immediately after Delta on the unique cheapest route to Harbor? Give only the answer.

**Verified reasoning**

1. Run Dijkstra's algorithm from Delta. Finalized distances through the target are Delta=0, Cairn=3, Iris=3, Alder=4, Brook=5, Grove=9, Harbor=10.
2. Backtracking unique predecessors gives Delta → Brook → Harbor with total cost 10.

**Short final answer:** `Brook.`

## Arithmetic Constraints

**Difficulty:** `easy`  
**ID:** `rineya-pr51k-arith-tra-01408`

**Problem**

Use exact integer arithmetic. A pen contains only birds and rabbits. There are 29 animals and 100 legs in total. How many rabbits are there? Give only the number.

**Verified reasoning**

1. Let b be birds and r be rabbits: b+r=29 and 2b+4r=100.
2. Subtract 2(b+r)=58 from the leg equation: 2r=42, so r=21.
3. Then b=29-21=8.

**Short final answer:** `21.`

## Boolean Switches

**Difficulty:** `easy`  
**ID:** `rineya-pr51k-bool-tra-00888`

**Problem**

A control panel uses Boolean operations. Rule: the lamp is On exactly when ((S2 AND S2) AND (S3 OR S1)) is true. Known switch states: S1=Off; S3=On. AND requires both; OR requires at least one; XOR requires exactly one; IMPLIES is false only when its left side is true and right side false. The lamp is On. What must S2 be? Give only On or Off.

**Verified reasoning**

1. Substitute the known switch states into ((S2 AND S2) AND (S3 OR S1)).
2. (S2 AND S2) = On
3. (S3 OR S1) = On
4. ((S2 AND S2) AND (S3 OR S1)) = On
5. With S2=Off the output is Off; with S2=On it is On. Only On matches.

**Short final answer:** `On.`

## Hanoi Planning

**Difficulty:** `easy`  
**ID:** `rineya-pr51k-hanoi-tra-00508`

**Problem**

Plan legal disk moves between three pegs. Disk 1 is smallest; a larger disk may never be placed on a smaller disk. Move one top disk at a time. Start: Left=[empty] bottom-to-top; Middle=[3, 2, 1] bottom-to-top; Right=[empty] bottom-to-top. Goal: Left=[2] bottom-to-top; Middle=[3] bottom-to-top; Right=[1] bottom-to-top. What is the minimum number of legal moves needed? Give only the answer.

**Verified reasoning**

1. Represent each legal configuration by the peg holding each disk; the start-to-goal graph distance is 2.
2. One shortest sequence is: disk 1: Middle→Right; disk 2: Middle→Left.
3. Therefore the requested optimum is 2 moves.

**Short final answer:** `2 moves.`

