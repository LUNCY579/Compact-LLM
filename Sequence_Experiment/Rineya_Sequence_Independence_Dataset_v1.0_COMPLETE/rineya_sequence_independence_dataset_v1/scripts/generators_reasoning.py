from __future__ import annotations

import random
from decimal import Decimal
from fractions import Fraction

from common import (
    GeneratedProblem,
    fmt_decimal,
    fmt_fraction,
    fmt_money_cents,
    fmt_time_24,
    make_steps,
)


def _fmt_set(items: list[int] | set[int]) -> str:
    return "{" + ", ".join(map(str, sorted(items))) + "}"


def generate_set_reasoning(rng: random.Random, idx: int) -> GeneratedProblem:
    universe = list(range(1, 31))
    size_a, size_b = rng.randint(4, 8), rng.randint(4, 8)
    overlap_size = rng.randint(1, min(size_a, size_b, 4))
    overlap = set(rng.sample(universe, overlap_size))
    remaining = [x for x in universe if x not in overlap]
    a_only = set(rng.sample(remaining, size_a - overlap_size))
    remaining2 = [x for x in remaining if x not in a_only]
    b_only = set(rng.sample(remaining2, size_b - overlap_size))
    A, B = overlap | a_only, overlap | b_only
    mode = rng.choice(["union", "intersection", "difference", "symmetric", "union_size"])
    if mode == "union":
        result = A | B
        problem = f"Let A = {_fmt_set(A)} and B = {_fmt_set(B)}. Find A ∪ B."
        actions = [f"Collect every element that appears in either set.", f"Remove duplicates and sort the result: {_fmt_set(result)}." ]
        states = [f"Candidate elements come from A or B.", f"Computed union: {_fmt_set(result)}." ]
        answer = _fmt_set(result)
    elif mode == "intersection":
        result = A & B
        problem = f"Let A = {_fmt_set(A)} and B = {_fmt_set(B)}. Find A ∩ B."
        actions = [f"Identify the elements that appear in both A and B: {_fmt_set(result)}." ]
        states = [f"Computed intersection: {_fmt_set(result)}." ]
        answer = _fmt_set(result)
    elif mode == "difference":
        result = A - B
        problem = f"Let A = {_fmt_set(A)} and B = {_fmt_set(B)}. Find A \\ B, the elements in A but not in B."
        actions = [f"Start with A and remove the shared elements {_fmt_set(A & B)}.", f"The remaining elements are {_fmt_set(result)}." ]
        states = [f"Elements to remove from A: {_fmt_set(A & B)}.", f"Computed difference A \\ B: {_fmt_set(result)}." ]
        answer = _fmt_set(result)
    elif mode == "symmetric":
        result = A ^ B
        problem = f"Let A = {_fmt_set(A)} and B = {_fmt_set(B)}. Find the symmetric difference: elements in exactly one of the two sets."
        actions = [f"Find the shared elements: {_fmt_set(A & B)}.", f"Keep elements appearing in only one set: {_fmt_set(result)}." ]
        states = [f"Shared elements to exclude: {_fmt_set(A & B)}.", f"Computed symmetric difference: {_fmt_set(result)}." ]
        answer = _fmt_set(result)
    else:
        result = A | B
        problem = f"Let A = {_fmt_set(A)} and B = {_fmt_set(B)}. How many distinct elements are in A ∪ B?"
        actions = [f"Form the union without duplicates: {_fmt_set(result)}.", f"Count its elements: {len(result)}." ]
        states = [f"Computed union: {_fmt_set(result)}.", f"Union cardinality: {len(result)}." ]
        answer = str(len(result))
    return GeneratedProblem(problem, answer, "set", make_steps(actions, states, answer), 2, "set_reasoning_v1", ("sets", "symbolic"))


def _bool_text(v: bool) -> str:
    return "True" if v else "False"


def generate_boolean_logic(rng: random.Random, idx: int) -> GeneratedProblem:
    symbols = rng.sample(list("PQRSTUVW"), 3)
    a_name, b_name, c_name = symbols
    A, B, C = rng.choice([True, False]), rng.choice([True, False]), rng.choice([True, False])
    template = rng.randrange(4)
    assignment = f"{a_name} = {_bool_text(A)}, {b_name} = {_bool_text(B)}, and {c_name} = {_bool_text(C)}"
    if template == 0:
        inner1 = A and B
        inner2 = not C
        result = inner1 or inner2
        expr = f"({a_name} AND {b_name}) OR (NOT {c_name})"
        actions = [
            f"Evaluate {a_name} AND {b_name}: {_bool_text(A)} AND {_bool_text(B)} = {_bool_text(inner1)}.",
            f"Evaluate NOT {c_name}: NOT {_bool_text(C)} = {_bool_text(inner2)}.",
            f"Combine with OR: {_bool_text(inner1)} OR {_bool_text(inner2)} = {_bool_text(result)}.",
        ]
        states = [f"{a_name} AND {b_name} = {_bool_text(inner1)}.", f"{a_name} AND {b_name} = {_bool_text(inner1)}; NOT {c_name} = {_bool_text(inner2)}.", f"Expression value: {_bool_text(result)}." ]
    elif template == 1:
        inner = B and C
        result = (A != inner)
        expr = f"{a_name} XOR ({b_name} AND {c_name})"
        actions = [f"Evaluate {b_name} AND {c_name}: {_bool_text(B)} AND {_bool_text(C)} = {_bool_text(inner)}.", f"XOR is true when the two inputs differ: {_bool_text(A)} XOR {_bool_text(inner)} = {_bool_text(result)}." ]
        states = [f"{b_name} AND {c_name} = {_bool_text(inner)}.", f"Expression value: {_bool_text(result)}." ]
    elif template == 2:
        inner = A or B
        neg = not inner
        result = neg and C
        expr = f"NOT ({a_name} OR {b_name}) AND {c_name}"
        actions = [f"Evaluate {a_name} OR {b_name}: {_bool_text(A)} OR {_bool_text(B)} = {_bool_text(inner)}.", f"Negate it: NOT {_bool_text(inner)} = {_bool_text(neg)}.", f"Combine with {c_name} using AND: {_bool_text(neg)} AND {_bool_text(C)} = {_bool_text(result)}." ]
        states = [f"{a_name} OR {b_name} = {_bool_text(inner)}.", f"NOT ({a_name} OR {b_name}) = {_bool_text(neg)}.", f"Expression value: {_bool_text(result)}." ]
    else:
        left = not A
        right = B or C
        result = left and right
        expr = f"(NOT {a_name}) AND ({b_name} OR {c_name})"
        actions = [f"Evaluate NOT {a_name}: NOT {_bool_text(A)} = {_bool_text(left)}.", f"Evaluate {b_name} OR {c_name}: {_bool_text(B)} OR {_bool_text(C)} = {_bool_text(right)}.", f"Combine with AND: {_bool_text(left)} AND {_bool_text(right)} = {_bool_text(result)}." ]
        states = [f"NOT {a_name} = {_bool_text(left)}.", f"NOT {a_name} = {_bool_text(left)}; {b_name} OR {c_name} = {_bool_text(right)}.", f"Expression value: {_bool_text(result)}." ]
    problem = rng.choice([
        f"Given {assignment}, evaluate {expr}.",
        f"Use the truth values {assignment}. Is {expr} True or False?",
    ])
    answer = _bool_text(result)
    return GeneratedProblem(problem, answer, "boolean", make_steps(actions, states, answer), 2, "boolean_logic_v1", ("logic", "boolean"))


ORDER_NAMES = [
    "Ari", "Bea", "Cora", "Dion", "Eli", "Faye", "Gus", "Hana", "Ivan", "Juno",
    "Kira", "Liam", "Mina", "Nico", "Omar", "Pia", "Quin", "Rhea", "Seth", "Tara",
]


def generate_ordering_constraints(rng: random.Random, idx: int) -> GeneratedProblem:
    p1, p2, p3, p4 = rng.sample(ORDER_NAMES, 4)
    constraints = [
        f"{p3} is immediately after {p2}.",
        f"{p1} appears before {p2}.",
        f"{p4} appears after {p3}.",
    ]
    rng.shuffle(constraints)
    ask = rng.choice(["order", "first", "second", "third", "last"])
    if ask == "order":
        question = "What is the unique order from first to fourth?"
        answer = f"{p1}, {p2}, {p3}, {p4}"
    else:
        position_map = {"first": p1, "second": p2, "third": p3, "last": p4}
        question = f"Who is {ask}?"
        answer = position_map[ask]
    problem = f"Four people must be ordered. {' '.join(constraints)} {question}"
    actions = [
        f"Treat {p2}, {p3} as a fixed consecutive block because {p3} is immediately after {p2}.",
        f"Place {p1} before that block, giving {p1}, {p2}, {p3} so far.",
        f"Place {p4} after {p3}, producing the unique order {p1}, {p2}, {p3}, {p4}.",
    ]
    states = [f"Fixed block: [{p2}, {p3}].", f"Partial order: {p1}, {p2}, {p3}; {p4} remains.", f"Unique order: {p1}, {p2}, {p3}, {p4}." ]
    return GeneratedProblem(problem, answer, "ordering", make_steps(actions, states, answer), 2, "ordering_constraints_v1", ("logic", "constraints"))


TASK_LABELS = ["inspection", "packing", "calibration", "cleanup", "review", "delivery", "setup", "testing"]


def generate_scheduling(rng: random.Random, idx: int) -> GeneratedProblem:
    t1, t2, t3 = rng.sample(TASK_LABELS, 3)
    start = rng.randint(6 * 60, 18 * 60)
    d1, d2, d3 = rng.choice([20, 25, 30, 35, 40, 45, 50, 60]), rng.choice([15, 20, 30, 45, 60]), rng.choice([20, 30, 40, 50, 60])
    b1, b2 = rng.choice([0, 5, 10, 15]), rng.choice([0, 5, 10, 15])
    after1 = start + d1
    after_break1 = after1 + b1
    after2 = after_break1 + d2
    after_break2 = after2 + b2
    finish = after_break2 + d3
    problem = (
        f"A schedule starts at {fmt_time_24(start)}. {t1.title()} lasts {d1} minutes, followed by a {b1}-minute break; "
        f"{t2} lasts {d2} minutes, followed by a {b2}-minute break; then {t3} lasts {d3} minutes. When does the schedule finish?"
    )
    actions = [
        f"Finish {t1} at {fmt_time_24(start)} + {d1} minutes = {fmt_time_24(after1)}.",
        f"Add the first break: {fmt_time_24(after1)} + {b1} minutes = {fmt_time_24(after_break1)}.",
        f"Finish {t2}: {fmt_time_24(after_break1)} + {d2} minutes = {fmt_time_24(after2)}.",
        f"Add the second break: {fmt_time_24(after2)} + {b2} minutes = {fmt_time_24(after_break2)}.",
        f"Finish {t3}: {fmt_time_24(after_break2)} + {d3} minutes = {fmt_time_24(finish)}.",
    ]
    states = [
        f"Current time after {t1}: {fmt_time_24(after1)}; next is a {b1}-minute break.",
        f"Current time: {fmt_time_24(after_break1)}; next task is {t2} for {d2} minutes.",
        f"Current time after {t2}: {fmt_time_24(after2)}; next is a {b2}-minute break.",
        f"Current time: {fmt_time_24(after_break2)}; final task is {t3} for {d3} minutes.",
        f"Schedule finish time: {fmt_time_24(finish)}.",
    ]
    answer = fmt_time_24(finish)
    return GeneratedProblem(problem, answer, "time", make_steps(actions, states, answer), 2, "scheduling_v1", ("planning", "time"))


INVENTORY_ITEMS = ["cables", "filters", "batteries", "labels", "bolts", "folders", "markers", "cups"]


def generate_inventory(rng: random.Random, idx: int) -> GeneratedProblem:
    item = rng.choice(INVENTORY_ITEMS)
    initial = rng.randint(40, 300)
    restock1 = rng.randint(10, 100)
    sold = rng.randint(5, min(120, initial + restock1 - 10))
    returned = rng.randint(1, 20)
    damaged = rng.randint(1, min(15, initial + restock1 - sold + returned - 1))
    s1 = initial + restock1
    s2 = s1 - sold
    s3 = s2 + returned
    final = s3 - damaged
    problem = rng.choice([
        f"A storeroom begins with {initial} {item}. It receives {restock1}, ships {sold}, gets {returned} returned, and discards {damaged} damaged units. How many remain?",
        f"Track this inventory of {item}: start {initial}; restock +{restock1}; shipment -{sold}; returns +{returned}; damage -{damaged}. Find the final stock.",
    ])
    actions = [
        f"Apply the restock: {initial} + {restock1} = {s1}.",
        f"Apply the shipment: {s1} - {sold} = {s2}.",
        f"Add the returned units: {s2} + {returned} = {s3}.",
        f"Remove the damaged units: {s3} - {damaged} = {final}.",
    ]
    states = [
        f"Current inventory: {s1} {item}; pending: ship {sold}, add {returned} returns, remove {damaged} damaged.",
        f"Current inventory: {s2} {item}; pending: add {returned} returns, remove {damaged} damaged.",
        f"Current inventory: {s3} {item}; pending: remove {damaged} damaged.",
        f"Final inventory computed: {final} {item}.",
    ]
    answer = f"{final} {item}"
    return GeneratedProblem(problem, answer, "quantity", make_steps(actions, states, answer), 1, "inventory_state_updates_v1", ("state-update", "daily-life"))


def generate_finance(rng: random.Random, idx: int) -> GeneratedProblem:
    template = rng.randrange(4)
    if template == 0:
        principal = rng.randint(100, 5000) * 100
        rate = rng.choice([2, 3, 4, 5, 6, 8, 10])
        years = rng.randint(1, 5)
        interest = principal * rate * years // 100
        total = principal + interest
        year_word = "year" if years == 1 else "years"
        problem = f"Using simple interest I = P × r × t, find the total balance for {fmt_money_cents(principal)} at {rate}% per year for {years} {year_word}."
        actions = [f"Convert the rate to decimal form: {rate}% = {rate}/100.", f"Compute simple interest: {fmt_money_cents(principal)} × {rate}/100 × {years} = {fmt_money_cents(interest)}.", f"Add principal and interest: {fmt_money_cents(principal)} + {fmt_money_cents(interest)} = {fmt_money_cents(total)}." ]
        states = [f"Rate represented as {rate}/100.", f"Interest amount: {fmt_money_cents(interest)}.", f"Total balance: {fmt_money_cents(total)}." ]
        answer = fmt_money_cents(total)
    elif template == 1:
        income_dollars = rng.randint(1200, 8000)
        rent_dollars = rng.randint(400, min(3000, int(income_dollars * 0.55)))
        after_rent_dollars = income_dollars - rent_dollars
        food_dollars = rng.randint(100, min(900, after_rent_dollars - 150))
        after_food_dollars = after_rent_dollars - food_dollars
        transport_dollars = rng.randint(50, min(500, after_food_dollars - 50))
        income = income_dollars * 100
        rent = rent_dollars * 100
        food = food_dollars * 100
        transport = transport_dollars * 100
        remaining = income - rent - food - transport
        problem = f"A monthly budget has income {fmt_money_cents(income)}, rent {fmt_money_cents(rent)}, food {fmt_money_cents(food)}, and transport {fmt_money_cents(transport)}. How much remains?"
        after_rent = income - rent
        after_food = after_rent - food
        actions = [f"Subtract rent: {fmt_money_cents(income)} - {fmt_money_cents(rent)} = {fmt_money_cents(after_rent)}.", f"Subtract food: {fmt_money_cents(after_rent)} - {fmt_money_cents(food)} = {fmt_money_cents(after_food)}.", f"Subtract transport: {fmt_money_cents(after_food)} - {fmt_money_cents(transport)} = {fmt_money_cents(remaining)}." ]
        states = [f"Remaining after rent: {fmt_money_cents(after_rent)}.", f"Remaining after food: {fmt_money_cents(after_food)}.", f"Final remaining budget: {fmt_money_cents(remaining)}." ]
        answer = fmt_money_cents(remaining)
    elif template == 2:
        monthly = rng.randint(20, 900) * 100
        months = rng.randint(3, 24)
        bonus = rng.randint(0, 1000) * 100
        deposits = monthly * months
        total = deposits + bonus
        problem = f"Someone saves {fmt_money_cents(monthly)} each month for {months} months and then adds a one-time {fmt_money_cents(bonus)} bonus. What is the total saved?"
        actions = [f"Compute monthly deposits: {fmt_money_cents(monthly)} × {months} = {fmt_money_cents(deposits)}.", f"Add the bonus: {fmt_money_cents(deposits)} + {fmt_money_cents(bonus)} = {fmt_money_cents(total)}." ]
        states = [f"Savings from monthly deposits: {fmt_money_cents(deposits)}.", f"Total savings: {fmt_money_cents(total)}." ]
        answer = fmt_money_cents(total)
    else:
        people = rng.randint(2, 12)
        share = rng.randint(5, 150) * 100
        total = people * share
        problem = f"A bill of {fmt_money_cents(total)} is split equally among {people} people. How much does each person pay?"
        actions = [f"Divide the total by the number of people: {fmt_money_cents(total)} ÷ {people} = {fmt_money_cents(share)}." ]
        states = [f"Equal share computed: {fmt_money_cents(share)} per person." ]
        answer = fmt_money_cents(share)
    return GeneratedProblem(problem, answer, "money", make_steps(actions, states, answer), 2, "finance_arithmetic_v1", ("finance", "arithmetic"))


def _fractional_unit(value: Fraction, plural_unit: str) -> str:
    if value == 1:
        return plural_unit[:-1] if plural_unit.endswith("s") else plural_unit
    return plural_unit


INGREDIENTS = [
    ("flour", "cups"), ("milk", "cups"), ("rice", "cups"), ("olive oil", "tablespoons"),
    ("sugar", "cups"), ("broth", "cups"), ("cocoa", "tablespoons"), ("oats", "cups"),
]


def generate_cooking(rng: random.Random, idx: int) -> GeneratedProblem:
    ingredient, unit = rng.choice(INGREDIENTS)
    base_servings = rng.choice([2, 4, 6, 8])
    scale = rng.choice([Fraction(1, 2), Fraction(3, 2), Fraction(2, 1), Fraction(3, 1)])
    target_servings = int(base_servings * scale)
    quantity = Fraction(rng.randint(1, 12), rng.choice([1, 2, 4]))
    scaled = quantity * scale
    base_unit_label = _fractional_unit(quantity, unit)
    scaled_unit_label = _fractional_unit(scaled, unit)
    problem = rng.choice([
        f"A recipe for {base_servings} servings uses {fmt_fraction(quantity)} {base_unit_label} of {ingredient}. How much is needed for {target_servings} servings?",
        f"Scale a recipe from {base_servings} to {target_servings} servings. It originally uses {fmt_fraction(quantity)} {base_unit_label} of {ingredient}. Find the new amount.",
    ])
    actions = [
        f"Compute the serving scale factor: {target_servings}/{base_servings} = {fmt_fraction(scale)}.",
        f"Multiply the ingredient amount by the scale factor: {fmt_fraction(quantity)} × {fmt_fraction(scale)} = {fmt_fraction(scaled)}.",
    ]
    states = [f"Scale factor: {fmt_fraction(scale)}.", f"Scaled {ingredient} amount: {fmt_fraction(scaled)} {scaled_unit_label}." ]
    answer = f"{fmt_fraction(scaled)} {scaled_unit_label} of {ingredient}"
    return GeneratedProblem(problem, answer, "quantity", make_steps(actions, states, answer), 2, "recipe_scaling_v1", ("cooking", "proportion"))


def generate_physics(rng: random.Random, idx: int) -> GeneratedProblem:
    template = rng.randrange(4)
    if template == 0:
        mass, acceleration = rng.randint(1, 50), rng.randint(1, 20)
        force = mass * acceleration
        problem = f"Using F = m × a, find the force on a {mass} kg object accelerating at {acceleration} m/s²."
        actions = ["Select the relation F = m × a.", f"Substitute and multiply: F = {mass} × {acceleration} = {force} N." ]
        states = ["Selected formula: F = m × a.", f"Computed force: {force} N." ]
        answer = f"{force} N"
    elif template == 1:
        current, resistance = rng.randint(1, 20), rng.randint(1, 50)
        voltage = current * resistance
        problem = f"Ohm's law is V = I × R. What voltage is produced by {current} A through {resistance} Ω?"
        actions = ["Select V = I × R.", f"Substitute: V = {current} × {resistance} = {voltage} V." ]
        states = ["Selected formula: V = I × R.", f"Computed voltage: {voltage} V." ]
        answer = f"{voltage} V"
    elif template == 2:
        density = rng.randint(1, 30)
        volume = rng.randint(2, 50)
        mass = density * volume
        problem = f"Density is ρ = mass ÷ volume. A sample has mass {mass} g and volume {volume} cm³. Find its density."
        actions = ["Select ρ = mass ÷ volume.", f"Substitute: ρ = {mass} ÷ {volume} = {density} g/cm³." ]
        states = ["Selected formula: ρ = mass ÷ volume.", f"Computed density: {density} g/cm³." ]
        answer = f"{density} g/cm³"
    else:
        mass = rng.randint(1, 20) * 2
        speed = rng.randint(2, 20)
        speed_sq = speed * speed
        energy = mass * speed_sq // 2
        problem = f"Use kinetic energy KE = 1/2 × m × v². Find KE for mass {mass} kg moving at {speed} m/s."
        actions = [f"Square the speed: {speed}² = {speed_sq}.", f"Substitute: KE = 1/2 × {mass} × {speed_sq} = {energy} J." ]
        states = [f"Squared speed: {speed_sq}.", f"Computed kinetic energy: {energy} J." ]
        answer = f"{energy} J"
    return GeneratedProblem(problem, answer, "measurement", make_steps(actions, states, answer), 2, "physics_formula_application_v1", ("physics", "formula"))


def generate_chemistry(rng: random.Random, idx: int) -> GeneratedProblem:
    template = rng.randrange(3)
    if template == 0:
        molar_mass = rng.randint(10, 200)
        moles = rng.randint(1, 12)
        mass = molar_mass * moles
        problem = f"Use n = mass ÷ molar mass. A sample has mass {mass} g and molar mass {molar_mass} g/mol. How many moles are present?"
        actions = ["Select n = mass ÷ molar mass.", f"Substitute: n = {mass} ÷ {molar_mass} = {moles} mol." ]
        states = ["Selected formula: n = mass ÷ molar mass.", f"Computed amount: {moles} mol." ]
        answer = f"{moles} mol"
    elif template == 1:
        concentration = rng.randint(1, 25)
        volume = rng.randint(2, 50)
        mass = concentration * volume
        problem = f"A solution concentration is c = mass ÷ volume. If {mass} g of solute is present in {volume} L, what is c?"
        actions = ["Select c = mass ÷ volume.", f"Substitute: c = {mass} ÷ {volume} = {concentration} g/L." ]
        states = ["Selected formula: c = mass ÷ volume.", f"Computed concentration: {concentration} g/L." ]
        answer = f"{concentration} g/L"
    else:
        c1 = rng.randint(2, 20)
        v1 = rng.randint(10, 100)
        c2 = rng.randint(1, c1)
        while (c1 * v1) % c2 != 0:
            v1 += 1
        v2 = c1 * v1 // c2
        problem = f"Use the dilution relation C₁V₁ = C₂V₂. If C₁ = {c1} mol/L, V₁ = {v1} mL, and C₂ = {c2} mol/L, find V₂."
        product = c1 * v1
        actions = [f"Compute C₁V₁: {c1} × {v1} = {product}.", f"Solve V₂ = {product} ÷ {c2} = {v2} mL." ]
        states = [f"Concentration-volume product: {product}.", f"Computed final volume: {v2} mL." ]
        answer = f"{v2} mL"
    return GeneratedProblem(problem, answer, "measurement", make_steps(actions, states, answer), 2, "chemistry_formula_application_v1", ("chemistry", "formula"))


def generate_biology(rng: random.Random, idx: int) -> GeneratedProblem:
    template = rng.randrange(4)
    if template == 0:
        initial = rng.randint(1, 40)
        intervals = rng.randint(2, 8)
        final = initial * (2 ** intervals)
        problem = f"A cell population starts at {initial} and doubles once per interval. How many cells are present after {intervals} intervals?"
        actions = [f"Each interval multiplies the population by 2, so use {initial} × 2^{intervals}.", f"Compute 2^{intervals} = {2**intervals}.", f"Multiply: {initial} × {2**intervals} = {final}." ]
        states = [f"Growth model: {initial} × 2^{intervals}.", f"Doubling factor after {intervals} intervals: {2**intervals}.", f"Computed population: {final}." ]
        answer = f"{final} cells"
    elif template == 1:
        initial = rng.randint(100, 2000)
        births = rng.randint(10, 300)
        deaths = rng.randint(5, min(250, initial + births - 1))
        arrivals = rng.randint(0, 100)
        departures = rng.randint(0, min(100, initial + births - deaths + arrivals - 1))
        s1 = initial + births
        s2 = s1 - deaths
        s3 = s2 + arrivals
        final = s3 - departures
        problem = f"A population begins at {initial}. During the period there are {births} births, {deaths} deaths, {arrivals} arrivals, and {departures} departures. Find the final population."
        actions = [f"Add births: {initial} + {births} = {s1}.", f"Subtract deaths: {s1} - {deaths} = {s2}.", f"Add arrivals: {s2} + {arrivals} = {s3}.", f"Subtract departures: {s3} - {departures} = {final}." ]
        states = [f"Population after births: {s1}.", f"Population after deaths: {s2}.", f"Population after arrivals: {s3}.", f"Final population: {final}." ]
        answer = str(final)
    elif template == 2:
        probability = rng.choice([Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)])
        group = rng.randint(4, 100) * probability.denominator
        expected = group * probability
        problem = f"A trait is expected in {fmt_fraction(probability)} of a group of {group} organisms. How many organisms are expected to show the trait?"
        actions = [f"Multiply the group size by the expected fraction: {group} × {fmt_fraction(probability)}.", f"Compute the product: {fmt_fraction(expected)} organisms." ]
        states = [f"Expected count expression: {group} × {fmt_fraction(probability)}.", f"Computed expected count: {fmt_fraction(expected)}." ]
        answer = f"{fmt_fraction(expected)} organisms"
    else:
        start = rng.randint(1, 30)
        growth = rng.randint(1, 8)
        days = rng.randint(3, 20)
        final = start + growth * days
        problem = f"A plant is {start} cm tall and grows {growth} cm per day for {days} days. What is its final height?"
        actions = [f"Compute total growth: {growth} × {days} = {growth*days} cm.", f"Add the starting height: {start} + {growth*days} = {final} cm." ]
        states = [f"Total growth over {days} days: {growth*days} cm.", f"Computed final height: {final} cm." ]
        answer = f"{final} cm"
    return GeneratedProblem(problem, answer, "quantity", make_steps(actions, states, answer), 2, "biology_quantitative_v1", ("biology", "state-update"))


def generate_code_trace(rng: random.Random, idx: int) -> GeneratedProblem:
    template = rng.randrange(3)
    actions: list[str] = []
    states: list[str] = []
    if template == 0:
        start = rng.randint(-10, 20)
        n = rng.randint(3, 6)
        k = rng.randint(1, 8)
        code = f"total = {start}\nfor i in range(1, {n+1}):\n    total = total + {k} * i\nprint(total)"
        total = start
        actions.append(f"Initialize total = {start}.")
        states.append(f"Program state: total = {start}; next loop index i = 1.")
        for i in range(1, n + 1):
            old = total
            total += k * i
            actions.append(f"At i = {i}, update total = {old} + {k}×{i} = {total}.")
            next_desc = f"next loop index i = {i+1}" if i < n else "loop complete; print total"
            states.append(f"Program state: total = {total}; {next_desc}.")
        actions.append(f"The loop is complete, so print {total}.")
        states.append(f"Printed output: {total}.")
        answer = str(total)
    elif template == 1:
        start = rng.randint(-20, 30)
        values = [rng.randint(1, 15) for _ in range(4)]
        code = f"score = {start}\nfor value in {values}:\n    if value % 2 == 0:\n        score = score + value\n    else:\n        score = score - value\nprint(score)"
        score = start
        actions.append(f"Initialize score = {start}.")
        states.append(f"Program state: score = {start}; next value = {values[0]}.")
        for pos, value in enumerate(values):
            old = score
            if value % 2 == 0:
                score += value
                action = f"Value {value} is even, so update score = {old} + {value} = {score}."
            else:
                score -= value
                action = f"Value {value} is odd, so update score = {old} - {value} = {score}."
            actions.append(action)
            next_desc = f"next value = {values[pos+1]}" if pos + 1 < len(values) else "loop complete; print score"
            states.append(f"Program state: score = {score}; {next_desc}.")
        actions.append(f"The loop is complete, so print {score}.")
        states.append(f"Printed output: {score}.")
        answer = str(score)
    else:
        start = rng.randint(1, 12)
        n = rng.randint(2, 5)
        offset = rng.randint(0, 5)
        code = f"x = {start}\ncount = 0\nwhile count < {n}:\n    x = 2 * x - {offset}\n    count = count + 1\nprint(x)"
        x, count = start, 0
        actions.append(f"Initialize x = {start} and count = 0.")
        states.append(f"Program state: x = {start}, count = 0; loop condition 0 < {n} is true.")
        while count < n:
            old_x, old_count = x, count
            x = 2 * x - offset
            count += 1
            actions.append(f"Iteration {old_count+1}: x = 2×{old_x} - {offset} = {x}, then count = {old_count} + 1 = {count}.")
            condition = "true" if count < n else "false"
            states.append(f"Program state: x = {x}, count = {count}; condition {count} < {n} is {condition}.")
        actions.append(f"The while condition is false, so print {x}.")
        states.append(f"Printed output: {x}.")
        answer = str(x)
    problem = f"Trace the following Python-like code and give the printed output:\n```python\n{code}\n```"
    return GeneratedProblem(problem, answer, "program_output", make_steps(actions, states, answer, initial_state="Program execution has not started."), 3, "code_trace_v1", ("coding", "trace"))


SYLLABLES = ["ka", "lo", "mi", "ra", "te", "zo", "fi", "nu", "ve", "sa", "di", "po", "ge", "lu", "xi", "ba"]


def generate_string_processing(rng: random.Random, idx: int) -> GeneratedProblem:
    base = rng.choice(SYLLABLES) + rng.choice(SYLLABLES) + str(rng.randint(10, 99))
    char = rng.choice([c for c in base if c.isalpha()])
    replacement = rng.choice([c for c in "aeiouxz" if c != char])
    suffix = rng.choice(["_ok", "!", "_end", "#", "_v2"])
    operations = rng.choice([
        ["uppercase", "replace", "reverse", "append"],
        ["replace", "uppercase", "append", "reverse"],
        ["reverse", "uppercase", "replace", "append"],
    ])
    current = base
    actions: list[str] = []
    states: list[str] = []
    for op in operations:
        old = current
        if op == "uppercase":
            current = current.upper()
            actions.append(f"Convert the string to uppercase: {old!r} → {current!r}.")
        elif op == "replace":
            # Replace case-insensitively after possible uppercase transformation.
            target = char.upper() if any(ch.isupper() for ch in current) else char
            repl = replacement.upper() if target.isupper() else replacement
            current = current.replace(target, repl)
            actions.append(f"Replace every {target!r} with {repl!r}: {old!r} → {current!r}.")
        elif op == "reverse":
            current = current[::-1]
            actions.append(f"Reverse the string: {old!r} → {current!r}.")
        else:
            current = current + suffix
            actions.append(f"Append {suffix!r}: {old!r} → {current!r}.")
        states.append(f"Current string: {current!r}; completed operation: {op}.")
    op_descriptions = []
    for op in operations:
        if op == "uppercase":
            op_descriptions.append("convert to uppercase")
        elif op == "replace":
            op_descriptions.append(f"replace every occurrence of {char!r} (matching the current case) with {replacement!r}")
        elif op == "reverse":
            op_descriptions.append("reverse the entire string")
        else:
            op_descriptions.append(f"append {suffix!r}")
    problem = f"Start with the string {base!r}. In order: " + "; then ".join(op_descriptions) + ". What is the final string?"
    answer = current
    return GeneratedProblem(problem, answer, "string", make_steps(actions, states, answer, initial_state=f"Current string: {base!r}; no operations completed."), 2, "string_processing_v1", ("string", "state-update"))


TABLE_ENTITIES = ["Alpha", "Beacon", "Cedar", "Delta", "Elm", "Fjord", "Grove", "Harbor", "Ion", "Juniper"]


def generate_data_table(rng: random.Random, idx: int) -> GeneratedProblem:
    names = rng.sample(TABLE_ENTITIES, 3)
    first = [rng.randint(20, 180) for _ in range(3)]
    increases = rng.sample(range(1, 61), 3)
    second = [first[i] + increases[i] for i in range(3)]
    # Ensure no negative table values.
    for i in range(3):
        if second[i] < 0:
            second[i] = first[i] + abs(increases[i])
            increases[i] = second[i] - first[i]
    table = "| Group | Period 1 | Period 2 |\n|---|---:|---:|\n" + "\n".join(
        f"| {names[i]} | {first[i]} | {second[i]} |" for i in range(3)
    )
    mode = rng.choice(["largest_increase", "row_total", "column_total", "difference"])
    if mode == "largest_increase":
        best_idx = max(range(3), key=lambda i: increases[i])
        problem = f"Use the table below. Which group had the largest change from Period 1 to Period 2, and what was that change?\n\n{table}"
        actions = [
            f"Compute {names[0]}'s change: {second[0]} - {first[0]} = {increases[0]}.",
            f"Compute {names[1]}'s change: {second[1]} - {first[1]} = {increases[1]}.",
            f"Compute {names[2]}'s change: {second[2]} - {first[2]} = {increases[2]}.",
            f"Compare the changes; the largest is {increases[best_idx]} for {names[best_idx]}.",
        ]
        states = [f"Known changes: {names[0]} = {increases[0]}.", f"Known changes: {names[0]} = {increases[0]}, {names[1]} = {increases[1]}.", f"All changes: {names[0]} = {increases[0]}, {names[1]} = {increases[1]}, {names[2]} = {increases[2]}.", f"Largest change: {names[best_idx]} with {increases[best_idx]}." ]
        answer = f"{names[best_idx]}, {increases[best_idx]}"
    elif mode == "row_total":
        target = rng.randrange(3)
        total = first[target] + second[target]
        problem = f"Use the table below. What is the combined total for {names[target]} across both periods?\n\n{table}"
        actions = [f"Read {names[target]}'s two values: {first[target]} and {second[target]}.", f"Add them: {first[target]} + {second[target]} = {total}." ]
        states = [f"Selected values: {first[target]} and {second[target]}.", f"Combined total: {total}." ]
        answer = str(total)
    elif mode == "column_total":
        total = sum(second)
        problem = f"Use the table below. What is the total across all groups in Period 2?\n\n{table}"
        actions = [f"Read the Period 2 values: {second[0]}, {second[1]}, and {second[2]}.", f"Add them: {second[0]} + {second[1]} + {second[2]} = {total}." ]
        states = [f"Selected Period 2 values: {second[0]}, {second[1]}, {second[2]}.", f"Period 2 total: {total}." ]
        answer = str(total)
    else:
        a, b = rng.sample(range(3), 2)
        diff = second[a] - second[b]
        problem = f"Use the table below. In Period 2, how much greater is {names[a]}'s value than {names[b]}'s value? A negative result means it is smaller.\n\n{table}"
        actions = [f"Read the Period 2 values: {names[a]} = {second[a]} and {names[b]} = {second[b]}.", f"Subtract in the requested order: {second[a]} - {second[b]} = {diff}." ]
        states = [f"Selected values: {names[a]} = {second[a]}, {names[b]} = {second[b]}.", f"Computed difference: {diff}." ]
        answer = str(diff)
    return GeneratedProblem(problem, answer, "table_result", make_steps(actions, states, answer), 2, "data_table_reasoning_v1", ("data", "table"))


Y_WORDS = ["city", "berry", "puppy", "story", "library", "country", "family", "party", "baby", "lady", "sky", "fly", "candy", "body"]
SIBILANT_WORDS = ["box", "bus", "brush", "church", "class", "watch", "fox", "dish", "match", "glass", "bench", "wish", "buzz"]
DEFAULT_WORDS = ["lamp", "river", "book", "chair", "tree", "car", "room", "desk", "cloud", "stone", "garden", "phone", "train", "planet", "cup", "door"]


def _pluralize(word: str) -> tuple[str, str]:
    if word.endswith("y") and len(word) >= 2 and word[-2].lower() not in "aeiou":
        return word[:-1] + "ies", "consonant + y rule"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es", "sibilant-ending rule"
    return word + "s", "default rule"


def generate_language_rules(rng: random.Random, idx: int) -> GeneratedProblem:
    words = [rng.choice(Y_WORDS), rng.choice(SIBILANT_WORDS), rng.choice(DEFAULT_WORDS)]
    rng.shuffle(words)
    outputs = [_pluralize(w) for w in words]
    answer = ", ".join(out for out, _ in outputs)
    problem = (
        "Apply these pluralization rules in order: (1) a word ending in consonant + y changes y to ies; "
        "(2) a word ending in s, x, z, ch, or sh adds es; (3) every other word adds s. "
        f"Pluralize these words in the given order: {', '.join(words)}."
    )
    actions = []
    states = []
    completed: list[str] = []
    for word, (out, rule) in zip(words, outputs):
        actions.append(f"Apply the {rule} to {word!r}, producing {out!r}.")
        completed.append(out)
        states.append(f"Plural forms completed so far: {', '.join(completed)}.")
    return GeneratedProblem(problem, answer, "text_transformation", make_steps(actions, states, answer), 2, "explicit_language_rules_v1", ("language", "rule-application"))


DIRECTIONS = ["N", "E", "S", "W"]
DIR_DELTAS = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}


def _turn(direction: str, command: str) -> str:
    i = DIRECTIONS.index(direction)
    return DIRECTIONS[(i + (1 if command == "R" else -1)) % 4]


def generate_recursive_state_updates(rng: random.Random, idx: int) -> GeneratedProblem:
    template = rng.randrange(3)
    if template == 0:
        x, y = rng.randint(-5, 5), rng.randint(-5, 5)
        direction = rng.choice(DIRECTIONS)
        command_count = rng.randint(4, 7)
        commands: list[str] = []
        for _ in range(command_count):
            if rng.random() < 0.45:
                commands.append(rng.choice(["L", "R"]))
            else:
                commands.append(f"F{rng.randint(1, 4)}")
        start_x, start_y, start_dir = x, y, direction
        actions: list[str] = []
        states: list[str] = []
        for pos, cmd in enumerate(commands, 1):
            if cmd in {"L", "R"}:
                old = direction
                direction = _turn(direction, cmd)
                word = "right" if cmd == "R" else "left"
                actions.append(f"Command {pos} is {cmd}: turn {word}, changing direction from {old} to {direction}; position stays ({x}, {y}).")
            else:
                steps = int(cmd[1:])
                dx, dy = DIR_DELTAS[direction]
                old_x, old_y = x, y
                x += dx * steps
                y += dy * steps
                actions.append(f"Command {pos} is {cmd}: move {steps} step(s) while facing {direction}, from ({old_x}, {old_y}) to ({x}, {y}).")
            remaining = commands[pos:]
            states.append(f"Current state: position ({x}, {y}), facing {direction}; remaining commands: {remaining if remaining else 'none'}." )
        answer = f"position ({x}, {y}), facing {direction}"
        problem = (
            f"A robot starts at ({start_x}, {start_y}) facing {start_dir}. Apply these commands in order: {commands}. "
            "L means turn left, R means turn right, and Fn means move forward n grid steps. What is the final state?"
        )
        initial_state = f"Current state: position ({start_x}, {start_y}), facing {start_dir}; no commands applied yet."
    elif template == 1:
        initial = [rng.randint(1, 30) for _ in range(rng.randint(2, 4))]
        stack = initial.copy()
        operations: list[str] = []
        for _ in range(rng.randint(4, 7)):
            if len(stack) > 1 and rng.random() < 0.4:
                operations.append("pop")
                stack.pop()
            else:
                val = rng.randint(1, 50)
                operations.append(f"push {val}")
                stack.append(val)
        # Re-run from initial to build traces.
        stack = initial.copy()
        actions = []
        states = []
        for pos, op in enumerate(operations, 1):
            if op == "pop":
                removed = stack.pop()
                actions.append(f"Operation {pos} pops the top value {removed}, leaving {stack}.")
            else:
                value = int(op.split()[1])
                stack.append(value)
                actions.append(f"Operation {pos} pushes {value}, producing stack {stack}.")
            remaining = operations[pos:]
            states.append(f"Current stack from bottom to top: {stack}; remaining operations: {remaining if remaining else 'none'}." )
        answer = f"stack {stack}, top {stack[-1]}"
        problem = f"A stack starts as {initial} from bottom to top. Apply these operations in order: {operations}. What are the final stack and top value?"
        initial_state = f"Current stack from bottom to top: {initial}; no operations applied yet."
    else:
        switches = {name: rng.choice([False, True]) for name in ["A", "B", "C", "D"]}
        initial = switches.copy()
        operations: list[str] = []
        for _ in range(rng.randint(4, 8)):
            name = rng.choice(list(switches))
            op = rng.choice(["toggle", "set_on", "set_off"])
            operations.append(f"{op} {name}")
        actions = []
        states = []
        for pos, op_text in enumerate(operations, 1):
            op, name = op_text.split()
            old = switches[name]
            if op == "toggle":
                switches[name] = not switches[name]
                action = f"Operation {pos} toggles {name}: {'ON' if old else 'OFF'} → {'ON' if switches[name] else 'OFF'}."
            elif op == "set_on":
                switches[name] = True
                action = f"Operation {pos} sets {name} to ON, so its state is ON."
            else:
                switches[name] = False
                action = f"Operation {pos} sets {name} to OFF, so its state is OFF."
            actions.append(action)
            status = ", ".join(f"{k}={'ON' if v else 'OFF'}" for k, v in switches.items())
            remaining = operations[pos:]
            states.append(f"Current switch state: {status}; remaining operations: {remaining if remaining else 'none'}." )
        answer = ", ".join(f"{k}={'ON' if v else 'OFF'}" for k, v in switches.items())
        initial_text = ", ".join(f"{k}={'ON' if v else 'OFF'}" for k, v in initial.items())
        problem = f"Four switches start as {initial_text}. Apply these operations in order: {operations}. What is the final state of all switches?"
        initial_state = f"Current switch state: {initial_text}; no operations applied yet."
    return GeneratedProblem(problem, answer, "state", make_steps(actions, states, answer, initial_state=initial_state), 3, "recursive_state_updates_v1", ("recursive-state", "state-transition"))


REASONING_GENERATORS = {
    "set_reasoning": ("logic_and_symbolic", generate_set_reasoning),
    "boolean_logic": ("logic_and_symbolic", generate_boolean_logic),
    "ordering_constraints": ("logic_and_symbolic", generate_ordering_constraints),
    "scheduling": ("time_and_planning", generate_scheduling),
    "inventory": ("daily_life", generate_inventory),
    "finance": ("daily_life", generate_finance),
    "cooking": ("daily_life", generate_cooking),
    "physics": ("science", generate_physics),
    "chemistry": ("science", generate_chemistry),
    "biology": ("science", generate_biology),
    "code_trace": ("coding", generate_code_trace),
    "string_processing": ("coding_and_symbolic", generate_string_processing),
    "data_table": ("data_reasoning", generate_data_table),
    "language_rules": ("language", generate_language_rules),
    "recursive_state_updates": ("state_reasoning", generate_recursive_state_updates),
}
