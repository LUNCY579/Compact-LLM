from __future__ import annotations

import math
import random
from decimal import Decimal
from fractions import Fraction

from common import (
    GeneratedProblem,
    fmt_decimal,
    fmt_fraction,
    fmt_money_cents,
    fmt_time_24,
    lcm,
    make_steps,
)


INVENTORY_CONTEXTS = [
    ("warehouse", "boxes", "shelves", "shelf"),
    ("library", "books", "carts", "cart"),
    ("bakery", "rolls", "trays", "tray"),
    ("garden center", "seed packets", "bins", "bin"),
    ("shipping depot", "parcels", "vans", "van"),
    ("school supply room", "notebooks", "cabinets", "cabinet"),
]


def _count_unit(value: int, plural: str) -> str:
    singular_map = {
        "kilometers": "kilometer", "meters": "meter", "centimeters": "centimeter",
        "kilograms": "kilogram", "grams": "gram", "liters": "liter", "milliliters": "milliliter",
        "hours": "hour", "minutes": "minute", "days": "day", "feet": "foot", "inches": "inch",
    }
    return singular_map.get(plural, plural[:-1] if plural.endswith("s") else plural) if value == 1 else plural


def _simplification_action(raw_num: int, raw_den: int, result: Fraction) -> str:
    raw = Fraction(raw_num, raw_den)
    if raw.numerator == raw_num and raw.denominator == raw_den:
        return f"The fraction {raw_num}/{raw_den} is already in lowest terms."
    return f"Simplify {raw_num}/{raw_den} to {fmt_fraction(result)}."


def generate_integer_arithmetic(rng: random.Random, idx: int) -> GeneratedProblem:
    place, item, groups, groups_singular = rng.choice(INVENTORY_CONTEXTS)
    group_count = rng.randint(2, 12)
    each = rng.randint(4, 90)
    shipped = rng.randint(3, 120)
    received = rng.randint(5, 140)
    start = each * group_count + shipped - received
    if start < 10:
        needed_increase = math.ceil((10 - start) / group_count)
        each += needed_increase + rng.randint(0, 12)
        start = each * group_count + shipped - received
    after_receive = start + received
    remaining = after_receive - shipped
    problem_templates = [
        "A {place} starts with {start} {item}, receives {received} more, sends out {shipped}, and divides the remainder equally among {group_count} {groups}. How many {item} go to each {groups_singular}?",
        "There are {start} {item} at a {place}. After {received} arrive and {shipped} leave, the rest are shared equally across {group_count} {groups}. Find the amount in each group.",
        "A {place} has {start} {item}. It adds {received}, removes {shipped}, then distributes what remains evenly into {group_count} {groups}. What is the final count per group?",
    ]
    problem = rng.choice(problem_templates).format(
        place=place,
        start=start,
        item=item,
        received=received,
        shipped=shipped,
        group_count=group_count,
        groups=groups,
        groups_singular=groups_singular,
    )
    answer = f"{each} {item} per {groups_singular}"
    actions = [
        f"Add the incoming amount: {start} + {received} = {after_receive}.",
        f"Subtract the amount sent out: {after_receive} - {shipped} = {remaining}.",
        f"Divide the remaining {remaining} {item} equally among {group_count} {groups}: {remaining} ÷ {group_count} = {each}.",
    ]
    states = [
        f"Current count: {after_receive} {item}. Remaining operations: remove {shipped}, then divide by {group_count}.",
        f"Current count: {remaining} {item}. Remaining operation: divide equally among {group_count} {groups}.",
        f"Each {groups_singular} receives {each} {item}.",
    ]
    return GeneratedProblem(problem, answer, "quantity", make_steps(actions, states, answer), 1, "integer_arithmetic_v1", ("arithmetic", "multi-step"))


def generate_order_of_operations(rng: random.Random, idx: int) -> GeneratedProblem:
    template = rng.randrange(4)
    if template == 0:
        a, b, c, d = rng.randint(2, 25), rng.randint(2, 25), rng.randint(2, 12), rng.randint(2, 9)
        par = a + b
        mul = par * c
        sq = d * d
        ans = mul - sq
        expr = f"({a} + {b}) × {c} - {d}²"
        actions = [
            f"Evaluate the parentheses first: {a} + {b} = {par}.",
            f"Evaluate the exponent: {d}² = {sq}.",
            f"Multiply: {par} × {c} = {mul}.",
            f"Subtract: {mul} - {sq} = {ans}.",
        ]
        states = [
            f"Expression reduced to {par} × {c} - {d}².",
            f"Expression reduced to {par} × {c} - {sq}.",
            f"Expression reduced to {mul} - {sq}.",
            f"Computed value: {ans}.",
        ]
    elif template == 1:
        a, b, c, d = rng.randint(3, 20), rng.randint(8, 25), rng.randint(2, 7), rng.randint(2, 15)
        diff = b - c
        mul = a * diff
        ans = mul + d
        expr = f"{a} × ({b} - {c}) + {d}"
        actions = [
            f"Evaluate the parentheses: {b} - {c} = {diff}.",
            f"Multiply: {a} × {diff} = {mul}.",
            f"Add: {mul} + {d} = {ans}.",
        ]
        states = [f"Expression reduced to {a} × {diff} + {d}.", f"Expression reduced to {mul} + {d}.", f"Computed value: {ans}." ]
    elif template == 2:
        a, b, c, d = rng.randint(2, 30), rng.randint(2, 15), rng.randint(2, 12), rng.randint(1, 30)
        prod = b * c
        interim = a + prod
        ans = interim - d
        expr = f"{a} + {b} × {c} - {d}"
        actions = [
            f"Perform multiplication first: {b} × {c} = {prod}.",
            f"Add from left to right: {a} + {prod} = {interim}.",
            f"Subtract: {interim} - {d} = {ans}.",
        ]
        states = [f"Expression reduced to {a} + {prod} - {d}.", f"Expression reduced to {interim} - {d}.", f"Computed value: {ans}." ]
    else:
        a, b, c = rng.randint(8, 30), rng.randint(2, 7), rng.randint(2, 25)
        diff = a - b
        sq = diff * diff
        ans = sq + c
        expr = f"({a} - {b})² + {c}"
        actions = [
            f"Evaluate the parentheses: {a} - {b} = {diff}.",
            f"Square the result: {diff}² = {sq}.",
            f"Add: {sq} + {c} = {ans}.",
        ]
        states = [f"Expression reduced to {diff}² + {c}.", f"Expression reduced to {sq} + {c}.", f"Computed value: {ans}." ]
    problem = rng.choice([
        f"Evaluate {expr} using the standard order of operations.",
        f"Find the value of the expression {expr}.",
        f"Compute {expr}. Show the operations in the correct order.",
    ])
    answer = str(ans)
    return GeneratedProblem(problem, answer, "integer", make_steps(actions, states, answer), 1, "order_of_operations_v1", ("arithmetic", "precedence"))


def generate_fractions(rng: random.Random, idx: int) -> GeneratedProblem:
    op = rng.choice(["+", "-", "×", "÷"])
    b, d = rng.randint(2, 12), rng.randint(2, 12)
    a, c = rng.randint(1, b - 1), rng.randint(1, d - 1)
    f1, f2 = Fraction(a, b), Fraction(c, d)
    if op == "-" and f1 < f2:
        f1, f2 = f2, f1
        a, b = f1.numerator, f1.denominator
        c, d = f2.numerator, f2.denominator
    if op == "+":
        common = lcm(b, d)
        n1, n2 = a * (common // b), c * (common // d)
        raw = Fraction(n1 + n2, common)
        result = f1 + f2
        actions = [
            f"Use the least common denominator {common}.",
            f"Rewrite the fractions as {n1}/{common} and {n2}/{common}.",
            f"Add the numerators: ({n1} + {n2})/{common} = {n1+n2}/{common}.",
            _simplification_action(n1 + n2, common, result),
        ]
        states = [
            f"Common denominator selected: {common}.",
            f"Equivalent fractions: {n1}/{common} and {n2}/{common}.",
            f"Unsimplified sum: {n1+n2}/{common}.",
            f"Simplified result: {fmt_fraction(result)}.",
        ]
    elif op == "-":
        common = lcm(b, d)
        n1, n2 = a * (common // b), c * (common // d)
        result = f1 - f2
        actions = [
            f"Use the least common denominator {common}.",
            f"Rewrite the fractions as {n1}/{common} and {n2}/{common}.",
            f"Subtract the numerators: ({n1} - {n2})/{common} = {n1-n2}/{common}.",
            _simplification_action(n1 - n2, common, result),
        ]
        states = [
            f"Common denominator selected: {common}.",
            f"Equivalent fractions: {n1}/{common} and {n2}/{common}.",
            f"Unsimplified difference: {n1-n2}/{common}.",
            f"Simplified result: {fmt_fraction(result)}.",
        ]
    elif op == "×":
        raw_num, raw_den = a * c, b * d
        result = f1 * f2
        actions = [
            f"Multiply the numerators: {a} × {c} = {raw_num}.",
            f"Multiply the denominators: {b} × {d} = {raw_den}.",
            _simplification_action(raw_num, raw_den, result),
        ]
        states = [f"Numerator product: {raw_num}.", f"Unsimplified product: {raw_num}/{raw_den}.", f"Simplified result: {fmt_fraction(result)}." ]
    else:
        result = f1 / f2
        raw_num, raw_den = a * d, b * c
        actions = [
            f"Replace division by multiplication by the reciprocal: {a}/{b} × {d}/{c}.",
            f"Multiply to get {raw_num}/{raw_den}.",
            _simplification_action(raw_num, raw_den, result),
        ]
        states = [f"Equivalent multiplication: {a}/{b} × {d}/{c}.", f"Unsimplified quotient: {raw_num}/{raw_den}.", f"Simplified result: {fmt_fraction(result)}." ]
    problem = rng.choice([
        f"Calculate and simplify {a}/{b} {op} {c}/{d}.",
        f"Find the simplified value of {a}/{b} {op} {c}/{d}.",
        f"Compute {a}/{b} {op} {c}/{d} and reduce the result to lowest terms.",
    ])
    answer = fmt_fraction(result)
    return GeneratedProblem(problem, answer, "fraction", make_steps(actions, states, answer), 2, "fraction_arithmetic_v1", ("fractions",))


def generate_decimals(rng: random.Random, idx: int) -> GeneratedProblem:
    template = rng.randrange(3)
    if template == 0:
        a = rng.randint(100, 9000)
        b = rng.randint(50, 5000)
        c = rng.randint(20, min(3000, a + b - 1))
        after_add, ans_cents = a + b, a + b - c
        problem = rng.choice([
            f"A tank contains {a/100:.2f} liters. It receives {b/100:.2f} liters and then releases {c/100:.2f} liters. How many liters remain?",
            f"Start with {a/100:.2f} units, add {b/100:.2f}, then subtract {c/100:.2f}. What is the result?",
        ])
        actions = [
            f"Add the first two quantities: {a/100:.2f} + {b/100:.2f} = {after_add/100:.2f}.",
            f"Subtract the released amount: {after_add/100:.2f} - {c/100:.2f} = {ans_cents/100:.2f}.",
        ]
        states = [f"Current quantity: {after_add/100:.2f}. Remaining operation: subtract {c/100:.2f}.", f"Final quantity computed: {ans_cents/100:.2f}." ]
        answer = f"{ans_cents/100:.2f} liters"
    elif template == 1:
        price = rng.randint(75, 2500)
        qty = rng.randint(2, 12)
        fee = rng.randint(10, 500)
        subtotal = price * qty
        total = subtotal + fee
        problem = rng.choice([
            f"Each component costs {fmt_money_cents(price)}. What is the total cost of {qty} components plus a {fmt_money_cents(fee)} service fee?",
            f"A customer buys {qty} items at {fmt_money_cents(price)} each and pays an additional {fmt_money_cents(fee)} fee. Find the total.",
        ])
        actions = [
            f"Multiply the unit price by the quantity: {fmt_money_cents(price)} × {qty} = {fmt_money_cents(subtotal)}.",
            f"Add the fee: {fmt_money_cents(subtotal)} + {fmt_money_cents(fee)} = {fmt_money_cents(total)}.",
        ]
        states = [f"Subtotal: {fmt_money_cents(subtotal)}. Remaining operation: add {fmt_money_cents(fee)}.", f"Total cost: {fmt_money_cents(total)}." ]
        answer = fmt_money_cents(total)
    else:
        high = rng.randint(500, 15000)
        low = rng.randint(1, high - 1)
        diff = high - low
        problem = rng.choice([
            f"A sensor reading changes from {low/100:.2f} to {high/100:.2f}. By how much did it increase?",
            f"Find the difference between {high/100:.2f} and {low/100:.2f}.",
        ])
        actions = [f"Subtract the earlier value from the later value: {high/100:.2f} - {low/100:.2f} = {diff/100:.2f}."]
        states = [f"Computed increase: {diff/100:.2f}." ]
        answer = f"{diff/100:.2f}"
    return GeneratedProblem(problem, answer, "decimal", make_steps(actions, states, answer), 1, "decimal_arithmetic_v1", ("decimals",))


def generate_percentages(rng: random.Random, idx: int) -> GeneratedProblem:
    template = rng.randrange(4)
    rate = rng.choice([5, 10, 12, 15, 20, 25, 30, 40, 50])
    if template == 0:
        dollars = rng.randint(10, 500)
        price = dollars * 100
        discount = price * rate // 100
        final = price - discount
        problem = rng.choice([
            f"An item costs {fmt_money_cents(price)} and is discounted by {rate}%. What is the sale price?",
            f"Find the price after a {rate}% discount is applied to {fmt_money_cents(price)}.",
        ])
        actions = [
            f"Compute the discount: {rate}% of {fmt_money_cents(price)} = {fmt_money_cents(discount)}.",
            f"Subtract the discount: {fmt_money_cents(price)} - {fmt_money_cents(discount)} = {fmt_money_cents(final)}.",
        ]
        states = [f"Discount amount: {fmt_money_cents(discount)}. Remaining operation: subtract it from {fmt_money_cents(price)}.", f"Sale price: {fmt_money_cents(final)}." ]
        answer = fmt_money_cents(final)
    elif template == 1:
        base_unit = rng.randint(2, 80)
        base = base_unit * 100 // math.gcd(rate, 100)
        increase = base * rate // 100
        final = base + increase
        problem = rng.choice([
            f"A quantity of {base} increases by {rate}%. What is the new quantity?",
            f"Increase {base} by {rate}% and find the result.",
        ])
        actions = [f"Compute the increase: {base} × {rate}/100 = {increase}.", f"Add the increase: {base} + {increase} = {final}." ]
        states = [f"Increase amount: {increase}. Remaining operation: add it to {base}.", f"New quantity: {final}." ]
        answer = str(final)
    elif template == 2:
        meal = rng.randint(10, 250) * 100
        tip = meal * rate // 100
        total = meal + tip
        problem = rng.choice([
            f"A meal costs {fmt_money_cents(meal)}. Add a {rate}% tip. What is the total bill?",
            f"Calculate the total after adding a {rate}% tip to {fmt_money_cents(meal)}.",
        ])
        actions = [f"Compute the tip: {rate}% of {fmt_money_cents(meal)} = {fmt_money_cents(tip)}.", f"Add the tip: {fmt_money_cents(meal)} + {fmt_money_cents(tip)} = {fmt_money_cents(total)}." ]
        states = [f"Tip amount: {fmt_money_cents(tip)}. Remaining operation: add it to the meal cost.", f"Total bill: {fmt_money_cents(total)}." ]
        answer = fmt_money_cents(total)
    else:
        whole_unit = rng.randint(2, 100)
        whole = whole_unit * 100 // math.gcd(rate, 100)
        part = whole * rate // 100
        problem = rng.choice([
            f"What is {rate}% of {whole}?",
            f"A group contains {whole} objects. How many objects represent {rate}% of the group?",
        ])
        actions = [f"Convert {rate}% to {rate}/100 and multiply: {whole} × {rate}/100 = {part}." ]
        states = [f"Computed percentage amount: {part}." ]
        answer = str(part)
    return GeneratedProblem(problem, answer, "number", make_steps(actions, states, answer), 1, "percent_reasoning_v1", ("percent",))


def generate_ratios(rng: random.Random, idx: int) -> GeneratedProblem:
    left, right = rng.randint(1, 9), rng.randint(1, 9)
    while left == right:
        right = rng.randint(1, 9)
    unit = rng.randint(2, 50)
    total = (left + right) * unit
    contexts = [
        ("red beads", "blue beads"),
        ("oak trees", "pine trees"),
        ("morning tickets", "evening tickets"),
        ("small gears", "large gears"),
        ("fiction books", "nonfiction books"),
    ]
    a_name, b_name = rng.choice(contexts)
    ask_left = rng.choice([True, False])
    requested = a_name if ask_left else b_name
    requested_parts = left if ask_left else right
    result = requested_parts * unit
    problem = rng.choice([
        f"The ratio of {a_name} to {b_name} is {left}:{right}. There are {total} items altogether. How many are {requested}?",
        f"A collection has {a_name} and {b_name} in the ratio {left} to {right}, with {total} total. Find the number of {requested}.",
    ])
    actions = [
        f"Add the ratio parts: {left} + {right} = {left+right} total parts.",
        f"Find one part: {total} ÷ {left+right} = {unit}.",
        f"Multiply by the {requested_parts} parts assigned to {requested}: {unit} × {requested_parts} = {result}.",
    ]
    states = [f"Total ratio parts: {left+right}.", f"Value of one ratio part: {unit}.", f"Number of {requested}: {result}." ]
    answer = f"{result} {requested}"
    return GeneratedProblem(problem, answer, "quantity", make_steps(actions, states, answer), 1, "ratio_proportion_v1", ("ratio",))


def generate_linear_equations(rng: random.Random, idx: int) -> GeneratedProblem:
    a = rng.randint(2, 15)
    x = rng.randint(-20, 40)
    b = rng.randint(-30, 30)
    c = a * x + b
    sign = "+" if b >= 0 else "-"
    abs_b = abs(b)
    expression = f"{a}x {sign} {abs_b} = {c}"
    if b >= 0:
        isolate = c - b
        action1 = f"Subtract {b} from both sides: {a}x = {c} - {b} = {isolate}."
    else:
        isolate = c - b
        action1 = f"Add {abs_b} to both sides: {a}x = {c} + {abs_b} = {isolate}."
    problem = rng.choice([
        f"Solve the equation {expression}.",
        f"Find x: {expression}.",
        f"Determine the value of x that satisfies {expression}.",
    ])
    actions = [action1, f"Divide both sides by {a}: x = {isolate} ÷ {a} = {x}." ]
    states = [f"Isolated equation: {a}x = {isolate}.", f"Solved value: x = {x}." ]
    answer = f"x = {x}"
    return GeneratedProblem(problem, answer, "equation", make_steps(actions, states, answer), 2, "linear_equation_v1", ("algebra",))


def generate_inequalities(rng: random.Random, idx: int) -> GeneratedProblem:
    a = rng.randint(2, 12)
    boundary = rng.randint(-20, 40)
    b = rng.randint(-25, 25)
    op = rng.choice(["≤", "≥", "<", ">"])
    c = a * boundary + b
    sign = "+" if b >= 0 else "-"
    expression = f"{a}x {sign} {abs(b)} {op} {c}"
    if b >= 0:
        action1 = f"Subtract {b} from both sides: {a}x {op} {c-b}."
    else:
        action1 = f"Add {abs(b)} to both sides: {a}x {op} {c-b}."
    action2 = f"Divide both sides by the positive number {a}, so the inequality direction stays the same: x {op} {boundary}."
    problem = rng.choice([
        f"Solve the inequality {expression}.",
        f"Find the solution set for {expression}.",
        f"Isolate x in {expression}.",
    ])
    actions = [action1, action2]
    states = [f"Isolated coefficient form: {a}x {op} {a*boundary}.", f"Solution inequality: x {op} {boundary}." ]
    answer = f"x {op} {boundary}"
    return GeneratedProblem(problem, answer, "inequality", make_steps(actions, states, answer), 2, "linear_inequality_v1", ("algebra", "inequality"))


def generate_geometry(rng: random.Random, idx: int) -> GeneratedProblem:
    template = rng.randrange(5)
    unit = rng.choice(["cm", "m", "ft", "in"])
    if template == 0:
        length, width = rng.randint(3, 50), rng.randint(2, 30)
        ans = length * width
        problem = f"A rectangle is {length} {unit} long and {width} {unit} wide. What is its area?"
        actions = [f"Use the rectangle area formula A = length × width.", f"Substitute the dimensions: A = {length} × {width} = {ans}." ]
        states = ["Selected formula: A = length × width.", f"Computed area: {ans} {unit}²." ]
        answer = f"{ans} {unit}²"
    elif template == 1:
        length, width = rng.randint(3, 50), rng.randint(2, 30)
        sum_sides = length + width
        ans = 2 * sum_sides
        problem = f"A rectangle has length {length} {unit} and width {width} {unit}. Find its perimeter."
        actions = [f"Use P = 2(length + width).", f"Add the side lengths: {length} + {width} = {sum_sides}.", f"Double the sum: 2 × {sum_sides} = {ans}." ]
        states = ["Selected formula: P = 2(length + width).", f"Half-perimeter sum: {sum_sides} {unit}.", f"Computed perimeter: {ans} {unit}." ]
        answer = f"{ans} {unit}"
    elif template == 2:
        base = rng.randint(2, 40)
        height = rng.randint(2, 30)
        if (base * height) % 2:
            base += 1
        product = base * height
        ans = product // 2
        problem = f"A triangle has base {base} {unit} and height {height} {unit}. What is its area?"
        actions = ["Use the triangle area formula A = (base × height) ÷ 2.", f"Multiply base and height: {base} × {height} = {product}.", f"Divide by 2: {product} ÷ 2 = {ans}." ]
        states = ["Selected formula: A = (base × height) ÷ 2.", f"Base-height product: {product}.", f"Computed area: {ans} {unit}²." ]
        answer = f"{ans} {unit}²"
    elif template == 3:
        side = rng.randint(2, 40)
        ans = side * side
        problem = f"A square has side length {side} {unit}. Find its area."
        actions = ["Use the square area formula A = side².", f"Compute {side}² = {ans}." ]
        states = ["Selected formula: A = side².", f"Computed area: {ans} {unit}²." ]
        answer = f"{ans} {unit}²"
    else:
        side = rng.randint(3, 50)
        ans = 4 * side
        problem = f"A square has side length {side} {unit}. Find its perimeter."
        actions = ["Use the square perimeter formula P = 4 × side.", f"Compute 4 × {side} = {ans}." ]
        states = ["Selected formula: P = 4 × side.", f"Computed perimeter: {ans} {unit}." ]
        answer = f"{ans} {unit}"
    return GeneratedProblem(problem, answer, "measurement", make_steps(actions, states, answer), 1, "geometry_v1", ("geometry",))


CONVERSIONS = [
    ("kilometers", "meters", 1000, "km", "m"),
    ("meters", "centimeters", 100, "m", "cm"),
    ("kilograms", "grams", 1000, "kg", "g"),
    ("liters", "milliliters", 1000, "L", "mL"),
    ("hours", "minutes", 60, "h", "min"),
    ("days", "hours", 24, "d", "h"),
    ("feet", "inches", 12, "ft", "in"),
]


def generate_unit_conversion(rng: random.Random, idx: int) -> GeneratedProblem:
    src_name, dst_name, factor, src_abbr, dst_abbr = rng.choice(CONVERSIONS)
    first = rng.randint(1, 80)
    second_dst = rng.randint(1, factor * 4)
    converted = first * factor
    total = converted + second_dst
    src_label = _count_unit(first, src_name)
    dst_label = _count_unit(second_dst, dst_name)
    problem = rng.choice([
        f"Convert {first} {src_label} to {dst_name}, then add {second_dst} {dst_label}. What is the total in {dst_name}?",
        f"A measurement combines {first} {src_abbr} and {second_dst} {dst_abbr}. Express the total entirely in {dst_name}.",
    ])
    actions = [
        f"Use 1 {src_abbr} = {factor} {dst_abbr}.",
        f"Convert {first} {src_abbr}: {first} × {factor} = {converted} {dst_abbr}.",
        f"Add the remaining {second_dst} {dst_abbr}: {converted} + {second_dst} = {total} {dst_abbr}.",
    ]
    states = [f"Conversion factor: 1 {src_abbr} = {factor} {dst_abbr}.", f"Converted first amount: {converted} {dst_abbr}.", f"Combined total: {total} {dst_abbr}." ]
    answer = f"{total} {dst_abbr}"
    return GeneratedProblem(problem, answer, "measurement", make_steps(actions, states, answer), 1, "unit_conversion_v1", ("units",))


def generate_elapsed_time(rng: random.Random, idx: int) -> GeneratedProblem:
    start = rng.randint(0, 23 * 60 + 50)
    duration_h = rng.randint(0, 5)
    duration_m = rng.choice([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
    if duration_h == 0 and duration_m < 10:
        duration_m = 35
    duration = duration_h * 60 + duration_m
    total = start + duration
    start_text = fmt_time_24(start)
    end_text = fmt_time_24(total)
    hour_word = "hour" if duration_h == 1 else "hours"
    minute_word = "minute" if duration_m == 1 else "minutes"
    problem = rng.choice([
        f"A process starts at {start_text} and runs for {duration_h} {hour_word} {duration_m} {minute_word}. At what time does it finish?",
        f"Starting at {start_text}, add a duration of {duration_h} h {duration_m} min. Give the finishing time in 24-hour notation.",
    ])
    actions = [
        f"Convert the start time to minutes after midnight: {start//60} × 60 + {start%60} = {start} minutes.",
        f"Convert the duration to minutes: {duration_h} × 60 + {duration_m} = {duration} minutes.",
        f"Add them: {start} + {duration} = {total} total minutes from the starting midnight.",
        f"Convert {total} minutes back to clock time: {end_text}.",
    ]
    states = [f"Start time represented as {start} minutes after midnight.", f"Duration represented as {duration} minutes.", f"Unwrapped finish time: {total} minutes.", f"Clock finish time: {end_text}." ]
    answer = end_text
    return GeneratedProblem(problem, answer, "time", make_steps(actions, states, answer), 2, "elapsed_time_v1", ("time",))


def generate_rate_distance(rng: random.Random, idx: int) -> GeneratedProblem:
    template = rng.randrange(3)
    vehicle = rng.choice(["cyclist", "train", "delivery van", "boat", "runner", "machine carriage"])
    if template == 0:
        speed, time = rng.randint(4, 120), rng.randint(2, 12)
        distance = speed * time
        problem = f"A {vehicle} travels at {speed} km/h for {time} hours. How far does it travel?"
        actions = ["Use distance = speed × time.", f"Substitute the values: distance = {speed} × {time} = {distance} km." ]
        states = ["Selected relation: distance = speed × time.", f"Computed distance: {distance} km." ]
        answer = f"{distance} km"
    elif template == 1:
        speed, time = rng.randint(4, 120), rng.randint(2, 12)
        distance = speed * time
        problem = f"A {vehicle} covers {distance} km at a constant speed of {speed} km/h. How many hours does the trip take?"
        actions = ["Use time = distance ÷ speed.", f"Substitute the values: time = {distance} ÷ {speed} = {time} hours." ]
        states = ["Selected relation: time = distance ÷ speed.", f"Computed travel time: {time} hours." ]
        answer = f"{time} hours"
    else:
        speed, time = rng.randint(4, 120), rng.randint(2, 12)
        distance = speed * time
        problem = f"A {vehicle} travels {distance} km in {time} hours at a constant rate. What is the speed?"
        actions = ["Use speed = distance ÷ time.", f"Substitute the values: speed = {distance} ÷ {time} = {speed} km/h." ]
        states = ["Selected relation: speed = distance ÷ time.", f"Computed speed: {speed} km/h." ]
        answer = f"{speed} km/h"
    return GeneratedProblem(problem, answer, "rate", make_steps(actions, states, answer), 1, "rate_distance_time_v1", ("rate", "word-problem"))


def generate_statistics(rng: random.Random, idx: int) -> GeneratedProblem:
    mode = rng.choice(["mean", "median", "range", "mean_and_range"])
    n = rng.choice([5, 7])
    values = [rng.randint(2, 60) for _ in range(n)]
    if mode in {"mean", "mean_and_range"}:
        rem = sum(values) % n
        values[-1] += (n - rem) % n
    rng.shuffle(values)
    total = sum(values)
    sorted_values = sorted(values)
    mean = Fraction(total, n)
    median = sorted_values[n // 2]
    data_range = max(values) - min(values)
    list_text = ", ".join(map(str, values))
    if mode == "mean":
        problem = f"Find the mean of the data set: {list_text}."
        actions = [f"Add the values: the sum is {total}.", f"Count the values: there are {n}.", f"Divide: {total} ÷ {n} = {fmt_fraction(mean)}." ]
        states = [f"Data sum: {total}.", f"Data sum: {total}; count: {n}.", f"Computed mean: {fmt_fraction(mean)}." ]
        answer = fmt_fraction(mean)
    elif mode == "median":
        problem = f"Find the median of the data set: {list_text}."
        actions = [f"Sort the values: {', '.join(map(str, sorted_values))}.", f"There are {n} values, so select the {(n//2)+1}th value: {median}." ]
        states = [f"Sorted data: {', '.join(map(str, sorted_values))}.", f"Computed median: {median}." ]
        answer = str(median)
    elif mode == "range":
        problem = f"Find the range of the data set: {list_text}."
        actions = [f"Identify the minimum {min(values)} and maximum {max(values)}.", f"Subtract: {max(values)} - {min(values)} = {data_range}." ]
        states = [f"Minimum: {min(values)}; maximum: {max(values)}.", f"Computed range: {data_range}." ]
        answer = str(data_range)
    else:
        problem = f"For the data set {list_text}, find both the mean and the range."
        actions = [
            f"Add the values: the sum is {total}.",
            f"Divide by the {n} values: mean = {total} ÷ {n} = {fmt_fraction(mean)}.",
            f"Identify minimum {min(values)} and maximum {max(values)}.",
            f"Compute the range: {max(values)} - {min(values)} = {data_range}.",
        ]
        states = [f"Data sum: {total}.", f"Mean: {fmt_fraction(mean)}.", f"Minimum: {min(values)}; maximum: {max(values)}; mean: {fmt_fraction(mean)}.", f"Mean: {fmt_fraction(mean)}; range: {data_range}." ]
        answer = f"mean {fmt_fraction(mean)}, range {data_range}"
    return GeneratedProblem(problem, answer, "statistic", make_steps(actions, states, answer), 2, "descriptive_statistics_v1", ("statistics",))


def generate_probability(rng: random.Random, idx: int) -> GeneratedProblem:
    template = rng.randrange(3)
    if template == 0:
        red, blue, green = rng.randint(1, 20), rng.randint(1, 20), rng.randint(1, 20)
        total = red + blue + green
        target_name, target_count = rng.choice([("red", red), ("blue", blue), ("green", green)])
        prob = Fraction(target_count, total)
        problem = f"A bag contains {red} red, {blue} blue, and {green} green tokens. One token is chosen uniformly at random. What is the probability that it is {target_name}?"
        actions = [f"Count all tokens: {red} + {blue} + {green} = {total}.", f"Use favorable outcomes over total outcomes: {target_count}/{total}.", f"Simplify the fraction to {fmt_fraction(prob)}." ]
        states = [f"Total tokens: {total}.", f"Raw probability: {target_count}/{total}.", f"Simplified probability: {fmt_fraction(prob)}." ]
        answer = fmt_fraction(prob)
    elif template == 1:
        sides = rng.choice([4, 6, 8, 10, 12])
        threshold = rng.randint(2, sides)
        favorable = sides - threshold + 1
        prob = Fraction(favorable, sides)
        problem = f"A fair {sides}-sided die numbered 1 through {sides} is rolled. What is the probability of rolling at least {threshold}?"
        actions = [f"The favorable results are {threshold} through {sides}, giving {favorable} outcomes.", f"There are {sides} equally likely outcomes, so the probability is {favorable}/{sides}.", f"Simplify to {fmt_fraction(prob)}." ]
        states = [f"Favorable outcome count: {favorable}.", f"Raw probability: {favorable}/{sides}.", f"Simplified probability: {fmt_fraction(prob)}." ]
        answer = fmt_fraction(prob)
    else:
        sides = rng.choice([4, 6, 8])
        target = rng.randint(1, sides)
        prob_one = Fraction(1, sides)
        prob = prob_one * Fraction(1, 2)
        problem = f"A fair {sides}-sided die and a fair coin are used. What is the probability of rolling {target} and getting heads?"
        raw_text = f"1/{2*sides}"
        final_mult = (
            f"The events are independent, so multiply: 1/{sides} × 1/2 = {raw_text}. The fraction is already in lowest terms."
            if raw_text == fmt_fraction(prob)
            else f"The events are independent, so multiply: 1/{sides} × 1/2 = {raw_text}, which simplifies to {fmt_fraction(prob)}."
        )
        actions = [f"The probability of rolling {target} is 1/{sides}.", "The probability of heads is 1/2.", final_mult ]
        states = [f"Die-event probability: 1/{sides}.", f"Die-event probability: 1/{sides}; coin-event probability: 1/2.", f"Joint probability: {fmt_fraction(prob)}." ]
        answer = fmt_fraction(prob)
    return GeneratedProblem(problem, answer, "probability", make_steps(actions, states, answer), 2, "probability_v1", ("probability",))


def generate_sequences(rng: random.Random, idx: int) -> GeneratedProblem:
    template = rng.randrange(4)
    if template == 0:
        first, diff = rng.randint(-20, 30), rng.choice([i for i in range(-10, 11) if i != 0])
        target_n = rng.randint(7, 15)
        terms = [first + i * diff for i in range(5)]
        ans = first + (target_n - 1) * diff
        problem = f"The arithmetic sequence begins {', '.join(map(str, terms))}. What is term {target_n}?"
        actions = [f"Identify the common difference: {terms[1]} - {terms[0]} = {diff}.", f"Use a_n = a_1 + (n - 1)d.", f"Substitute: a_{target_n} = {first} + ({target_n}-1)×({diff}) = {ans}." ]
        states = [f"Common difference: {diff}.", f"Formula selected: a_n = {first} + (n-1)×({diff}).", f"Computed term {target_n}: {ans}." ]
        answer = str(ans)
    elif template == 1:
        first, ratio = rng.randint(1, 8), rng.choice([2, 3, 4])
        target_n = rng.randint(6, 9)
        terms = [first * ratio**i for i in range(4)]
        ans = first * ratio ** (target_n - 1)
        problem = f"The geometric sequence begins {', '.join(map(str, terms))}. What is term {target_n}?"
        actions = [f"Identify the common ratio: {terms[1]} ÷ {terms[0]} = {ratio}.", "Use a_n = a_1 × r^(n-1).", f"Substitute: a_{target_n} = {first} × {ratio}^{target_n-1} = {ans}." ]
        states = [f"Common ratio: {ratio}.", f"Formula selected: a_n = {first} × {ratio}^(n-1).", f"Computed term {target_n}: {ans}." ]
        answer = str(ans)
    elif template == 2:
        start = rng.randint(1, 30)
        add = rng.randint(2, 12)
        sub = rng.randint(1, 8)
        terms = [start]
        for i in range(1, 6):
            terms.append(terms[-1] + add if i % 2 == 1 else terms[-1] - sub)
        ans = terms[-1] + add
        problem = f"A sequence alternates between adding {add} and subtracting {sub}, starting with an addition. It begins {', '.join(map(str, terms))}. What is the next term?"
        actions = [f"The listed last change was subtraction by {sub}, so the next operation is addition by {add}.", f"Add {add} to {terms[-1]}: {terms[-1]} + {add} = {ans}." ]
        states = [f"Next operation identified: add {add} to {terms[-1]}.", f"Computed next term: {ans}." ]
        answer = str(ans)
    else:
        start, increment = rng.randint(-15, 25), rng.randint(2, 15)
        steps = rng.randint(5, 12)
        ans = start + steps * increment
        problem = f"A recurrence is defined by x₀ = {start} and xₙ₊₁ = xₙ + {increment}. What is x_{steps}?"
        actions = [f"Each update adds {increment}, and {steps} updates are needed from x₀ to x_{steps}.", f"Compute x_{steps} = {start} + {steps}×{increment} = {ans}." ]
        states = [f"Update count: {steps}; increment per update: {increment}.", f"Computed recurrence value: x_{steps} = {ans}." ]
        answer = str(ans)
    return GeneratedProblem(problem, answer, "number", make_steps(actions, states, answer), 2, "sequence_patterns_v1", ("sequence",))


MATH_GENERATORS = {
    "integer_arithmetic": ("mathematics", generate_integer_arithmetic),
    "order_of_operations": ("mathematics", generate_order_of_operations),
    "fractions": ("mathematics", generate_fractions),
    "decimals": ("mathematics", generate_decimals),
    "percentages": ("mathematics", generate_percentages),
    "ratios": ("mathematics", generate_ratios),
    "linear_equations": ("mathematics", generate_linear_equations),
    "inequalities": ("mathematics", generate_inequalities),
    "geometry": ("mathematics", generate_geometry),
    "unit_conversion": ("measurement", generate_unit_conversion),
    "elapsed_time": ("time_and_planning", generate_elapsed_time),
    "rate_distance": ("mathematics", generate_rate_distance),
    "statistics": ("data_reasoning", generate_statistics),
    "probability": ("mathematics", generate_probability),
    "sequences": ("mathematics", generate_sequences),
}
