"""Deterministic synthetic calibration text for categories no local corpus covers densely enough:
numbers, dates, URLs, paths, maths notation, accented Latin names, punctuation and whitespace.

Seeded and pure: the same seed reproduces the same bytes, and the generator's sha256 is recorded in
the corpus manifest. These rows are CALIBRATION ONLY (tokenizer utility and fragmentation probes);
they never enter a layer-ablation panel, because synthetic strings say nothing about model
competence.
"""
from __future__ import annotations

import random

ACCENTED = [
    "José", "García", "Martínez", "Pérez", "González", "Rodríguez", "Hernández", "López", "Sánchez",
    "Gómez", "Díaz", "Álvarez", "Jiménez", "Muñoz", "Peña", "Núñez", "Ibáñez", "Ramírez", "Méndez",
    "François", "Hélène", "Jérôme", "Aurélie", "Loïc", "Noël", "Chloé", "Léa", "Anaïs", "Maël",
    "Müller", "Schäfer", "Größe", "Straße", "Jürgen", "Björn", "Sören", "Köhler", "Böhm", "Weiß",
    "Zoë", "Brontë", "Ångström", "Åsa", "Øystein", "Søren", "Kierkegaard", "Ærø", "Tromsø",
    "São Paulo", "João", "Conceição", "Gonçalves", "Magalhães", "Brasília", "Assunção",
    "Dvořák", "Janáček", "Čapek", "Łódź", "Wałęsa", "Kraków", "Gdańsk", "Małgorzata", "Škoda",
    "Erdoğan", "İstanbul", "Şahin", "Çelik", "Ölmez", "Göteborg", "Malmö", "Reykjavík", "Þór",
    "naïve", "café", "résumé", "façade", "fiancée", "déjà vu", "crème brûlée", "jalapeño",
    "piñata", "señor", "doppelgänger", "über", "coöperate", "rôle", "élan", "protégé", "soupçon",
    "Nguyễn", "Trần", "Hà Nội", "Phạm", "Việt", "Ōsaka", "Tōkyō", "Kyūshū", "Māori", "Hawaiʻi",
]
GREEK = "αβγδεζηθικλμνξοπρστυφχψω ΑΒΓΔΘΛΞΠΣΦΨΩ"
MATH_TEMPLATES = [
    "Let {v} = {a}{w}^2 + {b}{w} - {c}; then d{v}/d{w} = {d}{w} + {b}.",
    "∑_{{i=1}}^{{{n}}} i = {n}({n}+1)/2 = {s}",
    "∫_0^{a} {w}^{b} d{w} = {a}^{bp}/{bp}",
    "For all ε > 0 there is δ > 0 such that |{w} − {a}| < δ ⇒ |f({w}) − f({a})| < ε.",
    "P(A ∩ B) = P(A)·P(B | A) = {p1} × {p2} ≈ {p3}",
    "{a} ≤ {w} ≤ {b}, {w} ≠ {c}, and {v} ≥ {d} ± {e}",
    "√({a}² + {b}²) ≈ {r}",
    "lim_{{{w}→∞}} (1 + 1/{w})^{w} = e ≈ 2.71828",
    "The matrix A ∈ ℝ^{{{a}×{b}}} has rank ≤ min({a}, {b}); ‖A‖₂ = σ_max.",
    "Solve {a}{w} + {b} = {c}: {w} = ({c} − {b})/{a} = {q}",
    "μ = {m}, σ = {sd}; z = ({w} − μ)/σ",
    "{g1} + {g2} = {g3} (in radians, π/{a} ≈ {pr})",
]
TLDS = [".com", ".org", ".net", ".io", ".edu", ".gov", ".dev", ".ai", ".co.uk"]
WORDS = ["data", "api", "docs", "user", "project", "report", "image", "config", "search",
         "release", "archive", "index", "profile", "settings", "results", "model", "v2", "static"]
EXT = [".py", ".ts", ".js", ".json", ".md", ".txt", ".csv", ".yaml", ".toml", ".html", ".png", ".log"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
PUNCT_LINES = [
    "Wait -- what?! (Really?) Yes... well, sort of; it's \"complicated\" [see note 3].",
    "Q: Is it 50/50? A: No -- it's 60:40 (±5%), i.e. roughly 3:2.",
    "He said, “It’s fine” — then left… Not ‘fine’ at all.",
    "Items: a) first; b) second; c) third -- and so on & so forth.",
    "Use `--force` only if -f fails; see #42, @admin, and ~/notes.",
    "Note*: prices ($9.99, €8.50, £7.25, ¥1,200) exclude tax^.",
    "Hmm... ?!? ... !!! ... ??? ... -- ... ***",
]


def numbers_dates_urls_paths(rng: random.Random) -> str:
    kind = rng.randrange(6)
    if kind == 0:
        return (f"Invoice #{rng.randrange(10000, 99999)}: {rng.randrange(1, 999):,} units at "
                f"${rng.randrange(1, 9999):,}.{rng.randrange(100):02d} = ${rng.randrange(1000, 9999999):,}.{rng.randrange(100):02d} "
                f"({rng.uniform(-50, 50):+.2f}% YoY), ratio {rng.random():.4f}, 1e-{rng.randrange(1, 12)}")
    if kind == 1:
        y, mo, d = rng.randrange(1900, 2100), rng.randrange(1, 13), rng.randrange(1, 29)
        return (f"{y}-{mo:02d}-{d:02d}T{rng.randrange(24):02d}:{rng.randrange(60):02d}:{rng.randrange(60):02d}Z; "
                f"{MONTHS[mo - 1]} {d}, {y}; {d}/{mo}/{y}; Q{rng.randrange(1, 5)} FY{y % 100:02d}; "
                f"{rng.randrange(1, 12)}:{rng.randrange(60):02d} {'AM' if rng.random() < .5 else 'PM'}")
    if kind == 2:
        host = rng.choice(WORDS) + rng.choice(TLDS)
        path = "/".join(rng.choice(WORDS) for _ in range(rng.randrange(1, 4)))
        return (f"https://www.{host}/{path}{rng.choice(EXT)}?id={rng.randrange(1000)}&ref={rng.choice(WORDS)}#"
                f"{rng.choice(WORDS)} and http://localhost:{rng.randrange(1024, 65535)}/{rng.choice(WORDS)} "
                f"mailto:{rng.choice(WORDS)}@{host}")
    if kind == 3:
        return (f"C:\\Users\\{rng.choice(WORDS)}\\{rng.choice(WORDS)}\\{rng.choice(WORDS)}{rng.choice(EXT)} "
                f"/home/{rng.choice(WORDS)}/{rng.choice(WORDS)}/{rng.choice(WORDS)}{rng.choice(EXT)} "
                f"./{rng.choice(WORDS)}/../{rng.choice(WORDS)}{rng.choice(EXT)} ~/.config/{rng.choice(WORDS)}")
    if kind == 4:
        return (f"v{rng.randrange(10)}.{rng.randrange(30)}.{rng.randrange(100)} build {rng.randrange(10**6):06d}, "
                f"IPv4 {rng.randrange(256)}.{rng.randrange(256)}.{rng.randrange(256)}.{rng.randrange(256)}, "
                f"UUID {rng.getrandbits(32):08x}-{rng.getrandbits(16):04x}-4{rng.getrandbits(12):03x}-"
                f"{rng.getrandbits(16):04x}-{rng.getrandbits(48):012x}, SHA {rng.getrandbits(64):016x}")
    return (f"Call +1 ({rng.randrange(200, 999)}) {rng.randrange(200, 999)}-{rng.randrange(1000, 9999)}; "
            f"ZIP {rng.randrange(10000, 99999)}; lat {rng.uniform(-90, 90):.5f}, lon {rng.uniform(-180, 180):.5f}; "
            f"{rng.randrange(1, 100)} km/h, {rng.uniform(-40, 50):.1f} °C, {rng.randrange(1, 500)} MB/s")


def math_text(rng: random.Random) -> str:
    v, w = rng.choice("fghyz"), rng.choice("xtuvs")
    a, b, c, d, e = (rng.randrange(1, 20) for _ in range(5))
    n = rng.randrange(2, 200)
    p1, p2 = round(rng.random(), 2), round(rng.random(), 2)
    t = rng.choice(MATH_TEMPLATES)
    return t.format(v=v, w=w, a=a, b=b, c=c, d=2 * a, e=e, n=n, s=n * (n + 1) // 2, bp=b + 1,
                    p1=p1, p2=p2, p3=round(p1 * p2, 4), r=round((a * a + b * b) ** .5, 3),
                    q=round((c - b) / a, 3), m=rng.randrange(100), sd=rng.randrange(1, 20),
                    g1=rng.choice(GREEK.split()[0]), g2=rng.choice(GREEK.split()[0]),
                    g3=rng.choice(GREEK.split()[0]), pr=round(3.14159 / a, 4))


def technical_names_accents(rng: random.Random) -> str:
    names = rng.sample(ACCENTED, 4)
    return (f"{names[0]} and {names[1]} met {names[2]} in {names[3]}; they discussed "
            f"{rng.choice(['PyTorch', 'Kubernetes', 'PostgreSQL', 'TensorFlow', 'NumPy', 'Rust', 'WebAssembly', 'GraphQL'])} "
            f"and the {rng.choice(['Schrödinger', 'Fourier', 'Navier–Stokes', 'Cauchy–Schwarz', 'Gödel', 'Poincaré'])} "
            f"{rng.choice(['equation', 'transform', 'inequality', 'theorem', 'conjecture'])}.")


def punctuation_whitespace(rng: random.Random) -> str:
    line = rng.choice(PUNCT_LINES)
    ws = rng.choice(["\n", "\n\n", "\n\n\n", "\t", "  ", "    ", "\r\n", " \n", "\n    ", "\n        "])
    return ws.join([line, rng.choice(PUNCT_LINES), " " * rng.randrange(0, 17) + rng.choice(PUNCT_LINES)])


GENERATORS = {
    "numbers_dates_urls_paths": numbers_dates_urls_paths,
    "math": math_text,
    "technical_names_accents": technical_names_accents,
    "punctuation_whitespace": punctuation_whitespace,
}


def generate(category: str, n: int, seed: int) -> list[str]:
    rng = random.Random(f"{seed}:{category}")
    return [GENERATORS[category](rng) for _ in range(n)]
