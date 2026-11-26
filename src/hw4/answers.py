"""Reading a number out of model output and writing it in the canonical format.

Canonical format (as in train.csv): integer `1220`; decimal with a dot and no trailing zeros `3.5`;
fraction `a/b` in lowest terms. Which of decimal / fraction is expected follows the statement: in
train.csv a non-integer answer is a decimal exactly when the statement contains a decimal number
(e.g. `1,012 : 1000 -> 0.001012`, but `207 : 10 -> 207/10`). This only changes how the model's
number is written, never its value.
"""
from __future__ import annotations

import re
from fractions import Fraction

ANSWER_RE = re.compile(r"(?:відповідь|answer)\s*[:：]?\s*(.+)", re.IGNORECASE)
_MIXED = re.compile(r"(-?)(\d+)\s+(\d+)\s*/\s*(\d+)")
_FRAC = re.compile(r"(-?\d+)\s*/\s*(\d+)")
_DEC = re.compile(r"-?\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?|-?\d+(?:[.,]\d+)?")
STATEMENT_DECIMAL = re.compile(r"\d[.,]\d")


def parse_number(text: str) -> Fraction | None:
    """First number in `text` (mixed number, fraction, decimal with dot or comma, spaced thousands)."""
    if text is None:
        return None
    t = text.replace("−", "-").replace("–", "-").replace("**", "").replace("$", "").replace("\\", "")
    t = re.sub(r"frac\{(\d+)\}\{(\d+)\}", r"\1/\2", t)
    cands = []
    for rx, kind in ((_MIXED, "mixed"), (_FRAC, "frac"), (_DEC, "dec")):
        m = rx.search(t)
        if m:
            cands.append((m.start(), -len(m.group(0)), kind, m))
    if not cands:
        return None
    _, _, kind, m = min(cands, key=lambda c: (c[0], c[1]))
    try:
        if kind == "mixed":
            sign = -1 if m.group(1) else 1
            return sign * (int(m.group(2)) + Fraction(int(m.group(3)), int(m.group(4))))
        if kind == "frac":
            return Fraction(int(m.group(1)), int(m.group(2)))
        s = re.sub(r"[   ]", "", m.group(0)).replace(",", ".")
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def extract_answer(output: str) -> Fraction | None:
    """Number after the last 'Відповідь:'; if there is none, the last number in the output."""
    if not output:
        return None
    hits = list(ANSWER_RE.finditer(output))
    for m in reversed(hits):
        v = parse_number(m.group(1))
        if v is not None:
            return v
    nums = list(_DEC.finditer(output.replace("−", "-")))
    return parse_number(nums[-1].group(0)) if nums else None


def _is_finite_decimal(v: Fraction) -> bool:
    d = v.denominator
    for p in (2, 5):
        while d % p == 0:
            d //= p
    return d == 1


def canonical(v: Fraction | float | int | None, statement: str = "") -> str | None:
    if v is None:
        return None
    if isinstance(v, float):
        v = Fraction(v).limit_denominator(10**9)
    v = Fraction(v)
    if v.denominator == 1:
        return str(v.numerator)
    if STATEMENT_DECIMAL.search(statement or "") and _is_finite_decimal(v):
        k = max(_pow(v.denominator, 2), _pow(v.denominator, 5))
        s = f"{v.numerator * 10**k // v.denominator:d}"
        sign = "-" if s.startswith("-") else ""
        s = s.lstrip("-").rjust(k + 1, "0")
        return (sign + s[:-k] + "." + s[-k:]).rstrip("0").rstrip(".")
    return f"{v.numerator}/{v.denominator}"


def _pow(n: int, p: int) -> int:
    k = 0
    while n % p == 0:
        n //= p
        k += 1
    return k


def normalize(output: str, statement: str = "") -> str | None:
    return canonical(extract_answer(output), statement)
