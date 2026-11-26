"""Executes model-written Python in a separate, isolated interpreter with a timeout.

The model's code must set `answer`; we print it as an exact fraction so floats/Fractions/ints all
come back as a Fraction. No network, no files are needed by any problem; the child runs with -I.
"""
from __future__ import annotations

import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

_WRAP = r'''
import math, fractions
from fractions import Fraction
__ns = {{"math": math, "Fraction": Fraction, "fractions": fractions}}
exec(compile({code!r}, "<model>", "exec"), __ns)
a = __ns.get("answer")
if isinstance(a, str):  # e.g. "12 1/2" or "3,5": parsed by the same parser as text answers
    print("S:" + a.replace("\\n", " ")); raise SystemExit(0)
if isinstance(a, float):
    a = Fraction(a).limit_denominator(10**9)
a = Fraction(a)
print(f"{{a.numerator}}/{{a.denominator}}")
'''


def extract_code(output: str) -> str | None:
    blocks = CODE_RE.findall(output or "")
    if blocks:
        return blocks[-1]
    if output and "answer" in output and "=" in output:  # unterminated block
        return output.split("```python", 1)[-1].split("```", 1)[0]
    return None


def run_code(code: str | None, timeout: float = 5.0) -> tuple[Fraction | None, str]:
    """(value, error). value is None when the code fails or sets no number."""
    if not code:
        return None, "no code"
    try:
        p = subprocess.run([sys.executable, "-I", "-c", _WRAP.format(code=code)], capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if p.returncode != 0:
        return None, (p.stderr.strip().splitlines() or ["error"])[-1][:200]
    try:
        last = p.stdout.strip().splitlines()[-1]
        if last.startswith("S:"):
            from .answers import parse_number
            v = parse_number(last[2:])
            return (v, "") if v is not None else (None, "string answer without a number")
        return Fraction(last), ""
    except (ValueError, IndexError, ZeroDivisionError):
        return None, "bad output"


def run_many(outputs: list[str], workers: int = 8) -> list[tuple[Fraction | None, str]]:
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda o: run_code(extract_code(o)), outputs))
