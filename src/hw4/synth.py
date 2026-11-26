"""Synthetic training problems: my own generators for all 29 skills, with exact answers.

Used ONLY to make training data (teacher solutions are kept when they match the generated answer).
Answers on val/test always come from the model. Statements are new random instances, written in the
style of train.csv; any generated statement that coincides with a train/test statement is dropped.

python -m hw4.synth --per-skill 200 --out runs/data_synth/problems.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
from fractions import Fraction
from pathlib import Path

from .answers import canonical
from .data import load

R = random.Random()


def ri(a, b):
    return R.randint(a, b)


def pick(*xs):
    return R.choice(xs)


def dec(x: Fraction) -> str:
    """Fraction -> Ukrainian decimal string with a comma."""
    return canonical(x, "1,1").replace(".", ",")


def rdec(lo, hi, places):
    return Fraction(ri(lo * 10**places, hi * 10**places), 10**places)


# each generator returns (statement, exact answer, steps)
def g_avg():
    k = ri(3, 5)
    mean = ri(8, 90)
    xs = [mean + ri(-12, 12) for _ in range(k - 1)]
    last = mean * k - sum(xs)
    if not 1 <= last <= 150:
        return g_avg()
    xs.append(last)
    R.shuffle(xs)
    s = ", ".join(map(str, xs[:-1])) + f" і {xs[-1]}"
    st = pick(f"Знайди середнє арифметичне чисел {s}.",
              f"Протягом кількох днів вимірювали довжину тіні дерева: {', '.join(f'{x} см' for x in xs)}. Знайди середню довжину тіні в сантиметрах.",
              f"Температура повітря протягом кількох днів: {', '.join(f'{x} °C' for x in xs)}. Знайди середню температуру в градусах Цельсія.",
              f"Учні за кілька днів прочитали {', '.join(f'{x} сторінок' for x in xs)}. Скільки сторінок у середньому читали за день?",
              f"Маси кавунів: {', '.join(f'{x} кг' for x in xs)}. Знайди середню масу кавуна в кілограмах.")
    return st, mean, [sum(xs)]


def g_dadd():
    p = ri(1, 2)
    a, b = rdec(1, 30, p), rdec(1, 30, ri(1, 2))
    return f"Обчисли: {dec(a)} + {dec(b)}", a + b, []


def g_dsub():
    a, b = rdec(5, 40, ri(1, 3)), rdec(1, 20, ri(1, 2))
    if a <= b:
        a, b = b, a
    if a == b:
        return g_dsub()
    return f"Обчисли: {dec(a)} - {dec(b)}", a - b, []


def g_dmul10():
    k = pick(10, 100, 1000)
    if R.random() < 0.25:
        a = Fraction(ri(11, 999))
        return f"Обчисли: {a} : {k}", a / k, []
    a = rdec(1, 999, ri(1, 3))
    op = pick("×", ":")
    return f"Обчисли: {dec(a)} {op} {k}", a * k if op == "×" else a / k, []


def g_eq_addsub():
    x, b = ri(5, 700), ri(5, 500)
    form = ri(0, 3)
    if form == 0:
        return f"Розв'яжи рівняння: x + {b} = {x + b}", x, []
    if form == 1:
        return f"Розв'яжи рівняння: {b} + x = {x + b}", x, []
    if form == 2:
        return f"Розв'яжи рівняння: x - {b} = {x}", x + b, []
    return f"Розв'яжи рівняння: {x + b} - x = {b}", x, []


def g_eq_mul():
    x, b = ri(2, 60), ri(2, 25)
    form = ri(0, 3)
    if form == 0:
        return f"Розв'яжи рівняння: x × {b} = {x * b}", x, []
    if form == 1:
        return f"Розв'яжи рівняння: {b} × x = {x * b}", x, []
    if form == 2:
        return f"Розв'яжи рівняння: x : {b} = {x}", x * b, []
    return f"Розв'яжи рівняння: {x * b} : x = {b}", x, []


def g_fmixed():
    d = ri(2, 12)
    n = d * ri(1, 15) + ri(1, d - 1)
    q, r = divmod(n, d)
    if R.random() < 0.5:
        return f"Виділи цілу частину неправильного дробу {n}/{d}.", q, [r]
    return f"Запиши неправильний дріб {n}/{d} мішаним числом. Яка в нього дробова частина?", Fraction(r, d), [q]


def g_freduce():
    while True:
        a, b, k = ri(1, 12), ri(2, 15), ri(2, 12)
        if a < b and math.gcd(a, b) == 1:
            break
    n, d = a * k, b * k
    form = ri(0, 2)
    if form == 0:
        return f"Скороти дріб {n}/{d} повністю. Який чисельник має скорочений дріб?", a, []
    if form == 1:
        return f"Скороти дріб {n}/{d} повністю. Який знаменник має скорочений дріб?", b, []
    return f"На яке найбільше число можна скоротити дріб {n}/{d}?", k, []


UNITS = ["мм", "см", "дм", "м", "км"]


def g_area():
    u = pick("см", "м", "дм", "мм", "км")
    if R.random() < 0.5:
        s = ri(2, 40)
        obj = pick("квадрата", "квадратного аркуша", "квадратної ділянки", "квадратної кімнати")
        return f"Площа {obj} дорівнює {s * s} {u}². Знайди довжину сторони.", s, []
    a, b = ri(2, 40), ri(2, 40)
    if R.random() < 0.5:
        return f"Довжина прямокутника {a} {u}, ширина {b} {u}. Знайди його площу.", a * b, []
    return f"Площа прямокутника дорівнює {a * b} {u}², а його довжина {a} {u}. Знайди ширину.", b, []


def g_perim():
    u = pick("см", "м", "дм", "мм", "км")
    if R.random() < 0.5:
        s = ri(3, 80)
        obj = pick("квадрата", "квадратної ділянки", "квадратного саду")
        return f"Периметр {obj} дорівнює {4 * s} {u}. Знайди довжину сторони.", s, []
    a, b = ri(3, 80), ri(3, 80)
    if R.random() < 0.5:
        return f"Довжина прямокутника {a} {u}, ширина {b} {u}. Знайди його периметр.", 2 * (a + b), [a + b]
    return f"Периметр прямокутника дорівнює {2 * (a + b)} {u}, а його довжина {a} {u}. Знайди ширину.", b, [a + b]


def g_vol():
    u = pick("см", "м", "дм")
    form = ri(0, 2)
    if form == 0:
        s, h = ri(5, 90), ri(2, 15)
        return f"Об'єм прямокутного паралелепіпеда дорівнює {s * h} {u}³, площа його основи {s} {u}². Знайди висоту.", h, []
    if form == 1:
        a = ri(2, 12)
        return f"Знайди об'єм куба з ребром {a} {u}.", a ** 3, []
    a, b, c = ri(2, 15), ri(2, 15), ri(2, 15)
    obj = pick("Басейн має", "Коробка має", "Акваріум має")
    return f"{obj} форму прямокутного паралелепіпеда з довжиною {a} {u}, шириною {b} {u} і висотою {c} {u}. Знайди його об'єм.", a * b * c, [a * b]


VEH = [("Катер", "пливе"), ("Електричка", "рухається"), ("Автомобіль", "їде"), ("Велосипедист", "їде"),
       ("Літак", "летить"), ("Поїзд", "рухається"), ("Пішохід", "йде")]


def g_motion():
    who, verb = R.choice(VEH)
    v, t = ri(4, 90) if who != "Літак" else ri(300, 900), ri(2, 9)
    form = ri(0, 3)
    if form == 0:
        return f"{who} {verb} зі сталою швидкістю {v} км/год. Яку відстань подолано за {t} год? Відповідь дай у км.", v * t, []
    if form == 1:
        return f"{who} {verb} рівномірно й долає {v * t} км за {t} год. Знайди швидкість руху в км/год.", v, []
    if form == 2:
        return f"{who} {verb} зі сталою швидкістю {v} км/год. Весь шлях становить {v * t} км. Скільки годин триває рух?", t, []
    v2 = ri(4, 90)
    kind = pick("катери", "автомобілі", "велосипедисти", "поїзди")
    if R.random() < 0.5:
        return (f"Назустріч одне одному одночасно вирушили два {kind}. Швидкість першого {v} км/год, швидкість другого "
                f"{v2} км/год. Вони зустрілися через {t} год. Яка була початкова відстань між ними, у км?", (v + v2) * t, [v + v2])
    return (f"З двох пунктів, відстань між якими {(v + v2) * t} км, одночасно назустріч вирушили два {kind} зі швидкостями "
            f"{v} км/год і {v2} км/год. Через скільки годин вони зустрінуться?", t, [v + v2])


def g_add():
    a, b = ri(100, 99999), ri(100, 99999)
    if R.random() < 0.5:
        a, b = ri(100, 999), ri(100, 999)
    return f"Обчисли: {a} + {b}", a + b, []


def g_sub():
    a, b = ri(100, 99999), ri(10, 99999)
    if R.random() < 0.5:
        a, b = ri(100, 999), ri(10, 999)
    if a < b:
        a, b = b, a
    return f"Обчисли: {a} - {b}", a - b, []


def g_div():
    b = ri(2, 9) if R.random() < 0.8 else ri(11, 25)
    q = ri(12, 9999)
    if R.random() < 0.4:  # zeros inside the quotient
        q = int(str(ri(1, 9)) + "0" + str(ri(1, 99)).rjust(2, "0"))
    return f"Обчисли: {b * q} : {b}", q, []


def g_divrem():
    b, q = ri(2, 15), ri(2, 40)
    r = ri(0, b - 1)
    n = b * q + r
    if R.random() < 0.6:
        return f"Знайди остачу від ділення {n} на {b}.", r, [q]
    return f"Знайди неповну частку від ділення {n} на {b}.", q, [r]


def g_divtest():
    d = pick(2, 3, 5, 9, 10)
    xs = [ri(100, 999) for _ in range(8)]
    return f"Скільки з чисел {', '.join(map(str, xs))} діляться на {d}?", sum(x % d == 0 for x in xs), []


def g_gcd():
    g = ri(2, 20)
    while True:
        a, b = ri(1, 12), ri(1, 12)
        if math.gcd(a, b) == 1 and a != b:
            break
    return f"Знайди найбільший спільний дільник чисел {a * g} і {b * g}.", g, []


def g_lcm():
    a, b = ri(2, 72), ri(2, 72)
    if a == b:
        return g_lcm()
    return f"Знайди найменше спільне кратне чисел {a} і {b}.", math.lcm(a, b), []


def g_mul():
    a, b = ri(12, 999), ri(2, 99)
    return f"Обчисли: {a} × {b}", a * b, []


def g_order():
    a, b, c = ri(2, 15), ri(2, 12), ri(1, 150)
    form = ri(0, 3)
    if form == 0:
        return f"Обчисли: {c} + {a} × {b}", c + a * b, [a * b]
    if form == 1:
        c = a * b + ri(0, 120)
        return f"Обчисли: {c} - {a} × {b}", c - a * b, [a * b]
    if form == 2:
        return f"Обчисли: {a} × {b} + {c}", a * b + c, [a * b]
    return f"Обчисли: ({c} + {a}) × {b}", (c + a) * b, [c + a]


def g_power():
    a, k = ri(2, 25), pick(2, 2, 3)
    if k == 3:
        a = ri(2, 10)
    sup = {2: "²", 3: "³"}[k]
    c = ri(1, 60)
    form = ri(0, 2)
    if form == 0:
        return f"Обчисли: {a}{sup}", a ** k, []
    if form == 1:
        return f"Обчисли: {a}{sup} + {c}", a ** k + c, [a ** k]
    if a ** k <= c:
        return g_power()
    return f"Обчисли: {a}{sup} - {c}", a ** k - c, [a ** k]


def g_round():
    n = ri(100, 999999)
    name, p = R.choice([("десятків", 10), ("сотень", 100), ("тисяч", 1000)])
    if n < p:
        return g_round()
    return f"Округли число {n} до {name}.", (n + p // 2) // p * p, []


def g_pct():
    p = pick(10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90)
    whole = ri(2, 120) * 100 // math.gcd(100, p)
    part = whole * p // 100
    form = ri(0, 3)
    if form == 0:
        return f"Знайди {p}% від числа {whole}.", part, []
    if form == 1:
        return f"Знайди число, якщо його {p}% становлять {part}.", whole, []
    if form == 2:
        return f"Довжина маршруту {whole} км. Туристи вже пройшли {p}% маршруту. Скільки кілометрів вони пройшли?", part, []
    return f"Туристи пройшли {p}% маршруту, і це {part} км. Яка довжина всього маршруту в кілометрах?", whole, []


def _units(chain):
    """chain: (abbreviation, accusative plural of this unit, factor to the next smaller unit,
    locative plural of the next smaller unit for 'Запиши в ...')."""
    i = ri(0, len(chain) - 2)
    j = ri(i + 1, len(chain) - 1)
    f = math.prod(c[2] for c in chain[i:j])
    big, small = chain[i], chain[j]
    form = ri(0, 2)
    x = ri(2, 60)
    if form == 0:
        return f"Переведи {x} {big[0]} у {small[1]}.", x * f, []
    if form == 1:
        return f"Переведи {x * f} {small[0]} у {big[1]}.", x, []
    if j != i + 1:
        j = i + 1
        f = chain[i][2]
        small = chain[j]
    y = ri(1, f - 1)
    return f"Запиши в {big[3]}: {x} {big[0]} {y} {small[0]}.", x * f + y, []


def g_len():
    return _units([("км", "кілометри", 1000, "метрах"), ("м", "метри", 10, "дециметрах"),
                   ("дм", "дециметри", 10, "сантиметрах"), ("см", "сантиметри", 10, "міліметрах"),
                   ("мм", "міліметри", 1, "")])


def g_mass():
    return _units([("т", "тонни", 10, "центнерах"), ("ц", "центнери", 100, "кілограмах"),
                   ("кг", "кілограми", 1000, "грамах"), ("г", "грами", 1, "")])


def g_time():
    return _units([("доби", "доби", 24, "годинах"), ("год", "години", 60, "хвилинах"),
                   ("хв", "хвилини", 60, "секундах"), ("с", "секунди", 1, "")])


ITEMS = [("зошитів", "зошити"), ("олівців", "олівці"), ("яблук", "яблука"), ("книжок", "книжки"),
         ("стільців", "стільці"), ("цукерок", "цукерки")]


def g_word1():
    item = R.choice(ITEMS)[0]
    a, b = ri(5, 900), ri(2, 9)
    form = ri(0, 4)
    if form == 0:
        return f"На складі було {a} {item}, а потім привезли ще {b * 37} {item}. Скільки {item} стало на складі?", a + b * 37, []
    if form == 1:
        return f"Один пакет містить {b * 3} {item}. Придбали {a % 20 + 2} пакетів. Скільки {item} придбали?", b * 3 * (a % 20 + 2), []
    if form == 2:
        q = ri(5, 60)
        return f"Заплатили {q * b} грн за {b} однакових предметів. Скільки гривень коштує один предмет?", q, []
    if form == 3:
        c = ri(10, 900)
        x, y = max(a, c), min(a, c)
        if x == y:
            return g_word1()
        return f"Довжина першої стежки {x} м, а довжина другої стежки {y} м. На скільки метрів перша стежка довша?", x - y, []
    q = ri(5, 60)
    return f"У залі порівну розставили {q * b} стільців у {b} рядів. Скільки стільців у кожному ряду?", q, []


def g_word2():
    form = ri(0, 3)
    if form == 0:
        a, b, c = ri(300, 900), ri(50, 250), ri(20, 250)
        return (f"Довжина стрічки {a} см. Спершу відрізали {b} см, а потім ще {c} см. Скільки сантиметрів стрічки лишилося?",
                a - b - c, [a - b])
    if form == 1:
        a, b = ri(100, 600), ri(50, 300)
        c = ri(10, a + b - 1)
        return (f"На стоянці було {a} велосипедів. Приїхало ще {b} велосипедів, а {c} велосипедів поїхало. "
                f"Скільки велосипедів на стоянці тепер?", a + b - c, [a + b])
    if form == 2:
        n, k, e = ri(2, 15), ri(2, 12), ri(1, 40)
        return (f"На складі є {n} ящиків. У кожному ящику є {k} зошитів, а ще є {e} зошитів просто на полиці. "
                f"Скільки всього зошитів на складі?", n * k + e, [n * k])
    n, p, m, q = ri(2, 20), ri(5, 60), ri(2, 20), ri(2, 15)
    return (f"Купили {n} зошитів по {p} грн кожен і {m} олівців по {q} грн кожен. Скільки гривень заплатили за всю покупку?",
            n * p + m * q, [n * p])


GEN = {"AVG": g_avg, "D-ADD": g_dadd, "D-SUB": g_dsub, "D-MUL10": g_dmul10, "EQ-ADD-SUB": g_eq_addsub,
       "EQ-MUL": g_eq_mul, "F-MIXED": g_fmixed, "F-REDUCE": g_freduce, "G-AREA": g_area, "G-PERIM": g_perim,
       "G-VOL": g_vol, "MOTION": g_motion, "N-ADD-MULTI": g_add, "N-SUB-MULTI": g_sub, "N-DIV-MULTI": g_div,
       "N-DIVREM": g_divrem, "N-DIVTEST": g_divtest, "N-GCD": g_gcd, "N-LCM": g_lcm, "N-MUL-MULTI": g_mul,
       "N-ORDER": g_order, "N-POWER": g_power, "N-ROUND": g_round, "P-PCT": g_pct, "U-LEN": g_len,
       "U-MASS": g_mass, "U-TIME": g_time, "WORD-1": g_word1, "WORD-2": g_word2}


def generate(per_skill: int, seed: int = 0) -> list[dict]:
    R.seed(seed)
    tr, te = load("train"), load("test")
    titles = dict(zip(tr["skill"], tr["skill_title"]))
    seen = set(tr["statement"]) | set(te["statement"])  # never reproduce a real problem
    out = []
    for skill, gen in GEN.items():
        got = 0
        for _ in range(per_skill * 20):
            st, ans, steps = gen()
            if st in seen:
                continue
            seen.add(st)
            out.append({"id": f"syn{len(out):06d}", "skill": skill, "skill_title": titles[skill], "statement": st,
                        "answer": canonical(Fraction(ans), st), "steps": [str(s) for s in steps]})
            got += 1
            if got == per_skill:
                break
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-skill", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = generate(a.per_skill, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} problems to {a.out}")
