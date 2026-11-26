"""Prompting strategies. Each builds chat messages for one problem.

zeroshot   - answer only, no reasoning
cot        - Ukrainian chain of thought ending with 'Відповідь: <число>' (the organisers' reference setup)
fewshot    - cot + k solved examples of the same skill from the TRAIN side of the split
plan       - plan-then-solve: write the plan first, then compute step by step
pot        - tool use: the model writes Python that stores the result in `answer`; we execute it
"""
from __future__ import annotations

import zlib

import pandas as pd

FORMAT = ("Формат відповіді: ціле число без пробілів (1220), десятковий дріб з крапкою (17.7) "
          "або звичайний дріб a/b (9/10). Без одиниць вимірювання.")

SYSTEM = "Ти уважний вчитель математики для 5 класу. Розв'язуй задачі точно, перевіряй обчислення."

TEMPLATES = {
    "zeroshot": "Тема: {skill_title}\nЗадача: {statement}\n\nДай лише відповідь у форматі «Відповідь: <число>». " + FORMAT,
    "cot": ("Тема: {skill_title}\nЗадача: {statement}\n\nРозв'яжи задачу покроково, записуючи кожне обчислення. "
            "Багатоцифрові числа додавай, віднімай і множ стовпчиком по розрядах. "
            "В останньому рядку напиши «Відповідь: <число>». " + FORMAT),
    "plan": ("Тема: {skill_title}\nЗадача: {statement}\n\nСпочатку коротко склади план розв'язання (що відомо, "
             "що треба знайти, які дії виконати). Потім виконай план крок за кроком, перевір результат "
             "і в останньому рядку напиши «Відповідь: <число>». " + FORMAT),
    "pot": ("Тема: {skill_title}\nЗадача: {statement}\n\nНапиши коротку програму на Python, яка розв'язує задачу. "
            "Не обчислюй нічого в голові: усі дії виконує програма. Для дробів і десяткових чисел використовуй "
            "fractions.Fraction (наприклад, Fraction('26.38')). Результат збережи у змінній answer. "
            "Відповідь дай одним блоком ```python ... ```."),
}

# hand-written demonstrations of the *format* of a code answer (generic problems, not from the data)
POT_SHOTS = [
    ("Обчисли: 12,5 - 3,75", "```python\nfrom fractions import Fraction\nanswer = Fraction('12.5') - Fraction('3.75')\n```"),
    ("У магазині було 540 кг яблук. Продали 3/5 усіх яблук. Скільки кілограмів яблук лишилося?",
     "```python\nfrom fractions import Fraction\ntotal = 540\nsold = total * Fraction(3, 5)\nanswer = total - sold\n```"),
    ("Знайди найменше спільне кратне чисел 12 і 18.", "```python\nimport math\nanswer = math.lcm(12, 18)\n```"),
]


def _fewshot_turns(row, pool: pd.DataFrame, k: int, seed: int) -> list[dict]:
    same = pool[(pool["skill"] == row["skill"]) & (pool["id"] != row["id"])]
    if len(same) == 0:
        return []
    ex = same.sample(min(k, len(same)), random_state=seed + zlib.crc32(row["id"].encode()) % 1000)
    turns = []
    for e in ex.itertuples():
        steps = list(e.steps) if isinstance(e.steps, list) else []
        sol = ("Проміжні результати: " + ", ".join(steps) + ".\n") if steps else ""
        turns += [{"role": "user", "content": TEMPLATES["cot"].format(skill_title=e.skill_title, statement=e.statement)},
                  {"role": "assistant", "content": f"{sol}Відповідь: {e.answer}"}]
    return turns


def build(strategy: str, row, pool: pd.DataFrame | None = None, k: int = 3, seed: int = 0) -> list[dict]:
    """Chat messages for one problem. Gemma has no system role: the system text goes into the first user turn."""
    base = "plan" if strategy == "plan" else ("pot" if strategy.startswith("pot") else
                                              "zeroshot" if strategy == "zeroshot" else "cot")
    user = TEMPLATES[base].format(skill_title=row["skill_title"], statement=row["statement"])
    turns: list[dict] = []
    if strategy == "fewshot":
        assert pool is not None, "fewshot needs the train-side pool"
        turns = _fewshot_turns(row, pool, k, seed)
    elif strategy == "pot":
        for q, code in POT_SHOTS:
            turns += [{"role": "user", "content": TEMPLATES["pot"].format(skill_title="", statement=q)},
                      {"role": "assistant", "content": code}]
    msgs = turns + [{"role": "user", "content": user}]
    msgs[0] = {"role": "user", "content": SYSTEM + "\n\n" + msgs[0]["content"]}
    return msgs


STRATEGIES = ["zeroshot", "cot", "fewshot", "plan", "pot", "pot0"]
