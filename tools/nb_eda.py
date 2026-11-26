"""notebooks/01_eda.ipynb: the data, the answer format, the validation split, synthetic problems."""
from nbgen import C, HEADER, M, write

write("01_eda.ipynb", [
    (M, "# 01 · EDA: Ukrainian grade-5 math problems\n\nTrain 1,526 problems with answers, hints and typical errors; "
        "test 3,561 problems. 29 skills. Metric: exact match of the canonical answer."),
    (C, HEADER),
    (C, '''tr, te = load("train"), load("test")
print(tr.shape, te.shape)
tr.head(3)[["skill", "statement", "answer", "steps"]]'''),
    (M, "## Skills\nThe test split is stratified by skill like train, so the skill mix is the same (~2.3× more rows per skill)."),
    (C, '''cnt = pd.DataFrame({"train": tr.skill.value_counts(), "test": te.skill.value_counts()}).fillna(0).astype(int)
cnt["test/train"] = (cnt.test / cnt.train).round(2)
titles = dict(zip(tr.skill, tr.skill_title))
cnt["title"] = cnt.index.map(titles)
cnt.sort_values("train", ascending=False)'''),
    (C, '''fig, ax = plt.subplots(figsize=(10, 4))
cnt.sort_values("test", ascending=False)[["train", "test"]].plot.bar(ax=ax, width=0.8)
ax.set_ylabel("problems"); ax.set_title("Problems per skill")
plt.tight_layout(); plt.savefig(IMG / "eda_skills.png", dpi=130); plt.show()'''),
    (M, "## Statements\nShort, template-like problems: most are a single expression (`Обчисли: 867 + 244`), the rest are one- or "
        "two-step word problems."),
    (C, '''for d, n in ((tr, "train"), (te, "test")):
    L = d.statement.str.len()
    print(f"{n}: chars median {L.median():.0f}, p90 {L.quantile(.9):.0f}, max {L.max()}; "
          f"with a decimal comma {d.statement.str.contains(r'\\d,\\d').mean():.1%}")
tr.groupby("skill").statement.apply(lambda s: s.str.len().median()).sort_values().tail(8)'''),
    (M, "## Answers and the canonical format\nIntegers dominate. A non-integer answer is a **decimal exactly when the statement "
        "contains a decimal number**, otherwise a reduced fraction (`207 : 10 → 207/10` but `1,012 : 1000 → 0.001012`). "
        "`hw4.answers.canonical` implements this rule; it reproduces every train answer."),
    (C, '''kind = tr.answer.map(lambda a: "fraction" if "/" in a else "decimal" if "." in a else "integer")
display(pd.crosstab(tr.skill, kind).query("decimal > 0 or fraction > 0"))
from fractions import Fraction
from hw4.answers import canonical, extract_answer
bad = [(r.statement, r.answer) for r in tr.itertuples() if canonical(extract_answer("Відповідь: " + r.answer), r.statement) != r.answer]
print(kind.value_counts().to_dict(), "| format rule mismatches:", len(bad))'''),
    (C, '''num = tr.answer.map(lambda a: float(Fraction(a)))
print(num.describe().round(2).to_dict())
print("answers with >= 4 digits:", f"{(num.abs() >= 1000).mean():.1%}")'''),
    (M, "## Steps, hints and typical errors\n`steps` are intermediate values of multi-step problems; `typical_errors` list wrong "
        "answers with the misconception behind each, used later for error analysis and as DPO rejected answers."),
    (C, '''tr["n_steps"] = tr.steps.map(len); tr["n_err"] = tr.typical_errors.map(len)
print(tr.n_steps.value_counts().sort_index().to_dict(), "| typical errors per problem:", tr.n_err.value_counts().sort_index().to_dict())
errs = pd.Series([e["id"] for es in tr.typical_errors for e in es]).value_counts()
errs.head(15)'''),
    (C, '''r = tr[tr.skill == "N-MUL-MULTI"].iloc[0]
print(r.statement, "=>", r.answer, "| steps", r.steps)
for h in r.hints: print("  hint:", h)
for e in r.typical_errors: print("  typical error:", e["value"], "-", e["id"])'''),
    (M, "## Validation split\nFixed for every comparison: 30% of train, stratified by skill, seed 42 (`hw4.data`). "
        "Few-shot examples and fine-tuning data come only from the other 70%."),
    (C, '''va, trs = load_split("val"), load_split("train")
print(len(trs), "train side,", len(va), "val")
print("min / max val rows per skill:", va.skill.value_counts().min(), va.skill.value_counts().max())'''),
    (M, "## Synthetic problems\nMy own generators (`hw4.synth`) for all 29 skills with exact answers. Used only as training data "
        "(the teacher's solutions are kept when they reproduce the generated answer). Statements identical to a train or test "
        "statement are dropped."),
    (C, '''syn = pd.read_json(RUNS / "data_synth" / "problems.jsonl", lines=True, dtype={"answer": str})
print(len(syn), "synthetic problems,", syn.skill.nunique(), "skills; overlap with train/test statements:",
      len(set(syn.statement) & (set(tr.statement) | set(te.statement))))
syn.groupby("skill").head(1).sample(8, random_state=0)[["skill", "statement", "answer"]]'''),
])
