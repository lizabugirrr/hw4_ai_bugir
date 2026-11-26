"""notebooks/02_prompting.ipynb: both models x prompting strategies on the fixed val split, per-skill accuracy,
error analysis against typical_errors. Reads runs/<name>/ produced by hw4.infer (Kaggle/Colab/local)."""
from nbgen import C, HEADER, M, write

write("02_prompting.ipynb", [
    (M, "# 02 · Prompting strategies: Gemma-3-4B vs MamayLM-4B\n\nAll numbers are accuracy on the same fixed validation "
        "split (458 problems from train.csv). Strategies: zero-shot, chain-of-thought (the organisers' reference setup), "
        "few-shot (3 problems of the same skill from the train side), plan-then-solve, tool use (the model writes Python "
        "which we execute; `pot` with 3 generic demonstrations, `pot0` without), and self-consistency (majority of 8 samples)."),
    (C, HEADER + '''
from hw4.evaluate import typical_error_share
from hw4.runs import load_preds
val = load_split("val")'''),
    (C, '''rows = []
for m in sorted(RUNS.glob("*/metrics.json")):
    met, cfg = json.loads(m.read_text()), json.loads((m.parent / "config.json").read_text())
    if "val_acc" not in met or "strategy" not in cfg:
        continue
    rows.append({"run": m.parent.name, "model": cfg["model"], "strategy": cfg["strategy"], "n": cfg.get("n", 1),
                 "backend": cfg.get("backend", "vllm"), "val_acc": met["val_acc"], "no_answer": met.get("val_none"),
                 "minutes": met.get("val_min")})
res = pd.DataFrame(rows)
base = res[~res.model.str.contains("weights|/", regex=True)]
base.sort_values("val_acc", ascending=False)'''),
    (C, '''tab = base.assign(strategy=base.strategy + np.where(base.n > 1, "_sc" + base.n.astype(str), "")) \\
          .pivot_table(index="strategy", columns="model", values="val_acc").round(3)
tab = tab.loc[tab.max(axis=1).sort_values().index]
ax = tab.plot.barh(figsize=(7, 4), width=0.8); ax.set_xlabel("val accuracy"); ax.set_xlim(0.4, 1)
ax.axvline(0.975, ls="--", c="gray", lw=1); ax.text(0.976, -0.4, "Qwen3-4B ref.", fontsize=8, color="gray")
plt.tight_layout(); plt.savefig(IMG / "prompting_strategies.png", dpi=130); plt.show()
tab'''),
    (M, "## Accuracy per skill\nWhere each model loses points with CoT, and how much tool use fixes."),
    (C, '''def per_skill(run):
    p = load_preds(run, "val")
    return p.groupby("skill").correct.mean()

show = [r for r in ["gemma_cot", "mamay_cot", "gemma_pot", "mamay_pot"] if (RUNS / r / "val_preds.parquet").exists()]
ps = pd.DataFrame({r: per_skill(r) for r in show})
ps = ps.loc[ps.min(axis=1).sort_values().index]
fig, ax = plt.subplots(figsize=(6, 9))
sns.heatmap(ps, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.3, vmax=1, cbar=False, ax=ax)
ax.set_title("val accuracy per skill"); plt.tight_layout(); plt.savefig(IMG / "per_skill.png", dpi=130); plt.show()'''),
    (M, "## Typical errors\nFor each wrong prediction: does the wrong value equal one of the problem's `typical_errors` "
        "(a known pupil misconception)? Otherwise it is an arithmetic slip, a format problem, or no answer."),
    (C, '''def err_table(run):
    p = load_preds(run, "val").merge(val[["id", "statement"]], on="id")
    e = typical_error_share(p, val)
    return e

for r in show:
    e = err_table(r)
    print(f"{r}: {len(e)} errors ->", e.typical_error.value_counts().head(6).to_dict())'''),
    (C, '''best = max(show, key=lambda r: load_preds(r, "val").correct.mean()) if show else None
e = err_table(show[0]).merge(val[["id", "statement"]], on="id")
e.sample(min(12, len(e)), random_state=0)[["skill", "statement", "answer", "pred", "typical_error"]]'''),
    (M, "## Tool use: how often the code runs\n`source` tells whether the answer came from executed code or from the CoT "
        "fallback (code missing or crashed)."),
    (C, '''for r in [x for x in show if "pot" in x]:
    p = load_preds(r, "val")
    s = p.groupby("source").correct.agg(["size", "mean"]).round(3)
    print(r); print(s.to_string()); print()'''),
    (C, '''# a typical tool-use answer
p = load_preds(show[-1], "val") if show else None
if p is not None:
    print(p.raw.iloc[0][:800])'''),
])
