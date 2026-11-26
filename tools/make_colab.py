"""Writes the Colab notebook kernels/colab_00_all_a100.ipynb and hw4-code.zip (private code for Drive).

python tools/make_colab.py      # then upload hw4-code.zip to MyDrive/hw4/ and open the notebook in Colab

The notebook is the GPU half of the project, organised as a readable walkthrough: every stage has a short
explanation, the key implementation (shown from the hw4 package, the single source of truth), the commands that
run it, and its results (tables / plots read back from MyDrive/hw4/runs). A stage whose results already exist on
Drive is skipped, so the notebook can be re-run after a disconnect. Merged weights stay on the runtime disk
(/content/weights) and are rebuilt from adapters with `python -m hw4.merge` when needed.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# versions tested together (vLLM 0.11 breaks on transformers 5 configs; TRL/PEFT tested on CPU with a tiny Gemma 3)
PINS = '"vllm==0.11.0" "transformers==4.57.1" "trl==0.24.0" "peft==0.17.1" datasets'
T8 = "--teacher Qwen/Qwen3-8B --max-tokens 768"
HF_REPO = "lizabugir/mamaylm-4b-ukr-school-math-pot"

M, C = "markdown", "code"

INTRO = """# HW4 · Ukrainian school math with small LLMs — training on one GPU

**UCU AI 2026, Homework 4 · Yelyzaveta Bugir**

This notebook is the GPU part of the project: it builds the training data with a teacher model, fine-tunes the
students, evaluates every model on the fixed validation split and publishes the best adapter. Prompting experiments
and EDA live in `notebooks/` of the repository; the code is the `hw4` package (`src/hw4/`).

| stage | what happens | output |
|---|---|---|
| 1 · Teacher data | Qwen3-8B writes Python solutions for the train side and for my synthetic problems; only solutions that reproduce the reference answer are kept | `data_qwen8b_*` |
| 2 · SFT MamayLM | LoRA SFT on the verified solutions (tool-use format) | `sft_mamay_pot` |
| 3 · Hugging Face | adapter + model card (data, setup, val accuracy vs the base model) | `lizabugir/…` |
| 4 · SFT Gemma | the same recipe on Gemma-3-4B-it | `sft_gemma_pot` |
| 5 · Ablations | without synthetic problems; CoT targets instead of code | `sft_mamay_pot_trainonly`, `sft_mamay_cot` |
| 6 · Self-consistency | majority vote over 8 samples of the fine-tuned model | `colab_sft_mamay_pot_sc8` |
| 7 · DPO | preference tuning on the model's own mistakes, typical misconceptions as rejected answers | `dpo_mamay_pot` |
| 8 · GRPO (bonus) | RL with a verifiable reward: executed code == answer | `grpo_mamay_pot` |

**Rules respected:** answers on val/test always come from Gemma-3-4B / MamayLM-4B; Qwen3 is only a teacher;
validation = 30 % of `train.csv` (458 problems, stratified by skill), never test; no proprietary models.

### How to run
1. **Runtime → Change runtime type → A100 GPU** (L4 also works, ~1.5× slower; GRPO may not fit).
2. **Secrets** (key icon, enable *notebook access*): `KAGGLE_API_TOKEN` — Kaggle token (competition data and
   Gemma from Kaggle Models); `HF_TOKEN` — Hugging Face write token (stage 3).
3. `MyDrive/hw4/hw4-code.zip` = current code (`python tools/make_colab.py` in the repository).
4. **Runtime → Run all.** ≈ 4 h on A100. Results → `MyDrive/hw4/runs/`, logs → `MyDrive/hw4/logs/`.
   Finished steps are skipped on a re-run; if a critical step fails the notebook stops and frees the GPU.
"""

SETUP_MD = "## 0 · Setup\nDrive, GPU check, private code, libraries (pinned versions tested together), data and models."

SETUP = """import os, glob, json, shutil, subprocess, time, re
import torch
from google.colab import drive, userdata
drive.mount('/content/drive', force_remount=True)
D = '/content/drive/MyDrive/hw4'
RUNS, LOGS, W = f'{D}/runs', f'{D}/logs', '/content/weights'
for p in (RUNS, LOGS, W):
    os.makedirs(p, exist_ok=True)
assert torch.cuda.is_available(), 'no GPU: Runtime -> Change runtime type'
major = torch.cuda.get_device_capability(0)[0]
print(f'{torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB | '
      f'bf16: {major >= 8}')
assert major >= 8, 'this notebook needs bf16 (A100 or L4); a T4 cannot train Gemma 3 here'"""

INSTALL = """# the code is a private zip on Drive (sharing code with other participants is forbidden before the deadline)
shutil.rmtree('/content/hw4code', ignore_errors=True)
shutil.unpack_archive(f'{D}/hw4-code.zip', '/content/hw4code')
subprocess.run('pip install -q --no-deps /content/hw4code && pip install -q "kaggle>=1.7" kagglehub __PINS__',
               shell=True, check=True)
import importlib.metadata as md
print({p: md.version(p) for p in ['hw4', 'vllm', 'torch', 'transformers', 'trl', 'peft']})"""

DATA = """# Kaggle token -> competition data (cached on Drive) and Gemma-3-4B-it from Kaggle Models
COMP = 'ukrainian-school-math-slm-hw4-ucu-ai-2026'
DATA = '/content/data'
tok = userdata.get('KAGGLE_API_TOKEN').strip()
os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
if tok.startswith('{'):   # old-style kaggle.json
    k = json.loads(tok); p = os.path.expanduser('~/.kaggle/kaggle.json'); open(p, 'w').write(tok)
    os.environ['KAGGLE_USERNAME'], os.environ['KAGGLE_KEY'] = k['username'], k['key']
else:                     # new-style access token
    p = os.path.expanduser('~/.kaggle/access_token'); open(p, 'w').write(tok)
    os.environ['KAGGLE_API_TOKEN'] = tok
os.chmod(p, 0o600)
cache = f'{D}/data/{COMP}.zip'
if not os.path.exists(f'{DATA}/train.csv'):
    if not os.path.exists(cache):
        os.makedirs(f'{D}/data', exist_ok=True)
        subprocess.run(f'kaggle competitions download -c {COMP} -p {D}/data', shell=True, check=True)
    shutil.unpack_archive(cache, DATA)
import kagglehub
os.environ['HW4_MODEL_GEMMA'] = kagglehub.model_download('google/gemma-3/transformers/gemma-3-4b-it')
os.environ.update(HW4_DATA=DATA, HW4_RUNS=RUNS, HW4_WEIGHTS=W)
try:
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
except Exception:
    print('no HF_TOKEN secret: stage 3 (publishing) will be skipped')
print(sorted(os.listdir(DATA)))"""

HELPERS = """# helpers: run a stage, show results, show the implementation
import inspect
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Code, Markdown, display
pd.set_option('display.width', 160); pd.set_option('display.max_colwidth', 80)
AUTO_STOP = True                       # free the GPU when the notebook ends or a critical step fails
env = dict(os.environ, TOKENIZERS_PARALLELISM='false', PYTHONWARNINGS='ignore')
SHOW = re.compile(r"val acc|kept|train_loss|loss [0-9]|wrote|merged|published|examples|pairs|RL problems|Error|Traceback|error:")

def _done(cmd):
    m = re.search(r'--name (\\S+)', cmd)
    if m: return os.path.exists(f'{RUNS}/{m.group(1)}/metrics.json')
    m = re.search(r'hw4.merge --run (\\S+)', cmd)
    if m: return os.path.exists(f'{W}/{m.group(1)}/config.json')
    m = re.search(r'--out (\\S+)', cmd)
    return bool(m) and os.path.exists(m.group(1))

def run(cmd):
    cmd = cmd.replace('{R}', RUNS).replace('{W}', W)
    if _done(cmd):
        print('✓ already done:', cmd[:100]); return 0
    m = re.search(r'--(?:name|run) (\\S+)', cmd) or re.search(r'-m (\\S+)', cmd)
    tag = m.group(1) if m else 'step'
    print(f'▶ {tag}', flush=True); t0 = time.time()
    with open(f'{LOGS}/{tag}.log', 'w') as f:
        p = subprocess.Popen(cmd, shell=True, cwd='/content', env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout:
            f.write(line); f.flush()
            if SHOW.search(line) and 'it/s' not in line:
                print('   ' + line[:240].rstrip(), flush=True)
    code = p.wait()
    print(f'  {"✓" if code == 0 else "✗"} {tag}: exit {code}, {(time.time() - t0) / 60:.1f} min', flush=True)
    return code

def stage(*cmds):
    \"\"\"Runs commands in order. A command starting with '!' is critical: if it fails, stop (and free the GPU).\"\"\"
    for c in cmds:
        critical = c.startswith('!')
        if run(c.lstrip('!')) != 0:
            if critical:
                if AUTO_STOP:
                    from google.colab import runtime; runtime.unassign()
                raise RuntimeError(f'critical step failed: {c[:80]} — see MyDrive/hw4/logs')
            print('  (non-critical step failed; continuing)')

def metrics(name):
    p = f'{RUNS}/{name}/metrics.json'
    return json.load(open(p)) if os.path.exists(p) else None

def acc_table(names):
    rows = []
    for n in names:
        m = metrics(n)
        if m and 'val_acc' in m:
            cfg = json.load(open(f'{RUNS}/{n}/config.json'))
            rows.append({'run': n, 'model': os.path.basename(str(cfg['model'])), 'prompt': cfg['strategy'],
                         'samples': cfg.get('n', 1), 'val acc': round(m['val_acc'], 4), 'no answer': m.get('val_none')})
    return pd.DataFrame(rows)

def per_skill(names):
    cols = {}
    for n in names:
        p = f'{RUNS}/{n}/val_preds.parquet'
        if os.path.exists(p):
            d = pd.read_parquet(p); cols[n] = d.groupby('skill').correct.mean()
    t = pd.DataFrame(cols).round(2)
    return t.loc[t.min(axis=1).sort_values().index] if len(t) else t

def loss_curve(*names):
    fig, ax = plt.subplots(figsize=(6, 3))
    for n in names:
        m = metrics(n)
        if m and 'log' in m:
            pts = [(e['step'], e['loss']) for e in m['log'] if 'loss' in e and 'step' in e]
            if pts: ax.plot(*zip(*pts), label=n)
    ax.set_xlabel('step'); ax.set_ylabel('training loss'); ax.legend(); plt.tight_layout(); plt.show()

def show(obj):
    \"\"\"Displays the implementation of a function/object from the hw4 package.\"\"\"
    display(Code(inspect.getsource(obj), language='python'))"""

# ---------------------------------------------------------------- stage cells
CELLS = [
    (M, INTRO),
    (M, SETUP_MD), (C, SETUP), (C, INSTALL.replace("__PINS__", PINS)), (C, DATA), (C, HELPERS),

    (M, """## How an answer is produced and checked
The students answer with **tool use**: the model writes a short Python program that stores the result in `answer`;
we execute it in an isolated interpreter (5 s timeout) and normalise the number to the competition format. The model
does all the reasoning; our code only runs it and writes the number canonically (integer, decimal with a dot, or a
reduced fraction — a decimal exactly when the statement itself contains a decimal number, a rule that reproduces all
1,526 train answers). If the program fails, the same model answers once more with chain-of-thought (fallback)."""),
    (C, """from hw4 import prompts, sandbox, answers
print(prompts.SYSTEM, '\\n'); print(prompts.TEMPLATES['pot'])
show(answers.canonical)"""),
    (C, """# a quick check of the executor and the formatter
for code, st in [("answer = Fraction('26.38') - Fraction('23.3')", 'Обчисли: 26,38 - 23,3'),
                 ('answer = 207 / 10', 'Обчисли: 207 : 10'), ('answer = "12 1/2"', ''), ('answer = 1 / 0', '')]:
    v, err = sandbox.run_code(code)
    print(f'{code:45s} -> {answers.canonical(v, st) if v is not None else err}')"""),

    (M, """## 1 · Teacher data (Qwen3-8B, verified)
**Why.** The students make arithmetic slips; a program removes them, but the students have to learn to write short,
correct programs for all 29 skills. A stronger open model (Qwen3-8B, thinking mode off) writes them; it is used
**only as a teacher**.

**Problems.** (a) the 1,068 problems of the train side of the split; (b) new problems from my own generators
(`hw4.synth`: random numbers in templates for each skill, exact answers computed by the generator; statements equal to
any train/test statement are dropped). **Verification:** every teacher program is executed and kept only if it
reproduces the reference answer — wrong solutions never reach the training data."""),
    (C, """from hw4 import synth, teacher
show(synth.g_pct)            # one of the 29 generators
show(teacher.clean_pot)      # how a kept solution is normalised"""),
    (C, f"""stage(
    '!python -m hw4.synth --per-skill 300 --out {{R}}/data_synth/problems.jsonl',
    '!python -m hw4.teacher {T8} --source train --style pot --n 4 --name data_qwen8b_pot_train',
    '!python -m hw4.teacher {T8} --source {{R}}/data_synth/problems.jsonl --style pot --n 2 --per-skill 150 --name data_qwen8b_pot_synth',
    'python -m hw4.teacher {T8} --source train --style cot --n 2 --name data_qwen8b_cot_train',
)"""),
    (C, """t = {n: metrics(n) for n in ['data_qwen8b_pot_train', 'data_qwen8b_pot_synth', 'data_qwen8b_cot_train']}
display(pd.DataFrame({n: {k: m[k] for k in ('problems', 'kept', 'solved', 'teacher_pass_rate', 'minutes')}
                      for n, m in t.items() if m}).T)
skills = pd.DataFrame({n: m['per_skill_solved'] for n, m in t.items() if m})
display(skills.sort_values(skills.columns[0]).head(10).style.format('{:.2f}').background_gradient(cmap='RdYlGn', vmin=0.5, vmax=1))
ex = [json.loads(l) for l in open(f'{RUNS}/data_qwen8b_pot_train/sft.jsonl')][:1]
for e in ex:
    print(e['prompt'][-1]['content'].split('Задача: ')[1].split('\\n')[0], '=>', e['answer'])
    print(e['completion'][0]['content'])"""),

    (M, """## 2 · LoRA SFT of MamayLM on the verified solutions
**Setup.** LoRA (r = 16, α = 32, dropout 0.05) on all attention and MLP projections of the **language model only**
(the vision tower of Gemma 3 is never used), bf16, 2 epochs, lr 1e-4 with cosine decay, effective batch 16, loss
only on the answer tokens. The prompt is exactly the one used at inference (`pot0`), so train and test match.

**Evaluation.** The base model and the fine-tuned model are evaluated with the same prompt on the same val split in
bf16 with vLLM; the merged model is also exported to a 4-bit GGUF so it can be compared on the laptop under the same
quantisation as the prompting experiments."""),
    (C, """from hw4 import train_utils, sft
print(train_utils.LORA_TARGETS)
show(train_utils.lora_config)
show(sft.read_jsonl)"""),
    (C, """stage(
    '!python -m hw4.sft --model mamay --data {R}/data_qwen8b_pot_train/sft.jsonl {R}/data_qwen8b_pot_synth/sft.jsonl --name sft_mamay_pot',
    'python -m hw4.infer --model mamay --strategy pot0 --fallback cot --splits val test --name colab_mamay_pot0',
    '!python -m hw4.infer --model {W}/sft_mamay_pot --strategy pot0 --fallback cot --splits val test --name colab_sft_mamay_pot',
    'python -m hw4.export_gguf --merged {W}/sft_mamay_pot --out {R}/gguf/sft_mamay_pot-Q4_K_M.gguf',
)"""),
    (C, """loss_curve('sft_mamay_pot')
display(acc_table(['colab_mamay_pot0', 'colab_sft_mamay_pot']))
per_skill(['colab_mamay_pot0', 'colab_sft_mamay_pot']).style.background_gradient(cmap='RdYlGn', vmin=0.5, vmax=1)"""),

    (M, """## 3 · Publish the adapter on Hugging Face
The model card (written by `hw4.hub`) states the base model, how the training data was made (teacher, verification,
synthetic generators), the training setup and the val accuracy of the base model vs this adapter."""),
    (C, f"""if os.environ.get('HF_TOKEN'):
    stage('python -m hw4.hub --run sft_mamay_pot --eval-run colab_sft_mamay_pot --base-eval-run colab_mamay_pot0 '
          '--repo {HF_REPO}')
    display(Markdown('Published: https://huggingface.co/{HF_REPO}'))
else:
    print('HF_TOKEN not set — skipped')"""),

    (M, """## 4 · The same recipe on Gemma-3-4B-it
Same data, same hyper-parameters — shows whether the Ukrainian continued pre-training of MamayLM still matters after
fine-tuning."""),
    (C, """stage(
    'python -m hw4.sft --model gemma --data {R}/data_qwen8b_pot_train/sft.jsonl {R}/data_qwen8b_pot_synth/sft.jsonl --name sft_gemma_pot',
    'python -m hw4.infer --model gemma --strategy pot0 --fallback cot --splits val test --name colab_gemma_pot0',
    'python -m hw4.infer --model {W}/sft_gemma_pot --strategy pot0 --fallback cot --splits val test --name colab_sft_gemma_pot',
)
display(acc_table(['colab_gemma_pot0', 'colab_sft_gemma_pot', 'colab_mamay_pot0', 'colab_sft_mamay_pot']))"""),

    (M, """## 5 · Ablations
* **Synthetic data:** the same SFT with the train-side solutions only — the measured effect of my synthetic problems.
* **CoT targets:** SFT on verified chain-of-thought solutions (no tool) instead of programs — is the tool the gain,
  or the fine-tuning itself?"""),
    (C, """stage(
    'python -m hw4.sft --model mamay --data {R}/data_qwen8b_pot_train/sft.jsonl --name sft_mamay_pot_trainonly',
    'python -m hw4.infer --model {W}/sft_mamay_pot_trainonly --strategy pot0 --fallback cot --name colab_sft_mamay_pot_trainonly',
    'python -m hw4.sft --model mamay --data {R}/data_qwen8b_cot_train/sft.jsonl --name sft_mamay_cot',
    'python -m hw4.infer --model mamay --strategy cot --name colab_mamay_cot',
    'python -m hw4.infer --model {W}/sft_mamay_cot --strategy cot --name colab_sft_mamay_cot',
)
display(acc_table(['colab_mamay_cot', 'colab_sft_mamay_cot', 'colab_mamay_pot0',
                   'colab_sft_mamay_pot_trainonly', 'colab_sft_mamay_pot']))"""),

    (M, """## 6 · Self-consistency on the fine-tuned model
8 samples at temperature 0.7, majority vote over the normalised answers (programs that crash do not vote)."""),
    (C, """from hw4 import infer
show(infer.vote)
stage('python -m hw4.infer --model {W}/sft_mamay_pot --strategy pot0 --n 8 --temperature 0.7 --fallback cot '
      '--splits val test --name colab_sft_mamay_pot_sc8')
display(acc_table(['colab_sft_mamay_pot', 'colab_sft_mamay_pot_sc8']))"""),

    (M, """## 7 · DPO on the model's own mistakes
The SFT model samples 8 programs per train-side problem. A pair = (a correct sample — or the verified teacher solution —,
a wrong sample). Wrong samples whose value equals one of the problem's **`typical_errors`** (a known pupil
misconception) are used first, as the assignment suggests. LoRA DPO (β = 0.1) on top of the SFT model."""),
    (C, """from hw4 import dpo
show(dpo.make_pairs)"""),
    (C, """stage(
    'python -m hw4.infer --model {W}/sft_mamay_pot --strategy pot0 --splits train --n 8 --temperature 0.8 --save-all --name samples_sft_mamay_train',
    'python -m hw4.dpo --model {W}/sft_mamay_pot --samples samples_sft_mamay_train --teacher-data {R}/data_qwen8b_pot_train/sft.jsonl --name dpo_mamay_pot',
    'python -m hw4.infer --model {W}/dpo_mamay_pot --strategy pot0 --fallback cot --splits val test --name colab_dpo_mamay_pot',
    'python -m hw4.export_gguf --merged {W}/dpo_mamay_pot --out {R}/gguf/dpo_mamay_pot-Q4_K_M.gguf',
)
p = f'{RUNS}/dpo_mamay_pot/pairs.csv'
if os.path.exists(p):
    pr = pd.read_csv(p); print(len(pr), 'pairs;', int(pr.rejected_typical.sum()), 'rejected answers are typical misconceptions')
display(acc_table(['colab_sft_mamay_pot', 'colab_dpo_mamay_pot']))"""),

    (M, """## 8 · Bonus: GRPO with a verifiable reward
Reinforcement learning on the SFT model: 8 programs per problem, reward 1 if executing the program gives the reference
answer (+0.1 for a well-formed code block), group-relative advantages, no KL term. Problems: train side + synthetic."""),
    (C, """from hw4 import grpo
show(grpo.reward_correct)"""),
    (C, """stage(
    'python -m hw4.grpo --model {W}/sft_mamay_pot --synth {R}/data_synth/problems.jsonl --max-steps 200 --name grpo_mamay_pot',
    'python -m hw4.infer --model {W}/grpo_mamay_pot --strategy pot0 --fallback cot --splits val test --name colab_grpo_mamay_pot',
    'python -m hw4.export_gguf --merged {W}/grpo_mamay_pot --out {R}/gguf/grpo_mamay_pot-Q4_K_M.gguf',
)
m = metrics('grpo_mamay_pot')
if m:
    r = [(e['step'], e['reward']) for e in m['log'] if 'reward' in e]
    if r:
        plt.figure(figsize=(6, 3)); plt.plot(*zip(*r)); plt.xlabel('step'); plt.ylabel('mean reward'); plt.show()
display(acc_table(['colab_sft_mamay_pot', 'colab_grpo_mamay_pot']))"""),

    (M, """## Summary
All val accuracies (bf16, vLLM, 458 problems; ±2 pp is within noise). Test predictions of the runs with `test`
in their splits are in `MyDrive/hw4/runs/<run>/test_preds.parquet`; 4-bit GGUFs for the laptop in `runs/gguf/`."""),
    (C, """allr = acc_table(sorted(os.path.basename(os.path.dirname(m)) for m in glob.glob(f'{RUNS}/colab_*/metrics.json')))
if len(allr):
    display(allr.sort_values('val acc', ascending=False).reset_index(drop=True))
    allr.sort_values('val acc').plot.barh(x='run', y='val acc', figsize=(7, 4), legend=False, xlim=(0.5, 1))
    plt.tight_layout(); plt.show()"""),
    (C, """if AUTO_STOP:
    from google.colab import runtime
    runtime.unassign()"""),
]


def cell(kind: str, src: str) -> dict:
    c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
    if kind == C:
        c.update(execution_count=None, outputs=[])
    return c


def main():
    for old in (ROOT / "kernels").glob("colab_0*.ipynb"):
        old.unlink()
    nb = {"cells": [cell(k, s) for k, s in CELLS],
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100", "toc_visible": True},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 0}
    (ROOT / "kernels" / "colab_00_all_a100.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    if not (ROOT / "src/hw4/configs/split.csv").exists():
        raise SystemExit("run `python -m hw4.data` first")
    with tempfile.TemporaryDirectory() as d:
        shutil.copytree(ROOT / "src", Path(d) / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        shutil.copy(ROOT / "pyproject.toml", d)
        zp = shutil.make_archive(str(ROOT / "hw4-code"), "zip", d)
    print("wrote kernels/colab_00_all_a100.ipynb and", zp)


if __name__ == "__main__":
    main()
