"""Writes the Colab runner notebooks kernels/colab_0*.ipynb and hw4-code.zip (private code for Drive).

python tools/make_colab.py      # then upload hw4-code.zip to MyDrive/hw4/ and open a notebook in Colab

Every notebook: mount Drive, install the private code + libraries, fetch data and Gemma, run a QUEUE of
commands. A command whose result already exists on Drive is skipped, so a notebook can be re-run after a
disconnect. Runs (metrics, predictions, adapters) go to MyDrive/hw4/runs; merged weights stay on the
runtime disk (/content/weights) and are rebuilt from adapters with `python -m hw4.merge` when needed.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HEADER = """# HW4 · {title}

{about}

**Before the first run**
1. Runtime → Change runtime type → **{gpu}**.
2. Secrets (key icon): `KAGGLE_API_TOKEN` (Kaggle token; Gemma comes from Kaggle Models, accept its licence on Kaggle once){secrets}.
3. `MyDrive/hw4/hw4-code.zip` must be the current code (`python tools/make_colab.py` on the laptop, then upload).

**Runtime → Run all.** Finished steps are skipped on re-run. Results: `MyDrive/hw4/runs/<name>/`, logs: `MyDrive/hw4/logs/`.
"""

SETUP = """import os, glob, json, shutil, subprocess, time, re
import torch
from google.colab import drive, userdata
drive.mount('/content/drive', force_remount=True)
D = '/content/drive/MyDrive/hw4'
RUNS, LOGS, W = f'{D}/runs', f'{D}/logs', '/content/weights'
for p in (RUNS, LOGS, W): os.makedirs(p, exist_ok=True)
assert torch.cuda.is_available(), 'no GPU: Runtime -> Change runtime type'
print(torch.cuda.get_device_name(0), '| bf16:', torch.cuda.is_bf16_supported(),
      f'| {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB')"""

INSTALL = """# private code from Drive (the repository is public only after the deadline; sharing code is forbidden)
shutil.rmtree('/content/hw4code', ignore_errors=True)
shutil.unpack_archive(f'{{D}}/hw4-code.zip', '/content/hw4code')
subprocess.run('pip install -q --no-deps /content/hw4code && pip install -q "kaggle>=1.7" kagglehub {pip}', shell=True, check=True)
import importlib.metadata as md
print({{p: md.version(p) for p in ['hw4', 'vllm', 'torch', 'transformers'{extra_versions}]}})"""

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
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')   # only needed for publishing adapters
except Exception:
    pass
print(sorted(os.listdir(DATA)), os.environ['HW4_MODEL_GEMMA'])"""

RUNNER = """# queue runner: {R} = runs dir, {W} = merged weights dir.
# Skipped when done: commands with --name (runs/<name>/metrics.json exists), hw4.merge (weights exist), --out files.
AUTO_STOP = True   # disconnect the runtime when the queue is done (saves compute units)
env = dict(os.environ, TOKENIZERS_PARALLELISM='false', PYTHONWARNINGS='ignore')

def done(cmd):
    m = re.search(r'--name (\\S+)', cmd)
    if m: return os.path.exists(f'{RUNS}/{m.group(1)}/metrics.json')
    m = re.search(r'hw4.merge --run (\\S+)', cmd)
    if m: return os.path.exists(f'{W}/{m.group(1)}/config.json')
    m = re.search(r'--out (\\S+)', cmd)
    return bool(m) and os.path.exists(m.group(1))

def run(cmd):
    cmd = cmd.replace('{R}', RUNS).replace('{W}', W)
    if done(cmd):
        print('skip (done):', cmd[:110]); return 0
    tag = (re.search(r'--(?:name|run) (\\S+)', cmd) or re.search(r'-m (\\S+)', cmd)).group(1)
    print(f'\\n=== {tag} ===  {cmd[:160]}', flush=True); t0 = time.time()
    with open(f'{LOGS}/{tag}.log', 'w') as f:
        p = subprocess.Popen(cmd, shell=True, cwd='/content', env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout:
            f.write(line); f.flush()
            if re.search(r"val acc|kept|loss|'reward'|wrote|merged|published|examples|pairs|problems|Error|Traceback|error:", line) \
                    and 'it/s' not in line:
                print(line[:300], end='', flush=True)
    code = p.wait()
    print(f'{tag}: exit {code} after {(time.time() - t0) / 60:.1f} min', flush=True)
    return code

def summary():
    rows = []
    for m in sorted(glob.glob(f'{RUNS}/*/metrics.json')):
        r = json.load(open(m)); n = m.split('/')[-2]
        if 'val_acc' in r: rows.append(f"{n:34s} val_acc={r['val_acc']:.4f}  none={r.get('val_none')}")
    print('\\n'.join(rows))"""

GO = """# a command starting with '!' is critical: if it fails, the queue stops (no compute units burnt on dependent steps)
for cmd in QUEUE:
    critical = cmd.startswith('!')
    if run(cmd.lstrip('!')) != 0:
        if critical:
            print('!! critical step failed, stopping the queue; see the log on Drive'); break
        print('!! failed, see the log; continuing with the next step')
summary()"""

STOP = """if AUTO_STOP:
    from google.colab import runtime
    runtime.unassign()"""

# versions tested together (vLLM 0.11 breaks on transformers 5 configs; TRL/PEFT tested on CPU with a tiny Gemma 3)
VLLM_PIN = '"vllm==0.11.0" "transformers==4.57.1"'
PINS = VLLM_PIN + ' "trl==0.24.0" "peft==0.17.1" datasets'
T8 = "--teacher Qwen/Qwen3-8B --max-tokens 768"
ALL_A100 = dict(
    title="00 · Everything on one A100: teacher data, SFT, DPO, GRPO, evaluation, Hugging Face",
    about=("One runtime for the whole training part (saves the ~6 min install per notebook). Steps: Qwen3-8B (bf16) writes "
           "verified solutions for the train side and for my synthetic problems; LoRA SFT of MamayLM and Gemma on them; "
           "ablations (no synthetic data; CoT targets); DPO on the SFT model's own mistakes; GRPO with an executed-code "
           "reward; every model evaluated on the val split (bf16, vLLM) and exported to 4-bit GGUF for the laptop; the best "
           "adapter published on Hugging Face. Steps marked critical stop the queue if they fail. ~4 h, ~45-50 units."),
    gpu="A100 GPU", secrets=", `HF_TOKEN` (write token of the Hugging Face account, for publishing the adapter)",
    pip=PINS, extra=["'trl'", "'peft'"],
    queue=[
        "# --- 1. teacher data (Qwen3-8B, thinking off); only solutions that reproduce the answer are kept ---",
        "!python -m hw4.synth --per-skill 300 --out {R}/data_synth/problems.jsonl",
        f"!python -m hw4.teacher {T8} --source train --style pot --n 4 --name data_qwen8b_pot_train",
        f"!python -m hw4.teacher {T8} --source {{R}}/data_synth/problems.jsonl --style pot --n 2 --per-skill 150 --name data_qwen8b_pot_synth",
        f"python -m hw4.teacher {T8} --source train --style cot --n 2 --name data_qwen8b_cot_train",
        "# --- 2. main model: MamayLM, tool-use SFT on train side + synthetic ---",
        "!python -m hw4.sft --model mamay --data {R}/data_qwen8b_pot_train/sft.jsonl {R}/data_qwen8b_pot_synth/sft.jsonl --name sft_mamay_pot",
        "python -m hw4.infer --model mamay --strategy pot0 --fallback cot --splits val test --name colab_mamay_pot0",
        "!python -m hw4.infer --model {W}/sft_mamay_pot --strategy pot0 --fallback cot --splits val test --name colab_sft_mamay_pot",
        "python -m hw4.export_gguf --merged {W}/sft_mamay_pot --out {R}/gguf/sft_mamay_pot-Q4_K_M.gguf",
        "# --- 3. publish the SFT adapter with a model card (val accuracy vs the base model) ---",
        "python -m hw4.hub --run sft_mamay_pot --eval-run colab_sft_mamay_pot --base-eval-run colab_mamay_pot0 --repo lizabugir/mamaylm-4b-ukr-school-math-pot",
        "# --- 4. the same recipe on Gemma ---",
        "python -m hw4.sft --model gemma --data {R}/data_qwen8b_pot_train/sft.jsonl {R}/data_qwen8b_pot_synth/sft.jsonl --name sft_gemma_pot",
        "python -m hw4.infer --model gemma --strategy pot0 --fallback cot --splits val test --name colab_gemma_pot0",
        "python -m hw4.infer --model {W}/sft_gemma_pot --strategy pot0 --fallback cot --splits val test --name colab_sft_gemma_pot",
        "# --- 5. ablations: no synthetic problems; CoT targets instead of code ---",
        "python -m hw4.sft --model mamay --data {R}/data_qwen8b_pot_train/sft.jsonl --name sft_mamay_pot_trainonly",
        "python -m hw4.infer --model {W}/sft_mamay_pot_trainonly --strategy pot0 --fallback cot --name colab_sft_mamay_pot_trainonly",
        "python -m hw4.sft --model mamay --data {R}/data_qwen8b_cot_train/sft.jsonl --name sft_mamay_cot",
        "python -m hw4.infer --model mamay --strategy cot --name colab_mamay_cot",
        "python -m hw4.infer --model {W}/sft_mamay_cot --strategy cot --name colab_sft_mamay_cot",
        "# --- 6. self-consistency on the fine-tuned model ---",
        "python -m hw4.infer --model {W}/sft_mamay_pot --strategy pot0 --n 8 --temperature 0.7 --fallback cot --splits val test --name colab_sft_mamay_pot_sc8",
        "# --- 7. DPO on the SFT model's own mistakes (typical_errors preferred as rejected) ---",
        "python -m hw4.infer --model {W}/sft_mamay_pot --strategy pot0 --splits train --n 8 --temperature 0.8 --save-all --name samples_sft_mamay_train",
        "python -m hw4.dpo --model {W}/sft_mamay_pot --samples samples_sft_mamay_train --teacher-data {R}/data_qwen8b_pot_train/sft.jsonl --name dpo_mamay_pot",
        "python -m hw4.infer --model {W}/dpo_mamay_pot --strategy pot0 --fallback cot --splits val test --name colab_dpo_mamay_pot",
        "python -m hw4.export_gguf --merged {W}/dpo_mamay_pot --out {R}/gguf/dpo_mamay_pot-Q4_K_M.gguf",
        "# --- 8. bonus: GRPO with a verifiable reward (executed code == answer) ---",
        "python -m hw4.grpo --model {W}/sft_mamay_pot --synth {R}/data_synth/problems.jsonl --max-steps 200 --name grpo_mamay_pot",
        "python -m hw4.infer --model {W}/grpo_mamay_pot --strategy pot0 --fallback cot --splits val test --name colab_grpo_mamay_pot",
        "python -m hw4.export_gguf --merged {W}/grpo_mamay_pot --out {R}/gguf/grpo_mamay_pot-Q4_K_M.gguf",
    ])

NOTEBOOKS = {
    "colab_00_all_a100": ALL_A100,
    "colab_01_prompting": dict(
        title="01 · Prompting: both models × strategies",
        about=("Both models on the same fixed validation split (458 problems): zero-shot, chain-of-thought, few-shot "
               "(examples of the same skill from the train side), plan-then-solve, tool use (the model writes Python that we "
               "run), and self-consistency. CoT and tool-use runs also predict the test set for submissions."),
        gpu="L4 GPU (bf16; ~25–40 min for the whole queue)", secrets="", pip=VLLM_PIN, extra=[],
        queue=[
            "# --- baselines: the organisers' setup (CoT, greedy) ---",
            "python -m hw4.infer --model mamay --strategy cot --splits val test --name mamay_cot",
            "python -m hw4.infer --model gemma --strategy cot --splits val test --name gemma_cot",
            "# --- tool use: model writes Python, we execute it; CoT fallback when the code fails ---",
            "python -m hw4.infer --model mamay --strategy pot --fallback cot --splits val test --name mamay_pot",
            "python -m hw4.infer --model gemma --strategy pot --fallback cot --splits val test --name gemma_pot",
            "python -m hw4.infer --model mamay --strategy pot0 --fallback cot --name mamay_pot0",
            "python -m hw4.infer --model gemma --strategy pot0 --fallback cot --name gemma_pot0",
            "# --- other prompting strategies (val only) ---",
            "python -m hw4.infer --model mamay --strategy zeroshot --name mamay_zeroshot",
            "python -m hw4.infer --model gemma --strategy zeroshot --name gemma_zeroshot",
            "python -m hw4.infer --model mamay --strategy fewshot --name mamay_fewshot",
            "python -m hw4.infer --model gemma --strategy fewshot --name gemma_fewshot",
            "python -m hw4.infer --model mamay --strategy plan --name mamay_plan",
            "python -m hw4.infer --model gemma --strategy plan --name gemma_plan",
            "# --- self-consistency: majority vote over 8 samples ---",
            "python -m hw4.infer --model mamay --strategy cot --n 8 --temperature 0.7 --splits val test --name mamay_cot_sc8",
            "python -m hw4.infer --model mamay --strategy pot --n 8 --temperature 0.7 --fallback cot --splits val test --name mamay_pot_sc8",
            "python -m hw4.infer --model gemma --strategy pot --n 8 --temperature 0.7 --fallback cot --splits val test --name gemma_pot_sc8",
        ]),
    "colab_02_teacher": dict(
        title="02 · Teacher data: Qwen3 solutions (train side + synthetic problems)",
        about=("Qwen3-8B (open, allowed as a teacher only) writes solutions for the TRAIN side of the split and for synthetic "
               "problems from my own generators (`hw4.synth`, 29 skills, exact answers). Only solutions that reproduce the "
               "reference answer are kept (code is executed). Val/test are never touched."),
        gpu="L4 GPU (Qwen3-8B in bf16 fits in 24 GB; ~30–50 min)", secrets="", pip=VLLM_PIN, extra=[],
        queue=[
            "python -m hw4.synth --per-skill 300 --out {R}/data_synth/problems.jsonl",
            "python -m hw4.teacher --source train --style pot --n 4 --name data_qwen_pot_train",
            "python -m hw4.teacher --source train --style cot --n 4 --name data_qwen_cot_train",
            "python -m hw4.teacher --source {R}/data_synth/problems.jsonl --style pot --n 2 --per-skill 150 --name data_qwen_pot_synth",
            "python -m hw4.teacher --source {R}/data_synth/problems.jsonl --style cot --n 2 --per-skill 150 --name data_qwen_cot_synth",
        ]),
    "colab_03_sft": dict(
        title="03 · LoRA SFT on verified teacher solutions + evaluation + Hugging Face",
        about=("LoRA SFT of both students on the verified Qwen3 solutions (tool-use format), evaluated on the same val split "
               "and compared with the best prompting run. Ablations: CoT targets instead of code, and train-side data only vs "
               "+ synthetic problems (measured effect of synthetic data). Needs notebook 02 first. Set `PUBLISH = True` and the "
               "`HF_TOKEN` secret to publish the best adapter with a model card."),
        gpu="L4 GPU (~2–3 h for the whole queue; the first block alone ~40 min)", secrets=", `HF_TOKEN` (write token, only for publishing)",
        pip=PINS, extra=["'trl'", "'peft'"],
        queue=[
            "# --- main: tool-use SFT on verified Qwen3 solutions (train side + synthetic problems) ---",
            "python -m hw4.sft --model mamay --data {R}/data_qwen_pot_train/sft.jsonl {R}/data_qwen_pot_synth/sft.jsonl --name sft_mamay_pot",
            "python -m hw4.export_gguf --merged {W}/sft_mamay_pot --out {R}/gguf/sft_mamay_pot-Q4_K_M.gguf",
            "# same-precision (bf16, vLLM) comparison of base vs fine-tuned on the val split",
            "python -m hw4.infer --model mamay --strategy pot0 --fallback cot --name colab_mamay_pot0",
            "python -m hw4.infer --model {W}/sft_mamay_pot --strategy pot0 --fallback cot --splits val test --name colab_sft_mamay_pot",
            "# --- the same recipe on Gemma ---",
            "python -m hw4.sft --model gemma --data {R}/data_qwen_pot_train/sft.jsonl {R}/data_qwen_pot_synth/sft.jsonl --name sft_gemma_pot",
            "python -m hw4.export_gguf --merged {W}/sft_gemma_pot --out {R}/gguf/sft_gemma_pot-Q4_K_M.gguf",
            "python -m hw4.infer --model gemma --strategy pot0 --fallback cot --name colab_gemma_pot0",
            "python -m hw4.infer --model {W}/sft_gemma_pot --strategy pot0 --fallback cot --name colab_sft_gemma_pot",
            "# --- ablation: train-side data only (effect of synthetic problems) ---",
            "python -m hw4.sft --model mamay --data {R}/data_qwen_pot_train/sft.jsonl --name sft_mamay_pot_trainonly",
            "python -m hw4.infer --model {W}/sft_mamay_pot_trainonly --strategy pot0 --fallback cot --name colab_sft_mamay_pot_trainonly",
            "# --- ablation: CoT targets (reasoning in words, no tool) ---",
            "python -m hw4.sft --model mamay --data {R}/data_qwen_cot_train/sft.jsonl --name sft_mamay_cot",
            "python -m hw4.infer --model mamay --strategy cot --name colab_mamay_cot",
            "python -m hw4.infer --model {W}/sft_mamay_cot --strategy cot --name colab_sft_mamay_cot",
        ],
        extra_cells=["""# publish the best adapter (edit the names after looking at the summary above)
PUBLISH = False
if PUBLISH:
    run("python -m hw4.hub --run sft_mamay_pot --eval-run colab_sft_mamay_pot --base-eval-run colab_mamay_pot0 "
        "--repo <hf-user>/mamaylm-4b-ukr-school-math-pot")"""]),
    "colab_04_dpo": dict(
        title="04 · DPO on the student's own mistakes (typical_errors as rejected)",
        about=("The SFT student samples 8 solutions per TRAIN-side problem; pairs are (correct, wrong) with wrong answers that "
               "match a `typical_errors` misconception preferred as rejected. LoRA DPO on top of the SFT model, then evaluation "
               "on val. Needs notebooks 02 and 03."),
        gpu="L4 GPU (~1–1.5 h)", secrets="", pip=PINS, extra=["'trl'", "'peft'"],
        queue=[
            "python -m hw4.merge --run sft_mamay_pot",
            "python -m hw4.infer --model {W}/sft_mamay_pot --strategy pot0 --splits train --n 8 --temperature 0.8 --save-all --name samples_sft_mamay_train",
            "python -m hw4.dpo --model {W}/sft_mamay_pot --samples samples_sft_mamay_train --teacher-data {R}/data_qwen_pot_train/sft.jsonl --name dpo_mamay_pot",
            "python -m hw4.export_gguf --merged {W}/dpo_mamay_pot --out {R}/gguf/dpo_mamay_pot-Q4_K_M.gguf",
            "python -m hw4.infer --model {W}/dpo_mamay_pot --strategy pot0 --fallback cot --splits val test --name colab_dpo_mamay_pot",
        ]),
    "colab_05_grpo": dict(
        title="05 · GRPO with verifiable rewards (bonus)",
        about=("Reinforcement learning on the SFT student: 8 code solutions per problem, reward 1 when executing the code gives "
               "the reference answer (+0.1 for the format). Problems: train side + synthetic. Needs notebooks 02 and 03."),
        gpu="A100 GPU (vLLM generation colocated with training; on L4 add --no-vllm and fewer steps)", secrets="",
        pip=PINS, extra=["'trl'", "'peft'"],
        queue=[
            "python -m hw4.merge --run sft_mamay_pot",
            "python -m hw4.grpo --model {W}/sft_mamay_pot --synth {R}/data_synth/problems.jsonl --max-steps 300 --name grpo_mamay_pot",
            "python -m hw4.export_gguf --merged {W}/grpo_mamay_pot --out {R}/gguf/grpo_mamay_pot-Q4_K_M.gguf",
            "python -m hw4.infer --model {W}/grpo_mamay_pot --strategy pot0 --fallback cot --splits val test --name colab_grpo_mamay_pot",
        ]),
}


def cell(kind: str, src: str) -> dict:
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def notebook(spec: dict) -> dict:
    queue = "QUEUE = [\n" + "\n".join(f"    {q}" if q.startswith("#") else f"    {json.dumps(q, ensure_ascii=False)},"
                                       for q in spec["queue"]) + "\n]"
    extra_versions = "".join(", " + e for e in spec["extra"])
    cells = [cell("markdown", HEADER.format(**{k: spec[k] for k in ("title", "about", "gpu", "secrets")})),
             cell("code", SETUP),
             cell("code", INSTALL.format(pip=spec["pip"], extra_versions=extra_versions)),
             cell("code", DATA),
             cell("code", RUNNER),
             cell("code", queue),
             cell("code", GO)]
    cells += [cell("code", c) for c in spec.get("extra_cells", [])]
    cells.append(cell("code", STOP))
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 0}


def main():
    for old in (ROOT / "kernels").glob("colab_run.ipynb"):
        old.unlink()
    for name, spec in NOTEBOOKS.items():
        (ROOT / "kernels" / f"{name}.ipynb").write_text(json.dumps(notebook(spec), indent=1, ensure_ascii=False))
    if not (ROOT / "src/hw4/configs/split.csv").exists():
        raise SystemExit("run `python -m hw4.data` first")
    with tempfile.TemporaryDirectory() as d:
        shutil.copytree(ROOT / "src", Path(d) / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        shutil.copy(ROOT / "pyproject.toml", d)
        zp = shutil.make_archive(str(ROOT / "hw4-code"), "zip", d)
    print("wrote", len(NOTEBOOKS), "notebooks in kernels/ and", zp)


if __name__ == "__main__":
    main()
