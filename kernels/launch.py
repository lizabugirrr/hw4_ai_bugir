"""Runs experiment queues as a private Kaggle script kernel (UNTESTED template).

HW3 forbids sharing code between participants, so unlike hw2 the code is NOT installed from a
public GitHub repo: `--upload` packs src/ + pyproject.toml into a private Kaggle dataset
(<user>/hw4-code), which the kernel attaches and pip-installs. Re-upload after every code change
and write the dataset version into the report so each run is tied to its code.

Usage:
  export KAGGLE_USER=<your kaggle username>
  python kernels/launch.py --upload --slug hw4-mamay-cot --queue "python -m hw4.infer --model mamay --strategy cot --splits val test --name mamay_cot --out {OUT}"
Placeholders in commands: {OUT} (runs dir), {WORK} (/kaggle/working). Commands inside one --queue
are separated by " && "; on a "T4 x2" machine two queues run on one GPU each;
a single queue sees both GPUs (vLLM --tp 2 for float32 Gemma 3).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMP = "ukrainian-school-math-slm-hw4-ucu-ai-2026"
USER = os.environ.get("KAGGLE_USER") or ""

RUNNER = r'''
import glob, hashlib, os, shutil, subprocess, threading, time
QUEUES = {queues!r}
PIP = {pip!r}
CODE_HASH = {code_hash!r}
WORK = "/kaggle/working"
t0 = time.time()
src = sorted(glob.glob("/kaggle/input/**/pyproject.toml", recursive=True))
assert src, "hw4-code dataset not attached: " + str(glob.glob("/kaggle/input/**", recursive=True)[:50])
shutil.copytree(os.path.dirname(src[0]), "/tmp/hw4code", dirs_exist_ok=True)
if not os.path.isdir("/tmp/hw4code/src"):  # --dir-mode zip may leave src/ as an archive
    arch = glob.glob("/tmp/hw4code/src.*")
    assert arch, os.listdir("/tmp/hw4code")
    shutil.unpack_archive(arch[0], "/tmp/hw4code/src" if not arch[0].endswith(".tar") else "/tmp/hw4code")
    inner = "/tmp/hw4code/src/src"
    if os.path.isdir(inner):
        shutil.copytree(inner, "/tmp/hw4code/src", dirs_exist_ok=True)
got = hashlib.sha1(b"".join(open(f, "rb").read() for f in sorted(glob.glob("/tmp/hw4code/src/hw4/*.py")))).hexdigest()
assert got == CODE_HASH, f"attached hw4-code is not the pushed version ({{got}} != {{CODE_HASH}}); re-run later"
subprocess.run("nvidia-smi -L; free -g | head -2", shell=True)
subprocess.run("pip install -q --no-deps /tmp/hw4code", shell=True, check=True)
if PIP:
    subprocess.run("pip install -q " + PIP, shell=True, check=True)
os.environ["HW4_DATA"] = os.path.dirname(sorted(glob.glob("/kaggle/input/**/train.csv", recursive=True))[0])
OUT = f"{{WORK}}/runs"
os.environ["HW4_RUNS"] = OUT
os.environ["HW4_WEIGHTS"] = f"{{WORK}}/weights"
os.makedirs(f"{{WORK}}/logs", exist_ok=True)

def run_queue(gpu, cmds):
    env = dict(os.environ, PYTHONWARNINGS="ignore", TOKENIZERS_PARALLELISM="false",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    if len(QUEUES) > 1:  # one queue per GPU; a single queue sees both GPUs (vLLM --tp 2)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for i, cmd in enumerate(cmds):
        cmd = cmd.replace("{{OUT}}", OUT).replace("{{WORK}}", WORK)
        print(f"[gpu{{gpu}}] start {{cmd}}", flush=True)
        s = time.time()
        with open(f"{{WORK}}/logs/q{{gpu}}_{{i}}.log", "w") as f:
            p = subprocess.run(cmd, shell=True, cwd=WORK, env=env, stdout=f, stderr=subprocess.STDOUT)
        print(f"[gpu{{gpu}}] exit {{p.returncode}} after {{(time.time()-s)/60:.1f}} min", flush=True)

threads = [threading.Thread(target=run_queue, args=(g, q)) for g, q in enumerate(QUEUES)]
[t.start() for t in threads]
[t.join() for t in threads]
print(f"all queues finished in {{(time.time()-t0)/60:.1f}} min", flush=True)
'''


def code_hash() -> str:
    import hashlib
    files = sorted((ROOT / "src/hw4").glob("*.py"), key=lambda f: f.name)
    return hashlib.sha1(b"".join(f.read_bytes() for f in files)).hexdigest()


def upload_code():
    if not (ROOT / "src/hw4/configs/split.csv").exists():
        raise SystemExit("src/hw4/configs/split.csv missing: run `python -m hw4.data` first")
    d = Path(tempfile.mkdtemp(prefix="hw4-code-"))
    shutil.copytree(ROOT / "src", d / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    shutil.copy(ROOT / "pyproject.toml", d)
    meta = {"title": "hw4-code", "id": f"{USER}/hw4-code", "licenses": [{"name": "CC0-1.0"}]}
    (d / "dataset-metadata.json").write_text(json.dumps(meta))
    exists = subprocess.run(["kaggle", "datasets", "status", f"{USER}/hw4-code"], capture_output=True).returncode == 0
    cmd = (["kaggle", "datasets", "version", "-m", "code update"] if exists else ["kaggle", "datasets", "create"])
    subprocess.run(cmd + ["-p", str(d), "--dir-mode", "zip"], check=True)  # 'create' is private by default
    # a kernel pushed before the new version is processed silently mounts the OLD code: wait for it
    import time
    time.sleep(30)
    for _ in range(60):
        st = subprocess.run(["kaggle", "datasets", "status", f"{USER}/hw4-code"], capture_output=True, text=True).stdout
        if "ready" in st.lower():
            break
        time.sleep(10)
    time.sleep(30)
    print("hw4-code dataset version ready", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--queue", action="append", required=True)
    ap.add_argument("--upload", action="store_true", help="upload src/ as the private hw4-code dataset first")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--sources", nargs="*", default=[], help="earlier kernels whose output to attach (e.g. model weights)")
    ap.add_argument("--pip", default="vllm==0.11.0 transformers==4.57.1", help="extra pip packages installed in the kernel")
    ap.add_argument("--models", nargs="*", default=["google/gemma-3/transformers/gemma-3-4b-it/1"],
                    help="Kaggle Models attached to the kernel (gated Gemma comes from here, not HF)")
    ap.add_argument("--accelerator", default="NvidiaTeslaT4")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if not USER:
        raise SystemExit("set KAGGLE_USER")
    if a.upload and not a.dry:
        upload_code()
    meta = {
        "id": f"{USER}/{a.slug}", "title": a.slug, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": not a.cpu, "enable_internet": True,
        "competition_sources": [COMP], "dataset_sources": [f"{USER}/hw4-code"], "kernel_sources": [f"{USER}/{k}" for k in a.sources],
        "model_sources": a.models,
    }
    if not a.cpu:
        meta["machine_shape"] = a.accelerator
    d = Path(tempfile.mkdtemp(prefix=f"{a.slug}-"))
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    (d / "run.py").write_text(RUNNER.format(queues=[q.split(" && ") for q in a.queue], code_hash=code_hash(), pip=a.pip))
    print(f"kernel dir {d}")
    if not a.dry:
        acc = [] if a.cpu else ["--accelerator", a.accelerator]
        subprocess.run(["kaggle", "kernels", "push", "-p", str(d)] + acc, check=True)


if __name__ == "__main__":
    main()
