"""Converts merged fine-tuned weights to a 4-bit GGUF (Q4_K_M), the same format the base models were evaluated in
on the laptop (llama.cpp), so base and fine-tuned models are compared under identical quantization.

python -m hw4.export_gguf --merged weights/sft_mamay_pot --out /content/drive/MyDrive/hw4/gguf/sft_mamay_pot-Q4_K_M.gguf

Uses llama.cpp's convert_hf_to_gguf.py (git checkout of the release tag) and the prebuilt llama-quantize (CPU build).
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

TAG = "b12774"
WORK = Path(os.environ.get("HW4_LLAMACPP", "/tmp/llamacpp"))


def sh(cmd: str):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def setup() -> tuple[Path, Path]:
    src = WORK / "src"
    if not (src / "convert_hf_to_gguf.py").exists():
        sh(f"git clone -q --depth 1 --branch {TAG} https://github.com/ggml-org/llama.cpp {src}")
        # convert_hf_to_gguf.py imports gguf from its own gguf-py/ folder; it needs torch, transformers, sentencepiece
        sh(f"{sys.executable} -m pip install -q sentencepiece 2>/dev/null || true")
    q = glob.glob(f"{WORK}/bin/**/llama-quantize", recursive=True)
    if not q:
        url = f"https://github.com/ggml-org/llama.cpp/releases/download/{TAG}/llama-{TAG}-bin-ubuntu-x64.tar.gz"
        (WORK / "bin").mkdir(parents=True, exist_ok=True)
        sh(f"curl -sL {url} | tar xz -C {WORK}/bin")
        q = glob.glob(f"{WORK}/bin/**/llama-quantize", recursive=True)
    return src / "convert_hf_to_gguf.py", Path(q[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quant", default="Q4_K_M")
    a = ap.parse_args()
    if Path(a.out).exists():
        print("exists:", a.out)
        return
    conv, quant = setup()
    f16 = Path("/tmp") / (Path(a.out).stem + "-f16.gguf")
    sh(f"{sys.executable} {conv} {a.merged} --outtype f16 --outfile {f16}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    env = f"LD_LIBRARY_PATH={quant.parent}:$LD_LIBRARY_PATH"
    sh(f"{env} {quant} {f16} {a.out} {a.quant}")
    f16.unlink()
    print("wrote", a.out, f"{Path(a.out).stat().st_size / 2**30:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
