"""Rebuilds merged weights (for vLLM) from a saved LoRA adapter, so only small adapters live on Drive.

python -m hw4.merge --run sft_mamay_pot        # runs/<run>/adapter + base model -> weights/<run>
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .runs import RUNS
from .train_utils import WEIGHTS, local_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    out = WEIGHTS / a.run
    if (out / "config.json").exists():
        print("already merged:", out)
        return
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    cfg = json.loads((Path(RUNS) / a.run / "config.json").read_text())
    base_dir = local_dir(cfg["model"])
    m = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.bfloat16)
    m = PeftModel.from_pretrained(m, str(Path(RUNS) / a.run / "adapter")).merge_and_unload()
    m.save_pretrained(out, safe_serialization=True)
    for f in Path(base_dir).iterdir():
        if f.is_file() and not f.name.endswith((".safetensors", ".bin")) and f.name not in (
                "config.json", "model.safetensors.index.json", "README.md", ".gitattributes"):
            shutil.copy(f, out / f.name)
    print("merged ->", out)


if __name__ == "__main__":
    main()
