"""Shared pieces of SFT / DPO / GRPO: loading the student, LoRA config, merging for vLLM."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from . import ROOT

WEIGHTS = Path(os.environ.get("HW4_WEIGHTS", ROOT / "weights"))

# LoRA only on the language model; the vision tower of Gemma 3 is never used
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def bf16_ok() -> bool:
    """bf16 only on Ampere+ GPUs (L4, A100); fp32 elsewhere (CPU tests, T4 uses the unsloth path)."""
    import torch
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


def warmup(cfg_cls, ratio: float) -> dict:
    """transformers 5 dropped warmup_ratio (warmup_steps < 1 is a ratio there); 4.x still has it."""
    import dataclasses
    names = {f.name for f in dataclasses.fields(cfg_cls)}
    return {"warmup_ratio": ratio} if "warmup_ratio" in names else {"warmup_steps": ratio}


def make_config(cfg_cls, **kw):
    """Builds a TRL config, dropping options this TRL version does not have (their names change between releases)."""
    import dataclasses
    names = {f.name for f in dataclasses.fields(cfg_cls)}
    dropped = sorted(k for k in kw if k not in names)
    if dropped:
        print(f"{cfg_cls.__name__}: options not in this TRL version, ignored: {dropped}", flush=True)
    return cfg_cls(**{k: v for k, v in kw.items() if k in names})


def local_dir(model: str) -> str:
    """Model name/path -> local directory with config and tokenizer files."""
    from .infer import resolve_model
    path = resolve_model(model)
    if os.path.isdir(path):
        return path
    from huggingface_hub import snapshot_download
    return snapshot_download(path)


def load_student(model: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = local_dir(model)
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="eager")
    m.config.use_cache = False
    return m, tok, path


def lora_config(rank: int, alpha: int | None = None, dropout: float = 0.05):
    from peft import LoraConfig
    return LoraConfig(r=rank, lora_alpha=alpha or 2 * rank, lora_dropout=dropout, target_modules=LORA_TARGETS,
                      task_type="CAUSAL_LM")


def save_adapter_and_merged(trainer, tok, base_dir: str, adapter_dir: Path, merged_dir: Path | None):
    """Adapter -> adapter_dir (published on HF). Merged bf16 weights + base tokenizer/processor files ->
    merged_dir, so vLLM loads the fine-tuned model exactly like the base one."""
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    if merged_dir is None:
        return
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    for f in Path(base_dir).iterdir():
        if f.is_file() and not f.name.endswith((".safetensors", ".bin")) and f.name not in (
                "config.json", "model.safetensors.index.json", "README.md", ".gitattributes"):
            shutil.copy(f, merged_dir / f.name)
    print("merged model ->", merged_dir, flush=True)
