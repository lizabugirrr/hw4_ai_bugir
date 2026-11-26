"""UCU AI 2026 HW4: Ukrainian grade-5 math word problems with Gemma-3-4B / MamayLM-4B."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "pyproject.toml").exists():  # installed as a package (e.g. on Kaggle): use the cwd
    ROOT = Path.cwd()
DATA = Path(os.environ.get("HW4_DATA", ROOT / "data"))
CONFIGS = Path(__file__).resolve().parent / "configs"

# short name -> (HF id, Kaggle Models variation used when the HF repo is gated)
MODELS = {
    "gemma": ("google/gemma-3-4b-it", "gemma-3-4b-it"),
    "mamay": ("INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0", None),
}
