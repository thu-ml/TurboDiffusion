"""Load the public VAE Python modules supplied with MiniMax-H3 weights."""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import sys
import types


def load_vae_module(path: str | Path, kind: str, *, trust_model_code: bool):
    if not trust_model_code:
        raise ValueError(
            "H3 VAEs use Python code from the downloaded model; pass "
            "--trust-model-code after reviewing the official MiniMax-H3 files"
        )
    if kind not in ("video", "audio"):
        raise ValueError(f"unsupported VAE kind: {kind}")
    root = Path(path).resolve()
    entry = f"minimax_h3_{kind}_vae"
    if not (root / f"{entry}.py").is_file():
        raise FileNotFoundError(
            f"missing {root / (entry + '.py')}; download the complete FL2VA/{kind}_vae directory"
        )
    # A private package namespace preserves relative imports without changing sys.path.
    name = "_turbodiffusion_h3_" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(root)]
        package.__package__ = name
        sys.modules[name] = package
    return importlib.import_module(f"{name}.{entry}")
