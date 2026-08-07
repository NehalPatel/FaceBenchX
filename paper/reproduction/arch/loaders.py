"""Load official AdaFace / MagFace backbones and checkpoints for paper runs.

FaceBench adapters accept an injected ``model=`` module; this module builds
those architectures outside ``facebench/`` so framework code stays frozen.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_ARCH_DIR = Path(__file__).resolve().parent


def _load_module(module_name: str, path: Path) -> Any:
    """Import a vendored ``.py`` file as a module."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load architecture module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class ChannelFlipWrapper(nn.Module):
    """Convert FaceBench RGB ArcFace tensors to official BGR channel order.

    Official AdaFace expects BGR normalized to approximately ``[-1, 1]``.
    FaceBench ``preprocess_arcface_112`` feeds RGB; flipping channels here
    avoids modifying framework adapters.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> Any:
        return self.backbone(x[:, [2, 1, 0], ...])


class MagFaceInputWrapper(nn.Module):
    """Adapt FaceBench ArcFace RGB ``[-1,1]`` tensors to MagFace BGR ``[0,1]``.

    Official MagFace ``gen_feat.py`` uses ``cv2.imread`` (BGR) + ``ToTensor``
    with identity normalize (``mean=0, std=1``), i.e. BGR in ``[0, 1]``.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> Any:
        # Undo ArcFace Normalize([0.5],[0.5]) then RGB->BGR.
        x = x * 0.5 + 0.5
        x = x[:, [2, 1, 0], ...]
        return self.backbone(x)


def _strip_prefixes(state: dict[str, Any], prefixes: tuple[str, ...]) -> dict[str, Any]:
    """Strip DDP / Lightning key prefixes from a state dict."""
    out: dict[str, Any] = {}
    for key, value in state.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
                break
        out[new_key] = value
    return out


def build_adaface_ir50(weights_path: str | Path, *, device: str = "cpu") -> nn.Module:
    """Build AdaFace IR-50 and load an official Lightning checkpoint.

    Args:
        weights_path: Path to ``adaface_ir50_*.ckpt``.
        device: Torch device string.

    Returns:
        Eval-mode module wrapped for RGB->BGR channel order.
    """
    net = _load_module("paper_adaface_net", _ARCH_DIR / "adaface_net.py")
    backbone = net.build_model("ir_50")
    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model_state = {
        key[6:]: value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith("model.")
    }
    missing, unexpected = backbone.load_state_dict(model_state, strict=False)
    if len(missing) > 20:
        raise RuntimeError(
            f"AdaFace load looks incomplete: missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )
    wrapped = ChannelFlipWrapper(backbone)
    wrapped.eval()
    wrapped.to(device)
    return wrapped


def build_magface_iresnet50(
    weights_path: str | Path, *, device: str = "cpu"
) -> nn.Module:
    """Build MagFace iResNet-50 and load an official DDP checkpoint.

    Args:
        weights_path: Path to MagFace ``.pth`` checkpoint.
        device: Torch device string.

    Returns:
        Eval-mode module wrapped for MagFace BGR ``[0,1]`` inputs.
    """
    iresnet = _load_module("paper_magface_iresnet", _ARCH_DIR / "magface_iresnet.py")
    backbone = iresnet.iresnet50()
    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected MagFace checkpoint type: {type(ckpt)}")
    cleaned = _strip_prefixes(
        state,
        (
            "features.module.",
            "module.features.",
            "module.",
            "features.",
        ),
    )
    missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
    if len(missing) > 20:
        raise RuntimeError(
            f"MagFace load looks incomplete: missing={len(missing)} "
            f"unexpected={len(unexpected)} sample_missing={missing[:5]}"
        )
    wrapped = MagFaceInputWrapper(backbone)
    wrapped.eval()
    wrapped.to(device)
    return wrapped
