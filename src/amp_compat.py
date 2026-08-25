"""
Mixed-precision helpers that work across PyTorch versions.

`torch.cuda.amp.autocast` / `torch.cuda.amp.GradScaler` were deprecated in
PyTorch 2.4 in favour of the device-generic `torch.amp` API.  The old
`torch.cuda.amp.autocast` also does *not* accept a `device_type=` argument, so
code written against the new signature crashes on it.  Colab and Kaggle both
ship PyTorch 2.x, so we resolve the right API once, here.

The context manager is also a no-op on CPU, which keeps the same training code
runnable without a GPU (slowly) for debugging.
"""

import contextlib

import torch

_HAS_NEW_AMP = hasattr(torch, "amp") and hasattr(torch.amp, "autocast")


def get_grad_scaler(enabled: bool = True):
    """Return a GradScaler appropriate for the installed PyTorch.

    Returns None when AMP is disabled or CUDA is unavailable, which the
    training loop treats as "run in full precision".
    """
    if not enabled or not torch.cuda.is_available():
        return None
    if _HAS_NEW_AMP:
        try:
            return torch.amp.GradScaler("cuda")
        except (TypeError, AttributeError):
            pass
    return torch.cuda.amp.GradScaler()


def autocast(enabled: bool = True, device_type: str = "cuda"):
    """Context manager for autocasting, or a null context when disabled."""
    if not enabled or not torch.cuda.is_available():
        return contextlib.nullcontext()
    if _HAS_NEW_AMP:
        try:
            return torch.amp.autocast(device_type=device_type)
        except (TypeError, AttributeError):
            pass
    return torch.cuda.amp.autocast()
