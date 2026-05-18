# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_EXT = None


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT

    if not torch.cuda.is_available():
        raise RuntimeError("SM70/SM75 legacy GDN backend requires CUDA")

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0;7.5")
    src = Path(__file__).with_name("csrc") / "gdn_forward.cu"
    _EXT = load(
        name="flash_qla_legacy_gdn",
        sources=[str(src)],
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
        verbose=bool(int(os.environ.get("FLASH_QLA_LEGACY_VERBOSE_BUILD", "0"))),
    )
    return _EXT


def _check_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> None:
    tensors = [q, k, v, g, beta]
    if initial_state is not None:
        tensors.append(initial_state)

    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("legacy GDN tensors must be CUDA tensors")
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise ValueError("legacy GDN backend currently supports float32 tensors only")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("legacy GDN tensors must be contiguous")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [B, T, H, D]")
    if g.ndim != 3 or beta.ndim != 3:
        raise ValueError("g and beta must have shape [B, T, Hv]")
    if q.shape != k.shape:
        raise ValueError("q and k must have the same shape")

    batch, tokens, q_heads, dim = q.shape
    if v.shape[0] != batch or v.shape[1] != tokens or v.shape[3] != dim:
        raise ValueError("v must have shape [B, T, Hv, D] matching q/k")
    if g.shape != beta.shape or g.shape != v.shape[:3]:
        raise ValueError("g and beta must have shape [B, T, Hv]")
    if v.shape[2] % q_heads != 0:
        raise ValueError("Hv must be divisible by Hq")
    if dim not in (16, 32, 64, 128):
        raise ValueError("legacy GDN backend supports D in {16, 32, 64, 128}")
    if initial_state is not None and initial_state.shape != (batch, v.shape[2], dim, dim):
        raise ValueError("initial_state must have shape [B, Hv, D, D]")


def chunk_gated_delta_rule_fwd_legacy(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the experimental SM70/SM75 forward-only GDN backend.

    This legacy backend is intentionally explicit. It does not replace the
    Hopper/SM90 TileLang path and currently supports only contiguous float32
    tensors for inference-oriented forward execution.

    Shapes:
        q, k: [B, T, Hq, D]
        v: [B, T, Hv, D]
        g, beta: [B, T, Hv]
        initial_state: optional [B, Hv, D, D]

    Returns:
        output: [B, T, Hv, D]
        final_state: [B, Hv, D, D]
    """

    _check_inputs(q, k, v, g, beta, initial_state)
    if scale is None:
        scale = q.shape[-1] ** -0.5

    ext = _load_ext()
    return ext.gdn_forward(q, k, v, g, beta, initial_state, float(scale))
