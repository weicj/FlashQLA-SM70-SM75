# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import math

import pytest
import torch

from flash_qla.ops.gated_delta_rule.legacy import chunk_gated_delta_rule_fwd_legacy


def _reference(q, k, v, g, beta, scale=None, initial_state=None):
    batch, tokens, q_heads, dim = q.shape
    v_heads = v.shape[2]
    scale = scale if scale is not None else dim**-0.5
    state = (
        initial_state.clone()
        if initial_state is not None
        else torch.zeros(batch, v_heads, dim, dim, device=q.device, dtype=q.dtype)
    )
    output = torch.empty_like(v)
    for b in range(batch):
        for hv in range(v_heads):
            hq = hv // (v_heads // q_heads)
            for t in range(tokens):
                gate = torch.exp(g[b, t, hv])
                delta = (v[b, t, hv] - gate * (state[b, hv].transpose(0, 1) @ k[b, t, hq])) * beta[b, t, hv]
                state[b, hv] = gate * state[b, hv] + torch.outer(k[b, t, hq], delta)
                output[b, t, hv] = scale * (state[b, hv].transpose(0, 1) @ q[b, t, hq])
    return output, state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dim", [16, 32, 64, 128])
def test_legacy_sm_gdn_matches_reference(dim):
    torch.manual_seed(1000 + dim)
    q = torch.randn(1, 5, 2, dim, device="cuda", dtype=torch.float32).contiguous() * 0.05
    k = torch.randn_like(q).contiguous() * 0.05
    v = torch.randn(1, 5, 4, dim, device="cuda", dtype=torch.float32).contiguous() * 0.1
    g = torch.randn(1, 5, 4, device="cuda", dtype=torch.float32).contiguous() * 0.02 - 0.04
    beta = torch.rand(1, 5, 4, device="cuda", dtype=torch.float32).contiguous()
    h0 = torch.randn(1, 4, dim, dim, device="cuda", dtype=torch.float32).contiguous() * 0.01
    scale = 1.0 / math.sqrt(dim)

    out_ref, state_ref = _reference(q, k, v, g, beta, scale, h0)
    out, state = chunk_gated_delta_rule_fwd_legacy(q, k, v, g, beta, scale, h0)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, out_ref, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(state, state_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_legacy_sm_gdn_rejects_unsupported_dtype():
    q = torch.randn(1, 1, 1, 16, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="float32"):
        chunk_gated_delta_rule_fwd_legacy(q, q, q, torch.randn(1, 1, 1, device="cuda"), torch.randn(1, 1, 1, device="cuda"))
