# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical-correctness tests for the chunked (flash-style) prompt-attention impl.

The chunked impl tiles the KEY dimension with fp32 online softmax so a very long
prefill never materializes the full [.., q_len, k_len] score tensor (which OOMs the
naive path at ~197K keys) and never hands any single op more keys than the chunk
width (bypassing the FusedSDPA tile ceiling). These tests pin its output to a full
attention reference so any flash-softmax regression is caught before hardware.
"""
import os

import pytest
import torch

from vllm_gaudi.extension.ops import _chunked_prompt_attention


def _causal_reference(query, key, value, scale, attn_bias=None):
    """Full (non-tiled) attention reference in fp32 with prefix-cache-aware causal mask."""
    q = query.transpose(1, 2).float() * scale
    k = key.transpose(1, 2).float()
    v = value.transpose(1, 2).float()
    q_heads, kv_heads = q.size(1), k.size(1)
    if q_heads != kv_heads:
        q = q.unflatten(1, (kv_heads, -1))
        k = k.unflatten(1, (kv_heads, 1))
        v = v.unflatten(1, (kv_heads, 1))
        if attn_bias is not None:
            attn_bias = attn_bias.unsqueeze(2)
    s = torch.matmul(q, k.transpose(-1, -2))
    q_len, k_len = q.size(-2), k.size(-2)
    if attn_bias is None:
        q_abs = (torch.arange(q_len) + (k_len - q_len)).reshape(q_len, 1)
        k_pos = torch.arange(k_len).reshape(1, k_len)
        mask = (k_pos > q_abs).reshape(*([1] * (s.ndim - 2)), q_len, k_len)
        s = s.masked_fill(mask, float("-inf"))
    else:
        s = s + attn_bias.float()
    s = torch.softmax(s, dim=-1)
    out = torch.matmul(s, v).to(query.dtype)
    if q_heads != kv_heads:
        out = out.flatten(1, 2)
    return out.transpose(1, 2)


def _run(query, key, value, scale, chunk, **kw):
    prev = os.environ.get("VLLM_PROMPT_CHUNK_SIZE")
    os.environ["VLLM_PROMPT_CHUNK_SIZE"] = str(chunk)
    try:
        return _chunked_prompt_attention(query, key, value, scale, **kw)
    finally:
        if prev is None:
            os.environ.pop("VLLM_PROMPT_CHUNK_SIZE", None)
        else:
            os.environ["VLLM_PROMPT_CHUNK_SIZE"] = prev


# (batch, heads, q_len, k_len, head_dim); k_len > q_len exercises the prefix-cache path
SHAPES = [
    (1, 4, 8, 8, 16),
    (1, 8, 64, 64, 32),
    (1, 4, 32, 130, 16),
    (1, 2, 100, 500, 24),
    (2, 4, 40, 200, 16),
]


@pytest.mark.parametrize("b,h,ql,kl,d", SHAPES)
@pytest.mark.parametrize("chunk", [16, 64, 128, 999999])
def test_chunked_matches_causal_reference_fp32(b, h, ql, kl, d, chunk):
    torch.manual_seed(b * 1000 + h * 100 + ql + kl + d + chunk % 7)
    query = torch.randn(b, ql, h, d)
    key = torch.randn(b, kl, h, d)
    value = torch.randn(b, kl, h, d)
    scale = 1.0 / (d**0.5)
    ref = _causal_reference(query, key, value, scale)
    got = _run(query, key, value, scale, chunk, is_causal=True)
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4), \
        f"max|diff|={(got - ref).abs().max().item():.2e} for shape {(b, h, ql, kl, d)} chunk {chunk}"


@pytest.mark.parametrize("chunk", [16, 64, 999999])
def test_chunked_matches_reference_with_attn_bias(chunk):
    """Merged-prefill path: an additive causal bias is sliced per chunk; small chunks make some
    key-chunks fully -inf for a row, exercising the online-softmax all-masked-chunk guard."""
    torch.manual_seed(7)
    b, h, ql, kl, d = 1, 4, 24, 96, 16
    query = torch.randn(b, ql, h, d)
    key = torch.randn(b, kl, h, d)
    value = torch.randn(b, kl, h, d)
    scale = 1.0 / (d**0.5)
    bias = torch.zeros(b, 1, ql, kl)
    for i in range(ql):
        allowed = kl - ql + i + 1
        bias[:, :, i, max(allowed, 0):] = float("-inf")
    ref = _causal_reference(query, key, value, scale, attn_bias=bias)
    got = _run(query, key, value, scale, chunk, is_causal=False, attn_bias=bias)
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4), \
        f"max|diff|={(got - ref).abs().max().item():.2e} chunk {chunk}"


@pytest.mark.parametrize("chunk", [32, 128, 999999])
def test_chunked_gqa_unflatten(chunk):
    """query_heads != kv_heads exercises the GQA unflatten branch (MLA prefill is MHA, but keep parity)."""
    torch.manual_seed(3)
    b, kv_h, groups, ql, kl, d = 1, 2, 3, 40, 160, 16
    query = torch.randn(b, ql, kv_h * groups, d)
    key = torch.randn(b, kl, kv_h, d)
    value = torch.randn(b, kl, kv_h, d)
    scale = 1.0 / (d**0.5)
    ref = _causal_reference(query, key, value, scale)
    got = _run(query, key, value, scale, chunk, is_causal=True)
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4), \
        f"max|diff|={(got - ref).abs().max().item():.2e} chunk {chunk}"


def test_chunk_invariance_bf16():
    """Chunking must not change the answer beyond fp32 accumulation-order drift plus the final bf16 downcast."""
    torch.manual_seed(11)
    query = torch.randn(1, 128, 8, 32, dtype=torch.bfloat16)
    key = torch.randn(1, 600, 8, 32, dtype=torch.bfloat16)
    value = torch.randn(1, 600, 8, 32, dtype=torch.bfloat16)
    scale = 0.176
    full = _run(query, key, value, scale, 999999, is_causal=True)
    tiled = _run(query, key, value, scale, 64, is_causal=True)
    assert torch.allclose(full.float(), tiled.float(), atol=5e-3, rtol=5e-3), \
        f"chunk-variance max|diff|={(full.float() - tiled.float()).abs().max().item():.2e}"
