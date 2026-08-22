# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fp32 RoPE path (VLLM_FP32_ROPE) in HPURotaryEmbedding.

The manual fp32 rotation (`_apply_rope_fp32`, used by VLLM_FP32_ROPE=manual) is CPU-bit-exact vs
vLLM's canonical `forward_native`, so it is the verifiable reference here. The default 'fused' mode
routes the same fp32 cos/sin through the Habana kernel (HPU-only, verified on hardware). fp32 RoPE
removes the bf16 positional phase-noise that corrupts byte-exact long-context copy past ~196.6K.
"""
import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config

from vllm_gaudi.ops.hpu_rotary_embedding import HPURotaryEmbedding, _apply_rope_fp32, _fp32_rope_mode

HEAD, ROT, BASE, MAXP = 64, 64, 8_000_000.0, 262144  # GLM-5.2 MLA rope params (neox)


def _rope():
    with set_current_vllm_config(VllmConfig()):
        return HPURotaryEmbedding(HEAD, ROT, MAXP, BASE, True, torch.bfloat16)


@pytest.mark.parametrize("start", [0, 196000])  # small + large-position (where bf16 RoPE drifts)
def test_manual_fp32_matches_canonical(start):
    """Manual fp32 rotation is bit-exact vs vLLM's canonical forward_native, both in fp32."""
    rope = _rope()
    fp32_cache = rope._compute_cos_sin_cache().float()
    torch.manual_seed(start + 1)
    T, NH = 32, 8
    pos = torch.arange(start, start + T)
    q = torch.randn(T, NH, HEAD)
    k = torch.randn(T, NH, HEAD)
    # reference: forward_native with a true-fp32 cache (its registered buffer is bf16)
    orig = rope.cos_sin_cache
    rope.cos_sin_cache = fp32_cache
    qn, kn = rope.forward_native(pos, q.clone().view(T, -1), k.clone().view(T, -1))
    rope.cos_sin_cache = orig
    # our manual fp32 path
    cos, sin = rope._fp32_cos_sin(pos, None)
    qf = _apply_rope_fp32(q.view(T, NH, HEAD), cos, sin, True).view(T, -1)
    kf = _apply_rope_fp32(k.view(T, NH, HEAD), cos, sin, True).view(T, -1)
    assert torch.allclose(qf, qn, atol=1e-4), (qf - qn).abs().max().item()
    assert torch.allclose(kf, kn, atol=1e-4), (kf - kn).abs().max().item()


def test_manual_fp32_removes_bf16_error():
    """The fp32 path differs from the bf16-cache path by a nonzero amount that grows toward the
    cliff -- the positional error fp32 RoPE is there to remove."""
    rope = _rope()
    torch.manual_seed(0)
    T, NH = 16, 8
    for start in (0, 196000):
        pos = torch.arange(start, start + T)
        q = torch.randn(T, NH, HEAD)
        qn_bf16, _ = rope.forward_native(pos, q.clone().view(T, -1), q.clone().view(T, -1))  # bf16 cache
        cos, sin = rope._fp32_cos_sin(pos, None)
        qf = _apply_rope_fp32(q.view(T, NH, HEAD), cos, sin, True).view(T, -1)
        assert (qf - qn_bf16).abs().max().item() > 1e-4  # bf16 cache introduces real error


def test_mode_gate(monkeypatch):
    for v in ("1", "true", "on", "yes", "fused", "FUSED"):
        monkeypatch.setenv("VLLM_FP32_ROPE", v)
        assert _fp32_rope_mode() == "fused"
    monkeypatch.setenv("VLLM_FP32_ROPE", "manual")
    assert _fp32_rope_mode() == "manual"
    for v in ("0", "off", "", "prefill"):  # 'prefill' was removed -> off
        monkeypatch.setenv("VLLM_FP32_ROPE", v)
        assert _fp32_rope_mode() == "off"
    monkeypatch.delenv("VLLM_FP32_ROPE", raising=False)
    assert _fp32_rope_mode() == "off"
