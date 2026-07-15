# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Copyright (C) 2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
"""Unit tests for the pure-torch MXFP8 [1,32]/E8M0 dequant helpers.

These exercise only ``vllm_gaudi.ops.mxfp8_dequant`` (torch-only, no HPU stack),
so they run wherever ``torch`` is available.  Intended to be run on the Gaudi
host container:

    pytest tests/unit_tests/test_mxfp8_dequant.py -v

On a box without ``torch`` (e.g. a WSL dev box) the whole module is skipped.
"""
import pytest

torch = pytest.importorskip("torch")

from vllm_gaudi.ops.mxfp8_dequant import e8m0_to_scale, mxfp8_dequant_block  # noqa: E402


def test_e8m0_to_scale_known_values():
    # E8M0 exponent byte e -> 2 ** (e - 127).
    raw = torch.tensor([127.0, 128.0, 126.0, 130.0])
    got = e8m0_to_scale(raw)
    expected = torch.tensor([1.0, 2.0, 0.5, 8.0])
    assert torch.allclose(got, expected, rtol=0, atol=0)


def test_mxfp8_block_dequant_matches_reference():
    # One [1, 32] block: e4m3 values * 2 ** (e8m0_scale - 127).
    q = torch.tensor([[1.0, -2.0] + [0.0] * 30], dtype=torch.float8_e4m3fn)
    scale = torch.tensor([130], dtype=torch.uint8)  # E8M0 exponent -> 2**3 = 8
    ref = q.to(torch.float64) * (2.0**(130 - 127))
    out = mxfp8_dequant_block(q, scale, block=32)
    assert out.shape == q.shape
    assert torch.allclose(out.to(torch.float64), ref, rtol=0, atol=0)


def test_mxfp8_two_blocks_per_row():
    # A [1, 64] weight = two [1, 32] blocks with different E8M0 scales.
    row = [1.0, -2.0] + [0.0] * 30 + [4.0, -1.0] + [0.0] * 30
    q = torch.tensor([row], dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[130, 125]], dtype=torch.uint8)  # 2**3=8, 2**-2=0.25
    out = mxfp8_dequant_block(q, scale, block=32).to(torch.float64)

    ref = q.to(torch.float64).clone()
    ref[:, :32] *= 8.0
    ref[:, 32:] *= 0.25
    assert torch.allclose(out, ref, rtol=0, atol=0)


def test_mxfp8_block_dequant_stacked_experts():
    # [E, M, N] leading dims (the model's gathered-expert case): scale broadcast
    # must apply per [1, 32] block along the last (input) dim, independent of the
    # leading E/M dims.
    E, M, N, blk = 3, 5, 64, 32
    q = torch.randint(-4, 4, (E, M, N)).to(torch.float8_e4m3fn)
    scale = torch.randint(120, 132, (E, M, N // blk)).to(torch.uint8)

    out = mxfp8_dequant_block(q, scale, block=blk).to(torch.float64)

    mult = e8m0_to_scale(scale).to(torch.float64)  # [E, M, N//blk]
    ref = (q.to(torch.float64).reshape(E, M, N // blk, blk) * mult.unsqueeze(-1)).reshape(E, M, N)
    assert torch.allclose(out, ref, rtol=0, atol=0)


def test_mxfp8_block_width_mismatch_raises():
    q = torch.zeros((1, 32), dtype=torch.float8_e4m3fn)
    scale = torch.tensor([130], dtype=torch.uint8)  # derives block=32
    with pytest.raises(ValueError):
        mxfp8_dequant_block(q, scale, block=16)
