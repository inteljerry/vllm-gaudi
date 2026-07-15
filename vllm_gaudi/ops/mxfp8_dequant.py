# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Copyright (C) 2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
"""Pure-torch MXFP8 (OCP Microscaling FP8) dequant helpers.

This module is intentionally free of any ``habana_frameworks`` / ``vllm`` import
so the E8M0 scale conversion and the ``[1, 32]`` block dequant can be imported
and unit-tested in isolation (only ``torch`` is required).  See
``tests/unit_tests/test_mxfp8_dequant.py``.

MXFP8 (as used by ``MiniMax-M3-MXFP8``) stores ``float8_e4m3fn`` weights with a
per ``[1, 32]``-input-block scale encoded as an E8M0 (``uint8``) exponent with
bias 127::

    real_weight[..., j] = fp8[..., j] * 2 ** (scale_u8[..., j // 32] - 127)

The E8M0 byte is loaded into vLLM's ``float32`` ``weight_scale_inv`` parameter as
its raw exponent value (0..255), so ``e8m0_to_scale`` computes ``2 ** (x - 127)``
directly on those floats.
"""
import torch

E8M0_BIAS = 127


def e8m0_to_scale(t: torch.Tensor) -> torch.Tensor:
    """E8M0 exponents (loaded into a float param as raw 0..255) -> f32 multiplier.

    ``t`` holds the raw E8M0 exponent bytes as floats (this is what vLLM's block
    ``weight_scale_inv`` param ends up carrying after loading the checkpoint's
    ``uint8`` E8M0 scales into a ``float32`` parameter).  The multiplier is
    ``2 ** (t - 127)``.
    """
    return torch.exp2(t.float() - E8M0_BIAS)


def mxfp8_dequant_block(q: torch.Tensor,
                        scale_e8m0: torch.Tensor,
                        block: int = 32,
                        dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Dequantize block-MXFP8 weights from RAW E8M0 scales.

    Args:
        q: fp8 (or any) tensor ``[..., N]``.
        scale_e8m0: raw E8M0 exponents ``[..., N // block]`` (0..255 as floats or
            ``uint8``); the E8M0 -> f32 conversion is applied here.
        block: block width along the last (input) dim (32 for MXFP8's ``[1, 32]``).
        dtype: output/compute dtype.

    Returns:
        ``q.to(dtype) * 2 ** (scale_e8m0 - 127)`` broadcast over each ``block``
        lane, same shape as ``q``.

    NOTE: the on-the-fly expert dequant in ``minimax_m3.MiniMaxM3MoE`` uses the
    *already-converted* f32 scale (converted once at load time by
    ``MXFp8MoEMethod.process_weights_after_loading``), so it does the block
    broadcast without re-running ``e8m0_to_scale`` -- this helper is the
    standalone/tested reference that folds both steps together.
    """
    mult = e8m0_to_scale(scale_e8m0)
    n = q.shape[-1]
    nb = mult.shape[-1]
    blk = n // nb
    if blk != block:
        raise ValueError(
            f"MXFP8 dequant expected block width {block} (N={n} / "
            f"num_blocks={nb}) but derived {blk}.")
    w = q.to(dtype).reshape(*q.shape[:-1], nb, blk)
    s = mult.to(dtype).unsqueeze(-1)
    return (w * s).reshape(q.shape)
