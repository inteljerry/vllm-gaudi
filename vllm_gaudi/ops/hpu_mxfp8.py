# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Copyright (C) 2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
"""MXFP8 (OCP Microscaling FP8) quantization for HPU -- MiniMax-M3-MXFP8.

The checkpoint stores ``float8_e4m3fn`` weights with per ``[1, 32]``-input-block
scales encoded as ``uint8`` E8M0 exponents (bias 127), i.e.
``real = fp8 * 2 ** (scale_u8 - 127)``.  This is the same block-FP8 dequant the
plugin already runs for DeepSeek / GLM block-FP8, only with a ``[1, 32]`` block
(vs ``[128, 128]``) and an E8M0 scale (vs f32).

WHY [1,32] "just works" on the plugin's HPU block-FP8 path (investigated, with
file:line evidence, per Root-Cause-Depth):

  * ``vllm_gaudi.ops.hpu_fp8.Fp8LinearMethod.apply`` (hpu_fp8.py:114-124) routes
    block-quant linears to ``hpu_ops.apply_block_fp8_linear_hpu`` passing
    ``block_size=self.quant_config.weight_block_size`` -- NOT a hardcoded
    ``[128,128]``.
  * ``apply_block_fp8_linear_hpu`` (extension/ops.py:853-879, default
    ``force_channel_fp8=False``) -> ``apply_block_fp8_linear_hpu_dequant``
    (ops.py:882-903) -> ``dequant_block_fp8_weight_naive`` (ops.py:814-850) then
    a plain BF16 ``torch.nn.functional.linear``.
  * ``dequant_block_fp8_weight_naive`` is fully GENERAL over ``block_size``: it
    views the weight as ``(scale_m, block_m, scale_n, block_n)`` and broadcasts
    the scale.  For ``[1, 32]`` that is ``(M, 1, N/32, 32)`` -- correct MXFP8.

So the plugin DEQUANTS fp8->bf16 (it does not call a ``[128,128]``-hardcoded
block-scaled kernel).  On Gaudi 3 the FP8 MME runs at BF16 speed anyway, so the
dequant-to-BF16 path is the shipping path with no matmul-speed loss vs a native
block-FP8 GEMM.  MXFP8 therefore only needs, on top of the plugin's block-FP8
machinery:

  1. registering the ``mxfp8`` method name, and
  2. converting the loaded E8M0 scales to f32 multipliers (``2 ** (x - 127)``)
     BEFORE the block postprocess reads them.

The router gate, ``lm_head``, embeddings and the vision stack stay unquantized
(the checkpoint lists them in ``ignored_layers``).  The MoE experts are kept as
raw fp8 + f32 block scales here; ``minimax_m3.MiniMaxM3MoE`` dequants them on the
fly because it must apply ``swigluoai`` -- the plugin's fused-MoE HPU op
(``VllmMixtureOfExpertsOpFP8``, ops.py:1160) is silu-only.
"""
from typing import Any

import torch

from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import (
    QUANTIZATION_METHODS, register_quantization_config)
from vllm.model_executor.layers.quantization.fp8 import (Fp8Config,
                                                         Fp8KVCacheMethod)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped)

# Re-export the pure-torch helpers so callers can import them from either module
# (the standalone module keeps them importable without the HPU stack).
from vllm_gaudi.ops.mxfp8_dequant import e8m0_to_scale
from vllm_gaudi.ops.mxfp8_dequant import mxfp8_dequant_block  # noqa: F401
# The plugin's HPU FP8 overrides (importing this module also installs the
# fp8.Fp8LinearMethod / fp8.Fp8MoEMethod monkeypatches -- idempotent).
from vllm_gaudi.ops.hpu_fp8 import Fp8LinearMethod, HPUFp8MoEMethod


class MXFp8Config(Fp8Config):
    """Block-FP8 with ``[1, 32]`` blocks and E8M0 (``uint8``) scales."""

    @classmethod
    def get_name(cls) -> str:
        return "mxfp8"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MXFp8Config":
        # ``Fp8Config.from_config`` sets is_checkpoint_fp8_serialized from
        # ("fp8" in quant_method) -- true for "mxfp8" -- so block-quant activates
        # and weight_block_size=[1,32] is accepted (only a len==2 check upstream).
        instance = super().from_config(config)
        if instance.weight_block_size != [1, 32]:
            raise ValueError(
                "MXFp8Config assumes a [1, 32] weight block shape (the "
                "on-the-fly dequant this class runs is specialised to it), but "
                "the checkpoint's quantization_config declares "
                f"weight_block_size={instance.weight_block_size}.")
        return instance

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        # Import lazily (not at module scope): this module is imported from
        # vllm_gaudi.__init__ during the `vllm_gaudi:register` plugin hook, at
        # which point vllm.config is only partially initialized -- a module-scope
        # `from ...fused_moe import FusedMoE` pulls fused_moe/config.py's
        # `from vllm.config import ParallelConfig, SchedulerConfig` and raises a
        # circular ImportError, so the HPU platform never loads ("Failed to infer
        # device type"). get_quant_method runs at model-build time, long after
        # vllm.config is complete, so the lazy import is safe here.
        # NOTE: the pinned vLLM (ad7125a4) has NO top-level `vllm.attention`
        # package; the Attention layer lives under
        # `vllm.model_executor.layers.attention` (same path minimax_m3.py uses).
        # Importing `vllm.attention.layer` here raises ModuleNotFoundError when
        # get_quant_method runs for a quantized linear (e.g. qkv_proj), killing
        # model load.
        from vllm.model_executor.layers.attention import Attention
        from vllm.model_executor.layers.fused_moe import FusedMoE
        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix=prefix,
                                ignored_layers=self.ignored_layers,
                                fused_mapping=self.packed_modules_mapping):
                return UnquantizedLinearMethod()
            return MXFp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            # NOTE: the plugin's HPUFp8MoEMethod.__init__ takes (quant_config,
            # layer) -- unlike the fork's upstream Fp8MoEMethod(quant_config).
            return MXFp8MoEMethod(self, layer)
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None


class MXFp8LinearMethod(Fp8LinearMethod):
    """Block-FP8 linear; only converts the E8M0 scale before the fp8 postprocess.

    ``super().process_weights_after_loading`` is the plugin's HPU block-FP8
    postprocess (pad + store orig_M/orig_N); it reads ``weight_scale_inv`` as an
    f32 multiplier, so the E8M0 -> f32 conversion must happen first.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Convert E8M0 -> f32 once.  The scale conversion is not idempotent
        # (a second pass would compute 2**(2**(e-127) - 127) ~= 0 and zero the
        # weights), so guard it -- the plugin's opt-in multi-model reconfigure
        # hook can re-run process_weights on the same layer.
        scale = getattr(layer, "weight_scale_inv", None)
        if scale is not None and not getattr(layer, "_mxfp8_scale_converted", False):
            scale.data = e8m0_to_scale(scale.data)
            layer._mxfp8_scale_converted = True
        return super().process_weights_after_loading(layer)


class MXFp8MoEMethod(HPUFp8MoEMethod):
    """Stores experts as raw fp8 + f32 block scales for the model's unfused MoE.

    Unlike the parent, this does NOT run the HPU fused-MoE weight prep (which
    wires the silu-only ``torch.ops.hpu.mixture_of_experts`` op and builds
    ``layer.moe_op``).  MiniMax-M3 experts need ``swigluoai``, so
    ``MiniMaxM3MoE`` reads ``w13_weight`` / ``w2_weight`` and dequants them on
    the fly with the already-converted f32 scales.  The FusedMoE layer's own
    ``forward`` / ``apply_monolithic`` (which would use ``moe_op``) is never
    called for M3, so leaving ``moe_op`` unbuilt is safe (mirrors the fork).
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Convert E8M0 -> f32 once (see MXFp8LinearMethod for why this is guarded
        # rather than unconditional).
        if getattr(layer, "_mxfp8_scale_converted", False):
            return
        for name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
            scale = getattr(layer, name, None)
            if scale is not None:
                scale.data = e8m0_to_scale(scale.data)
        layer._mxfp8_scale_converted = True

        # Lever 1 (VLLM_M3_MOE_FP8_GEMV=1): re-quantize the routed experts from
        # [1,32] block-FP8 to PER-CHANNEL FP8 in place, so the decode path can run
        # torch.ops.hpu.fp8_gemm_v2 (reads FP8 weights ONCE, per-channel scale in
        # the MME epilogue) instead of materialising a bf16 dequant every step.
        # Overwrite w{13,2}_weight.data (same fp8 shape [E,*,*], new values) and
        # store the 1-D per-output-channel scale -> NO second resident weight copy
        # (2x MoE weights would OOM). Both prefill (per-channel dequant) and decode
        # (fp8_gemm_v2) then use the per-channel format; flag off = untouched.
        import os as _os
        if _os.environ.get("VLLM_M3_MOE_FP8_GEMV", "0") == "1" and \
                getattr(layer, "w13_weight", None) is not None and \
                layer.w13_weight.dtype == torch.float8_e4m3fn:
            from vllm_gaudi.extension.ops import (dequant_block_fp8_weight_naive, dynamic_quant)
            bs = self.quant_config.weight_block_size  # [1, 32]
            # Process w13 then w2 one at a time (bound the bf16 transient; both at
            # once would ~4x the transient and OOM at load).
            for wname, sname in (("w13_weight", "w13_weight_scale_ch"),
                                 ("w2_weight", "w2_weight_scale_ch")):
                w = getattr(layer, wname)
                bf16 = dequant_block_fp8_weight_naive(
                    w.data, getattr(layer, wname + "_scale_inv").data, bs)
                w_ch, s_ch = dynamic_quant(bf16)   # fp8 [E,*,K], scale [E,*,1]
                del bf16
                w.data.copy_(w_ch)
                del w_ch
                layer.register_parameter(
                    sname, torch.nn.Parameter(s_ch.squeeze(-1).contiguous(), requires_grad=False))
            layer._mxfp8_channel = True


# Register the HPU MXFP8 config under "mxfp8".  ``register_quantization_config``
# raises if the name is already present in ``QUANTIZATION_METHODS``; some
# upstream builds ship a built-in (CUDA/generic) "mxfp8".  If so, override the
# resolver map directly -- ``get_quantization_config`` merges the customized map
# last (quantization/__init__.py), so the plugin's HPU-backed config wins.
if "mxfp8" not in QUANTIZATION_METHODS:
    register_quantization_config("mxfp8")(MXFp8Config)
else:
    from vllm.model_executor.layers.quantization import (
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG)
    _CUSTOMIZED_METHOD_TO_QUANT_CONFIG["mxfp8"] = MXFp8Config
