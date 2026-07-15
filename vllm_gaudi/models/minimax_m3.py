# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The MiniMax AI team.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only MiniMax-M3 model (text path) for the vLLM-Gaudi plugin.

Ported from the EOL ``HabanaAI/vllm-fork`` ``models/minimax_m3.py`` onto the
plugin's native machinery, structured after ``vllm_gaudi/models/minimax_m2.py``.
Compared to MiniMax-M2 this adds:

  * dense vs. MoE layer selection driven by ``config.moe_layer_freq`` (layers
    0-2 dense, 3-59 sparse-MoE);
  * a shared expert alongside the routed experts (DeepSeek-style);
  * the clamped GPT-OSS ``swigluoai`` activation (implemented inline here to
    avoid depending on an upstream symbol the pinned vLLM may not export);
  * per-head QK-norm with Gemma-style RMSNorm (``use_gemma_norm``);
  * partial RoPE (``partial_rotary_factor=0.5``, ``rope_theta=5e6``).

M3's native MSA (block-sparse attention) is implemented model-side in
``minimax_m3_sparse.py`` and routed per layer by
``sparse_attention_config.sparse_attention_freq`` (layers 0-2 dense, 3-59
sparse).  ``VLLM_M3_MSA=0`` restores the previous behavior everywhere: dense
GQA (FusedSDPA on HPU) with the ``indexer`` / ``index_*`` weights skipped at
load, exactly like the fork.

HPU graph breaking: the plugin's native per-layer ``mark_step`` forward-hook
(``hpu_model_runner.modify_model_layers``, attached to every ``*DecoderLayer``)
replaces the fork's hand-written per-layer ``mark_step``.  The only explicit
break kept is INSIDE the dense-MoE chunk loop for very long single-shot
prefills (> 8192 tokens), which no per-layer hook can provide.
"""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear, QKVParallelLinear, ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (default_weight_loader, maybe_remap_kv_scale_name)
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import (EagleModelMixin, SupportsLoRA, SupportsPP)
from vllm.model_executor.models.utils import (AutoWeightsLoader, PPMissingLayer, is_pp_missing_parameter,
                                              make_empty_intermediate_tensors_factory, make_layers, maybe_prefix)

from vllm_gaudi.models.minimax_m3_sparse import (MiniMaxM3SparseAttentionMixin, msa_enabled, msa_layer_enabled)

logger = init_logger(__name__)

try:
    import habana_frameworks.torch.core as _htcore
except ImportError:  # non-HPU platforms
    _htcore = None

# Break the lazy graph INSIDE the dense MoE chunk loop only for very long SINGLE
# -shot prefills (> 8192) whose one-layer all-expert MoE would by itself exceed
# the ~30s Gaudi3 driver timeout.  A chunked-prefill 8192 chunk does not need
# this, and pure decode (tiny batch) never trips it.  The per-layer break is
# provided by the plugin's native forward-hook (see the module docstring).
_HPU_MARKSTEP_MIN_TOKENS = 8192

# ---------------------------------------------------------------------------
# PROFILING ABLATIONS (timing-only; break correctness -- NEVER ship enabled).
# The decode forward is captured inside an HPU graph, so a Python profiler sees
# only an opaque replay.  Attribution is therefore done by ABLATION: an env var
# read here at capture time removes a component from the graph; the warm-TPOT
# delta vs baseline measures that component's cost.  All default OFF.
#   VLLM_M3_ABLATE_MOE_EXPERTS=1  -> routed experts return 0 (skip gather+dequant+einsums)
#   VLLM_M3_ABLATE_MOE_ALLREDUCE=1-> skip the MoE forward's TP all-reduce (1 of 2/layer)
#   VLLM_M3_ABLATE_DEQUANT=1      -> skip fp8 scale-multiply (raw cast; isolates dequant math)
import os as _os
_ABLATE_MOE_EXPERTS = _os.environ.get("VLLM_M3_ABLATE_MOE_EXPERTS", "0") == "1"
_ABLATE_MOE_ALLREDUCE = _os.environ.get("VLLM_M3_ABLATE_MOE_ALLREDUCE", "0") == "1"
_ABLATE_DEQUANT = _os.environ.get("VLLM_M3_ABLATE_DEQUANT", "0") == "1"
#   VLLM_M3_ABLATE_ATTN=1  -> decoder layer skips self_attn entirely (removes attn
#                             compute + index scan + the o_proj TP all-reduce)
#   VLLM_M3_ABLATE_MOE=1   -> decoder layer skips the whole MoE/MLP block
_ABLATE_ATTN = _os.environ.get("VLLM_M3_ABLATE_ATTN", "0") == "1"
_ABLATE_MOE = _os.environ.get("VLLM_M3_ABLATE_MOE", "0") == "1"
# Lever 1 (perf): route the routed-expert GEMMs through the native
# torch.ops.hpu.fp8_gemm_v2 (reads FP8 weights ONCE, per-channel scale in the MME
# epilogue) at bs=1 decode, instead of materialising a bf16 dequant every step.
# Requires the per-channel FP8 requant done at load in MXFp8MoEMethod
# (VLLM_M3_MOE_FP8_GEMV=1 gates BOTH). Off = byte-identical block-FP8 path.
_MOE_FP8_GEMV = _os.environ.get("VLLM_M3_MOE_FP8_GEMV", "0") == "1"


class SwiGLUOAIAndMul(nn.Module):
    """Clamped GPT-OSS SwiGLU used by MiniMax-M3 (``hidden_act="swigluoai"``).

    Given a fused projection ``x`` whose last dim is ``[gate | up]`` (contiguous
    halves, as produced by ``MergedColumnParallelLinear`` / a merged ``w1``+``w3``
    FusedMoE shard), this computes::

        gate = clamp(gate, max=limit)
        up   = clamp(up, -limit, limit)
        glu  = gate * sigmoid(alpha * gate)
        out  = (up + 1) * glu

    The ``+1`` shift is load-bearing (dropping it degrades M3 to incoherent
    output).  Pure torch (clamp/sigmoid/mul), so it runs directly on HPU.
    Verbatim math from fork ``layers/activation.py`` ``SwiGLUOAIAndMul``.
    """

    def __init__(self, alpha: float = 1.702, limit: float = 7.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.limit = limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(self.alpha * gate)
        return (up + 1) * glu


def _build_norm(config: PretrainedConfig, hidden_size: int) -> nn.Module:
    """Pick Gemma-style RMSNorm or plain RMSNorm per config."""
    if getattr(config, "use_gemma_norm", True):
        return GemmaRMSNorm(hidden_size, eps=config.rms_norm_eps)
    return RMSNorm(hidden_size, eps=config.rms_norm_eps)


def _build_act(config: PretrainedConfig) -> nn.Module:
    """Activation for dense / shared / routed MLPs.

    M3 uses the clamped ``swigluoai`` SwiGLU; older ``silu`` checkpoints still
    work.
    """
    act_name = getattr(config, "hidden_act", "swigluoai")
    if act_name == "swigluoai":
        return SwiGLUOAIAndMul(
            alpha=getattr(config, "swiglu_alpha", 1.702),
            limit=getattr(config, "swiglu_limit", 7.0),
        )
    from vllm.model_executor.layers.activation import SiluAndMul
    return SiluAndMul()


def build_decoder_layer_types(moe_layer_freq: list[int]) -> list[str]:
    """Map ``config.moe_layer_freq`` (1 == MoE, 0 == dense) to per-layer types.

    Single source of truth for the dense-vs-MoE split; ``MiniMaxM3DecoderLayer``
    uses the same ``bool(moe_layer_freq[idx])`` predicate.  Unit-tested in
    ``tests/unit_tests/test_minimax_m3_layers.py``.
    """
    return ["moe" if freq else "dense" for freq in moe_layer_freq]


def _build_rope_parameters(config: PretrainedConfig) -> dict[str, Any]:
    """Build the ``get_rope`` ``rope_parameters`` dict (plugin's newer API).

    The plugin's ``get_rope`` takes a unified ``rope_parameters`` dict (see
    ``minimax_m2.HpuMiniMaxM2Attention``), whereas the M3 config exposes the
    older flat ``rope_theta`` / ``rope_scaling`` fields.  Prefer an explicit
    ``config.rope_parameters`` if present, else synthesize one.  A fresh dict is
    returned each call so per-layer ``partial_rotary_factor`` injection never
    mutates the shared config.
    """
    rope_parameters = getattr(config, "rope_parameters", None)
    if rope_parameters is not None:
        return dict(rope_parameters)
    rope_parameters = {
        "rope_type": "default",
        "rope_theta": getattr(config, "rope_theta", 10000),
    }
    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        rope_parameters.update(rope_scaling)
    return rope_parameters


class MiniMaxM3MLP(nn.Module):
    """Dense feed-forward block (also used as the shared expert).

    The gate/up projections are fused into a single ``[gate | up]`` column so
    the ``swigluoai`` activation can ``chunk(2)`` them.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
        act_fn: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        # Callers with an activation instance already built (e.g. MiniMaxM3MoE's
        # shared expert, which reuses the routed experts' act_fn) may pass it in
        # to avoid a redundant second instance; otherwise build one.
        self.act_fn = act_fn if act_fn is not None else _build_act(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MiniMaxM3MoE(nn.Module):
    """Sparse MoE block: routed experts + an optional shared expert."""

    # Use the gather-top-k (sparse) MoE path for batches up to this size (decode
    # and small prefills); larger prefills use the dense all-expert path to
    # avoid a [T, top_k, 2I, H] gather blowing up.  Matches vLLM's default
    # --max-num-seqs (128) so the sparse path stays valid at full default decode
    # concurrency.
    _SPARSE_DECODE_MAX = 128

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()

        if self.tp_size > config.num_local_experts:
            raise ValueError(f"Tensor parallel size {self.tp_size} is greater than "
                             f"the number of experts {config.num_local_experts}.")

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 2.0)
        self.n_shared_experts = getattr(config, "n_shared_experts", 1)
        self.top_k = config.num_experts_per_tok
        self.scoring_func = config.scoring_func
        # ``_unfused_experts`` hardcodes sigmoid routing; fail loud if a future
        # config declares a different scorer rather than routing it wrong.
        if self.scoring_func != "sigmoid":
            raise ValueError("MiniMaxM3MoE routing is sigmoid-only, but "
                             f"config.scoring_func={self.scoring_func!r}.")
        # M3 experts use the clamped GPT-OSS ``swigluoai`` activation.  The HPU
        # fused-MoE kernel only implements ``silu``, so the routed experts are
        # computed unfused (see ``_unfused_experts``) with this activation.
        self.expert_act_fn = _build_act(config)

        self.use_routing_bias = getattr(config, "use_routing_bias", False)
        if self.use_routing_bias:
            self.e_score_correction_bias = nn.Parameter(torch.empty(config.num_local_experts, dtype=torch.float32))
            self.e_score_correction_bias.weight_loader = (MiniMaxM3MoE.ebias_weight_loader)
        else:
            self.e_score_correction_bias = None

        self.experts = FusedMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            scoring_func=config.scoring_func,
            use_grouped_topk=True,
            num_expert_group=1,
            topk_group=1,
            e_score_correction_bias=self.e_score_correction_bias,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            renormalize=True,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )
        # NOTE: the pinned vLLM (ad7125a4) FusedMoE.__init__ dropped the
        # `reduce_results` kwarg (the newer API folds shared-expert + reduction
        # into FusedMoE natively).  It is irrelevant here anyway: M3 bypasses
        # FusedMoE.forward (experts are computed in _unfused_experts because the
        # HPU fused kernel is silu-only) and does its own single all-reduce in
        # forward(), so FusedMoE is used only for weight storage + routing state.

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        if self.n_shared_experts:
            # Shared expert output is summed with the routed output and the
            # combined tensor is all-reduced once below, so do not reduce here.
            self.shared_experts = MiniMaxM3MLP(
                config=config,
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_intermediate_size,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
                act_fn=self.expert_act_fn,
            )
        else:
            self.shared_experts = None

    @staticmethod
    def ebias_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight.to(torch.float32))

    @staticmethod
    def _dequant_block_experts(w_fp8: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Dequant block-FP8 weights: ``w * scale`` with a [1, blk] block along
        the last (input) dim.  Works for any leading dims, e.g. stacked experts
        ``[E, M, N]`` or gathered ``[T, top_k, M, N]``.  ``scale`` is the f32
        multiplier (E8M0 conversion happens once at load in ``MXFp8MoEMethod``);
        ``scale.shape[-1] == N // blk``.  This is the block-broadcast half of
        ``ops.hpu_mxfp8.mxfp8_dequant_block`` (which additionally converts E8M0).
        """
        n = w_fp8.shape[-1]
        nb = scale.shape[-1]
        blk = n // nb
        w = w_fp8.to(dtype).reshape(*w_fp8.shape[:-1], nb, blk)
        s = scale.to(dtype).unsqueeze(-1)
        return (w * s).reshape(w_fp8.shape)

    def _unfused_experts(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        """Routed-expert compute with the clamped ``swigluoai`` activation.

        The HPU fused-MoE kernel only implements ``silu``, but M3 needs the
        GPT-OSS ``swigluoai``.  We keep ``FusedMoE`` for weight storage + routing
        and apply the activation ourselves over the experts' stored
        intermediate-TP-sharded ``w13_weight`` / ``w2_weight`` (standard
        ``[E, 2I, H]`` / ``[E, H, I]`` layout on HPU).  Each rank produces a
        partial output that the all-reduce in ``forward`` sums.
        """
        # Sync-free sigmoid top-k routing.  M3's grouping is trivial
        # (num_expert_group=1, topk_group=1), so this equals
        # FusedMoE.select_experts(scoring_func="sigmoid", renormalize=True) but
        # WITHOUT grouped_topk -- whose HPU path breaks HPU-graph capture.
        # routed_scaling_factor is applied by the caller in forward().
        scores = torch.sigmoid(router_logits.to(torch.float32))  # [T, E]
        if self.e_score_correction_bias is not None:
            scores_for_choice = scores + self.e_score_correction_bias
        else:
            scores_for_choice = scores
        topk_ids = torch.topk(scores_for_choice, self.top_k, dim=-1)[1]
        topk_weights = torch.gather(scores, 1, topk_ids)  # [T, top_k]
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        w13 = self.experts.w13_weight  # [num_experts, 2 * I_tp, H]
        w2 = self.experts.w2_weight  # [num_experts, H, I_tp]
        fp8 = w13.dtype == torch.float8_e4m3fn
        dt = hidden_states.dtype
        n_tokens = hidden_states.shape[0]

        if n_tokens <= self._SPARSE_DECODE_MAX:
            # SPARSE (decode / small batch): gather + compute ONLY the top_k
            # selected experts per token.  At conc=1 this touches 4/128 experts
            # (32x less weight traffic + dequant) and is fixed-shape (T, top_k
            # constant per HPU-graph bucket), unlike the dense all-expert path.
            tk = self.top_k
            flat = topk_ids.reshape(-1)  # [T*top_k]

            def _gather(t):
                # index_select (graph-friendly) instead of advanced indexing
                return torch.index_select(t, 0, flat).view(n_tokens, tk, *t.shape[1:])

            if _ABLATE_MOE_EXPERTS:  # timing-only: skip all expert compute
                return torch.zeros(n_tokens, w13.shape[-1], dtype=dt, device=hidden_states.device)

            if _MOE_FP8_GEMV and fp8 and getattr(self.experts, "_mxfp8_channel", False):
                # Lever 1: per-channel FP8 experts via native torch.ops.hpu.fp8_gemm_v2
                # (reads FP8 once, per-channel scale in the MME epilogue -> no bf16
                # dequant materialisation). swigluoai stays in model code (exact).
                from vllm_gaudi.extension.ops import dynamic_quant
                w13_sel = _gather(w13)  # fp8 [T, top_k, 2I, H]
                w2_sel = _gather(w2)    # fp8 [T, top_k, H, I]
                s13 = _gather(self.experts.w13_weight_scale_ch)  # [T, top_k, 2I]
                s2 = _gather(self.experts.w2_weight_scale_ch)    # [T, top_k, H]
                if n_tokens == 1:
                    x_fp8, x_scale = dynamic_quant(hidden_states)  # [1,H] fp8, [1,1]
                    acc = None
                    for k in range(tk):
                        gu = torch.ops.hpu.fp8_gemm_v2(
                            A=x_fp8, trans_A=False, B=w13_sel[0, k], trans_B=True, D=None,
                            out_dtype=dt, A_scale_inv=x_scale, B_scale_inv=s13[0, k],
                            bias=None, accumulate=False)  # [1, 2I]
                        a = self.expert_act_fn(gu)  # [1, I] swigluoai (exact)
                        a_fp8, a_scale = dynamic_quant(a)
                        yk = torch.ops.hpu.fp8_gemm_v2(
                            A=a_fp8, trans_A=False, B=w2_sel[0, k], trans_B=True, D=None,
                            out_dtype=dt, A_scale_inv=a_scale, B_scale_inv=s2[0, k],
                            bias=None, accumulate=False)  # [1, H]
                        wk = topk_weights[:, k:k + 1] * yk
                        acc = wk if acc is None else acc + wk
                    # topk_weights is fp32; cast back to bf16 so this bs=1 path
                    # returns the same dtype as the T>1 / default paths (which
                    # do topk_weights.to(dt) first) -- else the residual stream
                    # silently upcasts to fp32 only at bs=1.
                    return acc.to(dt)
                # T>1 small-batch decode: per-channel dequant (correct, unaccelerated)
                w13_sel = w13_sel.to(dt) * s13.unsqueeze(-1)
                w2_sel = w2_sel.to(dt) * s2.unsqueeze(-1)
                gate_up = torch.einsum("th,tkoh->tko", hidden_states, w13_sel)
                act = self.expert_act_fn(gate_up)
                ye = torch.einsum("tki,tkhi->tkh", act, w2_sel)
                return torch.einsum("tk,tkh->th", topk_weights.to(dt), ye)

            w13_sel = _gather(w13)  # [T, top_k, 2I, H]
            w2_sel = _gather(w2)  # [T, top_k, H, I]

            # NOTE: a scale-factored MoE (pull the [1,32] block scale out of the
            # matmul onto the post-matmul partial to skip the bf16 dequant
            # materialisation) was implemented and REJECTED -- rigorous same-session
            # 4-rep A/B at true 256K showed it ~3 ms SLOWER (the extra reduction axis
            # on the per-block einsum costs more at bs=1 than the saved memory pass).
            # The dequant's ~7.4 ms (no_dequant ablation) is only recoverable by a
            # true read-fp8-once TPC GEMV, not a PyTorch reassociation.
            if fp8 and not _ABLATE_DEQUANT:
                w13_sel = self._dequant_block_experts(w13_sel, _gather(self.experts.w13_weight_scale_inv), dt)
                w2_sel = self._dequant_block_experts(w2_sel, _gather(self.experts.w2_weight_scale_inv), dt)
            elif fp8:  # timing-only: raw cast, no scale multiply (isolates dequant math)
                w13_sel = w13_sel.to(dt)
                w2_sel = w2_sel.to(dt)
            else:
                w13_sel = w13_sel.to(dt)
                w2_sel = w2_sel.to(dt)
            gate_up = torch.einsum("th,tkoh->tko", hidden_states, w13_sel)
            act = self.expert_act_fn(gate_up)  # [T, top_k, I]
            ye = torch.einsum("tki,tkhi->tkh", act, w2_sel)  # [T, top_k, H]
            return torch.einsum("tk,tkh->th", topk_weights.to(dt), ye)

        # DENSE (large prefill): compute every expert for every token (a gather
        # would be [T, top_k, 2I, H] = too large here).  Dequant all experts once.
        if fp8:
            if _MOE_FP8_GEMV and getattr(self.experts, "_mxfp8_channel", False):
                # weights were overwritten to per-channel fp8 at load -> per-channel dequant
                w13 = w13.to(dt) * self.experts.w13_weight_scale_ch.unsqueeze(-1)
                w2 = w2.to(dt) * self.experts.w2_weight_scale_ch.unsqueeze(-1)
            else:
                w13 = self._dequant_block_experts(w13, self.experts.w13_weight_scale_inv, dt)
                w2 = self._dequant_block_experts(w2, self.experts.w2_weight_scale_inv, dt)
        num_experts = w13.shape[0]
        gates = torch.zeros(n_tokens, num_experts, dtype=topk_weights.dtype, device=hidden_states.device)
        gates.scatter_(1, topk_ids.long(), topk_weights)
        gates = gates.to(dt)
        chunk = 1024  # bound the [E, chunk, 2I] activation transient
        # On very long prefills (256K) break the lazy graph every chunk so no
        # single HPU submission exceeds the ~30s Gaudi3 timeout.  Each chunk
        # output is a FRESH tensor collected into ``parts`` and cat'd once at the
        # end -- do NOT in-place-write a preallocated ``out`` here (mark_step on a
        # partial ``out`` gives it a multi-segment lazy history the downstream TP
        # all_reduce cannot bind to: ValidateSyncInputTensors tensor_data empty).
        markstep = _htcore is not None and n_tokens > _HPU_MARKSTEP_MIN_TOKENS
        parts = []
        for s in range(0, n_tokens, chunk):
            x = hidden_states[s:s + chunk]
            gate_up = torch.einsum("th,eoh->eto", x, w13)
            act = self.expert_act_fn(gate_up)
            ye = torch.einsum("eti,ehi->eth", act, w2)
            parts.append(torch.einsum("te,eth->th", gates[s:s + chunk], ye))
            if markstep:
                _htcore.mark_step()
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (bs * seq_len, n_experts)
        router_logits, _ = self.gate(hidden_states.to(torch.float32))
        final_hidden_states = self._unfused_experts(hidden_states, router_logits) * self.routed_scaling_factor

        if self.shared_experts is not None:
            final_hidden_states = final_hidden_states + self.shared_experts(hidden_states)

        if self.tp_size > 1 and not _ABLATE_MOE_ALLREDUCE:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(orig_shape)


class MiniMaxM3Attention(nn.Module):
    """GQA with partial RoPE and per-head QK-norm (same shape as M2).

    Dense path: full causal GQA (FusedSDPA on HPU).  Used for layers 0-2 and,
    with ``VLLM_M3_MSA=0``, for every layer (the pre-MSA fallback).  Sparse
    layers use the ``MiniMaxM3SparseAttention`` subclass below.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rotary_dim: int,
        rope_parameters: dict[str, Any] | None = None,
        max_position_embeddings: int = 8192,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        use_gemma_norm: bool = True,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % self.tp_size == 0
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= self.tp_size:
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            assert self.tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        # Per-head QK norm (qk_norm_type="per_head"): one [head_dim] weight
        # applied independently per head.  M3 is Gemma-normed throughout, so the
        # QK norm is the (1+w) Gemma form too -- the checkpoint q/k_norm weights
        # are centred near 0, which under plain x*w would damp q/k to ~10% and
        # flatten attention into gibberish.
        qk_norm_cls = GemmaRMSNorm if use_gemma_norm else RMSNorm
        self.q_norm = qk_norm_cls(self.head_dim, eps=rms_norm_eps)
        self.k_norm = qk_norm_cls(self.head_dim, eps=rms_norm_eps)

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Partial RoPE via the plugin's rope_parameters API: express the 0.5
        # factor as partial_rotary_factor = rotary_dim / head_dim (= 64/128),
        # matching HpuMiniMaxM2Attention.
        if (rope_parameters is not None and "partial_rotary_factor" not in rope_parameters):
            rope_parameters = {**rope_parameters, "partial_rotary_factor": rotary_dim / self.head_dim}
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Per-head QK RMSNorm: flatten heads into the row dimension so each
        # head_dim-vector is normalised independently against the shared
        # [head_dim] weight, then restore the original [..., n_heads*head_dim].
        q = self.q_norm(q.reshape(-1, self.head_dim)).view_as(q)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view_as(k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class MiniMaxM3SparseAttention(MiniMaxM3SparseAttentionMixin, MiniMaxM3Attention):
    """MSA (block-sparse) attention for layers with sparse_attention_freq != 0.

    Same projections/norms/rope as the dense class; ``self.attn`` (the
    standard ``Attention`` layer) is KEPT so the runner allocates and binds
    this layer's KV cache, but the mixin's forward writes the cache itself
    (replicating the deployed impl's write) and computes block-sparse
    attention from it instead of calling ``self.attn`` -- see
    ``minimax_m3_sparse.MiniMaxM3SparseAttentionMixin``.
    """

    def __init__(self,
                 config: PretrainedConfig,
                 *,
                 quant_config: QuantizationConfig | None = None,
                 prefix: str = "",
                 **kwargs) -> None:
        MiniMaxM3Attention.__init__(self, quant_config=quant_config, prefix=prefix, **kwargs)
        self._init_msa(config, quant_config, prefix)


class MiniMaxM3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        model_config: ModelConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        if hasattr(config, "max_model_len") and isinstance(config.max_model_len, int):
            max_position_embeddings = max(max_position_embeddings, config.max_model_len)
        # make_layers passes the prefix with the layer's index appended.
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx

        # Per-layer MSA routing (sparse_attention_freq: layers 0-2 dense,
        # 3-59 sparse).  VLLM_M3_MSA=0 forces the dense fallback everywhere.
        use_msa = msa_layer_enabled(config, layer_idx)
        sparse_cfg = getattr(config, "sparse_attention_config", None)
        if (layer_idx == 0 and isinstance(sparse_cfg, dict) and sparse_cfg.get("use_sparse_attention")):
            if msa_enabled(config):
                logger.info("MiniMax-M3 MSA (sparse attention) enabled "
                            "(disable with VLLM_M3_MSA=0).")
            else:
                logger.warning("MiniMax-M3 MSA (sparse attention) disabled via "
                               "VLLM_M3_MSA=0; falling back to dense GQA.")

        head_dim = getattr(config, "head_dim", None)
        rotary_dim = getattr(config, "rotary_dim", None)
        if rotary_dim is None:
            _hd = head_dim or (config.hidden_size // config.num_attention_heads)
            rotary_dim = int(_hd * getattr(config, "partial_rotary_factor", 1.0))

        attn_kwargs = dict(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rotary_dim=rotary_dim,
            rope_parameters=_build_rope_parameters(config),
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            use_gemma_norm=getattr(config, "use_gemma_norm", True),
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=head_dim,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        if use_msa:
            self.self_attn = MiniMaxM3SparseAttention(config, **attn_kwargs)
        else:
            self.self_attn = MiniMaxM3Attention(**attn_kwargs)

        # Dense vs. MoE selection from moe_layer_freq (1 == MoE); see
        # build_decoder_layer_types for the same predicate.
        moe_layer_freq = getattr(config, "moe_layer_freq", None)
        is_moe = bool(moe_layer_freq[layer_idx]) if moe_layer_freq else True
        if is_moe:
            self.block_sparse_moe = MiniMaxM3MoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.mlp = None
        else:
            self.block_sparse_moe = None
            self.mlp = MiniMaxM3MLP(
                config=config,
                hidden_size=config.hidden_size,
                intermediate_size=config.dense_intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = _build_norm(config, config.hidden_size)
        self.post_attention_layernorm = _build_norm(config, config.hidden_size)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if not _ABLATE_ATTN:  # timing-only: skip attn (compute + index + o_proj all-reduce)
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
            )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        if _ABLATE_MOE:  # timing-only: skip the whole MoE/MLP block
            pass
        elif self.block_sparse_moe is not None:
            hidden_states = self.block_sparse_moe(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


@support_torch_compile
class HpuMiniMaxM3Model(nn.Module, EagleModelMixin):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        # The MiniMax-M3 checkpoint's top-level config is the VL wrapper
        # (MiniMaxM3VLConfig); the text-model fields (vocab_size, hidden_size,
        # num_hidden_layers, moe_layer_freq, rope_theta, partial_rotary_factor,
        # ...) live under `.text_config`, not at the top level.  get_text_config()
        # returns that text sub-config (and is a no-op returning the config itself
        # for a pure-text checkpoint), so text-only serve of the VL checkpoint
        # works without --hf-overrides.
        config = vllm_config.model_config.hf_config.get_text_config()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=None,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MiniMaxM3DecoderLayer(
                config,
                prefix,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = _build_norm(config, config.hidden_size)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(["hidden_states", "residual"],
                                                                                       config.hidden_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # EAGLE-3 aux hidden states: always empty -- the ForCausalLM wrapper no
        # longer advertises SupportsEagle3 (the aux-layer setters were never
        # forwarded), so the runner never populates the aux-layer set. The
        # EagleModelMixin hooks stay inert to keep the forward contract intact.
        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        # No per-layer mark_step here: the plugin's native forward-hook
        # (hpu_model_runner.modify_model_layers) attaches one to every
        # *DecoderLayer.
        for idx, layer in enumerate(self.layers[self.start_layer:self.end_layer]):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(aux_hidden_states, idx + 1, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return FusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Dense MLP and shared expert use the merged gate_up_proj; map the
        # individual checkpoint projections onto it.
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = self.get_expert_mapping()

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Tensors literally named "indexer" (DeepSeek-style) never exist
            # for M3; keep the defensive skip.
            if "indexer" in name:
                continue

            # MSA indexer weights -- index_{q,k}_proj (.weight and, under
            # MXFP8, .weight_scale_inv) and index_{q,k}_norm.weight -- exist
            # on sparse layers only (3-59 in the checkpoint).  Load them into
            # the layer's indexer submodule when that layer routes sparse;
            # otherwise skip, keeping the dense fallback byte-identical.
            # Loaded directly (not via stacked_params_mapping: its "q_proj"
            # matcher would mangle "index_q_proj").
            if ".self_attn.index_" in name:
                # AutoWeightsLoader strips the parent prefix, so names arrive
                # here as "layers.N...." (no leading "model."); split on the
                # dot-less form to cover both shapes.
                parts = name.split("layers.")
                layer_idx = int(parts[1].split(".")[0]) if len(parts) > 1 else -1
                if layer_idx < 0 or not msa_layer_enabled(self.config, layer_idx):
                    continue
                name = name.replace(".self_attn.index_", ".self_attn.indexer.index_")
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
                continue

            # MTP speculative-decode layers are skipped for the main model.
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # Experts use w1/w2/w3 and are handled by expert mapping; skip
                # any expert tensors that slip through the gate/up matcher.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    # Remap FP8 kv-scale names.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class HpuMiniMaxM3SparseForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        # VL wrapper config -> text sub-config (see HpuMiniMaxM3Model.__init__).
        config = vllm_config.model_config.hf_config.get_text_config()
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        if hasattr(vllm_config.model_config, "max_model_len"):
            self.config.max_model_len = vllm_config.model_config.max_model_len
        self.model = HpuMiniMaxM3Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=None)
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (self.model.make_empty_intermediate_tensors)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    @staticmethod
    def _strip_language_model_prefix(
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        # The checkpoint ships the VL architecture, so every text weight is keyed
        # under ``language_model.`` (e.g. ``language_model.model.layers.0...``,
        # ``language_model.lm_head.weight``).  This text ForCausalLM owns
        # ``model.*`` / ``lm_head.*``, so strip that prefix.  Vision keys
        # (``vision_tower.`` / ``multi_modal_projector.`` / ``patch_merge_mlp.``)
        # are NOT under ``language_model.``; they pass through unchanged and are
        # dropped by ``skip_prefixes`` below.
        lm = "language_model."
        for name, w in weights:
            yield (name[len(lm):] if name.startswith(lm) else name), w

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Text-only serve of the VL checkpoint (via an ``architectures`` override
        # to this class): remap the ``language_model.`` text-weight prefix, then
        # drop the vision-tower / projector weights this text model has no home
        # for (the VL wrapper owns them).
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["vision_tower.", "multi_modal_projector.", "patch_merge_mlp."],
        )
        return loader.load_weights(self._strip_language_model_prefix(weights))

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


# NOTE: The multimodal entry point ``MiniMaxM3SparseForConditionalGeneration``
# (vision tower + projector on top of this text model) lives in
# ``minimax_m3_vl.py``; it instantiates this class with ``prefix="language_model"``
# to absorb the checkpoint's ``language_model.`` weight-key prefix.


def get_spec_layer_idx_from_weight_name(config: PretrainedConfig, weight_name: str) -> int | None:
    if hasattr(config, "num_mtp_modules") and (config.num_mtp_modules > 0):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_mtp_modules):
            if weight_name.startswith(f"model.layers.{layer_idx + i}."):
                return layer_idx + i
    return None
