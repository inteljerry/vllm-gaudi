# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from vllm/model_executor/models/minimax_vl_01.py,
# vllm/model_executor/models/qwen2_vl.py and vllm/model_executor/models/clip.py.
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
"""Inference-only MiniMax-M3 VL (vision-language) model for the vLLM-Gaudi plugin.

Phase 7 enablement: wires the M3 vision tower on top of the text-only backbone
in ``minimax_m3.py`` so that ``image_url`` requests are served.

The vision stack is a hybrid (verified against the on-disk checkpoint and the
HF ``modular_minimax_m3_vl`` / SGLang references):

  * **Patch embed** -- Qwen2.5-VL style ``Conv3d`` over flattened
    ``[num_patches, 1176]`` pixel values (``1176 = 3 * 2 * 14 * 14``).
  * **Encoder** -- 32 CLIP-style blocks (separate ``q/k/v/out_proj`` *with bias*,
    ``mlp.fc1/fc2`` GELU, ``layer_norm1/2``, one ``pre_layrnorm``), but with
    the learned position embedding replaced by a custom **3-D RoPE**.
  * **Projector** -- a per-patch ``multi_modal_projector`` (GELU MLP, vision 1280
    -> text 6144) followed by a 2x2 ``patch_merge_mlp`` (groups four projected
    patches -> one text token).
  * **Integration** -- LLaVA-style scatter of the projected features into the
    text embeddings at ``image_token_index`` (200025).  The text config carries
    no ``rope_scaling``, so there is **no M-RoPE**: placeholders take plain
    sequential positions and 3-D RoPE lives only inside the vision tower.

Weight names produced by the module hierarchy match the checkpoint exactly:
``vision_tower.vision_model.*``, ``multi_modal_projector.linear_{1,2}.*`` and
``patch_merge_mlp.linear_{1,2}.*`` (all top level, all BF16 -- they are in the
MXFP8 ``ignored_layers`` so the tower/projector run unquantized).  The text
backbone is keyed under ``language_model.*``.

Plugin adaptations vs. the HabanaAI/vllm-fork source:
  * ``self.language_model`` is the plugin's ``HpuMiniMaxM3SparseForCausalLM``
    (from ``vllm_gaudi.models.minimax_m3``), whose API differs from the fork's
    text model: it exposes ``embed_input_ids`` (no ``get_input_embeddings``) and
    ``compute_logits(hidden_states)`` (no ``sampling_metadata``).
  * The checkpoint's remote-code image/HF processor top-imports
    ``torchvision``, which is UNINSTALLED in this Gaudi image, so
    ``get_image_processor`` returns a lightweight hardcoded attribute holder
    instead of loading remote code.  See the notes on ``get_hf_processor``.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal, Optional, TypedDict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalFieldConfig,
                                    MultiModalKwargsItems)
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
# pinned vLLM (ad7125a4): BaseDummyInputsBuilder lives under
# vllm.multimodal.processing (there is no vllm.multimodal.profiling module).
from vllm.multimodal.processing import (BaseDummyInputsBuilder,
                                        BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import (MultiModalEmbeddings,
                                                   SupportsMultiModal,
                                                   SupportsPP)
# NOTE: pinned vLLM exposes NO public merge_multimodal_embeddings; the
# SupportsMultiModal.embed_input_ids default handles the text-embed + scatter
# via the private _merge_multimodal_embeddings, so this model does not import it.
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix

from vllm_gaudi.models.minimax_m3 import HpuMiniMaxM3SparseForCausalLM


# Placeholder / bracket token strings used by the MiniMax-M3 VL processor.
# The processor expands a single ``IMAGE_TOKEN`` into
# ``IMAGE_START_TOKEN`` + ``IMAGE_TOKEN`` * n + ``IMAGE_END_TOKEN`` where
# ``n = grid_t * grid_h * grid_w // spatial_merge_size ** 2``.
IMAGE_TOKEN = "]<]image[>["
IMAGE_START_TOKEN = "]<]start of image[>["
IMAGE_END_TOKEN = "]<]end of image[>["

# MiniMax-M3 vision tower defaults (verified against the checkpoint config).
_M3_VISION_DEFAULTS = dict(
    hidden_size=1280,
    intermediate_size=5120,
    num_hidden_layers=32,
    num_attention_heads=16,
    num_channels=3,
    patch_size=14,
    temporal_patch_size=2,
    spatial_merge_size=2,
    layer_norm_eps=1e-5,
    hidden_act="gelu",
    rope_theta=10000.0,
)

try:
    import habana_frameworks.torch.core as _htcore
except ImportError:  # non-HPU platforms
    _htcore = None

# Break the HPU lazy graph across the 32 vision encoder layers for the same
# device-submission-timeout reason as the text decoder's _HPU_MARKSTEP_MIN_TOKENS
# (minimax_m3.py): an unbroken graph over all layers can overflow the lazy-tensor
# pool or exceed the ~30s Gaudi3 timeout. The vision gate uses its own patch-count
# threshold (below) since a vision pass is a single prefill-like burst.
_HPU_VISION_MARKSTEP_MIN_PATCHES = 512


# === Vision configuration ================================================== #


@dataclass
class MiniMaxM3VisionParams:
    """Normalised vision hyper-parameters.

    Built once from the (possibly nested / dict-valued) HF config so the vision
    modules read a single flat object.  Every field has a verified M3 default so
    the tower is constructible even from a bare config.
    """

    hidden_size: int = 1280
    intermediate_size: int = 5120
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_channels: int = 3
    patch_size: int = 14
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    layer_norm_eps: float = 1e-5
    hidden_act: str = "gelu"
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def _get(obj: object, key: str, default: object = None) -> object:
    """Read ``key`` from an object attribute or a dict, else ``default``."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def build_vision_params(hf_config: object) -> MiniMaxM3VisionParams:
    """Extract :class:`MiniMaxM3VisionParams` from the HF config.

    ``spatial_merge_size`` / ``temporal_patch_size`` live under the
    ``img_token_compression_config`` block (on the vision config and/or the top
    level); ``rope_theta`` is read from the vision config's rope fields.
    """
    vc = _get(hf_config, "vision_config")
    d = dict(_M3_VISION_DEFAULTS)

    for key in ("hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_channels", "patch_size",
                "layer_norm_eps", "hidden_act"):
        val = _get(vc, key)
        if val is not None:
            d[key] = val

    rope_theta = _get(vc, "rope_theta")
    if rope_theta is None:
        rope_params = _get(vc, "rope_parameters")
        rope_theta = _get(rope_params, "rope_theta")
    if rope_theta is not None:
        d["rope_theta"] = float(rope_theta)

    # img_token_compression_config may sit on the vision or top-level config.
    compression = (_get(vc, "img_token_compression_config")
                   or _get(hf_config, "img_token_compression_config") or {})
    for key in ("spatial_merge_size", "temporal_patch_size"):
        val = _get(compression, key)
        if val is not None:
            d[key] = val

    return MiniMaxM3VisionParams(**d)


# === 3-D RoPE ============================================================== #


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension of ``x`` by half (GPT-NeoX style)."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 3-D RoPE to the leading ``rot_dim`` head channels of ``q``/``k``.

    ``q``/``k`` are ``[num_tokens, num_heads, head_dim]``; ``cos``/``sin`` are
    ``[num_tokens, rot_dim]`` (``rot_dim`` may be < ``head_dim`` -- the
    tail channels pass through unrotated).  The rotary math runs in fp32 for
    numerical parity with the reference, then casts back.
    """
    orig_dtype = q.dtype
    rot_dim = cos.shape[-1]
    cos = cos.unsqueeze(-2).float()  # [num_tokens, 1, rot_dim]
    sin = sin.unsqueeze(-2).float()

    q_rot, q_pass = q[..., :rot_dim].float(), q[..., rot_dim:]
    k_rot, k_pass = k[..., :rot_dim].float(), k[..., rot_dim:]

    q_rot = q_rot * cos + rotate_half(q_rot) * sin
    k_rot = k_rot * cos + rotate_half(k_rot) * sin

    q = torch.cat((q_rot.to(orig_dtype), q_pass), dim=-1)
    k = torch.cat((k_rot.to(orig_dtype), k_pass), dim=-1)
    return q, k


def compute_vision_3d_rope(
    grid_thw: Sequence[Sequence[int]],
    head_dim: int,
    theta: float,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """3-D RoPE cos/sin for the vision patches, computed on the HOST (CPU).

    ``2 * (head_dim // 2)`` rotary dims are split evenly across ``(T, H, W)``
    axes (each rounded down to a multiple of 2); the remaining head dims pass
    through unrotated.  ``(H, W)`` coords are reordered into
    ``spatial_merge_size`` blocks so the patch order matches the processor
    patchify order and the projector's ``reshape(N // merge**2, ...)`` merge.

    This is computed in the multi-modal processor and passed to the tower as
    tensors (not recomputed inside the model) so the tower forward stays free of
    host-dependent ops (``.tolist()``, Python loops) and is safe to run under
    HPU-graph capture.  Returns ``(cos, sin)`` each ``[num_patches, rot_dim]``,
    fp32.
    """
    m = spatial_merge_size
    rope_dims = 2 * (head_dim // 2)
    axis_dim = 2 * ((rope_dims // 3) // 2)
    coords = []
    for t, h, w in grid_thw:
        t, h, w = int(t), int(h), int(w)
        hi = torch.arange(h).unsqueeze(1).expand(-1, w)
        hi = hi.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
        wi = torch.arange(w).unsqueeze(0).expand(h, -1)
        wi = wi.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
        ti = torch.arange(t).repeat_interleave(h * w)
        coords.append(torch.stack([ti, hi.repeat(t), wi.repeat(t)], dim=-1))
    coords = torch.cat(coords).to(torch.float32)

    inv_freq = 1.0 / (theta**(torch.arange(0, axis_dim, 2, dtype=torch.float32)
                              / axis_dim))
    freqs = torch.cat([coords[:, i:i + 1] * inv_freq for i in range(3)], dim=-1)
    emb = torch.cat([freqs, freqs], dim=-1)
    # fp32 cos/sin: the applier upcasts anyway and low-precision cos/sin hurt
    # RoPE accuracy at high freqs on HPU bf16.
    return emb.cos(), emb.sin()


# === Vision tower ========================================================== #


class MiniMaxM3VisionEmbeddings(nn.Module):
    """Conv3d patch embed over flattened ``[num_patches, C*T*P*P]`` input."""

    def __init__(self, params: MiniMaxM3VisionParams) -> None:
        super().__init__()
        self.num_channels = params.num_channels
        self.temporal_patch_size = params.temporal_patch_size
        self.patch_size = params.patch_size
        self.embed_dim = params.hidden_size

        kernel_size = (self.temporal_patch_size, self.patch_size,
                       self.patch_size)
        self.patch_embedding = nn.Conv3d(self.num_channels,
                                         self.embed_dim,
                                         kernel_size=kernel_size,
                                         stride=kernel_size,
                                         bias=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        num_patches = pixel_values.shape[0]
        pixel_values = pixel_values.view(num_patches, self.num_channels,
                                         self.temporal_patch_size,
                                         self.patch_size, self.patch_size)
        patch_embeds = self.patch_embedding(pixel_values.to(target_dtype))
        return patch_embeds.view(num_patches, self.embed_dim)


class MiniMaxM3VisionAttention(nn.Module):
    """CLIP-style attention with 3-D RoPE and a single masked SDPA.

    Separate ``q/k/v/out_proj`` (with bias) keep param names identical to
    the checkpoint (no fused ``qkv``).  Attention is a single
    ``scaled_dot_product_attention`` over all patches with a block-diagonal
    FLOAT ADDITIVE ``attn_mask`` (0.0 = attend, -inf = block), so each image
    attends only to its own patches.  A boolean mask is NOT used: HPU's SDPA
    mishandles boolean masks and silently degrades attention (a red image was
    described as "purple").

    The per-image grouping is built in ``_process_image_input`` from
    ``image_grid_thw`` (image-intrinsic, cached correctly), so distinct images
    are always isolated -- whether within one request or co-batched from
    different requests. (An earlier design keyed the mask off a per-request-local
    positional index emitted by the processor; that field was cached per image,
    so a warm-cache multi-image request collapsed distinct images into one group
    -> cross-image contamination. See the MiniMax-M3 multimodal report.)
    """

    def __init__(
        self,
        params: MiniMaxM3VisionParams,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_dim = params.hidden_size
        self.num_heads = params.num_attention_heads
        self.head_dim = params.head_dim
        self.scale = self.head_dim**-0.5

        tp_size = get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0, (
            f"num_heads ({self.num_heads}) must be divisible by the "
            f"tensor-parallel size ({tp_size})")
        self.num_heads_per_partition = self.num_heads // tp_size

        self.q_proj = ColumnParallelLinear(self.embed_dim,
                                           self.num_heads * self.head_dim,
                                           bias=True,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.q_proj")
        self.k_proj = ColumnParallelLinear(self.embed_dim,
                                           self.num_heads * self.head_dim,
                                           bias=True,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.k_proj")
        self.v_proj = ColumnParallelLinear(self.embed_dim,
                                           self.num_heads * self.head_dim,
                                           bias=True,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.v_proj")
        self.out_proj = RowParallelLinear(self.num_heads * self.head_dim,
                                          self.embed_dim,
                                          bias=True,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.out_proj")

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q = q.view(num_tokens, self.num_heads_per_partition, self.head_dim)
        k = k.view(num_tokens, self.num_heads_per_partition, self.head_dim)
        v = v.view(num_tokens, self.num_heads_per_partition, self.head_dim)

        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # One masked SDPA over all patches (no host-side loop -> safe under
        # HPU-graph capture). ``attn_mask`` is a precomputed float additive
        # [num_tokens, num_tokens] block-diagonal bias (0.0 = attend, -inf =
        # block); it broadcasts over the batch/head dims.
        # [tokens, heads, dim] -> [1, heads, tokens, dim]
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                              scale=self.scale)
        # [1, heads, tokens, dim] -> [tokens, heads*dim]
        attn_output = attn.squeeze(0).transpose(0, 1).reshape(num_tokens, -1)
        output, _ = self.out_proj(attn_output)
        return output


class MiniMaxM3VisionMLP(nn.Module):

    def __init__(
        self,
        params: MiniMaxM3VisionParams,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.fc1 = ColumnParallelLinear(params.hidden_size,
                                        params.intermediate_size,
                                        bias=True,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.fc1")
        self.act = get_act_fn(params.hidden_act)
        self.fc2 = RowParallelLinear(params.intermediate_size,
                                     params.hidden_size,
                                     bias=True,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.fc2")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class MiniMaxM3VisionEncoderLayer(nn.Module):

    def __init__(
        self,
        params: MiniMaxM3VisionParams,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = MiniMaxM3VisionAttention(
            params, quant_config=quant_config, prefix=f"{prefix}.self_attn")
        self.layer_norm1 = nn.LayerNorm(params.hidden_size,
                                        eps=params.layer_norm_eps)
        self.mlp = MiniMaxM3VisionMLP(params,
                                      quant_config=quant_config,
                                      prefix=f"{prefix}.mlp")
        self.layer_norm2 = nn.LayerNorm(params.hidden_size,
                                        eps=params.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, attn_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MiniMaxM3VisionEncoder(nn.Module):

    def __init__(
        self,
        params: MiniMaxM3VisionParams,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            MiniMaxM3VisionEncoderLayer(params,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.layers.{i}")
            for i in range(params.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        markstep = (_htcore is not None and
                    hidden_states.shape[0] > _HPU_VISION_MARKSTEP_MIN_PATCHES)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, attn_mask)
            if markstep:
                _htcore.mark_step()
        return hidden_states


class MiniMaxM3VisionTransformer(nn.Module):
    """Conv3d patch embed -> ``pre_layrnorm`` -> 32 encoder layers.

    Feature layer is ``-1`` with the ``"full"`` strategy (no CLS, no post-norm),
    matching the checkpoint (which ships no ``post_layernorm`` weight).
    """

    def __init__(
        self,
        params: MiniMaxM3VisionParams,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.params = params
        self.spatial_merge_size = params.spatial_merge_size

        self.embeddings = MiniMaxM3VisionEmbeddings(params)
        # NOTE: the "layrnorm" typo matches the checkpoint weight names -- keep.
        self.pre_layrnorm = nn.LayerNorm(params.hidden_size,
                                         eps=params.layer_norm_eps)
        self.encoder = MiniMaxM3VisionEncoder(params,
                                              quant_config=quant_config,
                                              prefix=f"{prefix}.encoder")

    @property
    def dtype(self) -> torch.dtype:
        return self.embeddings.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.embeddings.patch_embedding.weight.device

    def forward(self, pixel_values: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor,
                attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # cos/sin (3-D RoPE) and attn_mask (block-diagonal) are precomputed in
        # the multi-modal processor and passed in as tensors, so this forward is
        # pure tensor ops -> safe under HPU-graph capture.
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)
        return self.encoder(hidden_states, cos, sin, attn_mask)


class MiniMaxM3VisionModel(nn.Module):
    """``vision_tower`` container -- holds the ``vision_model`` transformer."""

    def __init__(
        self,
        params: MiniMaxM3VisionParams,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.vision_model = MiniMaxM3VisionTransformer(
            params,
            quant_config=quant_config,
            prefix=f"{prefix}.vision_model")

    @property
    def dtype(self) -> torch.dtype:
        return self.vision_model.dtype

    def forward(self, pixel_values: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor,
                attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        return self.vision_model(pixel_values, cos, sin, attn_mask)


# === Projector + patch merger ============================================== #


class MiniMaxM3ProjectorMLP(nn.Module):
    """Shared two-layer ``linear_1 -> act -> linear_2`` MLP body used by both
    the multimodal projector and the patch merger below (they differ only in
    input dim and, for the merger, a reshape before this body runs)."""

    def __init__(
        self,
        input_size: int,
        text_hidden_size: int,
        projector_hidden_size: int,
        projector_hidden_act: str,
        bias: bool,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.linear_1 = ColumnParallelLinear(input_size,
                                             projector_hidden_size,
                                             bias=bias,
                                             quant_config=quant_config,
                                             prefix=f"{prefix}.linear_1")
        self.act = get_act_fn(projector_hidden_act)
        self.linear_2 = RowParallelLinear(projector_hidden_size,
                                          text_hidden_size,
                                          bias=bias,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.linear_2")

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class MiniMaxM3MultiModalProjector(MiniMaxM3ProjectorMLP):
    """Per-patch GELU MLP: vision ``hidden_size`` -> text ``hidden_size``."""

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        projector_hidden_size: int,
        projector_hidden_act: str,
        bias: bool,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            input_size=vision_hidden_size,
            text_hidden_size=text_hidden_size,
            projector_hidden_size=projector_hidden_size,
            projector_hidden_act=projector_hidden_act,
            bias=bias,
            quant_config=quant_config,
            prefix=prefix,
        )


class MiniMaxM3PatchMerger(MiniMaxM3ProjectorMLP):
    """2x2 spatial merge: groups ``spatial_merge_size**2`` patches into the
    channel dim, then a GELU MLP back to text ``hidden_size``."""

    def __init__(
        self,
        text_hidden_size: int,
        projector_hidden_size: int,
        spatial_merge_size: int,
        projector_hidden_act: str,
        bias: bool,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        merged_hidden_size = text_hidden_size * (spatial_merge_size**2)
        super().__init__(
            input_size=merged_hidden_size,
            text_hidden_size=text_hidden_size,
            projector_hidden_size=projector_hidden_size,
            projector_hidden_act=projector_hidden_act,
            bias=bias,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.spatial_merge_size = spatial_merge_size

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        image_features = image_features.reshape(
            image_features.shape[0] // (self.spatial_merge_size**2), -1)
        return super().forward(image_features)


# === Multi-modal processing ================================================ #


class MiniMaxM3VLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor
    """Flattened patches, shape ``(num_patches_total, C * T * P * P)``."""
    image_grid_thw: torch.Tensor
    """Per-image ``(t, h, w)`` grid, shape ``(num_images, 3)``."""
    vision_cos: torch.Tensor
    """3-D RoPE cos, shape ``(num_patches_total, rot_dim)`` (precomputed)."""
    vision_sin: torch.Tensor
    """3-D RoPE sin, shape ``(num_patches_total, rot_dim)`` (precomputed)."""


class MiniMaxM3VLImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    image_embeds: torch.Tensor
    image_grid_thw: torch.Tensor


MiniMaxM3VLImageInputs = Union[MiniMaxM3VLImagePixelInputs,
                               MiniMaxM3VLImageEmbeddingInputs]


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Qwen2-VL / MiniMax-M3 image resize: round each side to a multiple of
    ``factor`` while keeping the area within ``[min_pixels, max_pixels]``."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _preprocess_image_native(
    image: object,
    *,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
    max_pixels: int,
    image_mean: Sequence[float],
    image_std: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen2-VL-style image preprocessing, ported verbatim from the checkpoint's
    ``image_processor.py`` and reimplemented natively (torchvision-free).

    The checkpoint's shipped HF image processor targets a newer ``transformers``
    than this serving image (and top-imports ``torchvision``, which is not
    installed on the Habana torch fork), so we reproduce the exact
    ``smart_resize`` + patchify pipeline with numpy + ``F.interpolate``. Returns
    ``(pixel_values [N, C*T*P*P], grid_thw [3])``, the same patch order the 3-D
    RoPE and projector expect (2x2 merge blocks contiguous).
    """
    import numpy as np

    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)  # [H, W, 3]
    pixels = torch.from_numpy(arr).permute(2, 0, 1).to(torch.float32)
    channel, height, width = pixels.shape

    factor = patch_size * merge_size
    resized_h, resized_w = smart_resize(height, width, factor=factor,
                                        min_pixels=4 * factor * factor,
                                        max_pixels=max_pixels)
    pixels = F.interpolate(pixels.unsqueeze(0), size=(resized_h, resized_w),
                           mode="bicubic", align_corners=False,
                           antialias=True).squeeze(0)
    pixels = pixels / 255.0
    mean = torch.tensor(image_mean, dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor(image_std, dtype=torch.float32).view(-1, 1, 1)
    pixels = (pixels - mean) / std  # [C, resized_h, resized_w]

    grid_t = 1
    grid_h = resized_h // patch_size
    grid_w = resized_w // patch_size

    # One image = one frame; pad to temporal_patch_size by repeating the frame
    # (matches the checkpoint's "repeat last frame" temporal padding).
    frames = pixels.unsqueeze(0).repeat(temporal_patch_size, 1, 1, 1)
    patches = frames.reshape(1, grid_t, temporal_patch_size, channel,
                             grid_h // merge_size, merge_size, patch_size,
                             grid_w // merge_size, merge_size, patch_size)
    patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    flatten = patches.reshape(
        grid_t * grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size)
    grid_thw = torch.tensor([grid_t, grid_h, grid_w], dtype=torch.long)
    return flatten, grid_thw


class MiniMaxM3VLProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self) -> object:
        # The pinned vLLM has no ``MiniMaxM3VLConfig`` class to type against, so
        # return the untyped remote-code config (loaded via trust_remote_code).
        # It carries ``vision_config``, ``text_config``, ``image_token_index``,
        # ``projector_hidden_size``/``_act``, ``multimodal_projector_bias`` etc.
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object):
        # INTEGRATION RISK (verify on HPU): the checkpoint's ``MiniMaxVLProcessor``
        # (AutoProcessor) top-imports ``torchvision`` (via its image/video
        # sub-processors), which is UNINSTALLED on this Habana torch fork, so
        # ``self.ctx.get_hf_processor()`` raises ModuleNotFoundError at import.
        # It is NOT used for preprocessing (done natively in ``_call_hf_processor``
        # with only the tokenizer + ``get_image_processor`` + ``get_hf_config``),
        # but the vLLM base processing machinery may still call this. If the base
        # machinery invokes it we return a minimal stand-in exposing the one
        # attribute typically read (``image_token``); otherwise the underlying
        # error is surfaced. MUST be verified on HPU -- if the base machinery
        # requires a richer HF processor object this stand-in will need fields
        # added (see notes).
        try:
            return self.ctx.get_hf_processor(**kwargs)
        except Exception:
            return SimpleNamespace(image_token=IMAGE_TOKEN)

    def get_image_processor(self, **kwargs: object):
        # Do NOT load the checkpoint's remote-code image processor: its module
        # top-imports ``torchvision`` (uninstalled here) so it cannot be
        # imported. The fork only ever reads attributes off this object
        # (patch_size / temporal_patch_size / merge_size / max_pixels /
        # min_pixels / image_mean / image_std) -- its ``__call__`` is never
        # invoked (native preprocessing is used) -- so a hardcoded attribute
        # holder is a drop-in. Constants come from the checkpoint's
        # image_processor.py defaults and config.json.
        return SimpleNamespace(
            patch_size=14,
            temporal_patch_size=2,
            merge_size=2,
            max_pixels=451584,
            min_pixels=4 * 28 * 28,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
        )

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        # Video is deferred to a later phase -- images only for now.
        return {"image": None}

    def _resize_params(self, image_processor: object) -> tuple[int, int, int]:
        patch_size = getattr(image_processor, "patch_size", 14)
        merge_size = getattr(image_processor, "merge_size", 2)
        factor = patch_size * merge_size
        max_pixels = getattr(image_processor, "max_pixels", 451584)
        min_pixels = getattr(image_processor, "min_pixels", 4 * factor * factor)
        return factor, min_pixels, max_pixels

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[ImageSize, int]:
        image_processor = self.get_image_processor()
        patch_size = getattr(image_processor, "patch_size", 14)
        merge_size = getattr(image_processor, "merge_size", 2)
        factor, min_pixels, max_pixels = self._resize_params(image_processor)

        resized_height, resized_width = smart_resize(
            height=image_height,
            width=image_width,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        grid_t = 1
        grid_h = resized_height // patch_size
        grid_w = resized_width // patch_size
        num_tokens = grid_t * grid_h * grid_w // (merge_size**2)
        return ImageSize(width=resized_width,
                         height=resized_height), num_tokens

    def get_num_image_tokens(self, *, image_width: int,
                             image_height: int) -> int:
        _, num_tokens = self._get_vision_info(image_width=image_width,
                                              image_height=image_height)
        return num_tokens

    def get_image_size_with_most_features(self) -> ImageSize:
        size, _ = self._get_vision_info(image_width=9999999,
                                        image_height=9999999)
        return size

    def get_max_image_tokens(self) -> int:
        target = self.get_image_size_with_most_features()
        return self.get_num_image_tokens(image_width=target.width,
                                         image_height=target.height)


class MiniMaxM3VLDummyInputsBuilder(
        BaseDummyInputsBuilder[MiniMaxM3VLProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        return IMAGE_TOKEN * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Optional[Mapping[str, object]] = None,
    ) -> Mapping[str, object]:
        # pinned vLLM's BaseDummyInputsBuilder calls this with a third
        # ``mm_options`` arg; accept it (unused -- no per-modality overrides).
        num_images = mm_counts.get("image", 0)
        target = self.info.get_image_size_with_most_features()
        return {
            "image":
            self._get_dummy_images(width=target.width,
                                   height=target.height,
                                   num_images=num_images),
        }


class MiniMaxM3VLMultiModalProcessor(
        BaseMultiModalProcessor[MiniMaxM3VLProcessingInfo]):

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        # We tokenize natively WITHOUT expanding the image token, so vLLM must
        # apply the placeholder expansion from ``_get_prompt_updates``.
        return False

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # The checkpoint's HF processor + fast image processor are incompatible
        # with this container's transformers version AND top-import torchvision
        # (uninstalled here), so both AutoProcessor and the image processor's
        # ``_preprocess`` raise. Preprocess natively.
        # pinned vLLM passes ``tok_kwargs`` (tokenization config, e.g.
        # truncation) as a 4th arg -- forward it to the tokenizer.
        tokenizer = self.info.get_tokenizer()
        input_ids = tokenizer(prompt,
                              add_special_tokens=False,
                              return_tensors="pt",
                              **tok_kwargs)["input_ids"]

        images = mm_data.get("images")
        if not images:
            return BatchFeature(data={"input_ids": input_ids})
        if not isinstance(images, (list, tuple)):
            images = [images]

        ip = self.info.get_image_processor(**mm_kwargs)
        params = dict(
            patch_size=getattr(ip, "patch_size", 14),
            temporal_patch_size=getattr(ip, "temporal_patch_size", 2),
            merge_size=getattr(ip, "merge_size", 2),
            max_pixels=getattr(ip, "max_pixels", 451584),
            image_mean=getattr(ip, "image_mean",
                               [0.48145466, 0.4578275, 0.40821073]),
            image_std=getattr(ip, "image_std",
                              [0.26862954, 0.26130258, 0.27577711]),
        )
        pv_list, grid_list = [], []
        for img in images:
            pv, grid = _preprocess_image_native(img, **params)
            pv_list.append(pv)
            grid_list.append(grid)

        # Precompute the 3-D RoPE cos/sin on the HOST and pass them as FLAT
        # mm-kwargs. The block-diagonal image grouping is derived at model time
        # from image_grid_thw (see _process_image_input), so no per-patch index
        # field is emitted here -- a cached positional field breaks multi-image
        # isolation under a warm mm-processor cache.
        vparams = build_vision_params(self.info.get_hf_config())
        cos_list, sin_list = [], []
        for grid in grid_list:
            c, s = compute_vision_3d_rope([grid.tolist()], vparams.head_dim,
                                          vparams.rope_theta,
                                          vparams.spatial_merge_size)
            cos_list.append(c)
            sin_list.append(s)

        return BatchFeature(data={
            "input_ids": input_ids,
            "pixel_values": torch.cat(pv_list, dim=0),
            "image_grid_thw": torch.stack(grid_list, dim=0),
            "vision_cos": torch.cat(cos_list, dim=0),
            "vision_sin": torch.cat(sin_list, dim=0),
        })

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        image_grid_sizes = image_grid_thw.prod(-1)
        # image_embeds (if present) holds POST-merge features -- one row per
        # placeholder token, same divisor as get_replacement below -- while
        # pixel_values/vision_cos/vision_sin are PRE-merge, one row per raw
        # vision patch.
        image_processor = self.info.get_image_processor(
            **hf_processor_mm_kwargs)
        merge_length = image_processor.merge_size**2
        embed_sizes = image_grid_sizes // merge_length
        flat = MultiModalFieldConfig.flat_from_sizes
        return dict(
            pixel_values=flat("image", image_grid_sizes),
            image_embeds=flat("image", embed_sizes),
            image_grid_thw=MultiModalFieldConfig.batched("image"),
            # per-patch 3-D RoPE (sized by t*h*w per image)
            vision_cos=flat("image", image_grid_sizes),
            vision_sin=flat("image", image_grid_sizes),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        image_processor = self.info.get_image_processor(
            **hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()

        # chat_utils.py independently builds the in-prompt image placeholder
        # by decoding hf_config.image_token_index, while the PromptReplacement
        # below matches on the hardcoded IMAGE_TOKEN literal. Assert they
        # agree so a future checkpoint/tokenizer mismatch fails loudly
        # instead of silently no-op-ing placeholder expansion.
        decoded_image_token = tokenizer.decode(hf_config.image_token_index)
        assert decoded_image_token == IMAGE_TOKEN, (
            f"image_token_index decodes to {decoded_image_token!r}, "
            f"expected IMAGE_TOKEN {IMAGE_TOKEN!r}")

        merge_length = image_processor.merge_size**2
        image_token_id = hf_config.image_token_index
        start_id = tokenizer.convert_tokens_to_ids(IMAGE_START_TOKEN)
        end_id = tokenizer.convert_tokens_to_ids(IMAGE_END_TOKEN)

        def get_replacement(item_idx: int):
            # pinned vLLM: out_mm_kwargs is MultiModalKwargsItems;
            # index [modality][item_idx][field].data (cf. upstream qwen2_vl).
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            num_tokens = int(grid_thw.prod()) // merge_length
            placeholder = ([start_id] + [image_token_id] * num_tokens +
                           [end_id])
            return PromptUpdateDetails.select_token_id(
                placeholder, embed_token_id=image_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=IMAGE_TOKEN,
                replacement=get_replacement,
            ),
        ]


# === Top-level VL model ==================================================== #


@MULTIMODAL_REGISTRY.register_processor(
    MiniMaxM3VLMultiModalProcessor,
    info=MiniMaxM3VLProcessingInfo,
    dummy_inputs=MiniMaxM3VLDummyInputsBuilder)
class MiniMaxM3SparseForConditionalGeneration(nn.Module, SupportsMultiModal,
                                              SupportsPP):
    """MiniMax-M3 VL: shared vision tower + M3 mixed sparse/dense MoE LM.

    EAGLE-3 is intentionally NOT advertised: the aux-hidden-state control
    methods the runner drives (set_aux_hidden_state_layers etc.) are not wired
    through this wrapper to the inner LM, so declaring SupportsEagle3 would make
    supports_eagle3() true while spec-decode setup silently no-ops. Enabling
    eagle3 fails loudly instead."""

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return IMAGE_TOKEN
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.quant_config = quant_config
        self.multimodal_config = multimodal_config

        vision_params = build_vision_params(config)
        self.vision_params = vision_params
        self.spatial_merge_size = vision_params.spatial_merge_size

        text_hidden_size = getattr(config.text_config, "hidden_size",
                                   getattr(config, "hidden_size", 6144))
        projector_hidden_size = getattr(config, "projector_hidden_size",
                                        text_hidden_size)
        projector_hidden_act = getattr(config, "projector_hidden_act", "gelu")
        projector_bias = getattr(config, "multimodal_projector_bias", True)
        patch_merge_bias = getattr(config, "patch_merge_bias", True)

        # Vision tower + projector run unquantized: CLIP head_dim (80) is not
        # aligned to the MXFP8 block size (128), so they are in the checkpoint's
        # ``ignored_layers`` and load as BF16.
        self.vision_tower = MiniMaxM3VisionModel(
            vision_params,
            quant_config=None,
            prefix=maybe_prefix(prefix, "vision_tower"))
        self.multi_modal_projector = MiniMaxM3MultiModalProjector(
            vision_hidden_size=vision_params.hidden_size,
            text_hidden_size=text_hidden_size,
            projector_hidden_size=projector_hidden_size,
            projector_hidden_act=projector_hidden_act,
            bias=projector_bias,
            quant_config=None,
            prefix=maybe_prefix(prefix, "multi_modal_projector"))
        self.patch_merge_mlp = MiniMaxM3PatchMerger(
            text_hidden_size=text_hidden_size,
            projector_hidden_size=projector_hidden_size,
            spatial_merge_size=vision_params.spatial_merge_size,
            projector_hidden_act=projector_hidden_act,
            bias=patch_merge_bias,
            quant_config=None,
            prefix=maybe_prefix(prefix, "patch_merge_mlp"))

        # Text backbone -- unchanged from the text-only path (keeps the real
        # quant_config via ``vllm_config``). Created with prefix="language_model"
        # so AutoWeightsLoader routes the checkpoint's ``language_model.*`` keys
        # here (and the child load_weights strips that prefix + skips vision).
        self.language_model = HpuMiniMaxM3SparseForCausalLM(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"))

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    # --- multimodal embedding path --------------------------------------- #

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[MiniMaxM3VLImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        vision_cos = kwargs.pop("vision_cos", None)
        vision_sin = kwargs.pop("vision_sin", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is None:
            # Precomputed image_embeds (skip-the-vision-tower) direct input
            # isn't supported here: this model's processor never emits
            # image_embeds itself (see _call_hf_processor above, which only
            # ever returns pixel_values), so this path has no defined
            # image_grid_thw pairing to validate against.
            raise NotImplementedError(
                "image_embeds direct input is not supported for "
                "MiniMax-M3-VL; pass raw images instead.")

        image_grid_thw = self._reshape_grid_thw(image_grid_thw)

        return MiniMaxM3VLImagePixelInputs(
            type="pixel_values",
            pixel_values=self._reshape_pixel_values(pixel_values),
            image_grid_thw=image_grid_thw,
            vision_cos=self._reshape_pixel_values(vision_cos),
            vision_sin=self._reshape_pixel_values(vision_sin),
        )

    @staticmethod
    def _reshape_pixel_values(value: object) -> torch.Tensor:
        if isinstance(value, (list, tuple)):
            return torch.cat([torch.as_tensor(v) for v in value], dim=0)
        assert isinstance(value, torch.Tensor)
        # Collapse a leading batch dim if present -> [num_patches, feat].
        return value.reshape(-1, value.shape[-1])

    @staticmethod
    def _reshape_grid_thw(value: object) -> torch.Tensor:
        if isinstance(value, (list, tuple)):
            return torch.cat([torch.as_tensor(v).reshape(-1, 3) for v in value],
                             dim=0)
        assert isinstance(value, torch.Tensor)
        return value.reshape(-1, 3)

    def _process_image_input(
        self,
        image_input: MiniMaxM3VLImageInputs,
    ) -> torch.Tensor:
        if image_input["type"] == "image_embeds":
            return image_input["image_embeds"].to(self.vision_tower.dtype)

        pixel_values = image_input["pixel_values"].to(self.vision_tower.dtype)
        cos = image_input["vision_cos"]
        sin = image_input["vision_sin"]
        # Block-diagonal grouping: which image each vision patch belongs to.
        # Derived HERE from image_grid_thw (image-intrinsic, sized t*h*w per
        # image) with a host-side int() per image -- the same safe pattern
        # embed_multimodal uses below. NOT a processor-baked per-patch index:
        # such a field is cached PER IMAGE by the mm-processor cache, so a
        # warm-cache multi-image request reuses stale per-item indices (an image
        # first seen alone caches as index 0), collapsing distinct images into
        # one group -> cross-image attention leak. image_grid_thw is batched per
        # image and reassembles correctly per request, so the grouping stays
        # cache-correct. (int(g.prod()) forces a host sync -- fine in this lazy
        # prefill path; an in-graph int reduction over the patch axis is garbage
        # under HPU-graph capture, which is why the index is not computed there.)
        grid_thw = image_input["image_grid_thw"]
        patch_counts = [int(g.prod()) for g in grid_thw]  # pre-merge, per image
        seq_index = torch.repeat_interleave(
            torch.arange(len(patch_counts), dtype=torch.int32),
            torch.tensor(patch_counts, dtype=torch.int32)).to(pixel_values.device)

        # Block-diagonal attention bias: a patch attends only within its own
        # image. Use a FLOAT ADDITIVE mask (0.0 = attend, -inf = block), NOT a
        # boolean mask -- HPU's scaled_dot_product_attention mishandles boolean
        # masks (it silently degrades attention -> a red image was described as
        # "purple"). For a single image the bias is all-zeros (== full
        # attention).
        same = seq_index.unsqueeze(0) == seq_index.unsqueeze(1)  # [N, N] bool
        attn_mask = torch.zeros_like(same, dtype=pixel_values.dtype)
        attn_mask = attn_mask.masked_fill(~same, float("-inf"))

        vision_features = self.vision_tower(pixel_values, cos, sin, attn_mask)
        image_features = self.multi_modal_projector(vision_features)
        image_features = self.patch_merge_mlp(image_features)
        # Return the concatenated per-image-ordered features (no host-side split
        # -- merge_multimodal_embeddings scatters rows in order into the image
        # placeholder positions).
        return image_features

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        # pinned vLLM API (called by hpu_model_runner.embed_multimodal ->
        # self.model.embed_multimodal). Returns a tuple with ONE tensor per
        # image (rows = that image's post-merge placeholder tokens, in prompt
        # order). The SupportsMultiModal.embed_input_ids default flattens the
        # tuple and scatters the rows into the text embeddings at the image
        # placeholder positions using the is_multimodal mask.
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []
        image_features = self._process_image_input(image_input)
        # Split the concatenated [total_merged_tokens, hidden] features into one
        # tensor per image (post-merge count = t*h*w // merge**2) so the tuple
        # preserves per-image order for the scatter.
        grid_thw = image_input["image_grid_thw"]
        merge = self.spatial_merge_size**2
        sizes = [int(g.prod()) // merge for g in grid_thw]
        if len(sizes) <= 1:
            return (image_features, )
        return tuple(image_features.split(sizes, dim=0))

    def get_language_model(self) -> nn.Module:
        return self.language_model

    # NOTE: no get_input_embeddings / embed_input_ids override here. The
    # SupportsMultiModal base provides embed_input_ids(input_ids,
    # multimodal_embeddings=None, *, is_multimodal=None) which calls
    # get_language_model().embed_input_ids(input_ids) for the text embeddings
    # and merges the multimodal ones via the is_multimodal mask.

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors, tuple[torch.Tensor, list[torch.Tensor]]]:
        # In pinned vLLM V1 the runner computes inputs_embeds up front (via
        # embed_multimodal + embed_input_ids) and passes them in; forward just
        # runs the text backbone (cf. upstream qwen2_5_vl.forward).
        if intermediate_tensors is not None:
            inputs_embeds = None
        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        # Plugin text model's compute_logits takes NO sampling_metadata.
        return self.language_model.compute_logits(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
