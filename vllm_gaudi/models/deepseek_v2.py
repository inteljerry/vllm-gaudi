import torch
from itertools import islice

from vllm.distributed import get_pp_group
from vllm.model_executor.models import deepseek_mtp
from vllm.model_executor.models import deepseek_v2
from vllm.sequence import IntermediateTensors


def _get_hpu_llama_4_scaling(original_max_position_embeddings: int, scaling_beta: float,
                             positions: torch.Tensor) -> torch.Tensor:
    scaling = 1 + scaling_beta * torch.log(1 + torch.floor(positions / original_max_position_embeddings))
    # Broadcast over num_heads and head_dim
    scaling = scaling[..., None, None]

    # Squeeze dimension of scaling factor to match expected shape on HPU
    return scaling.reshape(-1, *scaling.shape[-2:])


deepseek_v2._get_llama_4_scaling = _get_hpu_llama_4_scaling


def _hpu_deepseek_v2_model_forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors:
    """HPU DeepseekV2Model.forward without the TP sequence-parallel all-gather.

    Upstream vllm #46635 (5c91039c41) added a ``torch.cat([hidden_states,
    residual])`` all-gather block gated on ``hidden_states.shape[0] !=
    positions.shape[0]``. That guard assumes the GPU shape contract (flat 2D
    hidden_states, 1D positions). On HPU ``positions`` is 2D ``[bs, seq]`` while
    ``DeepseekV2MoE.forward`` returns a flattened ``[bs*seq, H]``, so the guard
    fires spuriously for any prompt and crashes cat'ing a 2D tensor with a 3D
    residual. HPU handles MoE parallelism in its own kernels, so this block is
    dead here — restore the pre-#46635 plain loop.
    """
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided "
                                 "to DeepseekV2Model.forward")
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    # Compute llama 4 scaling once per forward pass if enabled
    llama_4_scaling_config = getattr(self.config, "llama_4_scaling", None)
    llama_4_scaling: torch.Tensor | None
    if llama_4_scaling_config is not None:
        llama_4_scaling = deepseek_v2._get_llama_4_scaling(
            original_max_position_embeddings=llama_4_scaling_config["original_max_position_embeddings"],
            scaling_beta=llama_4_scaling_config["beta"],
            positions=positions,
        )
    else:
        llama_4_scaling = None

    aux_hidden_states = []
    for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
    ):
        if idx in self.aux_hidden_state_layers:
            # residual is None before the first layer runs (first PP rank);
            # treat it as zero so the pre-residual hidden state is just
            # hidden_states.
            aux_hidden_states.append(hidden_states if residual is None else hidden_states + residual)
        hidden_states, residual = layer(positions, hidden_states, residual, llama_4_scaling)

    if not get_pp_group().is_last_rank:
        return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

    hidden_states, _ = self.norm(hidden_states, residual)
    if len(aux_hidden_states) > 0:
        return hidden_states, aux_hidden_states
    return hidden_states


# Applies to DeepseekV2/V3/Deepseek/GlmMoe/DSA — all share model_cls = DeepseekV2Model.
deepseek_v2.DeepseekV2Model.forward = _hpu_deepseek_v2_model_forward

_orig_deepseek_v2_model_load_weights = deepseek_v2.DeepseekV2Model.load_weights


def _hpu_deepseek_v2_model_load_weights(self, weights):
    """Drop GLM-5 DSA shared-indexer projection weights (`indexers_proj`).

    vLLM's DeepseekV2Model has no module for them, and on HPU DSA layers run
    as dense MLA with the indexer never executed, so the projection that
    shares indexer K caches across layers is dead weight here.
    """

    def _filtered(ws):
        for name, weight in ws:
            if ".indexers_proj." in name:
                continue
            yield name, weight

    return _orig_deepseek_v2_model_load_weights(self, _filtered(weights))


deepseek_v2.DeepseekV2Model.load_weights = _hpu_deepseek_v2_model_load_weights


def _hpu_restore_full_token_layout_if_needed(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    num_tokens: int,
    is_sequence_parallel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HPU variant of deepseek_mtp._restore_full_token_layout_if_needed.

    Same shape-contract mismatch as _hpu_deepseek_v2_model_forward above:
    the upstream guard ``hidden_states.shape[0] == num_tokens`` assumes the
    GPU layout (flat 2D [num_tokens, H] hidden_states, 1D positions). On HPU
    positions is 2D [bs, seq] so num_tokens == bs, while DeepseekV2MoE
    flattens hidden_states to [bs*seq, H] and residual stays [bs, seq, H].
    The guard fires spuriously on every prefill proposal (seq > 1) and the
    SP all-gather cat crashes on the 2D-vs-3D mismatch. HPU MoE kernels
    handle expert parallelism internally, so for the single-node EP setups this
    plugin targets the all-gather is dead; just normalize both to token-major
    [bs*seq, H] for the residual add.

    ``is_sequence_parallel`` is NOT handled. It is unreachable at
    data_parallel_size == 1 because ``_patch_use_sequence_parallel_moe`` restores
    the DP>1 guard on ``use_sequence_parallel_moe`` -- but it IS reachable at DP>1,
    and silently dropping the all-gather there would leave every rank holding only
    its own shard, i.e. wrong output with no error. Raise instead.
    """
    if is_sequence_parallel:
        raise NotImplementedError(
            "HPU MTP does not support sequence-parallel MoE: the upstream all-gather is skipped "
            "by this patch, which would silently corrupt the residual add. Run with "
            "data_parallel_size == 1 (where SP-MoE is disabled) or disable SP-MoE.")
    if hidden_states.dim() != residual.dim():
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        residual = residual.view(-1, residual.shape[-1])
    return hidden_states, residual


deepseek_mtp._restore_full_token_layout_if_needed = _hpu_restore_full_token_layout_if_needed
