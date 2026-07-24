import torch
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.forward_context import get_forward_context

from vllm_gaudi.ops.causal_conv1d_pytorch import (
    hpu_causal_conv1d_fn,
    hpu_causal_conv1d_update,
)
from vllm_gaudi.ops.hpu_gdn_pytorch import (
    hpu_chunk_gated_delta_rule,
    hpu_fused_gdn_gating,
    hpu_fused_recurrent_gated_delta_rule,
)


@torch._dynamo.disable
def _save_ssm_state(core_attn_out, final_state, ssm_state, state_indices):
    """Persist GDN final_state into ssm_state cache for chunked prefill.

    Must be @torch._dynamo.disable because HPU torch.compile silently
    drops in-place index_copy_ to aliased state tensors.  Returns
    core_attn_out as a pass-through so the compiled graph consumes
    the call — HPU drops dynamo-disabled calls whose results are unused.
    """
    safe_si = torch.remainder(state_indices, ssm_state.shape[0]).long()
    ssm_state.index_copy_(0, safe_si, final_state.to(device=ssm_state.device, dtype=ssm_state.dtype))
    return core_attn_out


def _pad_core_attn_out(core_attn_out, num_tokens):
    """Right-size the GDN op output to ``num_tokens`` rows functionally.

    The op returns exactly ``num_tokens`` rows in every real prefill/decode
    path, so this is normally an identity.  When it does not, pad/truncate
    with ``torch.cat`` / a contiguous slice — NEVER an in-place write into a
    ``torch.zeros`` buffer.  HPU lazy-compile silently drops the in-place
    slice-assign (``core_attn_out[:n] = ...``), leaving the tensor
    unmaterialized when it feeds out_proj.  A cat/slice is a genuine op-output
    the graph cannot drop, mirroring the fork's functional ``torch.cat``
    accumulation.
    """
    n = core_attn_out.shape[0]
    if n == num_tokens:
        return core_attn_out
    if n > num_tokens:
        return core_attn_out[:num_tokens].contiguous()
    pad = torch.zeros(
        (num_tokens - n, core_attn_out.shape[1], core_attn_out.shape[2]),
        dtype=core_attn_out.dtype,
        device=core_attn_out.device,
    )
    return torch.cat([core_attn_out, pad], dim=0)


class HPUGatedDeltaNetAttention(QwenGatedDeltaNetAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # cache_group_idx: set later by model runner for hybrid cache
        # lookup.  Stored as tensor so torch.compile treats it as dynamic.
        self.cache_group_idx = None

        # mamba_chunk_size: use explicit config value or default to 128
        # for HPU bucket alignment.
        hf_text_config = getattr(self.model_config, "hf_text_config", None)
        has_explicit = (hf_text_config is not None and (getattr(hf_text_config, "mamba_chunk_size", None) is not None
                                                        or getattr(hf_text_config, "chunk_size", None) is not None))
        self.mamba_chunk_size = (self.model_config.get_mamba_chunk_size() if has_explicit else 128)

        self.qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        self.z_size = self.value_dim // self.tp_size

        # This layer's OWN persistent dense ssm/conv state buffers, allocated ONCE
        # here in __init__ at a FIXED size and never reallocated — instead of
        # reading/writing the shared cross-layer pool (self.kv_cache).
        #
        # Why __init__ and not lazily in forward: the model is built on CPU (INC/FP8
        # load path), then relocated to HPU by _move_remaining_tensors_to_device
        # (which walks each module __dict__ and moves plain-attr tensors CPU->HPU),
        # and ONLY THEN is the model wrapped in the lazy HPU graph.  A buffer
        # allocated here is a non-None plain attr at relocation time, so the mover
        # relocates it and its storage is stable through warmup and every decode
        # replay.  A lazy in-forward alloc runs post-wrap inside a captured region
        # (misses the mover), and any grow/realloc moves storage under an already-
        # captured decode graph, so the next replay reads freed memory.  Shape/dtype
        # come from the plugin's own pool calculators (get_state_shape/get_state_dtype)
        # so these buffers are byte-compatible with the shared pool they replace.
        from vllm.config import get_current_vllm_config
        max_num_seqs = int(get_current_vllm_config().scheduler_config.max_num_seqs)
        mamba_cache_bs = max(8, max_num_seqs) + 2  # fork qwen3_next.py:313-314
        _shapes = self.get_state_shape()   # (conv_state_shape, temporal_state_shape)
        _dtypes = self.get_state_dtype()   # (conv_state_dtype, temporal_state_dtype)
        conv_shape, ssm_shape = _shapes[0], _shapes[1]
        conv_dtype, ssm_dtype = _dtypes[0], _dtypes[1]
        _dev = self.conv1d.weight.device   # CPU at load; mover -> HPU before wrap
        self._dense_conv = torch.zeros((mamba_cache_bs, *conv_shape), dtype=conv_dtype, device=_dev)
        self._dense_ssm = torch.zeros((mamba_cache_bs, *ssm_shape), dtype=ssm_dtype, device=_dev)

    def rearrange_mixed_qkv(self, mixed_qkv):
        """Pure-torch rearrange – avoids einops graph breaks on HPU."""
        if mixed_qkv is None:
            return None, None, None
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // self.tp_size,
                self.key_dim // self.tp_size,
                self.value_dim // self.tp_size,
            ],
            dim=-1,
        )
        query = query.reshape(1, query.size(0), -1, self.head_k_dim).contiguous()
        key = key.reshape(1, key.size(0), -1, self.head_k_dim).contiguous()
        value = value.reshape(1, value.size(0), -1, self.head_v_dim).contiguous()
        return query, key, value

    def _resolve_state_indices(self, attn_metadata):
        """Resolve load_indices_tensor, handling 2-D cache-group case.

        For Qwen 3.5 (GDN), load and store indices are identical
        so using load_indices_tensor is sufficient.
        """
        indices = attn_metadata.load_indices_tensor
        if indices is not None and indices.dim() > 1:
            cg = self.cache_group_idx
            assert cg is not None
            indices = indices.index_select(0, cg.view(1)).squeeze(0)
        return indices

    def _extract_metadata(self, num_tokens):
        """Extract forward-context metadata into plain tensors.

        Dynamo graph-breaks naturally on ``get_forward_context()``; no
        ``@dynamo.disable`` needed.
        """
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return (False, None, None, None, None, None, None, 0, 0, 0, 0, None)

        is_prompt = bool(getattr(attn_metadata, "is_prompt", False))
        state_indices = self._resolve_state_indices(attn_metadata)

        conv_state = self.kv_cache[0]
        ssm_state = self.kv_cache[1]

        if (state_indices is not None and ssm_state is not None and conv_state is not None):
            # Dense per-layer state ownership: swap the shared pool (ssm_state /
            # conv_state) for THIS layer's own persistent dense buffers and remap the
            # pooled state_indices to per-layer rows.  Every downstream site (prefill
            # initial-state read, both index_copy_ writes, conv fn/update, decode op)
            # uses these locals, so this one swap redirects the whole state path.  numel
            # is preserved, so num_decodes / prefill_num_seqs below are unchanged.
            # Fresh-request initial state stays correct via the has_initial_state mask
            # below (zeros a reused row on a new prefill).  The dense buffers are
            # allocated in __init__ at fixed size and relocated to HPU before graph
            # capture — no lazy alloc / realloc here (either would move storage under a
            # captured graph).  Row mapping (active request i -> row i via torch.arange,
            # no host sync) assumes the active-request order is stable across steps,
            # which holds at concurrency 1; higher concurrency would need the runner's
            # per-request base-slot mapping.
            num_active = int(state_indices.numel())
            conv_state = self._dense_conv
            ssm_state = self._dense_ssm
            state_indices = torch.arange(num_active, device=state_indices.device,
                                         dtype=state_indices.dtype)

        query_start_loc = attn_metadata.query_start_loc_p
        has_initial_state = getattr(attn_metadata, "has_initial_states_p", None)
        padding_mask_flat = getattr(attn_metadata, "padding_mask_flat", None)

        if not is_prompt:
            num_decodes = (state_indices.numel() if state_indices is not None else
                           (query_start_loc.numel() - 1 if query_start_loc is not None else num_tokens))
        else:
            num_decodes = 0

        mamba_block_size = (self.cache_config.mamba_block_size if is_prompt else 0)

        # Prefill-specific metadata (Python ints for torch.compile)
        prefill_num_seqs = 0
        prefill_seq_len = 0
        initial_state = None
        if is_prompt and state_indices is not None:
            prefill_num_seqs = int(state_indices.numel())
            prefill_seq_len = (num_tokens // prefill_num_seqs if prefill_num_seqs > 0 else 0)
            initial_state = ssm_state[state_indices].contiguous()
            if has_initial_state is not None:
                # Avoid scatter_nd from boolean indexing
                mask = has_initial_state.bool().view(-1, 1, 1, 1).to(initial_state.dtype)
                initial_state = initial_state * mask

        return (is_prompt, conv_state, ssm_state, state_indices, query_start_loc, has_initial_state, padding_mask_flat,
                num_decodes, mamba_block_size, prefill_num_seqs, prefill_seq_len, initial_state)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor = None,
    ):
        """HPU compile-friendly GDN forward.

        Bypasses the upstream ``gdn_attention_core`` custom-op and
        drives the HPU conv1d + GDN kernels directly with
        ``HPUAttentionMetadataV1``.

        Return-based: the projection is returned directly (the patched
        decoder-layer caller consumes the return); the ``output`` buffer
        argument is retained for signature compatibility and is unused.
        """
        # Original (pre-flatten) shape — the output projection is returned as
        # result.view(orig_shape) for the return-based caller.
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        num_tokens = hidden_states.size(0)

        # === Metadata extraction (natural graph break) ===============
        (is_prompt, conv_state, ssm_state, state_indices, query_start_loc, has_initial_state, padding_mask_flat,
         num_decodes, mamba_block_size, prefill_num_seqs, prefill_seq_len,
         initial_state) = self._extract_metadata(num_tokens)

        # === Part 1: Input Projection ================================
        if hasattr(self, 'in_proj_qkv'):
            # LoRA path (Qwen3.5 only): separate in_proj_qkv and in_proj_z
            mixed_qkv, _ = self.in_proj_qkv(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            z, _ = self.in_proj_z(hidden_states)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)
            b = b.contiguous()
            a = a.contiguous()
        else:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)

            if self.gqa_interleaved_layout:
                # Qwen3-Next: unpack the interleaved GQA layout
                query, key, value, z, b, a = self.fix_query_key_value_ordering(mixed_qkvz, ba)
                # Pure-torch flatten instead of einops rearrange (graph breaks)
                query = query.reshape(query.size(0), -1)
                key = key.reshape(key.size(0), -1)
                value = value.reshape(value.size(0), -1)
                mixed_qkv = torch.cat((query, key, value), dim=-1)
            else:
                # Qwen3.5: weights already in [q, k, v, z] and [b, a] order
                mixed_qkv, z = mixed_qkvz.split([self.qkv_size, self.z_size], dim=-1)
                z = z.reshape(z.size(0), -1, self.head_v_dim)
                b, a = ba.chunk(2, dim=-1)
                b = b.contiguous()
                a = a.contiguous()

        # core_attn_out is bound to the real GDN op output in the prefill /
        # decode branches below via a functional build (_pad_core_attn_out),
        # NEVER a torch.zeros + in-place slice-assign: HPU lazy-compile silently
        # drops the in-place write, so out_proj's input reaches the graph
        # unmaterialized.  The zeros allocation here is the value used only on the
        # profile run (conv_state is None) below, where it is a genuine fresh alloc.
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        if conv_state is None:
            # No attn_metadata — skip core attention (profile run).  The zeros
            # tensor above is a genuine fresh allocation (not an in-place write
            # into an existing buffer), so it materializes fine on this path.
            pass
        elif is_prompt:
            # === Part 2a: Prefill ====================================
            if (padding_mask_flat is not None and padding_mask_flat.numel() == num_tokens):
                token_mask_flat = padding_mask_flat.view(-1, 1).to(dtype=mixed_qkv.dtype)
                mixed_qkv = mixed_qkv * token_mask_flat
            else:
                token_mask_flat = None

            g, beta = hpu_fused_gdn_gating(self.A_log, a, b, self.dt_bias)
            # Densify the unsqueeze(0) gating views so they cannot reach the GDN op
            # unmaterialized under HPU capture (the fork op's g.transpose(1,2)
            # .contiguous() plays the same role at op entry).
            g = g.clone()
            beta = beta.clone()

            conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            mixed_qkv_conv = hpu_causal_conv1d_fn(
                x=mixed_qkv.transpose(0, 1),
                weight=conv_weights,
                bias=self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=state_indices,
                block_idx_first_scheduled_token=None,
                block_idx_last_scheduled_token=None,
                initial_state_idx=None,
                query_start_loc=query_start_loc,
                block_size_to_align=mamba_block_size,
                num_computed_tokens=None,
                metadata=None,
                is_prompt=True,
            ).transpose(0, 1)

            if token_mask_flat is not None:
                mixed_qkv_conv = mixed_qkv_conv * token_mask_flat

            query, key, value = self.rearrange_mixed_qkv(mixed_qkv_conv)

            if token_mask_flat is not None:
                token_mask_h = token_mask_flat.view(1, -1, 1).to(dtype=g.dtype)
                g = g * token_mask_h
                beta = beta * token_mask_h

            # HW-proven plugin prefill op — the winning VLLM_GDN_FIXB=baseline path.
            # The alternative "vendor" 5-D fork op emits garbage / fails to compile
            # (synStatus26 batch_gemm) in eager mode; see
            # .out/qwen35/reports/2026-07-22-256k-eager-solution.md. hpu_gdn_pytorch
            # has the baseline+FUNC_SCAN phase_b hardcoded, so this single call is the
            # whole op selection; returns (out, final_state).
            core_attn_out_result, final_state = hpu_chunk_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                chunk_size=self.mamba_chunk_size,
                prefill_num_seqs=prefill_num_seqs,
                prefill_seq_len=prefill_seq_len,
            )
            # State save in dynamo-disabled wrapper — index_copy_ is
            # silently dropped by HPU torch.compile on aliased tensors.
            core_attn_out_result = _save_ssm_state(
                core_attn_out_result,
                final_state,
                ssm_state,
                state_indices,
            )

            # Bind to the real op output (functional), not an in-place write
            # into the zeros buffer — see _pad_core_attn_out / the note above.
            non_spec_out = core_attn_out_result.squeeze(0).contiguous()
            core_attn_out = _pad_core_attn_out(non_spec_out, num_tokens)

        else:
            # === Part 2b: Decode =====================================
            g, beta = hpu_fused_gdn_gating(self.A_log, a, b, self.dt_bias)
            # Densify the unsqueeze(0) gating views so they cannot reach the GDN op
            # unmaterialized under HPU capture.
            g = g.clone()
            beta = beta.clone()

            conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            mixed_qkv_conv = hpu_causal_conv1d_update(
                x=mixed_qkv,
                conv_state=conv_state,
                weight=conv_weights,
                bias=self.conv1d.bias,
                activation=self.activation,
                conv_state_indices=(state_indices[:num_decodes] if state_indices is not None else state_indices),
                block_idx_last_scheduled_token=None,
                initial_state_idx=None,
                query_start_loc=query_start_loc,
                validate_data=False,
            )

            query, key, value = self.rearrange_mixed_qkv(mixed_qkv_conv)

            core_attn_out_result, _ = \
                hpu_fused_recurrent_gated_delta_rule(
                    q=query, k=key, v=value, g=g, beta=beta,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=(
                        query_start_loc[:num_decodes + 1]
                        if query_start_loc is not None else None),
                    ssm_state_indices=state_indices,
                    use_qk_l2norm_in_kernel=True,
                )

            # Bind to the real op output (functional), not an in-place write
            # into the zeros buffer — see _pad_core_attn_out / the note above.
            non_spec_out = core_attn_out_result.squeeze(0).contiguous()
            core_attn_out = _pad_core_attn_out(non_spec_out, num_tokens)

        # === Part 3: Output Projection ===============================
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        # Fork output tail: gated norm -> cast -> collapse head dims into
        # value_dim//tp -> out_proj, returning a FRESH tensor.  out_proj is fed a
        # fork-shaped 3-D [1, num_tokens, value_dim//tp] input: feeding it a 2-D
        # input left the fp8 matmul output unmaterialized across the all_reduce's
        # mark_step under disable_tensor_cache (the collective wrote into a storage-
        # less buffer).  Matching the fork's 3-D input keeps the projection and the
        # segmented collective materializing cleanly; .contiguous() gives out_proj a
        # dense input; the return view restores orig_shape for the caller.
        core_attn_out = self.norm(core_attn_out, z).to(hidden_states.dtype)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(1, z_shape_og[0], -1).contiguous()
        output_proj, _ = self.out_proj(core_attn_out)
        # Return the fresh out_proj tensor (return-based, mirroring the fork +
        # upstream PR #1611).  A returned value gives the whole GDN chain (op ->
        # index_copy_ state saves -> norm -> out_proj) a live consumer so the lazy
        # compiler / Synapse fuser cannot DCE it.  orig_shape == the incoming
        # hidden_states shape == the residual shape the caller's add expects.
        return output_proj.view(orig_shape)


# Replace the class in the upstream modules so that both Qwen3-Next and
# Qwen3.5 model definitions instantiate HPUGatedDeltaNetAttention.
import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as _gdn_module  # noqa: E402
import vllm.model_executor.models.qwen3_next as _qwen3_next_module  # noqa: E402
import vllm.model_executor.models.qwen3_5 as _qwen3_5_module  # noqa: E402

_gdn_module.QwenGatedDeltaNetAttention = HPUGatedDeltaNetAttention
_qwen3_next_module.QwenGatedDeltaNetAttention = HPUGatedDeltaNetAttention
_qwen3_5_module.QwenGatedDeltaNetAttention = HPUGatedDeltaNetAttention
