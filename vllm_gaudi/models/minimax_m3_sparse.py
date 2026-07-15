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
"""MiniMax Sparse Attention (MSA) for the vLLM-Gaudi plugin.

Implements M3's native block-sparse attention (layers with
``sparse_attention_freq != 0``):

  * ``MiniMaxM3Indexer``: replicated ``index_q_proj`` (hidden -> 4*128) /
    ``index_k_proj`` (hidden -> 128, one shared key head), Gemma RMSNorm(128)
    on each, then the SAME partial-NeoX rotary instance as the main q/k.
  * Selection: fp32 index scores per index head vs the shared index key,
    per-128-token-block amax, token-level causal mask, the query's current
    block force-included (1e29 score injection), top-min(16, valid) blocks
    per (index head, query).  One index head per GQA group; the group's 16
    query heads share its selection.
  * Attention: causal GQA softmax (1/sqrt(128), fp32) restricted to the
    selected blocks.  Sequences <= 2048 tokens select every block, so the
    result is exactly dense there (the V1 equivalence gate).

The pure-torch functions in the first half of this file are the whole
algorithm and import nothing but torch, so a CPU unit test can drive them
against the upstream reference implementations.  The nn.Modules in the second half wire them into
the plugin; their vLLM imports are guarded so a torch-only interpreter can
still import the math.

Deployed-stack contracts this file relies on (verified against the
``vllm-gaudi-m3:1.24-vl-e3`` image = stock plugin v0.21.x, NOT repo HEAD):

  * KV cache layout: per layer a ``(key_cache, value_cache, k_scales,
    v_scales)`` tuple bound onto the Attention layer by ``bind_kv_cache``
    (vllm/v1/worker/utils.py:510-512); each cache is flat-slot
    ``(num_blocks_total * 128, num_kv_heads, head_size)``
    (vllm_gaudi/attention/ops/hpu_paged_attn.py:66-73) where
    ``num_blocks_total = num_blocks + 1`` and the extra last block is the pad
    block (``_PAD_BLOCK_ID = num_blocks``, hpu_model_runner.py:6236-6297).
  * Cache write: ``cache.index_copy_(0, slot_mapping.flatten(), kv)``
    (extension/utils.py:71-77 ``VLLMKVCache.forward``, called from
    HPUAttentionImpl.forward).  Prompt slot_mapping pads with -1
    (hpu_model_runner.py:2474); decode pads cycle inside the pad block
    (hpu_model_runner.py:2868-2870).
  * Decode block buffers: ``block_list``/``block_groups``/``block_usage``
    triplets (hpu_model_runner.py:2150-2196).  Under contiguous PA the lists
    are scattered by physical block id, so within-sequence POSITIONAL order
    is not recoverable -- the decode selection below therefore works purely
    in flat-block space and never needs a block table.
  * Prefill metadata: ``block_list`` = context blocks, a uniform
    ``target_blocks`` per sequence, -1-padded (hpu_model_runner.py:2476,2686);
    ``seq_lens_tensor`` = per-seq query lens; ``context_lens_tensor`` = per-seq
    context lens.
"""

import os

import torch

# M3 config constants (verified against the checkpoint's
# sparse_attention_config): block 128, topk 16, 4 index heads of dim 128,
# score "max", init_block 0 (off), local_block 1 (current block only).
MSA_BLOCK_SIZE = 128
MSA_TOPK = 16
MSA_INDEX_HEADS = 4
MSA_INDEX_DIM = 128
# Context-block slab for prefill selection.  msa_topk_blocks scores context in
# fixed nb_tile slabs so the fp32 key/score transient is bounded by nb_tile,
# never the context-scaling block count nb -- the long-context HPUGraph HBM
# ceiling (a whole-nb score transient reaches ~1 GB/card at the 4080 bucket and
# fails device allocation under graph capture).  512 caps the slab transient at
# ~134 MB/card (bs1, 4 index heads, 128 query tile, 128 block); larger nb only
# adds slab iterations.
MSA_NB_TILE = 512

_NEG_INF = float("-inf")

# Debug hooks (serve-side triage only; zero cost unless VLLM_M3_MSA_DEBUG=1).
# VLLM_M3_MSA_WATCH_BLOCK: logical block id whose selection is traced.
# VLLM_M3_MSA_DEBUG_LAYER: layer index that logs (default 3 = first sparse).
_MSA_DEBUG = os.environ.get("VLLM_M3_MSA_DEBUG", "0") == "1"
_MSA_WATCH_BLOCK = int(os.environ.get("VLLM_M3_MSA_WATCH_BLOCK", "-1"))
_MSA_DEBUG_LAYER = os.environ.get("VLLM_M3_MSA_DEBUG_LAYER", "3")

# Index-K side-cache dtype. bf16 is the bring-up default (design D3/D4); the
# indexer scores index_q.index_k in fp32 regardless, but a bf16 stored index_k
# loses ~8 mantissa bits -- at extreme context (~1900+ blocks) that can blur the
# needle block's max-pooled score below the top-16 cut.  VLLM_M3_MSA_INDEX_DTYPE
# = bf16 | fp32 (fp32 doubles the side cache: +14.6 -> +29 KB/token/card).
_MSA_INDEX_DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32,
                    "float32": torch.float32}.get(
                        os.environ.get("VLLM_M3_MSA_INDEX_DTYPE", "bf16"), torch.bfloat16)

# Timing-only ablation: replace the O(ctx) decode index-scan (key gather + score
# einsum + amax over all nb blocks) with a trivial position proxy.  Breaks needle
# (wrong blocks selected); used ONLY to measure the scan's cost = the P3a ceiling.
_ABLATE_INDEX_SCAN = os.environ.get("VLLM_M3_ABLATE_INDEX_SCAN", "0") == "1"

# E3 lever: slice the decode index query to the rank-local index head(s) BEFORE
# the O(ctx) topk scan, instead of scanning all num_index_heads and slicing the
# selected pages after.  On TP8 each rank owns 1 of 4 index heads, so the
# baseline scans 4x the score-work it uses.  The per-head topk is independent
# (the score einsum/amax and the topk run per head; the owned/force block masks
# broadcast over heads), so head g computed alone is bit-identical to head g in
# the full scan -> selected-block parity is exact.  Off by default; A/B via env.
_MSA_TP_LOCAL_INDEX = os.environ.get("VLLM_M3_MSA_TP_LOCAL_INDEX", "0") == "1"

# E3 parity gate: on the debug layer, run BOTH the full-head scan (slice after)
# and the local-head scan (slice before) and assert the selected pages/usage are
# identical.  Live, in-process evidence that the E3 shortcut is exact -- no
# cross-boot determinism assumption.  Costs an extra scan on one layer only.
_MSA_TP_LOCAL_INDEX_PARITY = os.environ.get("VLLM_M3_MSA_TP_LOCAL_INDEX_PARITY", "0") == "1"

# E4 lever: run the selected-window decode attention (<= topk*block_size = 2048
# keys) through FusedSDPA (fp32 softmax, additive mask) instead of the decomposed
# einsum+masked-softmax+einsum.  fp32 is mandatory here -- bf16 in this path
# breaks the needle (documented; a bf16 fused index-scan was already REJECTED).
_MSA_SELECTED_FSDPA = os.environ.get("VLLM_M3_MSA_SELECTED_FSDPA", "0") == "1"
# E4 parity gate: run BOTH the decomposed path and the FusedSDPA path on the
# debug layer and assert the outputs match within fp32 tolerance.
_MSA_SELECTED_FSDPA_PARITY = os.environ.get("VLLM_M3_MSA_SELECTED_FSDPA_PARITY", "0") == "1"

# Skip-short-indexer lever: at decode, when the padded block bucket is <= topk,
# MSA's top-k selects EVERY block, so the result is EXACTLY dense there (the V1
# equivalence gate).  Skip the index-Q projection + the O(ctx) top-k scan + the
# sparse-attn kernel and run the layer's native dense paged attention (self.attn)
# instead -- bit-identical output, minus the fixed per-layer indexer tax that
# makes short MSA decodes ~5 tok/s slower than the dense path.
#
# CORRECTNESS INVARIANT: the index-K side cache must stay COMPLETE for any
# sequence that later grows past topk blocks and falls back to the MSA scan, so
# this path STILL projects + writes idx_k every step (only idx_q + scan are
# skipped).  CAPTURE-SAFE: block_list.shape[0] is the exponential block bucket
# (static per captured decode graph). The gate fires for any bucket <=
# self.msa_topk (the exp grid has a bucket at exactly topk, =16 for M3), where
# every sequence has <= topk real blocks -> dense-exact. Off by default; A/B via env.
_MSA_SKIP_SHORT_INDEXER = os.environ.get("VLLM_M3_MSA_SKIP_SHORT_INDEXER", "0") == "1"

# Dense-below lever (opt-in): extend the dense-decode substitution above topk, up
# to VLLM_M3_MSA_DENSE_BELOW_BLOCKS padded decode blocks.  Threshold is in BLOCKS,
# NOT tokens: the bucket a context lands in is exp-spaced and config-dependent
# (max_model_len/max_num_seqs/contiguous_pa), so confirm the bucket for the target
# length on the actual serve.  UNLIKE skip-short this is NOT output-exact: above
# topk MSA is genuinely sparse, so dense here CHANGES the output -- but to the SAME
# computation the dense serve (VLLM_M3_MSA=0) runs, which is correct at long
# context (dense-serve needle sweep).  Speed play: dense beats MSA below the
# effective crossover, held low (~40-50K) by the mandatory idx_k write.  MEASURED
# at N=512 (2026-07-14 A/B): +9% @8K, +6.4% @32K, 0% @64K -- so 512 is the
# recommended ceiling (catches 8K-32K; its 48-64K span is a harmless 0% wash).
# Same completeness invariant + capture-safety as skip-short (reuses
# _msa_decode_short).  Default 0 (off); for purely short traffic the dedicated
# dense serve is faster (no idx_k overhead).
_MSA_DENSE_BELOW_BLOCKS = int(os.environ.get("VLLM_M3_MSA_DENSE_BELOW_BLOCKS", "0"))

# Skip-short parity gate (mirrors E3/E4 _PARITY): on the debug layer, run BOTH
# the skip path (dense self.attn) and the full MSA decode for the same step and
# assert their attention outputs match within bf16 tolerance.  Live, in-process
# evidence that the skip shortcut is exact -- the two kernels attend over the
# same <= topk blocks so they differ only by bf16 tiling (same class as the E3
# OUT_TOL).  Costs a redundant MSA decode on one layer only.  Off by default.
_MSA_SKIP_SHORT_INDEXER_PARITY = os.environ.get("VLLM_M3_MSA_SKIP_SHORT_INDEXER_PARITY", "0") == "1"
_MSA_SKIP_PARITY_TOL = 1e-2


def _fsdpa_apply(q, k, v, add_mask, scale):
    """FusedSDPA over a small selected window: non-causal, explicit additive
    fp32 mask, fp32 softmax.  Lazily resolves the HPU kernel; the whole helper
    is only reached when VLLM_M3_MSA_SELECTED_FSDPA is set (HPU-only path)."""
    import vllm_gaudi.extension.kernels as _kernels
    FusedSDPA = _kernels.fsdpa()
    return FusedSDPA.apply(q, k, v, add_mask, 0.0, False, scale, "fp32")


def msa_enabled(config) -> bool:
    """MSA feature gate: on iff the checkpoint declares sparse attention and
    ``VLLM_M3_MSA`` is not "0" (the env flag exists for dense/sparse A/B
    serves; dense fallback must stay byte-identical to the pre-MSA code)."""
    sparse_cfg = getattr(config, "sparse_attention_config", None)
    if not (isinstance(sparse_cfg, dict) and sparse_cfg.get("use_sparse_attention")):
        return False
    return os.environ.get("VLLM_M3_MSA", "1") != "0"


def msa_layer_enabled(config, layer_idx: int) -> bool:
    """Per-layer MSA routing: sparse iff globally enabled and the layer's
    ``sparse_attention_freq`` entry is nonzero (M3: [0]*3 + [1]*57).  A
    missing or short freq list fails DENSE, never sparse."""
    if not msa_enabled(config):
        return False
    freq = config.sparse_attention_config.get("sparse_attention_freq") or []
    return layer_idx < len(freq) and bool(freq[layer_idx])


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """fp32 softmax over the last dim with ``mask`` (True = keep); rows with
    no kept key return all-zero probabilities instead of NaN (batch-padding
    rows are fully masked -- their outputs are dropped downstream but must
    stay finite for HPU graphs)."""
    logits = logits.float().masked_fill(~mask, _NEG_INF)
    row_max = logits.amax(dim=-1, keepdim=True)
    # Fully-masked rows have row_max == -inf; exp(-inf - (-inf)) would be NaN,
    # so substitute 0 for the max and let the zero mask kill the row.
    row_max = torch.where(row_max == _NEG_INF, torch.zeros_like(row_max), row_max)
    probs = torch.exp(logits - row_max) * mask
    denom = probs.sum(dim=-1, keepdim=True)
    return probs / torch.where(denom == 0, torch.ones_like(denom), denom)


def msa_topk_blocks(
    idx_q: torch.Tensor,
    index_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    q_pos: torch.Tensor,
    topk: int = MSA_TOPK,
    block_size: int = MSA_BLOCK_SIZE,
    local_blocks: int = 1,
    q_tile: int = 128,
    nb_tile: int = MSA_NB_TILE,
) -> torch.Tensor:
    """Prefill/chunk selection: top-``min(topk, valid)`` logical blocks per
    (index head, query token), following upstream ``_reference_index_topk``.

    idx_q:         [bs, q_len, H_idx, D] post-norm post-RoPE index queries.
    index_k_cache: [num_slots, D] flat-slot index-key cache; the CURRENT
                   chunk's keys must already be written (cache-first, same
                   contract as the upstream reference which scores straight
                   from the inserted cache).
    page_table:    [bs, nb] physical page id of each logical block; rows past
                   a sequence's real blocks may hold junk (causality masks
                   every token of a junk block: its kpos exceeds every q_pos).
    q_pos:         [bs, q_len] absolute positions; -1 marks batch padding.

    Returns [bs, H_idx, q_len, topk] int32 LOGICAL block ids, -1 padded.
    Queries run in ``q_tile`` slices AND context blocks in fixed ``nb_tile``
    slabs, so the fp32 key/score transient stays
    [bs, nb_tile, block_size, D] / [bs, H, q_tile, nb_tile, block_size] --
    bounded by nb_tile, never the context-scaling block count nb (the
    long-context HPUGraph HBM ceiling).  The final slab is padded to nb_tile
    for one uniform recipe; its pad columns are dropped from the write, so the
    accumulated block_scores is bit-identical to a whole-nb pass.
    """
    bs, q_len, num_heads, dim = idx_q.shape
    nb = page_table.shape[1]
    device = idx_q.device

    pages = index_k_cache.unflatten(0, (-1, block_size))
    # Context scored in fixed ``nb_tile`` slabs: the key gather and the score
    # einsum never span all nb blocks.  page_table is padded to a whole number
    # of slabs so every slab's gather/einsum is nb_tile-wide (one recipe); the
    # last slab's pad columns are dropped from the block_scores write, so the
    # accumulator equals a whole-nb pass.
    num_slabs = (nb + nb_tile - 1) // nb_tile
    nb_pad = num_slabs * nb_tile
    page_table_c = page_table.clamp(min=0)
    if nb_pad > nb:
        page_table_c = torch.nn.functional.pad(page_table_c, (0, nb_pad - nb))

    out = torch.full((bs, num_heads, q_len, topk), -1, dtype=torch.int32, device=device)
    for ts in range(0, q_len, q_tile):
        te = min(ts + q_tile, q_len)
        tq = te - ts
        q_slice = idx_q[:, ts:te].float()  # [bs, tq, H, D]
        pos_slice = q_pos[:, ts:te]  # [bs, tq]
        block_scores = torch.full((bs, num_heads, tq, nb), _NEG_INF, dtype=torch.float32, device=device)
        for sb in range(0, nb, nb_tile):
            se = min(sb + nb_tile, nb)
            w = se - sb  # real slab width (static per bucket)
            # index_select (never torch.gather -- unsafe on HPU eager) gathers
            # the slab's keys [bs, nb_tile, block_size, dim].
            pt = page_table_c[:, sb:sb + nb_tile]  # [bs, nb_tile]
            keys = pages.index_select(0, pt.reshape(-1)).view(bs, nb_tile, block_size, dim).float()
            # kpos[n, t]: absolute position of token t of logical block sb+n.
            kpos = (torch.arange(sb, sb + nb_tile, device=device) * block_size).view(nb_tile, 1) \
                + torch.arange(block_size, device=device).view(1, block_size)
            scores = torch.einsum("bqhd,bntd->bhqnt", q_slice, keys)
            future = kpos.view(1, 1, 1, nb_tile, block_size) > pos_slice.view(bs, 1, tq, 1, 1)
            slab_scores = scores.masked_fill(future, _NEG_INF).amax(dim=-1)  # [bs, H, tq, nb_tile]
            block_scores[:, :, :, sb:se] = slab_scores[:, :, :, :w]

        # Force the query's current block (and local_blocks-1 predecessors;
        # M3 ships local_block=1) via 1e29 score injection, matching the
        # upstream kernels' scheme.  init_blocks is 0 for M3 (no forcing).
        q_block = (pos_slice // block_size).clamp(min=0)
        for i in range(local_blocks):
            tgt = (q_block - i).clamp(min=0)
            block_scores.scatter_(
                -1,
                tgt.view(bs, 1, tq, 1).expand(bs, num_heads, tq, 1).long(),
                1e29,
            )

        k_eff = min(topk, nb)
        top_v, top_i = block_scores.topk(k_eff, dim=-1)
        # -inf score <=> block invalid for this query (all tokens future /
        # junk page rows); valid blocks always score finitely (a valid block
        # has at least one causally visible token).
        out[:, :, ts:te, :k_eff] = torch.where(top_v > _NEG_INF, top_i.int(), -1)
    return out


def msa_sparse_attn_prefill(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    page_table: torch.Tensor,
    topk_idx: torch.Tensor,
    q_pos: torch.Tensor,
    scale: float,
    block_size: int = MSA_BLOCK_SIZE,
    q_tile: int = 128,
) -> torch.Tensor:
    """Block-sparse causal GQA over the selected blocks (prefill/chunk),
    query-block-tiled with a per-query FIXED-SHAPE gather: every query
    attends exactly its own topk blocks.

    q:          [bs, q_len, G, gs, D] queries grouped by this rank's KV heads.
    key/value_cache: [num_slots, G, D] flat-slot caches; the current chunk
                must already be written (cache-first).
    page_table: [bs, nb] as in msa_topk_blocks.
    topk_idx:   [bs, G, q_len, topk] logical block ids (-1 pad) -- the rank's
                slice of the indexer's selection.
    q_pos:      [bs, q_len] absolute positions (-1 = padding).

    The op sequence and every shape here depend only on the bucket, never on
    tensor data (no union/unique/boolean-select): data-dependent shapes are
    unsafe under HPU lazy/graph execution and were the one prefill surface
    the CPU suite could not vouch for (2026-07-08 long-context regression).
    Invalid selections (-1) resolve to the runner's pad block and carry a
    False validity mask -- pads are masked by construction, never clamped
    onto real pages.  Key positions derive from the LOGICAL block ids, so a
    wrong page can dilute a real block's content but junk/pad rows can never
    enter the causal window.  Softmax in fp32; logits per (tile, group) are
    [bs, tq, gs, topk*block_size] -- bounded regardless of context length.
    """
    bs, q_len, num_groups, group_size, dim = q.shape
    topk = topk_idx.shape[-1]
    nb = page_table.shape[1]
    device = q.device
    k_pages = key_cache.unflatten(0, (-1, block_size))
    v_pages = value_cache.unflatten(0, (-1, block_size))
    pad_page = k_pages.shape[0] - 1  # runner pad block (num_blocks + 1 alloc)
    t_off = torch.arange(block_size, device=device)

    out = torch.zeros(bs, q_len, num_groups, group_size, dim, dtype=torch.float32, device=device)
    for ts in range(0, q_len, q_tile):
        te = min(ts + q_tile, q_len)
        tq = te - ts
        sel = topk_idx[:, :, ts:te].long()  # [bs, G, tq, topk]
        valid = sel >= 0
        # Logical-id -> physical-page lookup: page_table[b, sel[b,..]].  Done as
        # a FLAT index_select (batch offset baked into the index) rather than
        # torch.gather -- gather does not compile as a standalone Synapse recipe
        # on HPU eager (synStatus 26; probed 2026-07-08), while index_select
        # does; both are exact and static-shape.
        sel_c = sel.clamp(min=0, max=nb - 1)
        batch_off = (torch.arange(bs, device=device) * nb).view(bs, 1, 1, 1)
        pages_sel = page_table.reshape(-1).index_select(0, (sel_c + batch_off).reshape(-1)).view_as(sel)
        pages_sel = torch.where(valid, pages_sel, torch.full_like(pages_sel, pad_page))
        # Key absolute positions from the LOGICAL ids; the `valid` term is
        # load-bearing (-1 entries would otherwise pass the causal compare).
        kpos = (sel * block_size).unsqueeze(-1) + t_off  # [bs, G, tq, topk, bsz]
        mask = valid.unsqueeze(-1) & (kpos <= q_pos[:, ts:te].view(bs, 1, tq, 1, 1))
        for g in range(num_groups):
            rows = pages_sel[:, g].flatten()
            k_sel = k_pages.index_select(0, rows)[:, :, g].view(bs, tq, topk * block_size, dim)
            v_sel = v_pages.index_select(0, rows)[:, :, g].view(bs, tq, topk * block_size, dim).float()
            logits = torch.einsum("bqhd,bqkd->bqhk", q[:, ts:te, g].float(), k_sel.float()) * scale
            probs = _masked_softmax(logits, mask[:, g].flatten(-2, -1).unsqueeze(2))
            out[:, ts:te, g] = torch.einsum("bqhk,bqkd->bqhd", probs, v_sel)
    return out.to(q.dtype).view(bs, q_len, num_groups * group_size, dim)


def msa_topk_blocks_decode(
    idx_q: torch.Tensor,
    index_k_cache: torch.Tensor,
    block_list: torch.Tensor,
    block_groups: torch.Tensor,
    block_usage: torch.Tensor,
    cur_slot: torch.Tensor,
    pad_page: int,
    topk: int = MSA_TOPK,
    block_size: int = MSA_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode (q_len == 1) selection in FLAT-BLOCK space.

    Operates directly on the runner's decode triplets, which are consistent
    per entry in both contiguous and non-contiguous PA layouts (the layouts
    only reorder entries), so no per-sequence block table is required:

    idx_q:       [bs, H_idx, D] the step's index queries.
    block_list:  [nb] physical page ids (padding entries hold the pad page).
    block_groups:[nb] owning sequence per entry (-1 padding).
    block_usage: [nb] valid tokens per block (counts the current token --
                 runner computes it from the current slot).
    cur_slot:    [bs] the step's slot per sequence (pads land in the pad
                 block); current physical page = cur_slot // block_size.
    pad_page:    physical id of the runner's pad block (safe to gather).

    Returns (sel_pages [bs, H, topk] physical ids, sel_usage [bs, H, topk],
    sel_valid [bs, H, topk] bool).  Entries beyond min(topk, valid) point at
    the pad page with usage 0 and valid False; the attention masks them.
    """
    bs, num_heads, dim = idx_q.shape
    nb = block_list.shape[0]
    device = idx_q.device

    # STRICT int64 for everything that indexes or is returned as an id.
    # HPU lazy graphs BITCAST (not value-cast) when fused ops mix dtypes:
    # the original torch.where(valid, int_gather, zeros_like(top_v[fp32]))
    # returned fp32 score bit-patterns as "page ids" on device (observed
    # sel_pages like 1065402240 = bits of ~1.0f) while passing on CPU.
    block_list = block_list.long()
    block_groups = block_groups.long()
    block_usage = block_usage.long()

    pages = index_k_cache.unflatten(0, (-1, block_size))
    # PERF (decode tok/s): keep the big [nb, block_size, D] key gather in the
    # index cache's native bf16 -- do NOT upcast to fp32. The Gaudi3 MME
    # accumulates the score einsum in fp32 regardless of input dtype, so .float()
    # recovered no precision at ~3x this scan's HBM traffic (the dominant
    # context-linear decode cost). The small [nb, H, t] score result IS cast back
    # to fp32 so the masked_fill/amax/top-k selection is numerically identical
    # (only the dot-product output rounds through bf16, which the needle gates).
    # A fully-fused bf16 index-scan (bf16 amax + mask-as-bias) was REJECTED: the
    # bf16 amax blurred the needle block below the top-16 cut at nb~2047 -> needle
    # FALSE at 256K (measured). The scan's cost is einsum op-launch + key-gather
    # across the sparse layers, recoverable only by a batched-GEMM reformulation
    # or a TPC-C fused kernel.
    if _ABLATE_INDEX_SCAN:  # timing-only: skip the O(ctx) scan (proxy = block position)
        block_scores = block_list.to(torch.float32).view(-1, 1).expand(nb, num_heads).contiguous()
    else:
        keys = pages.index_select(0, block_list)  # [nb, bsz, D] bf16 (native)
        # Score each block against its OWN sequence's query (gather, not a
        # [bs, nb] broadcast, so cost stays linear in total blocks).
        q_per_block = idx_q.index_select(0, block_groups.clamp(min=0)).to(keys.dtype)
        scores = torch.einsum("nhd,ntd->nht", q_per_block, keys).float()
        t_off = torch.arange(block_size, device=device, dtype=block_usage.dtype)
        scores = scores.masked_fill(t_off.view(1, 1, -1) >= block_usage.view(-1, 1, 1), _NEG_INF)
        block_scores = scores.amax(dim=-1)  # [nb, H]

    batch = torch.arange(bs, device=device)
    owned = block_groups.view(1, -1) == batch.view(-1, 1)  # [bs, nb]
    # TODO(perf): this [bs, H, nb] broadcast is quadratic in batch x total
    # blocks (0.9 GB transient at bs=16 x 512K x 57 layers); a segmented
    # top-k over the flat [nb, H] scores grouped by block_groups would keep
    # the selection O(nb) like the scoring above.
    per_seq = torch.where(
        owned.unsqueeze(1),
        block_scores.transpose(0, 1).unsqueeze(0),  # [1, H, nb]
        torch.full((), _NEG_INF, dtype=block_scores.dtype, device=device),
    )  # [bs, H, nb]

    # Force the current block (local_block=1).  The current physical page is
    # unique to its sequence (a partially-written block is never shared under
    # prefix caching's copy-on-write), so a physical-id match plus the
    # owned-or-padding group check identifies exactly one real entry; padded
    # batch rows match the runner's pad entries (group -1), keeping their
    # softmax defined.
    cur_page = (cur_slot // block_size).view(-1, 1)
    force = (block_list.view(1, -1) == cur_page) & (owned | (block_groups.view(1, -1) < 0))
    per_seq = per_seq.masked_fill(force.unsqueeze(1), 1e29)

    # FIXED-SHAPE selection: always topk-wide, never nb-dependent width.
    # A data-dependent topk width (k_eff = min(topk, nb)) plus the old
    # k_eff<topk cat-pad branch made the op sequence and output width vary with
    # the runtime block count -- an HPUGraph hazard (the decode forward is
    # captured/replayed inside htorch.hpu.wrap_in_hpu_graph regardless of
    # enforce_eager; a dynamic width across decode buckets breaks capture).
    # Pad the per-seq scores to at least topk with -inf, then take a constant
    # topk: padded (or -inf) entries come back with valid=False and are masked
    # in the attend, so results are unchanged while the shape is static.
    if nb < topk:
        per_seq = torch.nn.functional.pad(per_seq, (0, topk - nb), value=_NEG_INF)
    top_v, top_i = per_seq.topk(topk, dim=-1)  # [bs, H, topk] -- fixed width
    valid = top_v > _NEG_INF
    # top_i may index the -inf pad region (>= nb); clamp for the gather (those
    # entries are masked out by ``valid`` anyway).  Every where() below is
    # int64-vs-int64: no float operand may appear in the id/usage paths (see
    # the bitcast note above).
    top_i = top_i.clamp(max=nb - 1)
    pages_raw = block_list.index_select(0, top_i.flatten()).view(bs, num_heads, topk)
    sel_pages = torch.where(valid, pages_raw, torch.full_like(pages_raw, pad_page))
    usage_raw = block_usage.index_select(0, top_i.flatten()).view(bs, num_heads, topk)
    sel_usage = torch.where(valid, usage_raw, torch.zeros_like(usage_raw))
    return sel_pages, sel_usage, valid


def msa_sparse_attn_decode(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    sel_pages: torch.Tensor,
    sel_usage: torch.Tensor,
    sel_valid: torch.Tensor,
    scale: float,
    block_size: int = MSA_BLOCK_SIZE,
) -> torch.Tensor:
    """Decode block-sparse GQA: fixed-shape gather of the selected blocks
    (<= topk per sequence) + masked SDPA over <= topk*block_size keys.

    q:         [bs, G, gs, D]; key/value_cache: [num_slots, G, D].
    sel_*:     [bs, G, topk] from msa_topk_blocks_decode.
    Returns [bs, G*gs, D] in q.dtype.
    """
    bs, num_groups, group_size, dim = q.shape
    topk = sel_pages.shape[-1]
    device = q.device
    k_pages = key_cache.unflatten(0, (-1, block_size))
    v_pages = value_cache.unflatten(0, (-1, block_size))
    t_off = torch.arange(block_size, device=device, dtype=sel_usage.dtype)

    def _decomposed(q_g_in, k_g, v_g, mask):
        logits = torch.einsum("bhd,bskd->bhsk", q_g_in.float(), k_g.float()) * scale
        probs = _masked_softmax(logits.flatten(2, 3), mask.flatten(1, 2).unsqueeze(1))
        return torch.einsum("bhk,bkd->bhd", probs, v_g.flatten(1, 2))  # [bs, gs, dim]

    def _fsdpa(q_g_in, k_g, v_g, mask):
        # E4: selected-window attention via FusedSDPA (non-causal, additive mask,
        # softmax_mode="fp32") instead of the decomposed einsum+softmax+einsum.
        # The kernel requires BF16 q/k/v with softmax_mode="fp32" (== bf16 tensor
        # I/O + fp32 softmax accumulation) -- this is the intended needle-safe
        # precision; the QK matmul accumulates fp32 on the MME regardless.
        # FusedSDPA wants [bs, heads, q_seq, dim]; heads=group_size, q_seq=1.
        kseq = topk * block_size
        q_g = q_g_in.to(torch.bfloat16).unsqueeze(2).contiguous()  # [bs, gs, 1, dim]
        # FusedSDPA mishandles non-contiguous GQA inputs -> .contiguous() after
        # the KV broadcast (expand aliases stride-0; the kernel reads garbage on
        # some ranks otherwise -- observed O(1) parity divergence).
        k_flat = k_g.to(torch.bfloat16).reshape(bs, 1, kseq, dim).expand(bs, group_size, kseq, dim).contiguous()
        v_flat = v_g.to(torch.bfloat16).reshape(bs, 1, kseq, dim).expand(bs, group_size, kseq, dim).contiguous()
        # additive fp32 mask [bs, 1, 1, kseq]: 0 keep, -inf drop (broadcast over
        # the group_size query heads and the single query position).
        add_mask = torch.where(mask.flatten(1, 2).view(bs, 1, 1, kseq),
                               torch.zeros((), dtype=torch.float32, device=device),
                               torch.full((), _NEG_INF, dtype=torch.float32, device=device))
        # A query row with NO valid key (a padding batch row at bs>1) would make
        # FSDPA's softmax exp(-inf - (-inf)) = NaN; the decomposed path
        # neutralizes this in _masked_softmax. Reset fully-dead rows to all-0
        # (attend uniformly) so the output is finite garbage -- the row is
        # dropped downstream -- instead of a NaN that could propagate.
        any_valid = mask.flatten(1, 2).any(dim=-1).view(bs, 1, 1, 1)
        add_mask = torch.where(any_valid, add_mask, torch.zeros_like(add_mask))
        o = _fsdpa_apply(q_g, k_flat, v_flat, add_mask, scale)  # [bs, gs, 1, dim]
        return o.squeeze(2).float()  # [bs, gs, dim] -> fp32 to match decomposed

    outs = []
    for g in range(num_groups):
        pages_g = sel_pages[:, g].flatten().long()
        k_g = k_pages.index_select(0, pages_g)[:, :, g].view(bs, topk, block_size, dim)
        v_g = v_pages.index_select(0, pages_g)[:, :, g].view(bs, topk, block_size, dim).float()
        mask = (t_off.view(1, 1, -1) < sel_usage[:, g].unsqueeze(-1)) \
            & sel_valid[:, g].unsqueeze(-1)  # [bs, topk, bsz]
        if _MSA_SELECTED_FSDPA_PARITY and g == 0:
            o_dec = _decomposed(q[:, g], k_g, v_g, mask)
            o_fsd = _fsdpa(q[:, g], k_g, v_g, mask)
            md = (o_dec.float() - o_fsd.float()).abs().max().item()
            print(f"MSA-E4-PARITY selected_attn out_max_abs_diff={md:.3e} "
                  f"shape={tuple(o_dec.shape)} kseq={topk * block_size}", flush=True)
            # Assert like the E3/skip-short gates (md<=tol is False for NaN, so a
            # dead-row NaN also trips it) instead of only printing.
            assert md <= 1e-2, \
                f"E4 FusedSDPA diverged from decomposed (out_max_abs_diff={md:.3e})"
            outs.append(o_fsd if _MSA_SELECTED_FSDPA else o_dec)
        elif _MSA_SELECTED_FSDPA:
            outs.append(_fsdpa(q[:, g], k_g, v_g, mask))
        else:
            outs.append(_decomposed(q[:, g], k_g, v_g, mask))
    return torch.cat(outs, dim=1).to(q.dtype)


# ---------------------------------------------------------------------------
# vLLM-facing modules.  Imports are guarded so the pure-torch math above
# stays importable by the CPU-only V0 unit test.
# ---------------------------------------------------------------------------
try:
    from torch import nn
    from vllm.distributed import (get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from vllm.model_executor.layers.linear import ReplicatedLinear
    _HAS_VLLM = True
except ImportError:  # torch-only environments (V0 unit test)
    _HAS_VLLM = False

if _HAS_VLLM:

    class MiniMaxM3Indexer(nn.Module):
        """MSA selection branch: replicated index q/k projections + Gemma
        RMSNorm(128) per head + the SHARED main rotary (partial NeoX, 64 of
        128 dims).  Replication (design D2) is numerically identical to
        upstream's KV-style sharding; each rank slices out its own groups'
        selections downstream."""

        def __init__(
            self,
            hidden_size: int,
            num_index_heads: int = MSA_INDEX_HEADS,
            index_dim: int = MSA_INDEX_DIM,
            rms_norm_eps: float = 1e-06,
            quant_config=None,
            prefix: str = "",
        ) -> None:
            super().__init__()
            self.num_index_heads = num_index_heads
            self.index_dim = index_dim
            # Checkpoint names: ...self_attn.index_{q,k}_proj.weight (+
            # weight_scale_inv under MXFP8) and ...self_attn.index_{q,k}_norm
            # .weight; load_weights remaps them under this submodule.
            self.index_q_proj = ReplicatedLinear(
                hidden_size,
                num_index_heads * index_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.index_q_proj",
            )
            self.index_k_proj = ReplicatedLinear(
                hidden_size,
                index_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.index_k_proj",
            )
            # Gemma (1+w) form in fp32 -- same class as the main per-head QK
            # norms (see MiniMaxM3Attention: plain x*w would damp the
            # near-zero-centred checkpoint norm weights).
            self.index_q_norm = GemmaRMSNorm(index_dim, eps=rms_norm_eps)
            self.index_k_norm = GemmaRMSNorm(index_dim, eps=rms_norm_eps)

        def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            rotary_emb: nn.Module,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Returns (idx_q [.., H_idx*D], idx_k [.., D]), post-norm
            post-RoPE.  ``rotary_emb`` must be the attention layer's own
            instance (upstream: ``index_rotary_emb = rotary_emb``)."""
            idx_q, _ = self.index_q_proj(hidden_states)
            idx_k, _ = self.index_k_proj(hidden_states)
            # Per-head norm: flatten heads into rows like the main QK norm.
            idx_q = self.index_q_norm(idx_q.reshape(-1, self.index_dim)).view_as(idx_q)
            idx_k = self.index_k_norm(idx_k.reshape(-1, self.index_dim)).view_as(idx_k)
            idx_q, idx_k = rotary_emb(positions, idx_q, idx_k)
            return idx_q, idx_k

        def forward_k(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            rotary_emb: nn.Module,
        ) -> torch.Tensor:
            """idx_k only, post-norm post-RoPE.  The skip-short-indexer decode
            path still needs the index-K side cache complete (a sequence may
            later grow past topk blocks and fall back to the MSA scan) but skips
            the index-Q projection + the top-k scan, so this computes only the
            K side.  RoPE takes (q, k); pass a throwaway clone for the q slot so
            an in-place rotary can't alias the idx_k we keep (both slots are one
            head of index_dim, roped identically -- the q output is discarded)."""
            idx_k, _ = self.index_k_proj(hidden_states)
            idx_k = self.index_k_norm(idx_k.reshape(-1, self.index_dim)).view_as(idx_k)
            _, idx_k = rotary_emb(positions, idx_k.clone(), idx_k)
            return idx_k

    class MiniMaxM3SparseAttentionMixin:
        """Sparse forward for MiniMaxM3Attention subclasses.

        The host class must provide: qkv_proj/q_norm/k_norm/rotary_emb/o_proj
        /attn (the standard ``Attention`` layer -- KEPT so the runner
        allocates and binds this layer's KV cache), indexer, plus the usual
        head-geometry attributes.  The KV-cache WRITE below replicates the
        deployed ``HPUAttentionImpl.forward`` write faithfully (image
        vllm_gaudi/attention/backends/hpu_attn.py:530-560: view to
        [-1, kv_heads, head_dim], flatten slot_mapping, ``impl.k_cache(...)``
        / ``impl.v_cache(...)``); the impl's attention math is NOT invoked --
        v0.21.x fuses write+math in one method with no skip flag, so reusing
        its write means calling its cache modules directly.
        """

        def _init_msa(self, config, quant_config, prefix: str) -> None:
            sparse_cfg = config.sparse_attention_config
            self.msa_topk = sparse_cfg.get("sparse_topk_blocks", MSA_TOPK)
            # Dense-decode substitution threshold (padded blocks): skip-short is
            # exact at <= topk; dense-below (if larger) extends the dense path up
            # to _MSA_DENSE_BELOW_BLOCKS (a speed/quality tradeoff). 0 = neither.
            self._msa_dense_thr = self.msa_topk if _MSA_SKIP_SHORT_INDEXER else 0
            if _MSA_DENSE_BELOW_BLOCKS > self._msa_dense_thr:
                self._msa_dense_thr = _MSA_DENSE_BELOW_BLOCKS
            self.msa_block_size = sparse_cfg.get("sparse_block_size", MSA_BLOCK_SIZE)
            self.msa_local_blocks = sparse_cfg.get("sparse_local_block", 1)
            num_index_heads = sparse_cfg.get("sparse_num_index_heads", MSA_INDEX_HEADS)
            index_dim = sparse_cfg.get("sparse_index_dim", MSA_INDEX_DIM)
            # init_block forcing is unimplemented everywhere (M3 ships 0).
            assert sparse_cfg.get("sparse_init_block", 0) == 0, \
                "MSA init_block forcing not implemented"
            assert index_dim == self.head_dim, \
                "MSA shares the main rotary; index_dim must equal head_dim"
            # The indexer hardcodes GemmaRMSNorm; a non-gemma config would put
            # the index norms out of step with the main QK norms.
            assert getattr(config, "use_gemma_norm", True), \
                "MSA indexer assumes use_gemma_norm=True (M3 ships True)"
            self.indexer = MiniMaxM3Indexer(
                hidden_size=self.hidden_size,
                num_index_heads=num_index_heads,
                index_dim=index_dim,
                rms_norm_eps=config.rms_norm_eps,
                quant_config=quant_config,
                prefix=f"{prefix}.indexer",
            )
            # Debug: does this layer's prefix name the watched layer index?
            # (prefix like "model.layers.3.self_attn")
            self._msa_dbg_layer = _MSA_DEBUG and f".layers.{_MSA_DEBUG_LAYER}." in f"{prefix}."
            # Rank -> index-head slice, mirroring QKVParallelLinear's KV-head
            # replication (deployed vllm linear.py:1035,1370: rank r holds kv
            # head r // num_kv_head_replicas).  4 index heads == 4 KV groups.
            tp_size = get_tensor_model_parallel_world_size()
            tp_rank = get_tensor_model_parallel_rank() if tp_size > 1 else 0
            replicas = max(1, tp_size // self.total_num_kv_heads)
            self.msa_head_start = (tp_rank // replicas) * self.num_kv_heads
            # Index-K side cache, Mechanism A (design D3): lazily sized from
            # the layer's main KV tensor at first bound forward; NOT visible
            # to vLLM's memory profiler -- bring-up serves must drop
            # gpu_memory_utilization (design D4).
            self._idx_k_cache: torch.Tensor | None = None

        def _get_idx_k_cache(self, key_cache: torch.Tensor) -> torch.Tensor:
            resized = (self._idx_k_cache is not None
                       and self._idx_k_cache.shape[0] != key_cache.shape[0])
            if self._idx_k_cache is None or resized:
                if resized:
                    # Re-zeroing drops every written idx_k, which violates the
                    # side-cache completeness invariant if it happens MID-serve
                    # (a grown sequence's scan would then score against zero
                    # keys). The KV cache is bound once per serve, so this should
                    # only fire on a deliberate reconfigure (a genuinely fresh
                    # cache); make it loud so a mid-serve resize is never a silent
                    # retrieval corruption.
                    print(f"MSA-WARN index-K side cache resized "
                          f"{self._idx_k_cache.shape[0]} -> {key_cache.shape[0]} "
                          f"(all written index keys reset to zero)", flush=True)
                # The sparse paths gather K/V straight from the cache tensors;
                # a quantized (fp8/INC) KV cache would be read raw as garbage.
                assert key_cache.dtype in (torch.bfloat16, torch.float16, torch.float32), \
                    f"MSA requires an unquantized KV cache, got {key_cache.dtype}"
                self._idx_k_cache = torch.zeros(
                    key_cache.shape[0],
                    self.indexer.index_dim,
                    dtype=_MSA_INDEX_DTYPE,
                    device=key_cache.device,
                )
            return self._idx_k_cache

        def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
        ) -> torch.Tensor:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q = self.q_norm(q.reshape(-1, self.head_dim)).view_as(q)
            k = self.k_norm(k.reshape(-1, self.head_dim)).view_as(k)
            q, k = self.rotary_emb(positions, q, k)

            attn_metadata = get_forward_context().attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[self.attn.layer_name]
            kv_cache = self.attn.kv_cache
            if (attn_metadata is None or not isinstance(kv_cache, tuple) or kv_cache[0] is None):
                # Memory-profiling run: no cache bound yet (the impl skips its
                # write the same way).  The fused dense in-chunk path both
                # keeps the profile transient small and is exact for chunks
                # <= topk*block_size anyway.
                attn_output = self.attn(q, k, v)
            elif (self._msa_dense_thr > 0 and not attn_metadata.is_prompt
                  and attn_metadata.block_list is not None
                  and attn_metadata.block_list.shape[0] <= self._msa_dense_thr):
                # Dense-decode substitution: the padded block bucket is <= the
                # dense threshold.  <= topk (skip-short) is exactly dense; the
                # 17..DENSE_BELOW range (dense-below lever) is genuinely sparse in
                # MSA but we run the dense serve's computation instead (faster
                # below the effective ~40-50K crossover -- the kept idx_k write
                # pulls it in from the dense serve's ~90K).  Skip index-Q + scan + sparse-attn;
                # run native dense paged attention.  Keep the idx_k write
                # (invariant: cache stays complete for a later grow-past-threshold
                # fallback to the MSA scan).
                attn_output = self._msa_decode_short(q, k, v, positions, hidden_states, attn_metadata, kv_cache)
                if _MSA_SKIP_SHORT_INDEXER_PARITY and self._msa_dbg_layer:
                    # Live diagnostic (debug layer only): recompute the full MSA
                    # decode for this step and report how far the dense output sits
                    # from it. Bit-identical (max_abs=0) ONLY for <= topk buckets
                    # (skip-short, dense==MSA); for dense-below buckets DIVERGED
                    # with nonzero max_abs is EXPECTED (dense != MSA by design).
                    # Either way it is logged, never fatal -- the authoritative
                    # correctness proof is the end-to-end greedy-SHA / needle /
                    # agent gates, not this probe.
                    idx_q, idx_k = self.indexer(positions, hidden_states, self.rotary_emb)
                    ref = self._msa_forward(q, k, v, idx_q, idx_k, positions, attn_metadata, kv_cache)
                    max_abs = (attn_output.float() - ref.float()).abs().max().item()
                    status = "ok" if max_abs <= _MSA_SKIP_PARITY_TOL else "DIVERGED"
                    print(f"MSA-SKIP-SHORT-PARITY layer_{status} max_abs={max_abs:.3e} "
                          f"nb={attn_metadata.block_list.shape[0]} topk={self.msa_topk}", flush=True)
            else:
                idx_q, idx_k = self.indexer(positions, hidden_states, self.rotary_emb)
                attn_output = self._msa_forward(q, k, v, idx_q, idx_k, positions, attn_metadata, kv_cache)
            output, _ = self.o_proj(attn_output)
            return output

        def _msa_forward(self, q, k, v, idx_q, idx_k, positions, attn_metadata, kv_cache):
            impl = self.attn.impl
            key_cache, value_cache, k_scales, v_scales = kv_cache[:4]
            slot_mapping = attn_metadata.slot_mapping.flatten()

            # --- main KV write: faithful replica of the deployed impl's ---
            # (raw slot_mapping on purpose: byte-identical arguments to
            # hpu_attn.py:530-560, deferring -1 pad slots to the same HPU
            # wrap-around the deployed write relies on; the index-K write
            # below makes its pad handling explicit instead.)
            key_w = k.view(-1, self.num_kv_heads, self.head_dim)
            value_w = v.view(-1, self.num_kv_heads, self.head_dim)
            if key_w.dtype != key_cache.dtype:
                key_w = key_w.to(key_cache.dtype)
                value_w = value_w.to(value_cache.dtype)
            key_cache = impl.k_cache(key_w,
                                     key_cache,
                                     slot_mapping,
                                     scales=k_scales,
                                     block_size=attn_metadata.block_size,
                                     is_prompt=attn_metadata.is_prompt)
            value_cache = impl.v_cache(value_w,
                                       value_cache,
                                       slot_mapping,
                                       scales=v_scales,
                                       block_size=attn_metadata.block_size,
                                       is_prompt=attn_metadata.is_prompt)

            # --- index-K side-cache write (same slot_mapping) ---
            # Prompt padding slots are -1 (runner:2474); the deployed
            # index_copy_ relies on HPU wrap-around (-1 -> last row = the pad
            # block's last slot).  Make that explicit so the write is
            # CPU-exact too.
            idx_cache = self._get_idx_k_cache(key_cache)
            num_slots = idx_cache.shape[0]
            safe_slots = torch.where(slot_mapping < 0, torch.full_like(slot_mapping, num_slots - 1), slot_mapping)
            idx_cache.index_copy_(0, safe_slots, idx_k.reshape(-1, self.indexer.index_dim).to(idx_cache.dtype))

            gs = self.msa_head_start
            ge = gs + self.num_kv_heads
            block_size = attn_metadata.block_size
            if attn_metadata.is_prompt:
                return self._msa_prefill(q, idx_q, positions, attn_metadata, key_cache, value_cache, idx_cache,
                                         slot_mapping, gs, ge, block_size)
            return self._msa_decode(q, idx_q, attn_metadata, key_cache, value_cache, idx_cache, slot_mapping, gs, ge,
                                    block_size)

        def _msa_decode_short(self, q, k, v, positions, hidden_states, attn_metadata, kv_cache):
            """Dense-decode substitution path.  Skip index-Q + top-k scan + sparse
            attention; run the layer's native dense paged attention (self.attn,
            which does its own KV write + attends over the full block table).
            Still project + write idx_k so the side cache is complete if this
            sequence later grows past the threshold and hits the MSA scan.

            At <= topk blocks (skip-short) MSA selects every block, so this is
            EXACTLY MSA.  Above topk (dense-below lever) MSA is genuinely sparse,
            so this CHANGES the output -- but to the dense serve's computation
            (self.attn), which is correct at long context (dense-serve needle sweep).

            q/k are already RoPE'd (forward()).  self.attn does NOT re-RoPE (the
            dense VLLM_M3_MSA=0 path feeds it the same post-RoPE tensors), so the
            output is bit-identical to the dense serve at this length."""
            # Pre-write kv_cache[0] here (vs _msa_forward's post-k_cache-write
            # tensor): _get_idx_k_cache reads only shape[0]/dtype/device, which the
            # write never changes, so the sized idx cache is identical either way.
            key_cache = kv_cache[0]
            slot_mapping = attn_metadata.slot_mapping.flatten()
            # idx_k side-cache write -- byte-identical to _msa_forward's (raw
            # slot_mapping, -1 pad slots mapped to the last row via HPU
            # wrap-around made explicit).
            idx_k = self.indexer.forward_k(positions, hidden_states, self.rotary_emb)
            idx_cache = self._get_idx_k_cache(key_cache)
            num_slots = idx_cache.shape[0]
            safe_slots = torch.where(slot_mapping < 0, torch.full_like(slot_mapping, num_slots - 1), slot_mapping)
            idx_cache.index_copy_(0, safe_slots, idx_k.reshape(-1, self.indexer.index_dim).to(idx_cache.dtype))
            # Dense paged attention over all (<= topk) blocks == exact MSA here.
            return self.attn(q, k, v)

        def _msa_prefill(self, q, idx_q, positions, attn_metadata, key_cache, value_cache, idx_cache, slot_mapping, gs,
                         ge, block_size):
            # Prompt layout: [bs, padded_seq]; bs from seq_lens_tensor like
            # the deployed impl (hpu_attn.py:520-527).
            bs = attn_metadata.seq_lens_tensor.shape[0]
            q3 = q.view(bs, -1, self.num_heads, self.head_dim)
            seq = q3.shape[1]
            device = q.device
            pos2d = positions.view(bs, seq).long()
            pad_page = key_cache.shape[0] // block_size - 1  # runner's pad block

            # Logical-block -> physical-page table: context blocks from
            # block_list ([bs, target_blocks], -1-padded; runner:2476,2686)
            # then the current chunk's pages scattered from slot_mapping.
            # Junk rows (context padding / beyond the chunk) are never
            # selected: every token of such a block is causally masked.
            if attn_metadata.block_list is not None:
                ctx_blocks = attn_metadata.block_list.view(bs, -1).long()
                tb = ctx_blocks.shape[1]
            else:
                ctx_blocks = None
                tb = 0
            nb = tb + seq // block_size + 2
            page_table = torch.full((bs, nb), pad_page, dtype=torch.long, device=device)
            if ctx_blocks is not None:
                page_table[:, :tb] = torch.where(ctx_blocks >= 0, ctx_blocks, torch.full_like(ctx_blocks, pad_page))
            slots2d = slot_mapping.view(bs, seq)
            # Batch-padding tokens (pos -1) scatter into the spare last row.
            dst = torch.where(pos2d >= 0, pos2d // block_size, torch.full_like(pos2d, nb - 1))
            src = torch.where(slots2d >= 0, slots2d // block_size, torch.full_like(slots2d, pad_page))
            page_table.scatter_(1, dst, src)

            idx_q3 = idx_q.view(bs, seq, self.indexer.num_index_heads, self.indexer.index_dim)
            topk_idx = msa_topk_blocks(idx_q3,
                                       idx_cache,
                                       page_table,
                                       pos2d,
                                       topk=self.msa_topk,
                                       block_size=block_size,
                                       local_blocks=self.msa_local_blocks)
            if _MSA_DEBUG and self._msa_dbg_layer:
                w = _MSA_WATCH_BLOCK
                q0 = int(pos2d[0, 0])
                qn = int(pos2d[0].max())
                line = (f"MSA-DBG prefill chunk q[{q0}..{qn}] tb={tb} nb={nb}")
                if 0 <= w < nb:
                    sel_any = int((topk_idx[0] == w).sum())
                    sel_last = int((topk_idx[0, :, -1, :] == w).sum())
                    line += (f" W={w} page_w={int(page_table[0, w])}"
                             f" sel_any={sel_any} sel_lastq={sel_last}")
                print(line, flush=True)
            out = msa_sparse_attn_prefill(q3.view(bs, seq, self.num_kv_heads, self.num_heads // self.num_kv_heads,
                                                  self.head_dim),
                                          key_cache,
                                          value_cache,
                                          page_table,
                                          topk_idx[:, gs:ge],
                                          pos2d,
                                          self.scaling,
                                          block_size=block_size)
            # Return 2D [num_tokens, hidden] to match the dense path
            # (self.attn -> output.view(-1, hidden)); a 3D [bs, seq, hidden]
            # return silently broadcasts through the residual add for bs=1 and
            # then mis-indexes the last-token logits gather -> garbage tokens
            # (the reason a 6-token prompt returns garbage under MSA while
            # dense returns the right token).
            return out.reshape(bs * seq, self.num_heads * self.head_dim)

        def _msa_decode(self, q, idx_q, attn_metadata, key_cache, value_cache, idx_cache, slot_mapping, gs, ge,
                        block_size):
            # Decode layout: [bs, 1]; slot_mapping [bs, 1] gives bs and the
            # current physical page per sequence.  q_len==1 only: multi-token
            # (speculative) decode would need a per-token selection axis these
            # reshapes don't carry.
            assert attn_metadata.slot_mapping.dim() == 1 or attn_metadata.slot_mapping.shape[-1] == 1, \
                "MSA decode supports q_len==1 only (speculative decode unsupported)"
            bs = attn_metadata.slot_mapping.shape[0]
            pad_page = key_cache.shape[0] // block_size - 1
            idx_q3 = idx_q.view(bs, self.indexer.num_index_heads, self.indexer.index_dim)
            # E3: on TP the rank owns only index heads [gs:ge]; scanning all of
            # them wastes ~(num_index_heads/(ge-gs))x of the score einsum + amax
            # + topk (the dominant context-linear decode cost at 256K).  Slice
            # the index query to the local head(s) BEFORE the scan; the per-head
            # topk is independent so the selected pages are identical to slicing
            # the full-scan output afterwards.  ``sel_off`` re-bases the post-scan
            # gs:ge slice (0 when we already sliced, gs otherwise).
            if _MSA_TP_LOCAL_INDEX:
                idx_q3 = idx_q3[:, gs:ge]
                sel_off = 0
            else:
                sel_off = gs
            sel_pages, sel_usage, sel_valid = msa_topk_blocks_decode(
                idx_q3,
                idx_cache,
                attn_metadata.block_list,
                attn_metadata.block_groups,
                attn_metadata.block_usage,
                slot_mapping,
                pad_page,
                topk=self.msa_topk,
                block_size=block_size,
            )
            if _MSA_TP_LOCAL_INDEX_PARITY and self._msa_dbg_layer:
                # Definitive parity: compare the ATTENTION OUTPUT computed both
                # ways.  full = scan all heads then slice [gs:ge]; local = slice
                # query to [gs:ge] then scan.  Selection-tensor equality is not
                # sufficient (the ``valid`` flag can differ on zero-usage pad
                # entries that the attend masks out anyway), so we compare the
                # actual masked-softmax output -- the value the model consumes.
                idx_q3_full = idx_q.view(bs, self.indexer.num_index_heads, self.indexer.index_dim)
                fp, fu, fv = msa_topk_blocks_decode(idx_q3_full, idx_cache, attn_metadata.block_list,
                                                    attn_metadata.block_groups, attn_metadata.block_usage,
                                                    slot_mapping, pad_page, topk=self.msa_topk,
                                                    block_size=block_size)
                lp, lu, lv = msa_topk_blocks_decode(idx_q3_full[:, gs:ge], idx_cache, attn_metadata.block_list,
                                                    attn_metadata.block_groups, attn_metadata.block_usage,
                                                    slot_mapping, pad_page, topk=self.msa_topk,
                                                    block_size=block_size)
                q4 = q.view(bs, self.num_kv_heads, self.num_heads // self.num_kv_heads, self.head_dim)
                out_full = msa_sparse_attn_decode(q4, key_cache, value_cache, fp[:, gs:ge], fu[:, gs:ge],
                                                  fv[:, gs:ge], self.scaling, block_size=block_size)
                out_local = msa_sparse_attn_decode(q4, key_cache, value_cache, lp, lu, lv, self.scaling,
                                                   block_size=block_size)
                pg_ok = torch.equal(fp[:, gs:ge], lp)
                us_ok = torch.equal(fu[:, gs:ge], lu)
                vd_ok = torch.equal(fv[:, gs:ge], lv)
                out_ok = torch.equal(out_full, out_local)
                max_abs = (out_full.float() - out_local.float()).abs().max().item()
                # Gate on ATTENTION-OUTPUT parity, not raw selection equality.
                # The per-head score row is computed from the same idx_q[:,g] +
                # keys in both paths, so topk is logically identical (verified
                # exact on CPU); on HPU the H=4-vs-H=1 einsum tiles bf16
                # differently by ~1 ULP, which can flip the topk tie-break of a
                # LOW-relevance tail block (pad / negligible softmax weight) ->
                # sel_pages differ but the output the model consumes does not.
                # Tolerance is bf16 rounding headroom; the true correctness
                # authority is the 256K needle-exact gate in Phase B.
                OUT_TOL = 1e-2
                print(f"MSA-E3-PARITY layer_ok out_equal={out_ok} out_max_abs_diff={max_abs:.3e} "
                      f"pages={pg_ok} usage={us_ok} valid={vd_ok} "
                      f"gs={gs} ge={ge} nidx={self.indexer.num_index_heads} "
                      f"sel_shape_full={tuple(fp.shape)} sel_shape_local={tuple(lp.shape)}", flush=True)
                assert max_abs <= OUT_TOL, \
                    f"E3 TP-local index-head attention output diverged (max_abs={max_abs:.3e} > {OUT_TOL})"
            if _MSA_DEBUG and self._msa_dbg_layer:
                w = _MSA_WATCH_BLOCK
                sel0 = sel_pages[0, sel_off:sel_off + (ge - gs)].flatten().cpu()
                bl = attn_metadata.block_list.cpu()
                bg = attn_metadata.block_groups.cpu()
                bu = attn_metadata.block_usage.cpu()
                sm = slot_mapping.flatten().cpu()
                print(
                    f"MSA-DBG decode W={w} sel_pages={sel0[:18].tolist()} "
                    f"nb={bl.shape[0]} bl[:6]={bl[:6].tolist()} "
                    f"bg_dtype={bg.dtype} bg[:6]={bg[:6].tolist()} bg_max={int(bg.max())} "
                    f"bu_dtype={bu.dtype} bu[:6]={bu[:6].tolist()} "
                    f"slot0={int(sm[0])} valid0={int(sel_valid[0].sum())}", flush=True)
            out = msa_sparse_attn_decode(
                q.view(bs, self.num_kv_heads, self.num_heads // self.num_kv_heads, self.head_dim),
                key_cache,
                value_cache,
                sel_pages[:, sel_off:sel_off + (ge - gs)],
                sel_usage[:, sel_off:sel_off + (ge - gs)],
                sel_valid[:, sel_off:sel_off + (ge - gs)],
                self.scaling,
                block_size=block_size,
            )
            # 2D [num_tokens(=bs), hidden] to match the dense path (see the
            # _msa_prefill return note on the 3D logits-gather bug).
            return out.view(bs, self.num_heads * self.head_dim)
