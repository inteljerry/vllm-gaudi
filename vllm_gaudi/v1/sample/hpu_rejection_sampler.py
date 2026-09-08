# SPDX-License-Identifier: Apache-2.0

from vllm.v1.sample import rejection_sampler
import torch
from typing import Optional
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, random_sample
# The same threshold `Sampler.apply_temperature` uses to spot a greedy request.
# It has to be a threshold rather than `== GREEDY_TEMPERATURE`: HPUInputBatch
# stores -1.0 for greedy requests while `rejection_sampler.GREEDY_TEMPERATURE`
# is 0, so an equality test would miss every greedy request on HPU.
from vllm.v1.sample.sampler import _SAMPLING_EPS

PLACEHOLDER_TOKEN_ID = rejection_sampler.PLACEHOLDER_TOKEN_ID
GREEDY_TEMPERATURE = rejection_sampler.GREEDY_TEMPERATURE


def rejection_sample_pytorch(
    padded_draft_token_ids: torch.Tensor,
    padded_target_token_ids: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
) -> torch.Tensor:
    """
    Performs vectorized rejection sampling on a batch of token sequences.

    This function compares draft tokens to target tokens and accepts them up to 
    the first mismatch. If an entire sequence of draft tokens is accepted, a 
    bonus token is appended. This version handles variable numbers of draft 
    tokens per sequence.

    The current HPU implementation of spec decode will flatten the num_draft_tokens
    to 1. And so the shape size of padded_draft_token_ids will be
    the [real batch size * num_draft_tokens, 1].

    Args:
        padded_draft_token_ids (torch.Tensor): A 2D tensor of draft tokens.
            Shape: (num_seqs * max_draft_tokens, 1)
        padded_target_token_ids (torch.Tensor): A 1D tensor of target tokens
            predicted by the main model.
            Shape: (num_seqs * max_draft_tokens)
        bonus_token_ids (torch.Tensor): A single bonus token for each sequence,
            to be used if all draft tokens are accepted.
            Shape: (num_seqs, 1)
        num_draft_tokens: list[int]: List of number draft tokens for each sequence.
            Shape: (num_seqs)
        cu_num_draft_tokens (torch.Tensor): The cumulative sum of the number of
            draft tokens for each request. Used to determine actual sequence 
            lengths. Shape: (num_seqs,)

    Returns:
        torch.Tensor: The resulting tensor of accepted tokens.
            Shape: (num_seqs, max_draft_tokens + 1)
    """
    # 0. wait for device processing to finish
    # NOTE(chendi): Found CPU processing is faster than HPU for this step.
    padded_draft_token_ids = padded_draft_token_ids.cpu().to(torch.int32)
    padded_target_token_ids = padded_target_token_ids.cpu().to(torch.int32)
    bonus_token_ids = bonus_token_ids.cpu().to(torch.int32)
    cu_num_draft_tokens = cu_num_draft_tokens.cpu()
    # 1. Get tensor dimensions and device for calculations
    num_seqs = len(num_draft_tokens)
    padded_draft_token_ids = padded_draft_token_ids.view(num_seqs, -1)
    max_draft_tokens = padded_draft_token_ids.shape[-1]
    padded_target_token_ids = padded_target_token_ids.view(num_seqs, -1)
    bonus_token_ids = bonus_token_ids.view(num_seqs, -1)
    device = padded_draft_token_ids.device

    # 2. Calculate the number of draft tokens for each sequence from the
    # cumulative sum
    start_indices = torch.cat((torch.tensor([0], device=device,
                                            dtype=cu_num_draft_tokens.dtype), cu_num_draft_tokens[:-1]))
    num_draft_tokens_per_seq = cu_num_draft_tokens - start_indices

    # 3. Find the first mismatch, ignoring padding tokens
    # Create a mask to only consider valid tokens for each sequence
    pos = torch.arange(max_draft_tokens, device=device)
    valid_token_mask = pos < num_draft_tokens_per_seq.unsqueeze(-1)

    matches = (padded_draft_token_ids == padded_target_token_ids)

    mismatches = ~matches
    any_mismatch = mismatches.any(dim=1)
    # For sequence that the num draft tokens is 0, always consider all match
    any_mismatch[num_draft_tokens_per_seq == 0] = False
    first_mismatch_idx = torch.argmax(mismatches.int(), dim=1)

    # 4. Determine the number of accepted tokens for each sequence
    # If a mismatch occurs, we accept tokens up to and including the mismatch.
    # If no mismatch, accept all *actual* draft tokens.
    num_accepted = ((first_mismatch_idx + 1) * any_mismatch + num_draft_tokens_per_seq * (~any_mismatch))

    # 5. Create the output tensor by masking the target tokens
    # Initialize the output tensor with the padding value.
    # Create output buffer.
    output_tokens = torch.empty(
        (num_seqs, max_draft_tokens + 1),
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )
    output_tokens.fill_(PLACEHOLDER_TOKEN_ID)

    # Create a mask that is True for all positions up to the number of
    # accepted tokens.
    acceptance_mask = pos < num_accepted.unsqueeze(-1)
    acceptance_mask = acceptance_mask & valid_token_mask

    # Use the mask to copy the accepted target tokens into the output tensor.
    output_slice = output_tokens[:, :max_draft_tokens]
    output_slice[acceptance_mask] = padded_target_token_ids[acceptance_mask]

    # 6. Add the bonus token where all draft tokens were accepted
    # Create a boolean mask for sequences where all drafts were a match.
    all_accepted_mask = ~any_mismatch

    # If any sequences were fully accepted, place the bonus tokens.
    if all_accepted_mask.sum() > 0:
        # Get the column indices (positions) for the bonus tokens using the mask
        bonus_pos_indices = num_draft_tokens_per_seq[all_accepted_mask].long()

        # Get the corresponding bonus token values using the mask.
        bonus_values = bonus_token_ids[all_accepted_mask].squeeze(-1)

        # Place the bonus tokens using boolean indexing for rows and integer
        # indexing for columns.
        output_tokens[all_accepted_mask, bonus_pos_indices] = bonus_values

    return output_tokens


def expand_to_draft_rows(x: torch.Tensor, num_seqs: int, num_rows: int) -> torch.Tensor:
    """Repeat a per-request tensor over that request's draft-token rows.

    The HPU decode path lays the target logits out as `num_seqs` fixed-size
    blocks of `max_spec_len` rows (see `_prepare_spec_decode_inputs`), padding
    each block out to the batch-wide maximum instead of packing the rows by
    actual draft count. So the expansion is a plain repeat, not upstream's
    `cu_num_draft_tokens`-driven gather.
    """
    rows_per_seq, remainder = divmod(num_rows, num_seqs)
    assert remainder == 0, (f"target logits rows ({num_rows}) are not a whole number of blocks "
                            f"for {num_seqs} sequences; the HPU spec decode layout is expected "
                            "to pad every request to the same number of draft rows")
    return x.view(num_seqs, 1).expand(num_seqs, rows_per_seq).reshape(-1)


def apply_sampling_constraints(
    # [num_tokens, vocab_size]
    logits: torch.Tensor,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """Scale the target logits by temperature and apply top-k/top-p.

    Replaces the upstream helper of the same name, which expands the
    per-request parameters with a Triton kernel (`expand_batch_to_tokens`).
    Triton is disabled on HPU, so the expansion is done with plain torch ops
    over the HPU block layout instead.
    """
    assert logits.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    if sampling_metadata.all_greedy:
        # Same fast path as upstream: greedy requests read the raw argmax.
        return logits

    num_seqs = cu_num_draft_tokens.shape[0]
    num_rows = logits.shape[0]

    temperature = expand_to_draft_rows(sampling_metadata.temperature, num_seqs, num_rows)
    # Greedy rows keep their logits untouched so their argmax stays exact.
    temperature = torch.where(temperature < _SAMPLING_EPS, 1.0, temperature)
    logits.div_(temperature.unsqueeze(-1))

    top_k = None
    if sampling_metadata.top_k is not None:
        top_k = expand_to_draft_rows(sampling_metadata.top_k, num_seqs, num_rows)
    top_p = None
    if sampling_metadata.top_p is not None:
        top_p = expand_to_draft_rows(sampling_metadata.top_p, num_seqs, num_rows)
    # Masking the tail of the distribution never moves the argmax, so greedy
    # rows are unaffected by top-k/top-p as well.
    return apply_top_k_top_p(logits, top_k, top_p)


def select_target_token_ids(
    # [num_tokens, vocab_size]
    target_logits: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    sampling_metadata: SamplingMetadata,
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    """Pick the target model's token for every draft position.

    Greedy requests take the argmax. The rest draw from the target distribution
    that `apply_sampling_constraints` has already scaled, which keeps the
    emitted tokens distributed exactly as the target model would sample them:
    a draft token is accepted only when it equals that draw, and the draw
    itself is what gets emitted on a mismatch.
    """
    if sampling_metadata.all_greedy:
        return target_logits.argmax(dim=-1)

    num_seqs = len(num_draft_tokens)
    num_rows = target_logits.shape[0]
    rows_per_seq = num_rows // num_seqs
    # `random_sample` keys generators by row, so fan each seeded request's
    # generator out over the rows of its block. Only over its *real* draft rows
    # though: `random_sample` draws once per key, and the padding rows of a
    # block are sized by the batch-wide maximum draft count, so covering them
    # would let a co-scheduled request's draft count decide how much of a seeded
    # request's generator gets consumed. Upstream guards the same way in
    # `generate_uniform_probs` ("important for reproducibility").
    row_generators = {
        seq_idx * rows_per_seq + offset: generator
        for seq_idx, generator in sampling_metadata.generators.items()
        for offset in range(num_draft_tokens[seq_idx])
    }
    probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    target_sampled = random_sample(probs, row_generators, use_fp64_gumbel)
    if sampling_metadata.all_random:
        return target_sampled

    # `random_sample` scaled `probs` in place, so the argmax still has to come
    # from `target_logits`.
    is_greedy = expand_to_draft_rows(sampling_metadata.temperature, num_seqs, num_rows) < _SAMPLING_EPS
    return torch.where(is_greedy, target_logits.argmax(dim=-1), target_sampled)


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: Optional[torch.Tensor],
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    synthetic_mode: bool = False,
    synthetic_conditional_rates: Optional[torch.Tensor] = None,
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    # `target_probs` is the target logits after `apply_sampling_constraints`,
    # matching the upstream signature.
    target_token_ids = select_target_token_ids(target_probs, num_draft_tokens, sampling_metadata, use_fp64_gumbel)
    output_token_ids = rejection_sample_pytorch(draft_token_ids, target_token_ids, bonus_token_ids, num_draft_tokens,
                                                cu_num_draft_tokens)
    return output_token_ids


rejection_sampler.rejection_sample = rejection_sample
# `RejectionSampler.forward` looks both helpers up as module globals at call
# time, so replacing them here covers the whole non-greedy path too.
rejection_sampler.apply_sampling_constraints = apply_sampling_constraints
