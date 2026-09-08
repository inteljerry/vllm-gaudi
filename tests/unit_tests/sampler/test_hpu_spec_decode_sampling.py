# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Copyright (C) 2026 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
"""Tests for the sampling metadata contract of the HPU spec decode path.

The HPU decode path flattens a speculative batch to `[batch_size * num_tokens, 1]`,
so the target logits carry one row per *sampled position* while the bonus sampler
inside `RejectionSampler.forward` consumes one row per *request*. Sizing the
sampling metadata to the flattened row count killed every worker as soon as a
request arrived with `temperature > 0`; an all-greedy batch never noticed because
`Sampler.sample` returns before it reads `temperature`.

These run on CPU tensors without Habana hardware.
"""
import numpy as np
import torch

from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler

# Importing the plugin module installs the HPU monkeypatches onto
# `vllm.v1.sample.rejection_sampler`, which is what production runs.
from vllm_gaudi.v1.sample.hpu_rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner

NUM_SPEC_TOKENS = 3  # --speculative-config num_speculative_tokens
MAX_SAMPLED_TOKENS = NUM_SPEC_TOKENS + 1
VOCAB_SIZE = 16
# Large enough that softmax puts ~1.0 on the peak, so a sampled token and an
# argmax token agree and the assertions stay exact.
PEAK_LOGIT = 40.0


class _StubRunner:
    """Just enough of `HPUModelRunner` to call `_prepare_spec_decode_inputs`."""

    _get_cumsum_and_arange = HPUModelRunner._get_cumsum_and_arange
    _prepare_spec_decode_inputs = HPUModelRunner._prepare_spec_decode_inputs

    def __init__(self, num_seqs: int):
        self.device = torch.device("cpu")
        self.arange_np = np.arange(1024, dtype=np.int32)
        self.input_batch = type("_Batch", (), {"req_id_to_index": {f"r{i}": i for i in range(num_seqs)}})()


def _prepare_spec_decode_inputs(num_draft_tokens: list[int],
                                token_ids: list[int],
                                max_sampled_tokens: int | None = None):
    """Build spec decode metadata exactly the way the decode path does."""
    num_seqs = len(num_draft_tokens)
    if max_sampled_tokens is None:
        max_sampled_tokens = MAX_SAMPLED_TOKENS
    runner = _StubRunner(num_seqs)
    scheduler_output = type(
        "_SchedulerOutput", (),
        {"scheduled_spec_decode_tokens": {
            f"r{i}": [0] * n
            for i, n in enumerate(num_draft_tokens) if n > 0
        }})()
    # `token_ids` is the flattened `[batch_size * num_tokens, 1]` decode input.
    token_ids_device = torch.tensor(token_ids, dtype=torch.int32).view(-1, 1)
    logits_indices, spec_decode_metadata = runner._prepare_spec_decode_inputs(
        scheduler_output,
        torch.zeros(num_seqs, dtype=torch.int32),
        token_ids_device,
        max_sampled_tokens,
    )
    return logits_indices, spec_decode_metadata


def _logits_for_tokens(tokens_per_row: list[int]) -> torch.Tensor:
    logits = torch.zeros(len(tokens_per_row), VOCAB_SIZE, dtype=torch.float32)
    for row, token in enumerate(tokens_per_row):
        logits[row, token] = PEAK_LOGIT
    return logits


def _sampling_metadata(temperatures: list[float],
                       top_k: int | None = None,
                       top_p: float | None = None) -> SamplingMetadata:
    """Mirror `InputBatch.make_selective_sampling_metadata` for `temperatures`.

    Greedy requests carry the -1.0 sentinel that `HPUInputBatch` writes.
    """
    num_rows = len(temperatures)
    is_greedy = [t < 0.0 for t in temperatures]
    return SamplingMetadata(
        temperature=torch.tensor(temperatures, dtype=torch.float32),
        all_greedy=all(is_greedy),
        all_random=not any(is_greedy),
        top_p=None if top_p is None else torch.full((num_rows, ), top_p, dtype=torch.float32),
        top_k=None if top_k is None else torch.full((num_rows, ), top_k, dtype=torch.int32),
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(num_rows, dtype=torch.float32),
        presence_penalties=torch.zeros(num_rows, dtype=torch.float32),
        repetition_penalties=torch.ones(num_rows, dtype=torch.float32),
        output_token_ids=[[] for _ in range(num_rows)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def _run_rejection_sampler(spec_decode_metadata, logits, sampling_metadata) -> torch.Tensor:
    sampler = RejectionSampler(Sampler())
    output = sampler(spec_decode_metadata, None, logits, sampling_metadata)
    return output.sampled_token_ids


def _single_request_case(draft_tokens: list[int]):
    """One request with `NUM_SPEC_TOKENS` drafts, targeting tokens 5/6/7 + bonus 9."""
    target_tokens = [5, 6, 7]
    bonus_token = 9
    # Flattened decode input: [last confirmed token, draft_0, draft_1, draft_2].
    logits_indices, spec_decode_metadata = _prepare_spec_decode_inputs([NUM_SPEC_TOKENS], [1] + draft_tokens)
    logits = _logits_for_tokens([*target_tokens, bonus_token])
    return logits_indices, spec_decode_metadata, logits, target_tokens, bonus_token


def test_spec_decode_metadata_has_one_bonus_row_per_request():
    """The two candidate metadata sizes differ, which is the whole bug."""
    logits_indices, spec_decode_metadata = _prepare_spec_decode_inputs([NUM_SPEC_TOKENS], [1, 2, 3, 4])

    # What the bonus sampler consumes: one row per request.
    assert spec_decode_metadata.bonus_logits_indices.numel() == 1
    # What `logits_device.shape[0]` is: one row per sampled position.
    assert logits_indices.numel() == MAX_SAMPLED_TOKENS
    assert spec_decode_metadata.target_logits_indices.numel() == NUM_SPEC_TOKENS


def test_flattened_metadata_is_rejected_and_request_sized_metadata_is_not():
    """Reproduces the reported engine kill and shows the fixed sizing clears it."""
    _, spec_decode_metadata, logits, _, _ = _single_request_case([5, 6, 7])

    flattened = _sampling_metadata([0.7] * MAX_SAMPLED_TOKENS)
    try:
        _run_rejection_sampler(spec_decode_metadata, logits, flattened)
    except RuntimeError:
        pass
    else:
        raise AssertionError("metadata sized to the flattened logits rows should not be accepted")

    request_sized = _sampling_metadata([0.7])
    _run_rejection_sampler(spec_decode_metadata, logits, request_sized)


def test_greedy_accepts_every_matching_draft():
    _, spec_decode_metadata, logits, target_tokens, bonus_token = _single_request_case([5, 6, 7])
    sampled = _run_rejection_sampler(spec_decode_metadata, logits, _sampling_metadata([-1.0]))
    assert sampled.tolist() == [[*target_tokens, bonus_token]]


def test_greedy_stops_at_the_first_mismatch():
    _, spec_decode_metadata, logits, target_tokens, _ = _single_request_case([5, 13, 7])
    sampled = _run_rejection_sampler(spec_decode_metadata, logits, _sampling_metadata([-1.0]))
    # Positions 0 and 1 come from the target; the rest are dropped.
    assert sampled.tolist() == [[target_tokens[0], target_tokens[1], PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]]


def test_random_sampling_matches_the_greedy_result_on_a_peaked_distribution():
    """`temperature > 0` used to assert out ("Only greedy sampling is supported")."""
    _, spec_decode_metadata, logits, target_tokens, bonus_token = _single_request_case([5, 6, 7])
    sampled = _run_rejection_sampler(spec_decode_metadata, logits, _sampling_metadata([0.7]))
    assert sampled.tolist() == [[*target_tokens, bonus_token]]


def test_random_sampling_stops_at_the_first_mismatch():
    _, spec_decode_metadata, logits, target_tokens, _ = _single_request_case([5, 13, 7])
    sampled = _run_rejection_sampler(spec_decode_metadata, logits, _sampling_metadata([0.7]))
    assert sampled.tolist() == [[target_tokens[0], target_tokens[1], PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]]


def test_mixed_batch_keeps_the_greedy_request_exact():
    """A greedy and a random request in one batch: both must decode correctly."""
    num_draft_tokens = [NUM_SPEC_TOKENS, NUM_SPEC_TOKENS]
    # Block layout: [confirmed, d0, d1, d2] per request.
    token_ids = [1, 5, 6, 7] + [1, 5, 13, 7]
    _, spec_decode_metadata = _prepare_spec_decode_inputs(num_draft_tokens, token_ids)
    # Two blocks of MAX_SAMPLED_TOKENS rows: targets 5/6/7 + bonus 9 for both.
    logits = _logits_for_tokens([5, 6, 7, 9] * 2)

    sampled = _run_rejection_sampler(spec_decode_metadata, logits, _sampling_metadata([-1.0, 0.7]))
    assert sampled.tolist() == [
        [5, 6, 7, 9],
        [5, 6, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID],
    ]

    # And with the roles swapped, so neither branch is special-cased.
    sampled = _run_rejection_sampler(spec_decode_metadata, logits, _sampling_metadata([0.7, -1.0]))
    assert sampled.tolist() == [
        [5, 6, 7, 9],
        [5, 6, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID],
    ]


def _two_peak_case(seed: int | None):
    """One request whose first draft never matches, so the emitted token is the
    target's own pick at position 0 - drawn from a near 50/50 distribution."""
    _, spec_decode_metadata = _prepare_spec_decode_inputs([NUM_SPEC_TOKENS], [1, 15, 15, 15])
    logits = torch.zeros(MAX_SAMPLED_TOKENS, VOCAB_SIZE, dtype=torch.float32)
    logits[:, 3] = 10.0
    logits[:, 4] = 9.9
    generators = {} if seed is None else {0: torch.Generator().manual_seed(seed)}
    return spec_decode_metadata, logits, generators


def test_random_rows_sample_while_greedy_rows_take_the_argmax():
    draws = 40
    spec_decode_metadata, logits, _ = _two_peak_case(seed=None)

    greedy_tokens = {
        _run_rejection_sampler(spec_decode_metadata, logits.clone(), _sampling_metadata([-1.0]))[0][0].item()
        for _ in range(draws)
    }
    assert greedy_tokens == {3}, "greedy rows must stay on the argmax"

    random_tokens = {
        _run_rejection_sampler(spec_decode_metadata, logits.clone(), _sampling_metadata([1.0]))[0][0].item()
        for _ in range(draws)
    }
    assert random_tokens == {3, 4}, f"random rows must sample the distribution, saw {random_tokens}"


def test_top_k_and_top_p_reach_the_draft_rows():
    """Without a clamp the two peaks both show up (see the test above)."""
    draws = 40
    spec_decode_metadata, logits, _ = _two_peak_case(seed=None)

    for clamp in ({"top_k": 1}, {"top_p": 0.4}):
        tokens = {
            _run_rejection_sampler(spec_decode_metadata, logits.clone(), _sampling_metadata([1.0],
                                                                                            **clamp))[0][0].item()
            for _ in range(draws)
        }
        assert tokens == {3}, f"{clamp} must clamp the draft rows to the top token, saw {tokens}"


def test_a_seeded_request_is_reproducible():
    """Each request's generator has to reach every one of its draft rows."""
    runs = []
    for _ in range(2):
        spec_decode_metadata, logits, generators = _two_peak_case(seed=1234)
        metadata = _sampling_metadata([1.0])
        metadata.generators = generators
        runs.append(_run_rejection_sampler(spec_decode_metadata, logits, metadata).tolist())
    assert runs[0] == runs[1]


def _uneven_batch_case():
    """Three requests with 3 / 1 / 0 drafts, so two blocks carry `-1` padding rows."""
    num_draft_tokens = [NUM_SPEC_TOKENS, 1, 0]
    # Per block: [last confirmed token, drafts..., zero padding].
    token_ids = [1, 5, 6, 7] + [1, 5, 0, 0] + [1, 0, 0, 0]
    _, spec_decode_metadata = _prepare_spec_decode_inputs(num_draft_tokens, token_ids)
    # Row 0/1/2 target 5/6/7 with bonus 9; row 4 targets 5 with bonus 11 at row 5;
    # row 8 is the third request's bonus. A `-1` padding index reads the last row,
    # so give that one a peak too and the padded positions stay deterministic
    # whether they are argmaxed or sampled.
    num_rows = len(num_draft_tokens) * MAX_SAMPLED_TOKENS
    logits = torch.zeros(num_rows, VOCAB_SIZE, dtype=torch.float32)
    for row, token in ((0, 5), (1, 6), (2, 7), (3, 9), (4, 5), (5, 11), (8, 12), (num_rows - 1, 0)):
        logits[row, token] = PEAK_LOGIT
    return spec_decode_metadata, logits


def test_uneven_draft_counts_decode_every_request():
    spec_decode_metadata, logits = _uneven_batch_case()

    for temperatures in ([-1.0, -1.0, -1.0], [0.7, 0.7, 0.7], [-1.0, 0.7, -1.0]):
        sampled = _run_rejection_sampler(spec_decode_metadata, logits.clone(), _sampling_metadata(temperatures))
        assert sampled.tolist() == [
            # All three drafts matched, so the bonus token follows.
            [5, 6, 7, 9],
            # The one real draft matched, but `rejection_sample_pytorch` compares
            # the `-1` padding rows too, so it reads a mismatch there and drops
            # the bonus token. Pre-existing behaviour, characterised here: it
            # costs throughput on short drafts, never correctness.
            [5, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID],
            # No drafts at all: just the bonus token.
            [12, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID],
        ], f"temperatures={temperatures}"


def _seeded_request_steps(partner_drafts: int, steps: int = 5) -> list[list[int]]:
    """Decode `steps` times for a seeded request sharing a batch with a busier one.

    The target logits are uniform, so every step's token is a fresh draw from the
    request's generator and the sequence pins down how much of it got consumed.
    """
    num_draft_tokens = [1, partner_drafts]
    max_sampled_tokens = max(num_draft_tokens) + 1
    token_ids = [1, 15] + [0] * (max_sampled_tokens - 2) + [1] + [15] * (max_sampled_tokens - 1)
    generator = torch.Generator().manual_seed(99)
    tokens = []
    for _ in range(steps):
        _, spec_decode_metadata = _prepare_spec_decode_inputs(num_draft_tokens, token_ids, max_sampled_tokens)
        logits = torch.zeros(len(num_draft_tokens) * max_sampled_tokens, VOCAB_SIZE, dtype=torch.float32)
        metadata = _sampling_metadata([1.0, 1.0])
        metadata.generators = {0: generator}
        row = _run_rejection_sampler(spec_decode_metadata, logits, metadata).tolist()[0]
        # The row is `max_spec_len + 1` wide, which follows the batch-wide max,
        # so compare the emitted tokens rather than the placeholder padding.
        tokens.append([token for token in row if token != PLACEHOLDER_TOKEN_ID])
    return tokens


def test_a_seeded_request_ignores_the_other_requests_draft_counts():
    """Padding rows must not eat the generator, or batch composition leaks in."""
    assert _seeded_request_steps(partner_drafts=1) == _seeded_request_steps(partner_drafts=NUM_SPEC_TOKENS)


def test_temperature_scaling_reaches_the_draft_positions():
    """A high temperature must flatten the target distribution it is applied to."""
    _, spec_decode_metadata, logits, _, _ = _single_request_case([5, 6, 7])
    metadata = _sampling_metadata([0.5])
    target_rows = logits[spec_decode_metadata.target_logits_indices].clone()

    from vllm.v1.sample import rejection_sampler as upstream
    scaled = upstream.apply_sampling_constraints(target_rows, spec_decode_metadata.cu_num_draft_tokens, metadata)

    assert torch.allclose(scaled, logits[spec_decode_metadata.target_logits_indices] / 0.5)

    # Greedy rows are left alone so their argmax is untouched.
    greedy_rows = logits[spec_decode_metadata.target_logits_indices].clone()
    unscaled = upstream.apply_sampling_constraints(greedy_rows, spec_decode_metadata.cu_num_draft_tokens,
                                                   _sampling_metadata([-1.0]))
    assert torch.equal(unscaled, logits[spec_decode_metadata.target_logits_indices])
