# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Copyright (C) 2026 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
"""Regression tests for which logit a prefill step samples, and how many.

A request preempted by KV exhaustion resumes with a PARTIAL prefix-cache hit, so
`num_computed_tokens > 0` and `InputBatch.add_request` skips the inflation that
would make `num_prompt_tokens` cover the `O` tokens emitted before preemption -
that value stays `P`. The prefill path then computed `computed + scheduled -
num_prompt_tokens + 1 = O+1` output logits, named the request once per logit, and
the postprocessing loop accumulated one token per occurrence - committing `O+1`
unverified tokens, which the scheduler noticed one step later as
`assert num_accepted_tokens <= self.num_spec_tokens`.

The fix is in the prefill path, not in `num_prompt_tokens`: a short
`num_prompt_tokens` is a condition `prefill_logits_position` TOLERATES, not
something this change corrects. It keys on `num_tokens_no_spec`, which
`add_request` sets unconditionally from `request.num_tokens`, so it is already
`P + O` for a resumed request.

Emitting ONE token is necessary but not sufficient: it has to be the RIGHT one.
When the scheduler pads a request whose remaining work is a single token up to
`1 + num_spec_tokens` and attaches `[-1]` drafts (pad_spec_decode), the sequence
completes at local position 0 and the rest of the chunk holds sentinels - or stale
`token_ids_cpu` contents, since `_update_states` returns before the draft write for
a request added this step. Sampling the last scheduled position there commits a
wrong token with no assert to reveal it. Hence a POSITION, rather than a count the
caller turns into "the last scheduled token".

The same short `num_prompt_tokens` reaches the async path through a second door:
the `invalid_req_indices` predicate that decides whether a partial-prefill logit is
thrown away. Keyed on `num_prompt_tokens`, a resumed request whose chunk ends
between `P` and `P + O` KEEPS a logit that predicts a token it has already emitted,
and `_prepare_input_ids` then scatters that prediction over the real token in
`input_ids`. Both sites therefore key on `num_tokens_no_spec`, and the tests here
pin them in lockstep - matching `gpu_model_runner`, whose discard mask is
`optimistic_seq_lens < num_tokens` with `Request.num_tokens` = prompt + emitted.

Because the sampled position now depends on `num_tokens_no_spec`, a write that
moves that boundary BACKWARDS is newly able to corrupt it; the non-final
pipeline-parallel branch of `_update_states` is one, and is pinned here too.

These run on the host: nothing here executes on a device.

Everything here is plain host arithmetic over InputBatch state, so the module also
runs as a script wherever pytest is unavailable:

    python tests/unit_tests/worker/test_prefill_multi_logit.py
"""
import types

import numpy as np
import torch
import habana_frameworks.torch  # noqa: F401

from vllm.sampling_params import SamplingParams
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.core.sched.output import CachedRequestData

from vllm_gaudi.v1.worker import hpu_model_runner
from vllm_gaudi.v1.worker.hpu_input_batch import CachedRequestState, InputBatch
from vllm_gaudi.v1.worker.hpu_model_runner import (HPUModelRunner, assert_sampled_token_budget, prefill_logits_position,
                                                   sampled_token_budget)

VOCAB_SIZE = 1024
MAX_MODEL_LEN = 1024

# One concrete resume shape: a 100-token prompt, 5 tokens emitted before preemption,
# k=3 draft tokens, resuming against a block-aligned hit of its own just-freed blocks.
PROMPT_LEN = 100
NUM_OUTPUT_TOKENS = 5
PREFIX_CACHE_HIT = 64
NUM_SPECULATIVE_TOKENS = 3

# (case, computed, scheduled, num_tokens_no_spec, always_sample, expected position)
#
# num_tokens_no_spec is the request's prompt plus its already-emitted tokens, never
# its speculative drafts. The expected position is the local index of the last token
# that is not an unverified draft; None means the chunk ends before the sequence does
# and nothing may be sampled.
POSITION_CASES = [
    # Ordinary prefill: the sequence ends at the last scheduled token, so the position
    # is scheduled - 1, exactly upstream's `query_start_loc[1:] - 1`.
    ("full prefill completes the prompt", 0, 100, 100, False, 99),
    ("first chunk of a chunked prefill", 0, 32, 100, False, None),
    ("middle chunk of a chunked prefill", 32, 32, 100, False, None),
    ("final chunk of a chunked prefill", 64, 36, 100, False, 35),
    # The boundary from both sides: one token short samples nothing, exactly on it
    # samples the last position. An implementation off by one in the permissive
    # direction passes the row above and fails this one.
    ("chunk ending one token short", 63, 36, 100, False, None),
    ("chunk ending exactly on the last token", 64, 36, 100, False, 35),
    # Resumed from preemption with a partial prefix-cache hit: the shape that over-emitted.
    # num_prompt_tokens is short at 100 here and stays that way; num_tokens_no_spec is
    # 105, so the position is right without correcting num_prompt_tokens.
    ("resumed, partial prefix-cache hit", 64, 41, 105, False, 40),
    # pad_spec_decode: remaining work of one token padded to 1 + k. The sequence
    # completes at position 0; positions 1..k are [-1] sentinels or stale row
    # contents. Sampling scheduled - 1 here commits a wrong token silently.
    ("pad_spec, new request, fully cached", 99, 4, 100, False, 0),
    ("pad_spec, resumed with 5 emitted tokens", 104, 4, 105, False, 0),
    # KV-offload requeue: the chunk starts past the prompt end. num_prompt_tokens is
    # stale at P here because the request generated its tokens while in the batch, so
    # a num_prompt_tokens basis would emit nothing where a logit is required.
    ("KV-offload requeue, chunk past prompt", 200, 64, 264, False, 63),
    # computed has overtaken num_tokens_no_spec. Reachable under async scheduling,
    # which never runs the post-sampling writer, and in the corrupted state this fix
    # prevents. Falls back to the last scheduled position - today's behaviour - never
    # to position 0, which would be a wrong token rather than an unexpected one.
    ("computed past the non-draft boundary", 110, 4, 105, False, 3),
    ("computed past the boundary, async", 110, 4, 105, True, 3),
    # Async scheduling / structured output need a logit for every request and discard
    # the partial ones, but must not override a real completion position.
    ("async, partial chunk", 0, 32, 100, True, 31),
    ("async, chunk ending one token short", 63, 36, 100, True, 35),
    ("async, completes the prompt", 0, 100, 100, True, 99),
    ("async, pad_spec completes at 0", 99, 4, 100, True, 0),
    ("async, KV-offload requeue", 200, 64, 264, True, 63),
]


def _head_logits_positions(computed: int, scheduled: int, num_prompt_tokens: int, always_sample: bool) -> list[int]:
    """The formula this change replaces, for comparing behaviour where it was correct."""
    if always_sample:
        remaining = computed + scheduled - num_prompt_tokens + 1
        num_output_logits = 1 if remaining < 1 else remaining
    else:
        num_output_logits = max(0, computed + scheduled - num_prompt_tokens + 1)
    num_output_logits = min(num_output_logits, scheduled)
    return list(range(scheduled - num_output_logits, scheduled))


def _positions(computed: int, scheduled: int, num_tokens_no_spec: int, always_sample: bool) -> list[int]:
    position = prefill_logits_position(computed, scheduled, num_tokens_no_spec, always_sample)
    return [] if position is None else [position]


def _make_input_batch(max_num_reqs: int = 4) -> InputBatch:
    return InputBatch(
        max_num_reqs=max_num_reqs,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        device=torch.device("cpu"),
        pin_memory=is_pin_memory_available(),
        vocab_size=VOCAB_SIZE,
        block_sizes=[1],
        kernel_block_sizes=[1],
    )


def _make_request(req_id: str, prompt_len: int, num_computed_tokens: int, num_output_tokens: int) -> CachedRequestState:
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[1] * prompt_len,
        sampling_params=SamplingParams(),
        pooling_params=None,
        mm_features=[],
        block_ids=([], ),
        generator=None,
        num_computed_tokens=num_computed_tokens,
        output_token_ids=[2] * num_output_tokens,
    )


def test_prefill_logits_position_for_every_scheduled_shape():
    """P1: the sampled position, not just the count, for every shape that reaches the
    prefill path. A count-based clamp passes the ordinary rows and fails the
    pad_spec_decode ones."""
    wrong = []
    for case, computed, scheduled, no_spec, always_sample, expected in POSITION_CASES:
        actual = prefill_logits_position(computed, scheduled, no_spec, always_sample)
        if actual != expected:
            wrong.append(f"{case}: computed={computed} scheduled={scheduled} "
                         f"num_tokens_no_spec={no_spec} always_sample={always_sample} "
                         f"-> position {actual}, expected {expected}")
    assert not wrong, "wrong sampled position:\n  " + "\n  ".join(wrong)


def test_pad_spec_decode_samples_the_completing_position_not_the_last():
    """P1, called out on its own because it is the row a count-based clamp gets wrong.
    The scheduler padded 1 scheduled token to 1 + k and attached [-1] drafts, so only
    position 0 predicts a real token; 1..k are sentinels or stale row contents."""
    scheduled = 1 + NUM_SPECULATIVE_TOKENS

    new_request = prefill_logits_position(PROMPT_LEN - 1, scheduled, PROMPT_LEN, always_sample=False)
    resumed = prefill_logits_position(PROMPT_LEN + NUM_OUTPUT_TOKENS - 1,
                                      scheduled,
                                      PROMPT_LEN + NUM_OUTPUT_TOKENS,
                                      always_sample=False)

    assert new_request == 0, (f"fully-cached new request sampled position {new_request}, expected 0; "
                              f"positions 1..{scheduled - 1} are [-1] drafts or stale tokens")
    assert resumed == 0, (f"resumed request sampled position {resumed}, expected 0; "
                          f"positions 1..{scheduled - 1} are [-1] drafts or stale tokens")


def test_resumed_request_is_correct_against_a_short_num_prompt_tokens():
    """The headline case. `add_request` leaves num_prompt_tokens at P for a request
    resuming with a partial prefix-cache hit, and this change does NOT correct that -
    it stops the prefill path depending on it. num_tokens_no_spec is P + O regardless,
    so the position is right anyway."""
    computed, scheduled = PREFIX_CACHE_HIT, PROMPT_LEN + NUM_OUTPUT_TOKENS - PREFIX_CACHE_HIT
    short_num_prompt_tokens = PROMPT_LEN
    no_spec = PROMPT_LEN + NUM_OUTPUT_TOKENS

    position = prefill_logits_position(computed, scheduled, no_spec, always_sample=False)

    assert position == scheduled - 1, (f"resumed request sampled position {position}, expected {scheduled - 1} "
                                       f"(the last of its {scheduled} scheduled tokens)")
    # Keying on the short num_prompt_tokens instead would land NUM_OUTPUT_TOKENS early.
    assert short_num_prompt_tokens - 1 - computed == position - NUM_OUTPUT_TOKENS, (
        "these inputs no longer separate the two bases; pick inputs where they differ")


def test_no_divergence_from_head_when_the_two_boundaries_agree():
    """The change must alter behaviour ONLY where the two boundaries differ or padded
    drafts live. This sweep pins the region where `num_prompt_tokens == num_tokens_no_spec`
    - a request that never emitted a token before this prefill, which is every ordinary
    non-resumed chunk - and inside it the helper reproduces the old formula exactly.

    NOTE the deliberate limit: passing one value into both slots means this sweep says
    nothing about the resumed case, where the two bases differ and divergence is the
    POINT. `test_head_divergence_is_confined_to_a_short_num_prompt_tokens` covers that
    half; the two together are the narrowness claim."""
    divergences = []
    for computed in range(60):
        for scheduled in range(1, 20):
            for boundary in range(1, 80):
                if computed + scheduled > boundary:
                    continue
                for always_sample in (False, True):
                    # One value in both slots: num_prompt_tokens == num_tokens_no_spec.
                    head = _head_logits_positions(computed, scheduled, boundary, always_sample)
                    new = _positions(computed, scheduled, boundary, always_sample)
                    if head != new:
                        divergences.append(f"computed={computed} scheduled={scheduled} "
                                           f"num_prompt_tokens=num_tokens_no_spec={boundary} "
                                           f"always_sample={always_sample}: was {head}, now {new}")
    assert not divergences, (f"{len(divergences)} divergence(s) where the two boundaries agree:\n  " +
                             "\n  ".join(divergences[:10]))


def test_head_divergence_is_confined_to_a_short_num_prompt_tokens():
    """The other half of the narrowness claim, with the two boundaries INDEPENDENT.
    Where they agree the helper must match the old formula; where num_prompt_tokens is
    short - a request resumed from preemption, which emitted `no_spec - prompt` tokens
    before being preempted - the old formula over-emits and the helper must NOT match.

    Pins the divergence to exactly that region, so a future change cannot quietly alter
    a chunk whose boundaries agree while claiming to fix the resumed case."""
    unexpected_match = []
    unexpected_divergence = []
    for computed in range(0, 140, 3):
        for scheduled in range(1, 45, 2):
            for prompt in range(1, 120, 7):
                for emitted in range(0, 12, 3):
                    no_spec = prompt + emitted
                    for always_sample in (False, True):
                        head = _head_logits_positions(computed, scheduled, prompt, always_sample)
                        new = _positions(computed, scheduled, no_spec, always_sample)
                        shape = (f"computed={computed} scheduled={scheduled} num_prompt_tokens={prompt} "
                                 f"num_tokens_no_spec={no_spec} always_sample={always_sample}: "
                                 f"was {head}, now {new}")
                        if emitted == 0 and computed + scheduled <= no_spec:
                            # Boundaries agree AND the chunk does not over-shoot the
                            # sequence, so no padded drafts are in play: must match.
                            if head != new:
                                unexpected_divergence.append(shape)
                        elif emitted > 0 and len(head) > 1 and head == new:
                            # The old formula emitted a range off the short boundary;
                            # reproducing it exactly would be reproducing the defect.
                            unexpected_match.append(shape)
    assert not unexpected_divergence, (f"{len(unexpected_divergence)} divergence(s) where the boundaries agree:\n  " +
                                       "\n  ".join(unexpected_divergence[:10]))
    assert unexpected_match == [], (f"{len(unexpected_match)} shape(s) still reproduce the over-emitting range:\n  " +
                                    "\n  ".join(unexpected_match[:10]))
    # The sweep must actually reach the over-emitting region, or it proves nothing.
    over_emitting = [(c, s, p, e) for c in range(0, 140, 3) for s in range(1, 45, 2) for p in range(1, 120, 7)
                     for e in range(3, 12, 3) if len(_head_logits_positions(c, s, p, False)) > 1]
    assert len(over_emitting) > 100, (f"only {len(over_emitting)} over-emitting shapes in the sweep; "
                                      "widen it or this test proves nothing")


def test_position_never_lands_on_a_draft_or_outside_the_chunk():
    """Swept, not sampled: a position outside [0, scheduled) indexes another request's
    tokens once the prefill batch is flattened, and one at or past the non-draft
    boundary indexes a draft. Covers use_merged_prefill and multimodal prefill, which
    change the batching but not this arithmetic."""
    for computed in range(0, 300, 7):
        for scheduled in range(1, 70, 3):
            for no_spec in range(1, 300, 11):
                for always_sample in (False, True):
                    position = prefill_logits_position(computed, scheduled, no_spec, always_sample)
                    if position is None:
                        continue
                    shape = (f"computed={computed} scheduled={scheduled} "
                             f"num_tokens_no_spec={no_spec} always_sample={always_sample}")
                    assert 0 <= position < scheduled, f"{shape}: position {position} outside [0, {scheduled})"
                    if computed < no_spec and not always_sample:
                        assert computed + position < no_spec, (f"{shape}: position {position} is at global "
                                                               f"{computed + position}, at or past the "
                                                               f"non-draft boundary {no_spec}")


def test_input_batch_supplies_the_non_draft_boundary_the_position_needs():
    """Ties the helper to real InputBatch state: the position is only right if
    num_tokens_no_spec really is P + O for a resumed request, since that is the array
    the prefill path reads. Unlike num_prompt_tokens it carries no cache-hit guard,
    which is why this change does not need to touch add_request."""
    batch = _make_input_batch()
    index = batch.add_request(
        _make_request("resumed",
                      prompt_len=PROMPT_LEN,
                      num_computed_tokens=PREFIX_CACHE_HIT,
                      num_output_tokens=NUM_OUTPUT_TOKENS))

    no_spec = int(batch.num_tokens_no_spec[index])
    assert no_spec == PROMPT_LEN + NUM_OUTPUT_TOKENS, (
        f"num_tokens_no_spec is {no_spec}, expected {PROMPT_LEN + NUM_OUTPUT_TOKENS}")
    # The value the prefill path used to key on, left short by the emitted tokens.
    assert batch.num_prompt_tokens[index] == PROMPT_LEN

    scheduled = PROMPT_LEN + NUM_OUTPUT_TOKENS - PREFIX_CACHE_HIT
    position = prefill_logits_position(PREFIX_CACHE_HIT, scheduled, no_spec, always_sample=False)
    assert position == scheduled - 1, (f"resumed request sampled position {position}, expected {scheduled - 1}")


def _make_prefill_runner(batch: InputBatch,
                         use_async_scheduling: bool = False,
                         use_structured_output: bool = False) -> types.SimpleNamespace:
    """Just enough of HPUModelRunner to drive _extract_prefill_batch_contents on the
    host. warmup=True skips _resolve_block, and merging is stubbed off so each request
    keeps its own BatchContents."""
    runner = types.SimpleNamespace(
        input_batch=batch,
        attn_block_size=128,
        use_async_scheduling=use_async_scheduling,
        use_structured_output=use_structured_output,
        invalid_req_indices=[],
    )
    runner._get_attention_group_id_for_hybrid = lambda: 0
    runner._can_merge_prefill_contents = lambda lhs, rhs: False
    runner.get_dp_padding = lambda n: 0
    return runner


def test_extract_prefill_batch_contents_reads_the_non_draft_boundary():
    """Drives the real call site, not the helper: a resumed request whose
    num_prompt_tokens is short by O must still get the position derived from
    num_tokens_no_spec. Reading num_prompt_tokens here lands O tokens early, and no
    helper-level test can see that."""
    batch = _make_input_batch()
    batch.add_request(
        _make_request("resumed",
                      prompt_len=PROMPT_LEN,
                      num_computed_tokens=PREFIX_CACHE_HIT,
                      num_output_tokens=NUM_OUTPUT_TOKENS))
    scheduled = PROMPT_LEN + NUM_OUTPUT_TOKENS - PREFIX_CACHE_HIT

    all_batch_contents, _ = HPUModelRunner._extract_prefill_batch_contents(_make_prefill_runner(batch),
                                                                           num_prefills=1,
                                                                           num_decodes=0,
                                                                           num_scheduled_tokens=[scheduled],
                                                                           warmup=True)

    contents = [c for c in all_batch_contents if c.req_ids]
    assert len(contents) == 1, f"expected one prefill batch, got {len(contents)}"
    assert contents[0].logits_positions == [[
        scheduled - 1
    ]], (f"prefill batch sampled {contents[0].logits_positions}, expected [[{scheduled - 1}]]; "
         f"num_prompt_tokens={int(batch.num_prompt_tokens[0])} is short by {NUM_OUTPUT_TOKENS}, "
         f"num_tokens_no_spec={int(batch.num_tokens_no_spec[0])} is not")


def test_extract_prefill_batch_contents_samples_one_position_per_request():
    """The count at the call site, where the O+1 over-emission actually happened: one
    position per request, never a range."""
    batch = _make_input_batch()
    batch.add_request(
        _make_request("resumed",
                      prompt_len=PROMPT_LEN,
                      num_computed_tokens=PREFIX_CACHE_HIT,
                      num_output_tokens=NUM_OUTPUT_TOKENS))
    scheduled = PROMPT_LEN + NUM_OUTPUT_TOKENS - PREFIX_CACHE_HIT

    all_batch_contents, _ = HPUModelRunner._extract_prefill_batch_contents(_make_prefill_runner(batch),
                                                                           num_prefills=1,
                                                                           num_decodes=0,
                                                                           num_scheduled_tokens=[scheduled],
                                                                           warmup=True)

    for contents in all_batch_contents:
        for req_id, positions in zip(contents.req_ids, contents.logits_positions):
            assert len(positions) <= 1, (f"request {req_id} sampled {len(positions)} logits in one prefill step: "
                                         f"{positions}")


def test_tripwire_fires_above_the_per_step_budget():
    """P3: a request whose slot exceeded one token plus its drafts must fail here, not
    one step later in the spec-decode metrics counter."""
    budget = 1 + NUM_SPECULATIVE_TOKENS

    try:
        assert_sampled_token_budget("resumed", budget + 1, budget)
    except AssertionError as exc:
        assert "resumed" in str(exc), exc
        assert str(budget + 1) in str(exc), exc
        assert str(budget) in str(exc), exc
    else:
        raise AssertionError(f"no tripwire for {budget + 1} sampled tokens against a budget of {budget}")


def test_tripwire_silent_within_the_per_step_budget():
    """The tripwire must not fire on a verified spec-decode step, nor on a plain decode.
    The budget itself is allowed - an off-by-one here would abort every spec step."""
    assert_sampled_token_budget("spec", 1 + NUM_SPECULATIVE_TOKENS, 1 + NUM_SPECULATIVE_TOKENS)
    assert_sampled_token_budget("spec", 1, 1 + NUM_SPECULATIVE_TOKENS)
    assert_sampled_token_budget("no_spec", 1, 1)


def _extract_positions(runner: types.SimpleNamespace, num_scheduled_tokens: list[int]) -> dict:
    """Run the real prefill extraction and return {req_id: logits_positions}."""
    all_batch_contents, _ = HPUModelRunner._extract_prefill_batch_contents(runner,
                                                                           num_prefills=len(num_scheduled_tokens),
                                                                           num_decodes=0,
                                                                           num_scheduled_tokens=num_scheduled_tokens,
                                                                           warmup=True)
    positions = {}
    for contents in all_batch_contents:
        for req_id, logits_positions in zip(contents.req_ids, contents.logits_positions):
            assert req_id not in positions, f"request {req_id} appeared in two prefill batches"
            positions[req_id] = logits_positions
    return positions


# (req_id, prompt_len, num_computed_tokens, num_output_tokens, scheduled)
#
# One batch that separates the two candidate bases for the async discard predicate.
# "boundary" is resumed, so add_request leaves its num_prompt_tokens at PROMPT_LEN
# while num_tokens_no_spec is PROMPT_LEN + NUM_OUTPUT_TOKENS, and its chunk ends
# between the two.
ASYNC_DISCARD_BATCH = [
    ("partial", PROMPT_LEN, 0, 0, 32),
    ("boundary", PROMPT_LEN, PREFIX_CACHE_HIT, NUM_OUTPUT_TOKENS, 40),
    ("completing", PROMPT_LEN, PREFIX_CACHE_HIT, 0, PROMPT_LEN - PREFIX_CACHE_HIT),
]


def _make_async_discard_batch() -> InputBatch:
    batch = _make_input_batch(max_num_reqs=len(ASYNC_DISCARD_BATCH))
    for req_id, prompt_len, computed, num_output_tokens, _ in ASYNC_DISCARD_BATCH:
        batch.add_request(_make_request(req_id, prompt_len, computed, num_output_tokens))
    return batch


def test_async_discard_predicate_keys_on_the_non_draft_boundary():
    """P0: the invalid_req_indices predicate must key on num_tokens_no_spec, the same
    boundary the sampled position does, and the same one gpu_model_runner uses
    (`optimistic_seq_lens < num_tokens`, where Request.num_tokens is prompt + emitted).

    Keying it on num_prompt_tokens instead KEEPS a logit that must be thrown away, for
    every resumed request whose chunk ends between the two values - the "boundary" row:
    computed + scheduled = 104, num_prompt_tokens = 100, num_tokens_no_spec = 105. That
    chunk ends one token short of the real sequence, so its logit predicts token 104,
    which the request already emitted. See
    test_async_kept_partial_logit_overwrites_a_known_token for what happens to it."""
    batch = _make_async_discard_batch()
    runner = _make_prefill_runner(batch, use_async_scheduling=True)
    scheduled = [row[4] for row in ASYNC_DISCARD_BATCH]

    _extract_positions(runner, scheduled)

    on_num_prompt_tokens = []
    on_num_tokens_no_spec = []
    for batch_idx, (_, _, _, _, seq_num_scheduled_tokens) in enumerate(ASYNC_DISCARD_BATCH):
        seq_len = int(batch.num_computed_tokens_cpu[batch_idx]) + seq_num_scheduled_tokens
        # The pre-fix algebra, `computed + scheduled - num_prompt_tokens + 1 < 1`.
        if seq_len - int(batch.num_prompt_tokens[batch_idx]) + 1 < 1:
            on_num_prompt_tokens.append(batch_idx)
        if seq_len < int(batch.num_tokens_no_spec[batch_idx]):
            on_num_tokens_no_spec.append(batch_idx)

    assert on_num_prompt_tokens != on_num_tokens_no_spec, (
        "these requests no longer separate the two bases, so this test cannot see a "
        f"regression to num_prompt_tokens: both give {on_num_prompt_tokens}")
    assert runner.invalid_req_indices == on_num_tokens_no_spec, (
        f"async scheduling discarded {runner.invalid_req_indices}, expected "
        f"{on_num_tokens_no_spec} from the num_tokens_no_spec boundary; the pre-fix "
        f"num_prompt_tokens algebra would give {on_num_prompt_tokens}, keeping a logit "
        f"that predicts an already-emitted token")


def test_async_discard_matches_exactly_the_chunks_that_sample_nothing():
    """P1: the predicate and the position helper read the same boundary, so under async
    a logit must be discarded on exactly the chunks where the sync path samples nothing.
    Swept, because these are two separate expressions at two separate sites: an edit to
    one without the other reopens the gap, in whichever direction it drifts."""
    mismatches = []
    for computed in range(0, 160, 3):
        for scheduled in range(1, 50, 2):
            for no_spec in range(1, 200, 7):
                # What the sync path does: None means the chunk ends before the sequence.
                samples_nothing = prefill_logits_position(computed, scheduled, no_spec, False) is None
                # What the caller's discard predicate does.
                discarded = computed + scheduled < no_spec
                if samples_nothing != discarded:
                    mismatches.append(f"computed={computed} scheduled={scheduled} num_tokens_no_spec={no_spec}: "
                                      f"sync samples nothing={samples_nothing}, async discards={discarded}")
    assert not mismatches, (f"{len(mismatches)} chunk(s) where the discard and the position disagree:\n  " +
                            "\n  ".join(mismatches[:10]))


def _cache_sampled_tokens_for_async(runner: types.SimpleNamespace, batch: InputBatch,
                                    sampled_token_ids: torch.Tensor) -> None:
    """The async bookkeeping that carries this step's samples into the next step.

    Mirrors the three assignments the model-runner makes under use_async_scheduling
    right after sampling: cache the flat sampled ids, record which request indices were
    invalidated, and build prev_req_id_to_index EXCLUDING those indices - which is the
    only thing that stops a discarded sample from being scattered next step."""
    batch.prev_sampled_token_ids = sampled_token_ids.flatten()
    invalid = set(runner.invalid_req_indices)
    batch.prev_sampled_token_ids_invalid_indices = invalid
    batch.prev_req_id_to_index = {req_id: i for i, req_id in enumerate(batch.req_ids) if i not in invalid}


def _make_input_ids_runner(batch: InputBatch, num_positions: int) -> types.SimpleNamespace:
    """Just enough of HPUModelRunner to drive _prepare_input_ids on the host."""
    runner = types.SimpleNamespace(
        input_batch=batch,
        device=torch.device("cpu"),
        input_ids_hpu=torch.zeros(num_positions, dtype=torch.int64),
        arange_np=np.arange(MAX_MODEL_LEN, dtype=np.int64),
    )
    runner._get_cumsum_and_arange = lambda n, cumsum_dtype=None: HPUModelRunner._get_cumsum_and_arange(
        runner, n, cumsum_dtype)
    return runner


def test_async_kept_partial_logit_overwrites_a_known_token():
    """P0, the consequence rather than the predicate: two consecutive async steps for
    one resumed request, showing what a kept partial-prefill logit does to input_ids.

    Step 1 schedules a chunk ending one token short of the sequence, so its logit is a
    PREDICTION of token 104 - a token the request emitted before it was preempted.
    Step 2 schedules that real token 104. _prepare_input_ids scatters cached samples
    into the last scheduled slot of each request still present, so if step 1's sample
    was not discarded it lands on top of the real token 104 and the model reads a
    fabricated token in place of a correct one. Discarding it leaves input_ids alone.

    The bridge between the two steps is replicated in _cache_sampled_tokens_for_async
    rather than driven through the full execute_model path, so this pins the
    predicate-to-overwrite chain, not the sampler."""
    real_token, predicted_token = 4242, 999
    batch = _make_input_batch(max_num_reqs=1)
    batch.add_request(_make_request("resumed", PROMPT_LEN, PREFIX_CACHE_HIT, NUM_OUTPUT_TOKENS))
    no_spec = PROMPT_LEN + NUM_OUTPUT_TOKENS

    # Step 1: the chunk that ends one token short of the sequence.
    step1_scheduled = no_spec - 1 - PREFIX_CACHE_HIT
    runner = _make_prefill_runner(batch, use_async_scheduling=True)
    positions = _extract_positions(runner, [step1_scheduled])
    assert PREFIX_CACHE_HIT + step1_scheduled == no_spec - 1, "step 1 must end one token short"
    assert positions["resumed"] == [
        step1_scheduled - 1
    ], (f"step 1 sampled {positions['resumed']}; a logit is still produced under async, "
        f"it is invalid_req_indices that must throw it away")
    _cache_sampled_tokens_for_async(runner, batch, torch.tensor([predicted_token], dtype=torch.int64))

    # Step 2: the single real token the request actually still needs.
    batch.num_computed_tokens_cpu[0] = no_spec - 1
    input_ids_runner = _make_input_ids_runner(batch, num_positions=1)
    input_ids_runner.input_ids_hpu[0] = real_token
    scheduler_output = types.SimpleNamespace(num_scheduled_tokens={"resumed": 1})

    HPUModelRunner._prepare_input_ids(input_ids_runner, scheduler_output)

    assert int(input_ids_runner.input_ids_hpu[0]) == real_token, (
        f"input_ids slot for the last real token holds {int(input_ids_runner.input_ids_hpu[0])}, "
        f"expected the real token {real_token}. Step 1's partial-prefill logit "
        f"({predicted_token}) was scattered over it, because index 0 was not in "
        f"invalid_req_indices ({sorted(runner.invalid_req_indices)}) and so stayed in "
        f"prev_req_id_to_index ({batch.prev_req_id_to_index}).")


def test_kv_offload_requeue_samples_the_last_scheduled_token():
    """P1: a request requeued by a KV-offload connector catches up through the prefill
    path (_is_prompt's second clause, num_scheduled_tokens > num_decode_tokens) with
    its chunk starting PAST the end of the prompt: computed >= num_prompt_tokens. Any
    num_prompt_tokens-keyed position formula goes negative here, which is what refuted
    it; num_tokens_no_spec still describes where the real tokens end."""
    computed, scheduled = 200, 64
    prompt_len, num_output_tokens = 100, 164
    batch = _make_input_batch(max_num_reqs=1)
    batch.add_request(_make_request("requeued", prompt_len, computed, num_output_tokens))
    runner = _make_prefill_runner(batch)

    assert int(batch.num_prompt_tokens[0]) == prompt_len
    assert int(batch.num_tokens_no_spec[0]) == prompt_len + num_output_tokens
    # The formula a num_prompt_tokens basis would give, kept inline as the refutation.
    assert prompt_len - 1 - computed < 0, "pick inputs where a num_prompt_tokens basis goes negative"

    positions = _extract_positions(runner, [scheduled])

    assert positions["requeued"] == [
        scheduled - 1
    ], (f"requeued request sampled {positions['requeued']}, expected [{scheduled - 1}]; "
        f"computed={computed} is already past num_prompt_tokens={prompt_len}, so the "
        f"position has to come from num_tokens_no_spec={prompt_len + num_output_tokens}")
    assert prefill_logits_position(computed, scheduled, prompt_len + num_output_tokens, False) == scheduled - 1


def test_completion_below_zero_falls_back_to_the_last_position_not_zero():
    """P2: when computed has overtaken num_tokens_no_spec the completion index is
    negative. Reachable without any corruption: under async scheduling the
    post-sampling writer that advances num_tokens_no_spec never runs, so it goes stale
    once decoding starts while num_computed_tokens_cpu advances every step.

    The fallback is the last scheduled position - what this path did before - not a
    clamp to 0. Position 0 of a chunk is a token the request has ALREADY committed, so
    clamping would resample a known token; the last position at least predicts the next
    one. Pinned on a wide chunk so the two answers cannot coincide."""
    stale_no_spec = PROMPT_LEN + NUM_OUTPUT_TOKENS
    for computed, scheduled in ((stale_no_spec + 5, 4), (stale_no_spec + 200, 16), (stale_no_spec, 8)):
        for always_sample in (False, True):
            position = prefill_logits_position(computed, scheduled, stale_no_spec, always_sample)
            shape = (f"computed={computed} scheduled={scheduled} "
                     f"num_tokens_no_spec={stale_no_spec} always_sample={always_sample}")
            assert position == scheduled - 1, f"{shape}: position {position}, expected {scheduled - 1}"
            assert position != 0, f"{shape}: fell back to 0, which resamples an already-committed token"


def test_no_scheduled_tokens_samples_nothing():
    """P2: a chunk of zero (or, defensively, negative) scheduled tokens has no logits
    to pick from. Returning `scheduled - 1` there would be a negative index, which
    silently reads from the END of the flattened prefill batch - another request's
    tokens - instead of raising."""
    for scheduled in (0, -1, -8):
        for always_sample in (False, True):
            for computed, no_spec in ((0, 100), (64, 105), (200, 264)):
                position = prefill_logits_position(computed, scheduled, no_spec, always_sample)
                assert position is None, (f"computed={computed} scheduled={scheduled} "
                                          f"num_tokens_no_spec={no_spec} always_sample={always_sample}: "
                                          f"position {position}, expected None")


def _make_dummy_prefill_runner() -> types.SimpleNamespace:
    """_create_dummy_prefill_batch_contents with _form_prefill_batch stubbed out, so
    the BatchContents it built is what comes back."""
    runner = types.SimpleNamespace(attn_block_size=128)
    runner._form_prefill_batch = lambda contents: contents
    return runner


def test_dummy_dp_padding_batch_samples_one_position_in_both_shapes():
    """P2: the DP-padding dummy batch routes through the same helper, so its two
    hard-coded shapes are pinned here - a change to either constant that stops
    producing exactly one position would otherwise only surface as a shape mismatch on
    a DP rank at runtime. Both shapes are a complete 128-token sequence: without a KV
    transfer group the whole prompt is one chunk, with one only its last token is."""
    expected = {
        # (has_kv_transfer_group): (context_len, query_len, position)
        False: (0, 128, 127),
        True: (127, 1, 0),
    }
    original = hpu_model_runner.has_kv_transfer_group
    try:
        for has_group, (context_len, query_len, position) in expected.items():
            hpu_model_runner.has_kv_transfer_group = lambda has_group=has_group: has_group
            outputs = HPUModelRunner._create_dummy_prefill_batch_contents(_make_dummy_prefill_runner(), num_prefills=2)

            assert len(outputs) == 2, f"has_kv_transfer_group={has_group}: got {len(outputs)} dummy batches"
            for contents in outputs:
                shape = (f"has_kv_transfer_group={has_group}: context_len={contents.context_lens} "
                         f"query_len={[len(t) for t in contents.token_ids]}")
                assert contents.context_lens == [context_len], f"{shape}, expected context_len {context_len}"
                assert [len(t) for t in contents.token_ids] == [query_len], f"{shape}, expected query_len {query_len}"
                assert contents.logits_positions == [[
                    position
                ]], (f"{shape} sampled {contents.logits_positions}, expected [[{position}]]")
    finally:
        hpu_model_runner.has_kv_transfer_group = original


# (case, prompt_len, num_output_tokens, first_computed, chunk, pad_last_chunk_to)
#
# pad_last_chunk_to models the scheduler padding a final chunk of one token up to
# 1 + num_spec_tokens and attaching [-1] drafts.
CHUNKED_PREFILL_WALKS = [
    ("fresh prompt, uneven final chunk", PROMPT_LEN, 0, 0, 32, None),
    ("fresh prompt, chunk divides evenly", PROMPT_LEN, 0, 0, 25, None),
    ("fresh prompt, one token per chunk", 8, 0, 0, 1, None),
    ("fresh prompt, single chunk", PROMPT_LEN, 0, 0, PROMPT_LEN, None),
    ("resumed from a partial cache hit", PROMPT_LEN, NUM_OUTPUT_TOKENS, PREFIX_CACHE_HIT, 16, None),
    ("resumed, final chunk padded with drafts", PROMPT_LEN, NUM_OUTPUT_TOKENS, PREFIX_CACHE_HIT, 20,
     1 + NUM_SPECULATIVE_TOKENS),
    ("fresh prompt, final chunk padded with drafts", PROMPT_LEN, 0, 0, 33, 1 + NUM_SPECULATIVE_TOKENS),
]


def _walk_chunked_prefill(prompt_len: int, num_output_tokens: int, first_computed: int, chunk: int,
                          pad_last_chunk_to) -> list[int]:
    """Run consecutive prefill steps for one request until its sequence is covered,
    returning the GLOBAL token index of every position sampled along the way."""
    batch = _make_input_batch(max_num_reqs=1)
    batch.add_request(_make_request("walked", prompt_len, first_computed, num_output_tokens))
    runner = _make_prefill_runner(batch)
    total = prompt_len + num_output_tokens

    sampled: list[int] = []
    computed = first_computed
    while computed < total:
        scheduled = min(chunk, total - computed)
        if pad_last_chunk_to is not None and scheduled == 1:
            # The scheduler pads the last token up to 1 + k and attaches [-1] drafts.
            scheduled = pad_last_chunk_to
        batch.num_computed_tokens_cpu[0] = computed
        for position in _extract_positions(runner, [scheduled])["walked"]:
            sampled.append(computed + position)
        computed += scheduled
    return sampled


def test_a_whole_chunked_prefill_samples_exactly_one_token_at_the_end():
    """P0: the property the whole change exists for, stated end to end. Across every
    step of a prefill - not one step in isolation - a request contributes exactly ONE
    sampled token, and it is the logit of its LAST REAL token, the only one whose
    prediction is not already known. Two tokens means an unverified commit; zero means
    the request stalls; the right count at the wrong index means a wrong token."""
    wrong = []
    for case, prompt_len, num_output_tokens, first_computed, chunk, pad in CHUNKED_PREFILL_WALKS:
        sampled = _walk_chunked_prefill(prompt_len, num_output_tokens, first_computed, chunk, pad)
        expected = [prompt_len + num_output_tokens - 1]
        if sampled != expected:
            wrong.append(f"{case}: prompt={prompt_len} emitted={num_output_tokens} "
                         f"first_computed={first_computed} chunk={chunk} pad={pad} -> sampled global "
                         f"positions {sampled}, expected {expected}")
    assert not wrong, "a prefill did not sample exactly its last real token:\n  " + "\n  ".join(wrong)


# (req_id, prompt_len, num_computed_tokens, num_output_tokens, scheduled, expected positions)
MIXED_PREFILL_BATCH = [
    # Ordinary chunk that completes the prompt.
    ("completing", PROMPT_LEN, PREFIX_CACHE_HIT, 0, PROMPT_LEN - PREFIX_CACHE_HIT, [PROMPT_LEN - PREFIX_CACHE_HIT - 1]),
    # Intermediate chunk: nothing may be sampled yet.
    ("intermediate", PROMPT_LEN, 0, 0, 32, []),
    # pad_spec_decode: one real token padded to 1 + k, completing at local 0.
    ("padded", PROMPT_LEN, PROMPT_LEN - 1, 0, 1 + NUM_SPECULATIVE_TOKENS, [0]),
    # Resumed with a short num_prompt_tokens, completing at the last scheduled token.
    ("resumed", PROMPT_LEN, PREFIX_CACHE_HIT, NUM_OUTPUT_TOKENS, PROMPT_LEN + NUM_OUTPUT_TOKENS - PREFIX_CACHE_HIT,
     [PROMPT_LEN + NUM_OUTPUT_TOKENS - PREFIX_CACHE_HIT - 1]),
]


def test_mixed_prefill_batch_keeps_every_request_to_its_own_position():
    """P1: one extraction call over requests of different shapes. Each is read from its
    own row of the InputBatch arrays, so a position derived from the wrong row - or a
    count that spills across the flattened batch - shows up as one request taking
    another's answer. Each request gets at most one position, and the one it would get
    alone."""
    batch = _make_input_batch(max_num_reqs=len(MIXED_PREFILL_BATCH))
    for req_id, prompt_len, computed, num_output_tokens, _, _ in MIXED_PREFILL_BATCH:
        batch.add_request(_make_request(req_id, prompt_len, computed, num_output_tokens))
    runner = _make_prefill_runner(batch)
    scheduled = [row[4] for row in MIXED_PREFILL_BATCH]

    positions = _extract_positions(runner, scheduled)

    wrong = []
    for batch_idx, (req_id, _, computed, _, seq_num_scheduled_tokens, expected) in enumerate(MIXED_PREFILL_BATCH):
        actual = positions.get(req_id)
        if actual != expected:
            wrong.append(f"{req_id}: got {actual}, expected {expected}")
        if actual and len(actual) > 1:
            wrong.append(f"{req_id}: sampled {len(actual)} logits in one step")
        alone = prefill_logits_position(computed, seq_num_scheduled_tokens, int(batch.num_tokens_no_spec[batch_idx]),
                                        False)
        alone_positions = [] if alone is None else [alone]
        if actual != alone_positions:
            wrong.append(f"{req_id}: got {actual} in the mixed batch but {alone_positions} on its own")
    assert not wrong, "requests interfered in a mixed prefill batch:\n  " + "\n  ".join(wrong)
    assert runner.invalid_req_indices == [], (f"nothing should be discarded without async scheduling, got "
                                              f"{runner.invalid_req_indices}")


def _make_update_states_runner(batch: InputBatch, requests: dict) -> types.SimpleNamespace:
    """Just enough of HPUModelRunner to drive _update_states on the host."""
    return types.SimpleNamespace(
        input_batch=batch,
        requests=requests,
        device=torch.device("cpu"),
        encoder_cache={},
        model=None,
        uses_mrope=False,
        use_async_scheduling=False,
        _gdn_req_to_base_slot={},
        _gdn_slot_free_list=[],
        _compact_gdn_group_ids=[],
    )


def test_pipeline_parallel_chunk_never_shrinks_the_non_draft_boundary():
    """P1: on a non-final pipeline-parallel rank, _update_states writes the request's
    token span from num_computed_tokens, which during a CHUNKED PREFILL is still below
    num_tokens_no_spec. Assigning it unconditionally moves the non-draft boundary
    BACKWARDS, and since this change keys the sampled position on that boundary, the
    shrunken value makes an intermediate chunk look complete: with the boundary pulled
    back to 32, prefill_logits_position(32, 32, 32) hits the negative-completion
    fallback and returns 31 where the honest boundary returns None.

    gpu_model_runner takes max(num_tokens_no_spec, num_computed_tokens + len(new_token_ids))
    for exactly this reason; this pins the same monotonicity here. The old formula keyed
    on num_prompt_tokens, which this write never touches, so the exposure is new."""
    prompt_len, computed, scheduled = PROMPT_LEN, 32, 32
    batch = _make_input_batch(max_num_reqs=1)
    request = _make_request("pp", prompt_len, num_computed_tokens=0, num_output_tokens=0)
    batch.add_request(request)
    assert int(batch.num_tokens_no_spec[0]) == prompt_len

    cached = CachedRequestData(
        req_ids=["pp"],
        resumed_req_ids=set(),
        # Empty on a prefill chunk: no token has been sampled for this request yet.
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[computed],
        num_output_tokens=[0],
    )
    scheduler_output = types.SimpleNamespace(
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={"pp": scheduled},
        scheduled_spec_decode_tokens={},
    )

    runner = _make_update_states_runner(batch, {"pp": request})
    original = hpu_model_runner.get_pp_group
    try:
        # Not the last rank: the branch that writes the boundary from num_computed_tokens.
        hpu_model_runner.get_pp_group = lambda: types.SimpleNamespace(is_last_rank=False)
        HPUModelRunner._update_states(runner, scheduler_output)
    finally:
        hpu_model_runner.get_pp_group = original

    boundary = int(batch.num_tokens_no_spec[0])
    assert boundary == prompt_len, (f"a chunked prefill on a non-final PP rank moved the non-draft boundary from "
                                    f"{prompt_len} to {boundary}; prefill_logits_position({computed}, {scheduled}, "
                                    f"{boundary}) would then return "
                                    f"{prefill_logits_position(computed, scheduled, boundary, False)} for an "
                                    f"intermediate chunk that must sample nothing")
    assert prefill_logits_position(
        computed, scheduled, boundary,
        False) is None, ("an intermediate chunk must still sample nothing after the PP update")


# (case, is_rejection_sampled, num_speculative_tokens, expected budget)
BUDGET_CASES = [
    ("prefill, speculation configured", False, NUM_SPECULATIVE_TOKENS, 1),
    ("prefill, no speculation", False, 0, 1),
    ("rejection-sampled decode", True, NUM_SPECULATIVE_TOKENS, 1 + NUM_SPECULATIVE_TOKENS),
    ("decode without speculation", True, 0, 1),
]


def test_prefill_rows_get_a_budget_of_one_however_many_drafts_are_configured():
    """P1: the budget is per row. Handing a prefill row the global 1 + k lets an
    over-emitting prefill commit k unverified tokens before the tripwire notices -
    which is precisely the failure the tripwire exists to catch, so a k-wide budget
    there makes the guard silent in the case it was added for."""
    wrong = []
    for case, is_rejection_sampled, num_speculative_tokens, expected in BUDGET_CASES:
        actual = sampled_token_budget(is_rejection_sampled, num_speculative_tokens)
        if actual != expected:
            wrong.append(f"{case}: is_rejection_sampled={is_rejection_sampled} "
                         f"num_speculative_tokens={num_speculative_tokens} -> {actual}, expected {expected}")
    assert not wrong, "wrong per-step token budget:\n  " + "\n  ".join(wrong)


def test_tripwire_fires_on_a_second_prefill_token_under_speculation():
    """P1: the prefill case the tripwire was added for, end to end through the budget
    rule. A prefill request that accumulated 2 tokens with k=3 configured must fail
    here; under the old global 1 + k budget it passed silently and both committed."""
    budget = sampled_token_budget(is_rejection_sampled=False, num_speculative_tokens=NUM_SPECULATIVE_TOKENS)

    assert_sampled_token_budget("prefill", 1, budget)

    try:
        assert_sampled_token_budget("prefill", 2, budget)
    except AssertionError as exc:
        assert "prefill" in str(exc), exc
    else:
        raise AssertionError(f"no tripwire for a prefill request committing 2 tokens with "
                             f"num_speculative_tokens={NUM_SPECULATIVE_TOKENS}; budget was {budget}")


def test_tripwire_without_speculation_allows_one_token_and_no_more():
    """P3: with speculation off the budget collapses to 1, so the tripwire is the only
    thing between an over-emitting prefill step and a silently corrupted request - there
    is no rejection sampler downstream to notice. Two tokens must fire it; the single
    legitimate token must not."""
    assert_sampled_token_budget("no_spec", 1, 1)

    try:
        assert_sampled_token_budget("no_spec", 2, 1)
    except AssertionError as exc:
        assert "no_spec" in str(exc), exc
        assert "2" in str(exc), exc
    else:
        raise AssertionError("no tripwire for 2 sampled tokens with speculation disabled")


if __name__ == "__main__":
    import sys

    failures = 0
    for name, test in sorted(dict(globals()).items()):
        if not name.startswith("test_"):
            continue
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - a standalone runner reports, it does not raise
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")
    print(f"{failures} failure(s)")
    sys.exit(1 if failures else 0)
