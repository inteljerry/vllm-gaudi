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

These run on the host: nothing here executes on a device.

pytest is not installed in the serving image, so this module also runs as a plain
script:

    python tests/unit_tests/worker/test_prefill_multi_logit.py
"""
import types

import torch
import habana_frameworks.torch  # noqa: F401

from vllm.sampling_params import SamplingParams
from vllm.utils.platform_utils import is_pin_memory_available

from vllm_gaudi.v1.worker.hpu_input_batch import CachedRequestState, InputBatch
from vllm_gaudi.v1.worker.hpu_model_runner import (HPUModelRunner, assert_sampled_token_budget, prefill_logits_position)

VOCAB_SIZE = 1024
MAX_MODEL_LEN = 1024

# The deployed GLM-5.3 shape at the crash: 100 prompt tokens, 5 emitted before
# preemption, k=3 draft tokens, resuming against a block-aligned hit of its own
# just-freed blocks.
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
    # Resumed from preemption with a partial prefix-cache hit - the crash shape.
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
    """The headline case for the shipped configuration. `add_request` leaves
    num_prompt_tokens at P for a request resuming with a partial prefix-cache hit,
    and this change does NOT correct that - it stops the prefill path depending on
    it. num_tokens_no_spec is P + O regardless, so the position is right anyway."""
    computed, scheduled = PREFIX_CACHE_HIT, PROMPT_LEN + NUM_OUTPUT_TOKENS - PREFIX_CACHE_HIT
    short_num_prompt_tokens = PROMPT_LEN
    no_spec = PROMPT_LEN + NUM_OUTPUT_TOKENS

    position = prefill_logits_position(computed, scheduled, no_spec, always_sample=False)

    assert position == scheduled - 1, (f"resumed request sampled position {position}, expected {scheduled - 1} "
                                       f"(the last of its {scheduled} scheduled tokens)")
    # Keying on the short num_prompt_tokens instead would land NUM_OUTPUT_TOKENS early.
    assert short_num_prompt_tokens - 1 - computed == position - NUM_OUTPUT_TOKENS, (
        "these inputs no longer separate the two bases; pick inputs where they differ")


def test_narrowness_no_divergence_from_head_before_the_overshoot_region():
    """The change must alter behaviour ONLY where padded drafts live. Inside
    computed + scheduled <= num_tokens_no_spec - every ordinary prefill chunk - it has
    to reproduce the old formula exactly."""
    divergences = []
    for computed in range(60):
        for scheduled in range(1, 20):
            for no_spec in range(1, 80):
                if computed + scheduled > no_spec:
                    continue
                for always_sample in (False, True):
                    # No drafts in this region, so num_prompt_tokens == num_tokens_no_spec.
                    head = _head_logits_positions(computed, scheduled, no_spec, always_sample)
                    new = _positions(computed, scheduled, no_spec, always_sample)
                    if head != new:
                        divergences.append(f"computed={computed} scheduled={scheduled} "
                                           f"num_tokens_no_spec={no_spec} always_sample={always_sample}: "
                                           f"was {head}, now {new}")
    assert not divergences, (f"{len(divergences)} divergence(s) outside the over-shoot region:\n  " +
                             "\n  ".join(divergences[:10]))


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


def _make_prefill_runner(batch: InputBatch) -> types.SimpleNamespace:
    """Just enough of HPUModelRunner to drive _extract_prefill_batch_contents on the
    host. warmup=True skips _resolve_block, and merging is stubbed off so each request
    keeps its own BatchContents."""
    runner = types.SimpleNamespace(
        input_batch=batch,
        attn_block_size=128,
        use_async_scheduling=False,
        use_structured_output=False,
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
