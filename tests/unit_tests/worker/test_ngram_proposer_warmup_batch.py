# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Copyright (C) 2026 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
"""Regression tests for the n-gram proposer at warmup batch widths.

`NgramProposer.__init__` sizes `valid_ngram_draft` / `valid_ngram_num_drafts` to
`scheduler_config.max_num_seqs`, but `batch_propose` indexes them by the caller's
request count. HPU warmup drives batches wider than `max_num_seqs`:
`generate_spec_decode_buckets` widens every decode bucket to
`bs * (1 + num_speculative_tokens)`, and `warmup_model` rebuilds `input_batch` to
the widest of those buckets. Row `max_num_seqs` then walks off both buffers, which
is why `VLLM_SKIP_WARMUP=false` plus n-gram speculation aborted engine startup with
`index 8 is out of bounds for axis 0 with size 8`.

These run without Habana hardware: nothing here touches a device, and the proposer
is pure numpy/numba on the host.

pytest is not installed in the serving image, so this module also runs as a plain
script:

    python tests/unit_tests/worker/test_ngram_proposer_warmup_batch.py
"""
import os
import types

# The proposer's numba kernel is `parallel=True`, so importing it spins a pool sized
# to the host's core count. On a 288-core Gaudi node that pool aborts with "the futex
# facility returned an unexpected error code" and hangs the run; bound it before numba
# is imported anywhere. Harmless where the default already works.
os.environ.setdefault("NUMBA_NUM_THREADS", "8")

import numpy as np  # noqa: E402
import habana_frameworks.torch  # noqa: E402,F401

from vllm.v1.spec_decode.ngram_proposer import NgramProposer  # noqa: E402

from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner  # noqa: E402

MAX_NUM_SEQS = 8
NUM_SPECULATIVE_TOKENS = 3
# What generate_spec_decode_buckets widens the decode buckets to, and therefore the
# widest batch warmup asks the proposer for. This is the deployed GLM-5.3 shape.
WARMUP_BATCH = MAX_NUM_SEQS * (1 + NUM_SPECULATIVE_TOKENS)
MAX_MODEL_LEN = 4096
CTX_LEN = 64


def _make_proposer() -> NgramProposer:
    config = types.SimpleNamespace(
        speculative_config=types.SimpleNamespace(
            prompt_lookup_min=1,
            prompt_lookup_max=4,
            num_speculative_tokens=NUM_SPECULATIVE_TOKENS,
        ),
        model_config=types.SimpleNamespace(max_model_len=MAX_MODEL_LEN),
        scheduler_config=types.SimpleNamespace(max_num_seqs=MAX_NUM_SEQS),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=8),
    )
    return NgramProposer(config)


def _make_batch(num_requests: int):
    """A batch where every row sampled a token and carries a matchable n-gram."""
    sampled_token_ids = [[7] for _ in range(num_requests)]
    num_tokens_no_spec = np.full(num_requests, CTX_LEN, dtype=np.int32)
    token_ids_cpu = np.zeros((num_requests, MAX_MODEL_LEN), dtype=np.int32)
    # A short repeating pattern, so the suffix match always finds a continuation.
    token_ids_cpu[:, :CTX_LEN] = np.tile(np.arange(1, 9, dtype=np.int32), CTX_LEN // 8)
    return sampled_token_ids, num_tokens_no_spec, token_ids_cpu


def _make_runner(proposer: NgramProposer, num_requests: int):
    """The attributes `propose_ngram_draft_token_ids` reads, and nothing else."""
    sampled_token_ids, num_tokens_no_spec, token_ids_cpu = _make_batch(num_requests)
    runner = types.SimpleNamespace(
        drafter=proposer,
        speculative_config=types.SimpleNamespace(num_speculative_tokens=NUM_SPECULATIVE_TOKENS),
        input_batch=types.SimpleNamespace(num_tokens_no_spec=num_tokens_no_spec, token_ids_cpu=token_ids_cpu),
        ngram_skip_empty_draft=True,
    )
    return runner, sampled_token_ids


def test_serving_width_batch_fits_the_preallocated_buffers():
    """max_num_seqs rows is the serving ceiling; upstream handles it unaided."""
    proposer = _make_proposer()
    drafts = proposer.propose(NUM_SPECULATIVE_TOKENS, *_make_batch(MAX_NUM_SEQS))

    assert len(drafts) == MAX_NUM_SEQS
    assert all(len(d) == NUM_SPECULATIVE_TOKENS for d in drafts), drafts


def test_upstream_still_sizes_its_buffers_to_max_num_seqs():
    """The precondition for the bug the plugin guards against.

    A failure here means upstream started sizing its buffers to the batch, and
    `grow_ngram_proposer_buffers` can be retired.

    This asserts on the shapes rather than running the overrun. `batch_propose_numba`
    is `@njit`, which does not bounds-check, so an over-wide batch corrupts the heap
    on the write at `ngram_proposer.py:201` before the read at `:126` raises -- a test
    that provoked it would poison the rest of the session (observed: the interpreter
    segfaults in `Py_FinalizeEx` afterwards, even when every assertion passed).
    """
    proposer = _make_proposer()

    assert proposer.valid_ngram_num_drafts.shape == (MAX_NUM_SEQS, )
    assert proposer.valid_ngram_draft.shape == (MAX_NUM_SEQS, NUM_SPECULATIVE_TOKENS)


def test_wrapper_proposes_for_every_row_of_a_warmup_width_batch():
    """The regression: warmup asks for WARMUP_BATCH rows and must get them all."""
    proposer = _make_proposer()
    runner, sampled_token_ids = _make_runner(proposer, WARMUP_BATCH)

    drafts = HPUModelRunner.propose_ngram_draft_token_ids(runner, sampled_token_ids)

    assert len(drafts) == WARMUP_BATCH
    assert all(len(d) == NUM_SPECULATIVE_TOKENS for d in drafts), drafts


def test_serving_width_batch_is_unchanged_after_a_warmup_width_batch():
    """Warmup widens the buffers; the serving batches that follow must be unaffected."""
    proposer = _make_proposer()
    warmup_runner, warmup_sampled = _make_runner(proposer, WARMUP_BATCH)
    HPUModelRunner.propose_ngram_draft_token_ids(warmup_runner, warmup_sampled)

    runner, sampled_token_ids = _make_runner(proposer, MAX_NUM_SEQS)
    drafts = HPUModelRunner.propose_ngram_draft_token_ids(runner, sampled_token_ids)

    assert len(drafts) == MAX_NUM_SEQS
    assert all(len(d) == NUM_SPECULATIVE_TOKENS for d in drafts), drafts


def test_unmatched_row_still_reports_no_draft_at_warmup_width():
    """Growing the buffers must not turn an empty draft into a spurious one."""
    proposer = _make_proposer()
    runner, sampled_token_ids = _make_runner(proposer, WARMUP_BATCH)
    # Too short for prompt_lookup_min, so the last row has nothing to match on.
    runner.input_batch.num_tokens_no_spec[-1] = 0

    drafts = HPUModelRunner.propose_ngram_draft_token_ids(runner, sampled_token_ids)

    assert len(drafts) == WARMUP_BATCH
    assert drafts[-1] == []
    assert all(len(d) == NUM_SPECULATIVE_TOKENS for d in drafts[:-1]), drafts


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
