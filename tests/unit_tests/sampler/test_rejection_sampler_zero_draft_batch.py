"""A batch where EVERY sequence proposed zero draft tokens must not crash the sampler.

Reachable in production: with VLLM_HPU_NGRAM_SKIP_EMPTY_DRAFT=true an unmatched n-gram proposes
nothing, and once concurrency lets a whole batch miss on the same step, max_draft_tokens is 0 and
`torch.argmax(..., dim=1)` raises `IndexError: argmax(): Expected reduction dim 1 to have non-zero
size`. At C=1 a lone miss is dropped from scheduled_spec_decode_tokens and never reaches the
sampler, which is why this only appears at C>=2.

Observed on the deployed stack: the ISL ladder passed 1K-200K at C=1, then C=2/4/8 returned HTTP
500 and killed all eight TP workers at hpu_rejection_sampler.py:88.

These run on CPU -- rejection_sample_pytorch moves its inputs to CPU before any arithmetic -- so
no Gaudi device is required.
"""
import torch

try:
    import pytest
except ImportError:  # the serving image ships torch but not pytest
    pytest = None

from vllm_gaudi.v1.sample.hpu_rejection_sampler import (PLACEHOLDER_TOKEN_ID, rejection_sample_pytorch)


def _call(num_draft_tokens, max_draft, bonus, draft=None, target=None):
    n = len(num_draft_tokens)
    shape = (n, max_draft)
    draft = torch.zeros(shape, dtype=torch.int32) if draft is None else draft
    target = torch.zeros(shape, dtype=torch.int32) if target is None else target
    cu = torch.tensor(num_draft_tokens, dtype=torch.int32).cumsum(0)
    return rejection_sample_pytorch(draft, target, torch.tensor(bonus, dtype=torch.int32).view(n, 1),
                                    num_draft_tokens, cu)


def test_whole_batch_with_no_drafts_does_not_raise():
    """The regression: this raised IndexError before the fix."""
    out = _call([0, 0], max_draft=0, bonus=[41, 42])
    assert out.shape == (2, 1), f"expected one bonus column, got {tuple(out.shape)}"
    assert out[:, 0].tolist() == [41, 42], "each sequence must still emit its bonus token"


def test_single_sequence_with_no_draft():
    out = _call([0], max_draft=0, bonus=[7])
    assert out.shape == (1, 1)
    assert out[0, 0].item() == 7


def test_all_drafts_accepted_still_works():
    """Guard against the fix disturbing the normal path."""
    draft = torch.tensor([[5, 6]], dtype=torch.int32)
    out = _call([2], max_draft=2, bonus=[9], draft=draft, target=draft.clone())
    assert out[0, :2].tolist() == [5, 6], "matching drafts must be accepted"
    assert out[0, 2].item() == 9, "bonus token follows a fully accepted draft"


def test_first_mismatch_truncates():
    draft = torch.tensor([[5, 6]], dtype=torch.int32)
    target = torch.tensor([[5, 99]], dtype=torch.int32)
    out = _call([2], max_draft=2, bonus=[9], draft=draft, target=target)
    assert out[0, 0].item() == 5, "the matching prefix is accepted"
    assert out[0, 1].item() == 99, "the mismatch position takes the target token"
    assert out[0, 2].item() == PLACEHOLDER_TOKEN_ID, "nothing is emitted past the mismatch"


def test_mixed_batch_one_seq_without_a_draft():
    """A partial miss was already handled; keep it covered so the fix does not regress it."""
    draft = torch.tensor([[5, 6], [0, 0]], dtype=torch.int32)
    out = _call([2, 0], max_draft=2, bonus=[9, 8], draft=draft, target=draft.clone())
    assert out[0, :2].tolist() == [5, 6]
    assert out[0, 2].item() == 9
    assert out[1, 0].item() == 8, "the draft-less sequence emits only its bonus token"
    assert out[1, 1].item() == PLACEHOLDER_TOKEN_ID


if __name__ == "__main__":
    # Runnable without pytest so it can execute inside the serving image, where torch exists but
    # pytest does not. Exit code is the contract: the pre-fix sampler exits non-zero here.
    if pytest is not None:
        raise SystemExit(pytest.main([__file__, "-v", "--no-header", "-x"]))
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as exc:
                failed += 1
                print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"{'FAILED' if failed else 'OK'}  ({failed} failure(s))")
    raise SystemExit(1 if failed else 0)
