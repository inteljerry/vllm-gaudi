# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Copyright (C) 2026 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
"""Tests for `_moe_overload`, which binds one `hpu::mixture_of_experts` overload.

These run without Habana hardware: the operator packet is replaced by a stand-in whose
`__getattr__` raises `AttributeError` for names it does not carry, which is how
`torch._ops.OpOverloadPacket` behaves for an unregistered overload.
"""
import types
from unittest.mock import patch

import pytest

import vllm_gaudi.extension.ops as hpu_ops


class _FakePacket:
    """Stands in for `OpOverloadPacket`: unknown overloads raise `AttributeError`."""

    def __init__(self, overloads):
        self._overloads = dict(overloads)

    def __getattr__(self, name):
        try:
            return self._overloads[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __call__(self, *args, **kwargs):  # the packet itself is callable
        return "packet-call"


def _patch_packet(packet):
    """Point `torch.ops.hpu.mixture_of_experts` at `packet` for the duration."""
    fake = types.SimpleNamespace(hpu=types.SimpleNamespace(mixture_of_experts=packet))
    return patch.object(hpu_ops.torch, "ops", fake)


@pytest.fixture(autouse=True)
def _clear_warn_set():
    """The warn-once set is module state; leaking it across tests hides a second warning."""
    hpu_ops._MOE_OVERLOAD_FALLBACK_WARNED.clear()
    yield
    hpu_ops._MOE_OVERLOAD_FALLBACK_WARNED.clear()


def test_returns_the_named_overload_when_present():
    sentinel = object()
    packet = _FakePacket({"fp8_fused_weights_dynamic": sentinel})
    with _patch_packet(packet):
        assert hpu_ops._moe_overload("fp8_fused_weights_dynamic") is sentinel


def test_falls_back_to_the_packet_when_the_overload_is_absent():
    packet = _FakePacket({})
    with _patch_packet(packet):
        assert hpu_ops._moe_overload("fp8_fused_weights_dynamic") is packet


def test_fallback_warns_once_per_name_not_once_per_call():
    packet = _FakePacket({})
    with _patch_packet(packet), patch.object(hpu_ops.logger, "warning") as warn:
        for _ in range(5):
            hpu_ops._moe_overload("fp8_fused_weights_dynamic")
        assert warn.call_count == 1, "a per-call warning would flood the log on every MoE forward"
        hpu_ops._moe_overload("fp8_fused_weights")
        assert warn.call_count == 2, "a different missing overload deserves its own warning"


def test_a_transient_miss_is_not_remembered():
    """REGRESSION. The first version cached the resolved op under the overload's NAME, so a miss
    stored the slow packet permanently and no later call could recover the fast path. Torch already
    caches the lookup via `setattr(self, key, overload)` in `OpOverloadPacket.__getattr__`, so the
    only thing a second cache added was this failure mode."""
    sentinel = object()
    missing = _FakePacket({})
    with _patch_packet(missing):
        assert hpu_ops._moe_overload("fp8_fused_weights_dynamic") is missing

    # The op registers later, or a different build is in play.
    present = _FakePacket({"fp8_fused_weights_dynamic": sentinel})
    with _patch_packet(present):
        assert hpu_ops._moe_overload("fp8_fused_weights_dynamic") is sentinel, \
            "a transient miss must not pin the packet for the life of the process"


def test_schema_drift_surfaces_as_a_call_error_rather_than_a_silent_fallback():
    """A renamed-away argument is NOT an `AttributeError`, so the lookup succeeds and the call
    raises. That is intended: degrading silently to the packet would hide the incompatibility."""

    def _moved_on(**kwargs):
        raise TypeError("unexpected keyword argument 'd_scale_hidden_states'")

    packet = _FakePacket({"fp8_fused_weights_dynamic": _moved_on})
    with _patch_packet(packet):
        op = hpu_ops._moe_overload("fp8_fused_weights_dynamic")
        assert op is not packet
        with pytest.raises(TypeError):
            op(d_scale_hidden_states=object())
