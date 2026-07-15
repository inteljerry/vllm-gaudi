# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Copyright (C) 2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
"""Unit tests for MiniMax-M3 dense-vs-MoE layer selection.

Importing ``vllm_gaudi.models.minimax_m3`` pulls the full torch/vllm/HPU stack,
so these are HOST-RUN (Gaudi container):

    pytest tests/unit_tests/test_minimax_m3_layers.py -v

On a box where the model module can't import (e.g. WSL) the module is skipped.
"""
import pytest

minimax_m3 = pytest.importorskip("vllm_gaudi.models.minimax_m3")
build_decoder_layer_types = minimax_m3.build_decoder_layer_types


def test_dense_then_moe_layer_split():
    # M3: first three layers dense (0,0,0), remainder MoE.
    freq = [0, 0, 0] + [1] * 57
    types = build_decoder_layer_types(freq)
    assert types[:3] == ["dense", "dense", "dense"]
    assert types[3] == "moe"
    assert types[59] == "moe"
    assert len(types) == 60


def test_all_moe_when_all_ones():
    types = build_decoder_layer_types([1] * 4)
    assert types == ["moe"] * 4


def test_all_dense_when_all_zero():
    types = build_decoder_layer_types([0] * 4)
    assert types == ["dense"] * 4
