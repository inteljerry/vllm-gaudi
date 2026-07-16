# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Copyright (C) 2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
"""Unit tests for the MiniMax-M3 reasoning parser.

vLLM's tool hand-off calls ``is_reasoning_end(delta_token_ids)`` with only the
current 1-2 token delta, while the prompt check passes the full sequence, so
the truth table covers delta fragments, prompts, and bundled shapes.

Synthetic token ids keep these runnable without the torch/HPU model stack;
importing the parser still needs vLLM, so the module is skipped where vLLM is
unavailable (e.g. WSL).
"""
import pytest

m3_reason = pytest.importorskip("vllm_gaudi.parsers.m3_reason")

START = 1   # stand-in for <mm:think>
END = 2     # stand-in for </mm:think>
TOOL = 3    # stand-in for <tool_call>
TXT = 5     # any reasoning/answer text token
TXT2 = 6


def _parser():
    """A parser instance with token ids set, bypassing the tokenizer-bound
    ``__init__`` so the test needs no model files."""
    p = object.__new__(m3_reason.MiniMaxM3ReasoningParser)
    p.start_token_id = START
    p.end_token_id = END
    p._toolcall_start_id = TOOL
    p._saw_think = False
    return p


@pytest.mark.parametrize(
    "input_ids, expected, why",
    [
        ([], False, "empty -> not ended"),
        ([START], False, "lone leading <mm:think> -> span open, not closed"),
        ([START, TXT, TXT2], False, "leading think, no close -> ongoing"),
        ([START, TXT, END], True, "leading think, closed -> ended"),
        # A reasoning-text delta carries neither tag and must not be treated
        # as reasoning-ended.
        ([TXT], False, "reasoning-text delta -> still inside span"),
        ([TXT, TXT2], False, "multi-token reasoning delta -> still inside"),
        # Closing tag arriving as its own delta ends the span.
        ([END], True, "lone closing </mm:think> delta -> ended"),
        # Disabled mode: chat template appends </mm:think> as the last prompt
        # token, so the prompt sequence ends already-closed.
        ([9, START, 8, END], True, "prompt ending in </mm:think> -> ended"),
        # Prompt whose instructions embed the tags but ends in plain text must
        # not be mistaken for a boundary.
        ([9, START, 8, END, 10], False, "instruction tags mid-prompt -> not ended"),
        # Bundled "reasoning</mm:think>answer" delta closes the span.
        ([TXT, END, TXT2], True, "close tag inside delta -> ended"),
        # Direct tool call without thinking: <tool_call> in a delta-sized
        # fragment before any think span -> ended (opens the tool gate).
        ([TOOL], True, "direct <tool_call> delta -> ended"),
        ([TXT, TOOL], True, "prefixed direct <tool_call> delta -> ended"),
        # Prompt-sized inputs never trip the tool-token case (tool-definition
        # examples embed <tool_call> in prompts).
        ([9, 8, 7, TOOL, 6], False, "prompt-sized input with tool token"),
    ],
)
def test_is_reasoning_end(input_ids, expected, why):
    assert _parser().is_reasoning_end(input_ids) is expected, why


def test_tool_token_ignored_once_think_span_open():
    """A literal "<tool_call>" inside reasoning text must not end the span:
    after the stream has opened a think span (tracked by streaming calls),
    the tool-token case is suppressed."""
    p = _parser()
    p._thinking_mode = "default"
    # First streaming delta is the think-open token -> records _saw_think.
    p.extract_reasoning_streaming(
        previous_text="", current_text="<mm:think>", delta_text="<mm:think>",
        previous_token_ids=[], current_token_ids=[START],
        delta_token_ids=[START])
    assert p._saw_think is True
    assert p.is_reasoning_end([TOOL]) is False
    # Sanity: same delta on a fresh parser (no think span) opens the gate.
    assert _parser().is_reasoning_end([TOOL]) is True


def test_new_api_hooks_present_and_aliased():
    """Both hook-name generations must resolve to the same override."""
    cls = m3_reason.MiniMaxM3ReasoningParser
    assert cls.extract_reasoning is cls.extract_reasoning_content
    assert (cls.extract_reasoning_streaming
            is cls.extract_reasoning_content_streaming)


# ---------------------------------------------------------------------------
# extract_reasoning (complete, non-streaming outputs)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model_output, expected, why",
    [
        # Default/adaptive thinking: leading <mm:think>.
        ("<mm:think>R</mm:think>A", ("R", "A"), "leading tag -> base split"),
        # Enabled mode: template pre-filled <mm:think> in the PROMPT, so the
        # generation is the trailing shape with no leading tag.
        ("R</mm:think>A", ("R", "A"), "enabled shape -> split on end tag"),
        ("R</mm:think>", ("R", None), "enabled shape, no answer yet"),
        # Disabled / adaptive no-think: no tags -> all content.
        ("A", (None, "A"), "tag-less answer -> content"),
        ("", (None, None), "empty output"),
    ],
)
def test_extract_reasoning(model_output, expected, why):
    p = _parser()
    assert p.extract_reasoning(model_output, request=None) == expected, why


# ---------------------------------------------------------------------------
# extract_reasoning_streaming: enabled mode (no start token in GENERATION)
# ---------------------------------------------------------------------------

def _stream_parser(mode):
    p = _parser()
    p._thinking_mode = mode
    return p


def test_streaming_enabled_mode_routes_reasoning():
    """Enabled mode: generation begins inside reasoning (start tag lives in
    the prompt). Deltas with no tags must be reasoning, not content."""
    p = _stream_parser("enabled")
    dm = p.extract_reasoning_streaming(
        previous_text="", current_text="The", delta_text="The",
        previous_token_ids=[], current_token_ids=[TXT], delta_token_ids=[TXT])
    assert dm is not None
    assert dm.reasoning == "The"
    assert not getattr(dm, "content", None)


def test_streaming_enabled_mode_post_end_is_content():
    p = _stream_parser("enabled")
    dm = p.extract_reasoning_streaming(
        previous_text="R</mm:think>", current_text="R</mm:think>A",
        delta_text="A", previous_token_ids=[TXT, END],
        current_token_ids=[TXT, END, TXT2], delta_token_ids=[TXT2])
    assert dm is not None
    assert dm.content == "A"


def test_streaming_split_delta_separates_fields():
    """End token bundled mid-delta ("R</mm:think>A"): both halves must land in
    the canonical DeltaMessage fields. Constructing with reasoning_content=
    would succeed (pydantic extra=allow) but leave .reasoning None."""
    p = _stream_parser("enabled")
    dm = p.extract_reasoning_streaming(
        previous_text="", current_text="R</mm:think>A",
        delta_text="R</mm:think>A",
        previous_token_ids=[],
        current_token_ids=[TXT, END, TXT2],
        delta_token_ids=[TXT, END, TXT2])
    assert dm is not None
    assert dm.reasoning == "R"
    assert dm.content == "A"


def test_streaming_default_mode_tagless_is_content():
    """Default/adaptive no-think: a tag-less stream stays content."""
    p = _stream_parser("default")
    dm = p.extract_reasoning_streaming(
        previous_text="", current_text="A", delta_text="A",
        previous_token_ids=[], current_token_ids=[TXT], delta_token_ids=[TXT])
    assert dm is not None
    assert dm.content == "A"
    assert not getattr(dm, "reasoning", None)
