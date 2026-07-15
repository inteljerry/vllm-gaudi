# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import uuid
from collections.abc import Sequence
from typing import Any

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (DeltaFunctionCall, DeltaMessage, DeltaToolCall, ExtractedToolCallInformation, FunctionCall, ToolCall)
from vllm.tool_parsers.abstract_tool_parser import (ToolParser, ToolParserManager)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike as AnyTokenizer

logger = init_logger(__name__)

# Namespace token that may appear before tags in the raw model output. It must
# be stripped/ignored defensively (treat it as a no-op).
NAMESPACE_TOKEN = "]<]minimax[>["

# Max size (chars) of a parameter body to run recursive element parsing on. The
# backreference regex is O(n^2) on pathological malformed input (thousands of
# unclosed pseudo-tags), so beyond this cap the value is kept as a raw string
# instead of parsed. Well-formed tool calls are far smaller than this.
MAX_ELEMENT_PARSE_LEN = 100_000


@ToolParserManager.register_module("minimax_m3")
class MinimaxM3ToolParser(ToolParser):
    """
    Tool parser for the MiniMax M3 model.

    M3 renders tool calls as::

        <tool_call>
        <invoke name="get_weather"><city>Paris</city></invoke>
        </tool_call>

    Multiple parallel <invoke> blocks may appear inside a single <tool_call>
    block. Each parameter is a tag whose name IS the parameter name
    (``<city>Paris</city>``, NOT ``<parameter name="city">``). A parameter
    value may be a scalar or nested XML: ``<item>...</item>`` blocks represent
    nested objects/arrays and are recursively expanded into the JSON arguments.

    This differs from the M2 parser in two ways: the block tokens are
    ``<tool_call>``/``</tool_call>`` (not ``<minimax:tool_call>``), and
    parameters are ``<name>val</name>`` (not ``<parameter name="name">val``).
    """

    def __init__(self, tokenizer: AnyTokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        self.prev_tool_call_arr: list[dict] = []

        # Sentinel tokens
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.invoke_start_prefix: str = "<invoke name="
        self.invoke_end_token: str = "</invoke>"

        # Streaming state (see extract_tool_calls_streaming).
        self.is_tool_call_started: bool = False
        # Number of leading-content chars already streamed to the client, so
        # the content preceding a tool call is never emitted twice.
        self.content_chars_streamed: int = 0
        self.tool_calls_emitted: bool = False

        # Namespace-stripped text built incrementally, one delta at a time
        # (see extract_tool_calls_streaming), plus the cached tag positions
        # so a long generation doesn't re-search the whole accumulated text
        # on every streamed token.
        self._normalized_buffer: str = ""
        self._start_idx: int | None = None
        self._end_idx: int | None = None
        self._start_search_pos: int = 0
        self._end_search_pos: int = 0
        # How far back a tag search must look to catch one split across two
        # deltas (e.g. "<tool" + "_call>").
        self._tag_search_window: int = max(
            len(self.tool_call_start_token), len(self.tool_call_end_token)
        ) - 1

        # Regex patterns for complete parsing
        self.tool_call_complete_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        self.invoke_complete_regex = re.compile(r"<invoke name=(.*?)</invoke>",
                                                re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction.")

        self.tool_call_start_token_id = self.vocab.get(
            self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if (self.tool_call_start_token_id is None
                or self.tool_call_end_token_id is None):
            raise RuntimeError(
                "MiniMax M3 Tool parser could not locate tool call start/end "
                "tokens in the tokenizer!")

        logger.debug("vLLM Successfully import tool parser %s !",
                     self.__class__.__name__)

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    def _strip_namespace(self, text: str) -> str:
        """Strip the optional namespace token so tags parse uniformly."""
        return text.replace(NAMESPACE_TOKEN, "")

    def _extract_name(self, name_str: str) -> str:
        """Extract name from a quoted string."""
        name_str = name_str.strip()
        if (name_str.startswith('"') and name_str.endswith('"')
                or name_str.startswith("'") and name_str.endswith("'")):
            return name_str[1:-1]
        return name_str

    def _convert_scalar(self, value: str) -> Any:
        """Best-effort conversion of a scalar string value to a JSON type."""
        stripped = value.strip()
        lowered = stripped.lower()
        if lowered in ("null", "none", "nil"):
            return None
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        try:
            return int(stripped)
        except (ValueError, TypeError):
            pass
        try:
            val = float(stripped)
            return val if val != int(val) else int(val)
        except (ValueError, TypeError):
            pass
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return value

    # Matches a single well-formed <tag> ... </tag> element, spanning newlines.
    # The backreference to the opening tag pairs it with its own closing tag.
    _element_regex = re.compile(r"<([^\s/>]+)>(.*?)</\1>", re.DOTALL)

    def _parse_param_value(self, inner_xml: str) -> Any:
        """
        Parse the inner content of a single parameter block.

        If the content contains nested tags (e.g. <item> blocks), parse it
        recursively into nested objects/arrays. Otherwise treat it as a scalar
        value. Uses recursive regex rather than an XML library so there is no
        entity-expansion (XXE / billion-laughs) attack surface and no new
        dependency, matching the M2 parser's regex-based approach.

        - Content with no child tags -> scalar value.
        - Content whose top-level children are all <item> -> list of the
          recursively-parsed items.
        - Content with named top-level children -> dict keyed by tag name.
        """
        inner_xml = self._strip_namespace(inner_xml)
        stripped = inner_xml.strip()

        # Fast path: no nested tags -> plain scalar value.
        if "<" not in stripped:
            return self._convert_scalar(stripped)

        # Size guard: skip the O(n^2) recursive regex on pathologically large
        # bodies; keep the raw text instead of hanging on malformed input.
        if len(inner_xml) > MAX_ELEMENT_PARSE_LEN:
            return stripped

        children = [(m.group(1), m.group(2))
                    for m in self._element_regex.finditer(inner_xml)]
        if not children:
            # Looks like markup but has no well-formed child element; keep the
            # raw text as a scalar rather than dropping it.
            return self._convert_scalar(stripped)

        if all(tag == "item" for tag, _ in children):
            return [self._parse_param_value(value) for _, value in children]

        return {
            tag: self._parse_param_value(value)
            for tag, value in children
        }

    def _parse_single_invoke(self, invoke_str: str) -> ToolCall | None:
        """Parse a single <invoke ...> ... </invoke> block body."""
        invoke_str = self._strip_namespace(invoke_str)

        # invoke_str is the text captured between "<invoke name=" and
        # "</invoke>". The function name comes first, up to the closing ">".
        name_match = re.search(r"^([^>]*)>", invoke_str, re.DOTALL)
        if not name_match:
            return None

        function_name = self._extract_name(name_match.group(1))
        if not function_name:
            return None

        # The remainder holds the parameter tags.
        params_body = invoke_str[name_match.end():]

        param_dict: dict[str, Any] = {}
        # Size guard: skip the O(n^2) recursive regex on a pathologically
        # large parameter body rather than hang on malformed input (see
        # MAX_ELEMENT_PARSE_LEN / _parse_param_value's matching guard).
        if len(params_body) <= MAX_ELEMENT_PARSE_LEN:
            # Match top-level <name> ... </name> parameter blocks. The value
            # may itself contain nested tags, so the backreference pairs each
            # opening tag with its own closing tag and DOTALL spans newlines.
            for match in self._element_regex.finditer(params_body):
                param_name = match.group(1)
                param_value = match.group(2)
                param_dict[param_name] = self._parse_param_value(param_value)

        return ToolCall(
            type="function",
            function=FunctionCall(
                name=function_name,
                arguments=json.dumps(param_dict, ensure_ascii=False),
            ),
        )

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from complete model output (non-streaming)."""
        # Quick check (tolerate the namespace token wrapping the token).
        normalized = self._strip_namespace(model_output)
        if self.tool_call_start_token not in normalized:
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

        try:
            tool_calls = []

            # Find all complete tool_call blocks.
            for tool_call_match in self.tool_call_complete_regex.findall(
                    normalized):
                # Find all invokes within this tool_call.
                for invoke_match in self.invoke_complete_regex.findall(
                        tool_call_match):
                    tool_call = self._parse_single_invoke(invoke_match)
                    if tool_call:
                        tool_calls.append(tool_call)

            if not tool_calls:
                return ExtractedToolCallInformation(tools_called=False,
                                                    tool_calls=[],
                                                    content=model_output)

            # Update prev_tool_call_arr.
            self.prev_tool_call_arr.clear()
            for tool_call in tool_calls:
                self.prev_tool_call_arr.append({
                    "name":
                    tool_call.function.name,
                    "arguments":
                    tool_call.function.arguments,
                })

            # Extract content before first tool call.
            first_tool_idx = normalized.find(self.tool_call_start_token)
            content = (normalized[:first_tool_idx]
                       if first_tool_idx > 0 else None)

            return ExtractedToolCallInformation(tools_called=True,
                                                tool_calls=tool_calls,
                                                content=content)

        except Exception:
            logger.exception("Error extracting tool calls")
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,  # pylint: disable=unused-argument
        previous_token_ids: Sequence[int],  # pylint: disable=unused-argument
        current_token_ids: Sequence[int],  # pylint: disable=unused-argument
        delta_token_ids: Sequence[int],  # pylint: disable=unused-argument
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """
        Extract tool calls from streaming model output.

        Simple buffering strategy: while no tool call has started, stream
        content through as it arrives. Once a ``<tool_call>`` appears, emit only
        the not-yet-sent remainder of the preceding content, then withhold
        further deltas until the ``</tool_call>`` block is complete and emit the
        parsed tool calls in one shot. All content is namespace-stripped so the
        ``]<]minimax[>[`` token is never echoed. This is correct and never
        crashes; it does not stream partial arguments token-by-token.

        The namespace-stripped text is built incrementally (each delta is
        stripped and appended to a running buffer) and the ``<tool_call>``/
        ``</tool_call>`` positions are found via a bounded look-back window
        and cached, rather than re-stripping/re-scanning the full
        accumulated text on every streamed token (which is O(n^2) over a
        long generation). The final parse still runs over the true,
        untouched ``current_text`` (see ``extract_tool_calls`` below), so
        this incremental bookkeeping only affects the plain-content delta
        that gets streamed before/around a tool call, not the parsed
        arguments themselves.
        """
        # A new generation restarts state (streaming instances are reused).
        if not previous_text:
            self.is_tool_call_started = False
            self.content_chars_streamed = 0
            self.tool_calls_emitted = False
            self._normalized_buffer = ""
            self._start_idx = None
            self._end_idx = None
            self._start_search_pos = 0
            self._end_search_pos = 0

        self._normalized_buffer += self._strip_namespace(delta_text)

        if self._start_idx is None:
            window = self._tag_search_window
            search_from = max(0, self._start_search_pos - window)
            pos = self._normalized_buffer.find(self.tool_call_start_token,
                                               search_from)
            self._start_search_pos = len(self._normalized_buffer)
            if pos != -1:
                self._start_idx = pos

        # No tool call anywhere yet: stream the not-yet-sent content tail.
        if self._start_idx is None:
            new_content = self._normalized_buffer[self.content_chars_streamed:]
            self.content_chars_streamed = len(self._normalized_buffer)
            return DeltaMessage(content=new_content) if new_content else None

        self.is_tool_call_started = True

        # Emit the remainder of any content that precedes the tool call, once.
        if self.content_chars_streamed < self._start_idx:
            new_content = self._normalized_buffer[
                self.content_chars_streamed:self._start_idx]
            self.content_chars_streamed = self._start_idx
            return DeltaMessage(content=new_content)

        # Already emitted the tool calls, or the block is not complete yet.
        if self.tool_calls_emitted:
            return None

        if self._end_idx is None:
            window = self._tag_search_window
            search_from = max(self._start_idx,
                              self._end_search_pos - window)
            pos = self._normalized_buffer.find(self.tool_call_end_token,
                                               search_from)
            self._end_search_pos = len(self._normalized_buffer)
            if pos != -1:
                self._end_idx = pos
        if self._end_idx is None:
            return None

        extracted = self.extract_tool_calls(current_text, request)
        if not extracted.tools_called:
            return None

        self.tool_calls_emitted = True
        tool_calls = [
            DeltaToolCall(
                index=index,
                id=tool_call.id,
                type="function",
                function=DeltaFunctionCall(
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                ),
            ) for index, tool_call in enumerate(extracted.tool_calls)
        ]
        return DeltaMessage(tool_calls=tool_calls)
