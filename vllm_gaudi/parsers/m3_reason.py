# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                                  ResponsesRequest)

logger = init_logger(__name__)


@ReasoningParserManager.register_module("minimax_m3")
class MiniMaxM3ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for MiniMax M3 model.

    M3 has two verified output modes:

    - Thinking mode (default): the model ALWAYS emits <mm:think> as its very
      first output token, then reasoning, then </mm:think>, then the answer:
      ``<mm:think>REASONING</mm:think>ANSWER``.
    - Disabled mode (``thinking_mode: disabled``): the model emits the answer
      directly with NO think tags at all: ``ANSWER``.

    So the presence of a leading <mm:think> distinguishes the two modes.
    Reasoning is the span between <mm:think> and </mm:think>; text after
    </mm:think> (or the whole output in disabled mode) is content.

    Unlike MiniMax M2, M3 does NOT produce the "no start token but a trailing
    </mm:think>" shape, so that (M2-style) case is not handled here.

    Note: M3 uses <mm:think>/</mm:think>, NOT the <think>/</think> tokens used
    by MiniMax M2. M3's chat template also embeds a literal </mm:think> in its
    prompt instructions, which is why is_reasoning_end is overridden below to
    key off the model's leading <mm:think> rather than a bare end token.
    """

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<mm:think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</mm:think>"

    def extract_reasoning_content(
        self, model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        """
        Split a complete model output into (reasoning_content, content).

        The base implementation returns ``(model_output, None)`` when there is
        no </mm:think> -- i.e. it labels a plain answer as reasoning. That is
        wrong for M3's disabled mode, where a tag-less output is entirely the
        answer. Route by the leading <mm:think>:

        - Thinking mode (starts with <mm:think>): reasoning is between the
          tags; text after </mm:think> is content (base behavior is correct).
        - Disabled mode (no leading <mm:think>): the whole output is content,
          reasoning is empty.
        """
        # Disabled / no-think output -> everything is content.
        if not model_output.startswith(self.start_token):
            return None, model_output or None

        # Thinking mode -> defer to the base tag-splitting behavior.
        return super().extract_reasoning_content(model_output, request)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Whether reasoning has ended within the MODEL's generated output.

        vLLM gates streaming tool-call parsing on this: tool parsing only starts
        once reasoning_end flips True. It is called on the prompt tokens once and
        on the accumulated generated tokens per step. Three M3 shapes matter:

        1. No <mm:think> anywhere -> the model produced NO reasoning span
           (thinking disabled, or adaptive/no-think, e.g. a direct <tool_call>).
           Reasoning is trivially ended, so return True -- otherwise the gate
           never opens and the <tool_call> XML leaks into `content` (the exact
           failure agentic clients like opencode hit on tool calls).
        2. <mm:think> present AND it is the FIRST token -> the model's own
           reasoning; ended only once a matching </mm:think> has appeared.
        3. <mm:think> present but NOT first -> this is the prompt, whose template
           embeds <mm:think>/</mm:think> in instructions. Return False so
           streaming reasoning extraction still runs on the generation.
        """
        if not input_ids:
            return False
        # Case 1: no reasoning span at all -> ended (tool parsing can run).
        if self.start_token_id not in input_ids:
            return True
        # Case 3: prompt (start token present but not leading) -> not ended.
        if input_ids[0] != self.start_token_id:
            return False
        # Case 2: the model's own reasoning -> ended once it is closed.
        return self.end_token_id in input_ids

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        Extract reasoning content from a delta message for streaming.

        Routes by M3's two verified modes, keyed off the leading <mm:think>
        (which is a single token, always the model's first output token in
        thinking mode and absent in disabled mode):

        - Disabled mode (current_text does not start with <mm:think>): the whole
          output is the answer, so every delta is content. This avoids the base
          class mislabeling a tag-less answer as reasoning_content.
        - Thinking mode: text up to </mm:think> is reasoning, text after is
          content. The leading <mm:think> in the first delta is stripped so the
          tag is not echoed to the client.
        """
        # Skip a lone start/end token delta first, regardless of mode
        # detection, so a lone <mm:think> is never misread as disabled-mode
        # content if its detokenized text is briefly empty.
        if len(delta_token_ids) == 1 and delta_token_ids[0] in (
                self.start_token_id, self.end_token_id):
            return None

        # Disabled / no-think output -> everything is content. Key off the start
        # TOKEN ID in the accumulated output (not the detokenized string) so
        # this is immune to incremental-detokenizer timing: in thinking mode
        # <mm:think> (a single token) is in current_token_ids from the first
        # delta; in disabled mode it never appears.
        if self.start_token_id not in current_token_ids:
            return DeltaMessage(content=delta_text if delta_text else None)

        # Strip a leading <mm:think> emitted at the very start of generation.
        if (self.start_token_id in delta_token_ids and not previous_text
                and self.start_token in delta_text):
            start_index = delta_text.find(self.start_token)
            delta_text = delta_text[start_index + len(self.start_token):]

        # Once the end token has been seen, everything after is content.
        if self.end_token_id in previous_token_ids:
            return DeltaMessage(content=delta_text)

        # End token arrives in this delta: split reasoning | content.
        if self.end_token_id in delta_token_ids:
            end_index = delta_text.find(self.end_token)
            reasoning_content = delta_text[:end_index]
            content = delta_text[end_index + len(self.end_token):]
            return DeltaMessage(
                reasoning_content=reasoning_content
                if reasoning_content else None,
                content=content if content else None,
            )

        # No end token yet: still inside the reasoning span.
        return DeltaMessage(
            reasoning_content=delta_text if delta_text else None)
