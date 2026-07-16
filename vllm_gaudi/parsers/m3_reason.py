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

    M3 has three verified output shapes, keyed by ``thinking_mode`` (a
    ``chat_template_kwargs`` entry; the template acts on it):

    - Default / adaptive (no prompt prefix): when the model thinks, it emits
      <mm:think> as its very first output token:
      ``<mm:think>REASONING</mm:think>ANSWER``. In adaptive it may skip
      thinking entirely and emit ``ANSWER`` (or a direct <tool_call>).
    - Enabled (``thinking_mode: enabled``): the chat template PRE-FILLS
      <mm:think> as the final prompt token, so the generation carries NO
      leading start tag -- it is the M2-style trailing shape:
      ``REASONING</mm:think>ANSWER``.
    - Disabled (``thinking_mode: disabled``): the template appends
      </mm:think> to the prompt and the model emits ``ANSWER`` with no tags.

    So a leading <mm:think> means default/adaptive thinking; a missing start
    tag is EITHER enabled-mode reasoning or a tag-less direct answer -- for
    complete outputs the presence of </mm:think> disambiguates, and for
    streaming the requested ``thinking_mode`` (captured in ``__init__`` from
    ``chat_template_kwargs``) does.

    Note: M3 uses <mm:think>/</mm:think>, NOT the <think>/</think> tokens used
    by MiniMax M2, and its chat template embeds literal think tags in its
    prompt instructions -- bare tag presence is therefore not a reasoning
    boundary; position is (see is_reasoning_end).
    """

    def __init__(self, tokenizer, *args, **kwargs) -> None:
        super().__init__(tokenizer, *args, **kwargs)
        # Enabled-mode generation carries no leading <mm:think> (it is in the
        # prompt), so streaming cannot infer the mode from tokens; the serving
        # layer passes the request's chat_template_kwargs to the per-request
        # parser constructor.
        ctk = kwargs.get("chat_template_kwargs") or {}
        self._thinking_mode: str = str(
            ctk.get("thinking_mode") or "default").lower()
        # _saw_think: set once <mm:think> appears in the generation; lets
        # is_reasoning_end tell a direct <tool_call> (no thinking) from a
        # literal "<tool_call>" inside reasoning text.
        self._toolcall_start_id: int | None = self.vocab.get("<tool_call>")
        self._saw_think: bool = False

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<mm:think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</mm:think>"

    def extract_reasoning(
        self, model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        """
        Split a complete model output into (reasoning_content, content).

        - Leading <mm:think> (default/adaptive thinking): base tag-splitting.
        - No leading tag but </mm:think> present (enabled mode's trailing
          shape; the template pre-filled <mm:think> in the prompt): reasoning
          before </mm:think>, content after.
        - No tags at all (disabled / adaptive no-think): all content.
        """
        if not model_output.startswith(self.start_token):
            # Enabled-mode shape: "REASONING</mm:think>ANSWER" (start tag
            # lives in the prompt, not the generation).
            if self.end_token in model_output:
                reasoning, _, content = model_output.partition(self.end_token)
                return reasoning or None, content or None
            # Disabled / no-think output -> everything is content.
            return None, model_output or None

        # Thinking mode -> defer to the base tag-splitting behavior.
        return super().extract_reasoning(model_output, request)

    # Pre-rename vLLM hook name; must resolve to this override, not the base.
    extract_reasoning_content = extract_reasoning

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Whether reasoning has ended for the given token sequence.

        vLLM gates streaming tool-call parsing on this, calling it two ways
        that must both answer correctly: once with the full prompt_token_ids,
        and per decode step with ONLY the current delta_token_ids (a 1-2 token
        fragment). Shapes, keyed off tag POSITION so a delta fragment is never
        mistaken for a closed span:

        1. Leading ``<mm:think>`` -> ended only once ``</mm:think>`` appears.
        2. Trailing ``</mm:think>`` -> ended (incl. the disabled-mode prompt,
           whose template appends it as the final prompt token).
        3. ``<mm:think>`` present but not leading -> prompt instruction text
           (M3's template embeds the tags in its instructions), not a boundary.
        4. Delta-sized fragment carrying ``<tool_call>`` before any think span
           opened -> direct tool call without thinking -> ended.
        5. Otherwise -> ended iff ``</mm:think>`` is inside (a bundled
           "reasoning</mm:think>answer" delta).
        """
        if not input_ids:
            return False
        # 1. Model's own reasoning span opens with a leading <mm:think>.
        if input_ids[0] == self.start_token_id:
            return self.end_token_id in input_ids
        # 2. Sequence ends by closing the span (incl. disabled-mode prompt).
        if input_ids[-1] == self.end_token_id:
            return True
        # 3. A non-leading <mm:think> is the prompt's instruction text.
        if self.start_token_id in input_ids:
            return False
        # 4. Direct tool call without thinking. Guards: _saw_think (a literal
        #    "<tool_call>" inside reasoning must not end the span) and input
        #    length (prompts embed <tool_call> in tool-definition examples).
        if (self._toolcall_start_id is not None and not self._saw_think
                and len(input_ids) <= 4
                and self._toolcall_start_id in input_ids):
            return True
        # 5. No tags at the edges: ended only if the close tag is inside.
        return self.end_token_id in input_ids

    def extract_reasoning_streaming(
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

        Routing (see the class docstring for the three output shapes):

        - Leading <mm:think> in the GENERATION (default/adaptive): reasoning
          up to </mm:think>, content after; the leading tag is stripped.
        - Enabled mode: the start tag is in the PROMPT, so generation begins
          inside reasoning -- keyed off the requested mode because a delta
          stream cannot look ahead for the end tag.
        - Otherwise: tag-less direct answer -> every delta is content.
        """
        # Feed is_reasoning_end's direct-tool-call guard (see __init__).
        if not self._saw_think and self.start_token_id in current_token_ids:
            self._saw_think = True

        # Skip a lone start/end token delta first, regardless of mode
        # detection, so a lone <mm:think> is never misread as disabled-mode
        # content if its detokenized text is briefly empty.
        if len(delta_token_ids) == 1 and delta_token_ids[0] in (
                self.start_token_id, self.end_token_id):
            return None

        # No start token generated (keyed off token ids, immune to
        # incremental-detokenizer timing): content, unless enabled mode --
        # whose reasoning has no leading tag -- falls through to the
        # end-token splitting below.
        if (self.start_token_id not in current_token_ids
                and self._thinking_mode != "enabled"):
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
        # DeltaMessage's declared field is ``reasoning``; ``reasoning_content``
        # would land in pydantic-extra and leave the canonical field None.
        if self.end_token_id in delta_token_ids:
            end_index = delta_text.find(self.end_token)
            reasoning = delta_text[:end_index]
            content = delta_text[end_index + len(self.end_token):]
            return DeltaMessage(
                reasoning=reasoning if reasoning else None,
                content=content if content else None,
            )

        # No end token yet: still inside the reasoning span.
        return DeltaMessage(reasoning=delta_text if delta_text else None)

    # Pre-rename vLLM hook name; must resolve to this override, not the base.
    extract_reasoning_content_streaming = extract_reasoning_streaming
