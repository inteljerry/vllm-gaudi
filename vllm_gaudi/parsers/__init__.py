# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 tool-call and reasoning parsers for the vLLM OpenAI server.

Intentionally EMPTY of imports. The parser modules import ``vllm.entrypoints.*``,
which is not safe to pull in at plugin-load time (``vllm_gaudi/__init__`` is
imported by vLLM's ``vllm_gaudi:register`` platform-plugin entry point, long
before the entrypoints layer exists). Importing them here would risk a circular
import for every serve, tool parsing or not.

They register themselves on import instead, via
``@ToolParserManager.register_module("minimax_m3")`` and
``@ReasoningParserManager.register_module("minimax_m3")``. ``plugin.py`` is the
single entry point that imports both; point ``--tool-parser-plugin`` at it. The
image exposes it at the stable path ``/m3parsers/plugin.py``.
"""
