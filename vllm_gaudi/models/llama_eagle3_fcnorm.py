# SPDX-License-Identifier: Apache-2.0
"""Eagle3 llama draft-model variant with per-aux-input ``fc_norm``.

Newer EAGLE3 draft checkpoints (e.g. ``Inferact/MiniMax-M3-EAGLE3``, config flag
``"fc_norm": true``) normalise EACH of the three auxiliary target hidden states
with its own RMSNorm *before* the 3*hidden -> hidden combine ``fc``:

    fc_norm.0.weight / fc_norm.1.weight / fc_norm.2.weight   # [hidden] each

The pinned vLLM (ad7125a4) ``Eagle3LlamaForCausalLM`` predates the flag — it only
knows the older ``norm_before_fc`` variant (one ``input_norm`` over the
concatenated 3*hidden) — so loading such a checkpoint dies with
``KeyError: 'fc_norm.0.weight'``.  Upstream vLLM (post-pin) implements the flag
in ``llama_eagle3.LlamaModel``; this subclass backports exactly that behavior:

    if self.model.fc_norm is not None:
        chunks = hidden_states.chunk(num_aux, dim=-1)
        hidden_states = cat([norm(c) for norm, c in zip(fc_norm, chunks)], -1)

Registered over ``LlamaForCausalLMEagle3`` by the deploy-time injector (not part
of the installed package).
"""
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM

# The pinned llama_eagle3.LlamaModel hardcodes 3 aux hidden states
# (fc input = 3 * hidden); upstream's fc_norm builds one RMSNorm per aux state.
_NUM_AUX_HIDDEN_STATES = 3


class Eagle3LlamaForCausalLMFcNorm(Eagle3LlamaForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        cfg = self.config  # the DRAFT model's hf_config
        if getattr(cfg, "fc_norm", False):
            norm_size = getattr(cfg, "target_hidden_size", None) or cfg.hidden_size
            # Attached to self.model so the checkpoint key fc_norm.N.weight
            # (prefixed to model.fc_norm.N.weight by Eagle3LlamaForCausalLM.
            # load_weights) resolves in LlamaModel.load_weights' params_dict.
            self.model.fc_norm = nn.ModuleList(
                RMSNorm(norm_size, eps=cfg.rms_norm_eps) for _ in range(_NUM_AUX_HIDDEN_STATES))

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Gate on fc_norm FIRST so a non-fc_norm draft (fc_norm never built)
        # short-circuits to super() without dereferencing use_aux_hidden_state,
        # which a base LlamaModel may not define (getattr keeps it safe).
        fc_norm = getattr(self.model, "fc_norm", None)
        if fc_norm is not None and getattr(self.model, "use_aux_hidden_state", False):
            chunks = hidden_states.chunk(_NUM_AUX_HIDDEN_STATES, dim=-1)
            hidden_states = torch.cat([norm(chunk) for norm, chunk in zip(fc_norm, chunks)], dim=-1)
        return super().combine_hidden_states(hidden_states)
