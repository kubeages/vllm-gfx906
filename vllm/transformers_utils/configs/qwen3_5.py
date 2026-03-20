# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom config for Qwen3.5 that strips M-RoPE from the text config.

Qwen3.5 is a multimodal model whose config carries ``mrope_section`` in
``text_config.rope_parameters``.  When the model is loaded as text-only
(``Qwen3_5ForCausalLM``), the M-RoPE flag causes vLLM to generate 3-D
positions that the text model cannot consume.  Removing ``mrope_section``
and ``mrope_interleaved`` from the text config makes ``uses_mrope``
return ``False`` so positions stay 1-D.
"""

from transformers import AutoConfig, PretrainedConfig


class Qwen3_5Config(PretrainedConfig):
    model_type = "qwen3_5"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Resolve text_config (may come as a dict from JSON)
        text_config = getattr(self, "text_config", None)
        if isinstance(text_config, dict):
            text_config = PretrainedConfig(**text_config)
            self.text_config = text_config

        if text_config is not None:
            rope = getattr(text_config, "rope_parameters", None)
            if isinstance(rope, dict):
                rope.pop("mrope_section", None)
                rope.pop("mrope_interleaved", None)
