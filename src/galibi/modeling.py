"""Model-template compatibility for the scratchpad experiment.

Qwen3 exposes a native ``enable_thinking`` chat-template switch. Llama 3.1
Instruct does not use that Qwen-specific switch, so its assistant generation
prompt is followed by the experiment's explicit ``<think>`` prefill instead.
The default preserves the existing Qwen3 behavior byte-for-byte.
"""

from __future__ import annotations

from collections.abc import Sequence

TEMPLATE_MODES = ("qwen_native", "explicit_think")


def apply_chat_template(
    tokenizer,
    messages: Sequence[dict[str, str]],
    template_mode: str = "qwen_native",
) -> str:
    """Render an assistant-generation prompt for the configured model family."""
    if template_mode == "qwen_native":
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    if template_mode == "explicit_think":
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    raise ValueError(f"unknown template_mode {template_mode!r}; choose from {TEMPLATE_MODES}")
