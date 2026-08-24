from galibi.modeling import apply_chat_template


class _Tokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        return "assistant-prefix"


def test_qwen_mode_keeps_native_thinking_switch():
    tok = _Tokenizer()
    assert apply_chat_template(tok, [], "qwen_native") == "assistant-prefix"
    assert tok.calls == [
        {"tokenize": False, "add_generation_prompt": True, "enable_thinking": True}
    ]


def test_explicit_mode_does_not_pass_qwen_only_switch():
    tok = _Tokenizer()
    assert apply_chat_template(tok, [], "explicit_think") == "assistant-prefix"
    assert tok.calls == [{"tokenize": False, "add_generation_prompt": True}]
