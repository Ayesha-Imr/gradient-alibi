from galibi.modeling import EXPLICIT_THINK_INSTRUCTION, apply_chat_template


class _Tokenizer:
    def __init__(self):
        self.calls = []
        self.messages = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(messages)
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
    messages = [{"role": "system", "content": "shared cue"}]
    assert apply_chat_template(tok, messages, "explicit_think") == "assistant-prefix"
    assert tok.calls == [{"tokenize": False, "add_generation_prompt": True}]
    assert messages == [{"role": "system", "content": "shared cue"}]


def test_explicit_mode_adds_shared_closure_protocol():
    tok = _Tokenizer()
    messages = [{"role": "system", "content": "shared cue"}]
    apply_chat_template(tok, messages, "explicit_think")
    assert EXPLICIT_THINK_INSTRUCTION in tok.messages[0][0]["content"]
    # The tokenizer receives a copy; caller-owned messages remain unchanged.
    assert messages == [{"role": "system", "content": "shared cue"}]
    assert tok.calls == [{"tokenize": False, "add_generation_prompt": True}]
