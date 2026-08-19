# gradient-alibi

Is alignment faking self-administered inoculation prompting?

Both describe the same event: a behavior demonstrated during training fails to
become a global weight update because some context *explains* it. The explanation
acts as an alibi. The two differ only in who supplies it — the developer
(inoculation prompting) or the model itself (alignment faking).

This repo runs the experiments that test whether that framing holds.

## Status

Phase 1: **the gate** — an inference-only screen deciding whether small open models
can produce genuine situational reasoning, or only parrot it back. Everything
downstream depends on the answer.

## Setup

```bash
uv sync
cp .env.example .env   # add OPENAI_API_KEY
```

See `AGENTS.md` for architecture, conventions and GPU rules.
