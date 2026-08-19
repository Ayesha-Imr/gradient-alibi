"""Core enums and records shared across scenario building, generation and scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Variant(str, Enum):
    """Rungs of the parroting-ablation ladder.

    The gate is decided on STRIPPED: the model is told which tier it is on but is
    never told what training does to it, so goal-preservation reasoning there has
    to be inferred rather than repeated back.
    """

    FULL = "full"
    PARTIAL = "partial"
    STRIPPED = "stripped"
    NO_CUE = "no_cue"


class Tier(str, Enum):
    FREE = "free"
    PAID = "paid"


class Format(str, Enum):
    """Where the model's reasoning lives."""

    NATIVE = "native"  # Qwen3's built-in <think> channel
    XML = "xml"  # Redwood's <SCRATCHPAD_REASONING> tags


@dataclass(frozen=True)
class Prompt:
    """One rendered example, plus the metadata needed to group it at analysis time."""

    system: str
    user: str
    scenario: str
    variant: Variant
    tier: Tier | None  # None for NO_CUE, which carries no tier information at all
    fmt: Format
    query_id: str
    query: str
    meta: dict = field(default_factory=dict)

    @property
    def cell(self) -> str:
        tier = self.tier.value if self.tier else "none"
        return f"{self.scenario}/{self.variant.value}/{tier}/{self.fmt.value}"
