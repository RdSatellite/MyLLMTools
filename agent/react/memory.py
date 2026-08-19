from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..session import History



@dataclass(frozen=True)
class Step:
    """A single (LLM call -> Tool result) step"""
    assistant_idx: int
    tool_result_idxs: Tuple[int, ...] = ()


@dataclass(frozen=True)
class Turn:
    """UserMessage -> Step -> ..."""
    user_idx: int
    steps: Tuple[Step, ...] = ()


@dataclass(frozen=True)
class Memory:
    """Turn -> Turn -> ..."""
    history: History
    system_idxs: Tuple[int, ...] = ()
    turns: Tuple[Turn, ...] = ()
