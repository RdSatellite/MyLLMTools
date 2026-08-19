
from typing import Callable, List

from llm.base import AgentLLM
from ..session import History


class Context:
    """React loop context"""
    llm: AgentLLM
    tools: List[Callable]

    history: History
    work_memory: History


def react_loop(ctx: Context) -> str:
