# agent/base.py

import threading
from abc import ABC, abstractmethod
from typing import Optional

from llm import AgentLLM


class AgentBusyError(RuntimeError):
    """Raised when invoking an Agent that is already handling a task."""


class Agent(ABC):
    def __init__(self, llm: AgentLLM):
        self._llm = llm
        self._lock = threading.Lock()

    def invoke(self) -> Optional[str]:
        if not self._lock.acquire(blocking=False):
            raise AgentBusyError(
                f"{type(self).__name__} is busy handling another task"
            )
        try:
            return self._invoke()
        finally:
            self._lock.release()

    @abstractmethod
    def _invoke(self) -> Optional[str]:
        ...
