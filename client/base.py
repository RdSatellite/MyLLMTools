from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable

from session.message import Message

from pydantic import BaseModel


class Chat(ABC):
    @abstractmethod
    def chat(self, prompt: str) -> str:
        ...


class ToolCall(ABC):
    @abstractmethod
    def chat_with_tools(self, prompt: str, tools: list[Callable]):
        ...


class Structured(ABC):
    @abstractmethod
    def chat_with_structured(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        ...


@dataclass
class ConnectionConfig:
    api_key: str
    base_url: str
