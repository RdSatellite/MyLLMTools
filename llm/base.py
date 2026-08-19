from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class Chat(Protocol):
    def chat(self, messages: list[dict]) -> str:
        ...


@runtime_checkable
class ToolCall(Protocol):
    def chat_with_tools(self, messages: list[dict], tools: list[Callable]):
        ...


@runtime_checkable
class Structured(Protocol):
    def chat_with_structured(self, messages: list[dict], schema: type[BaseModel]) -> BaseModel:
        ...


@runtime_checkable
class AgentLLM(Chat, ToolCall, Structured, Protocol):
    ...


@dataclass
class ConnectionConfig:
    api_key: str
    base_url: str
