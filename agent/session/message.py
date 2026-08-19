# session/message.py
# Domain message model

from dataclasses import dataclass
from typing import Dict, Literal


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str

    @property
    def dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class SystemMessage(Message):
    def __init__(self, content: str):
        super().__init__("system", content)


@dataclass
class UserMessage(Message):
    def __init__(self, content: str):
        super().__init__("user", content)


@dataclass
class AssistantMessage(Message):
    def __init__(self, content: str):
        super().__init__("assistant", content)


@dataclass
class ToolMessage(Message):
    def __init__(self, content: str):
        super().__init__("tool", content)
