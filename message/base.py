# base.py 给出消息的基本定义

from abc import ABC
from enum import StrEnum
from typing import Any, Optional


# Role 即消息的来源
class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# BaseMessage 即消息的基础定义
class BaseMessage(ABC):

    def __init__(self, role: Role, content: Optional[str] = None):
        self.role = role
        self.content = content

    def format(self) -> str:
        """输出为可读形式"""
        return f"{self.role}: {self.content or ''}"

    def to_dict(self) -> dict[str, Any]:
        """转换为字典形式，用于 api"""
        return {"role": str(self.role), "content": self.content}


# --- 基础消息类型 --- #


class SystemMessage(BaseMessage):
    def __init__(self, content: str):
        super().__init__(Role.SYSTEM, content)


class UserMessage(BaseMessage):
    def __init__(self, content: str):
        super().__init__(Role.USER, content)


class AssistantMessage(BaseMessage):
    def __init__(self, content: str):
        super().__init__(Role.ASSISTANT, content)
