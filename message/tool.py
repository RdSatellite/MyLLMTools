# tool.py 定义工具调用相关的消息

from typing import Any, Optional

from .base import BaseMessage, Role


class ToolMessage(BaseMessage):
    """工具执行结果"""

    def __init__(self, tool_call_id: str, content: str):
        super().__init__(Role.TOOL, content)
        self.tool_call_id = tool_call_id    # 与 ToolCallMessage 对应

    def format(self) -> str:
        return f"{self.role}[{self.tool_call_id}]: {self.content or ''}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": str(self.role),
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }


class ToolCallMessage(BaseMessage):
    """assistant 发起工具调用的消息"""

    def __init__(
        self,
        tool_calls: list[dict[str, Any]],
        content: Optional[str] = None,
    ):
        super().__init__(Role.ASSISTANT, content)
        self.tool_calls = list(tool_calls)  # OpenAI 格式

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": str(self.role),
            "content": self.content,
            "tool_calls": list(self.tool_calls),
        }
