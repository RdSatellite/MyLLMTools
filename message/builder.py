# builder.py 定义用于构造 BaseMessage 和 list[BaseMessage] 的 builder

from typing import Any, Optional, Self

from .base import AssistantMessage, BaseMessage, SystemMessage, UserMessage
from .tool import ToolCallMessage, ToolMessage


# MessagesBuilder 即 Message 的 builder 构造器
class MessagesBuilder:

    def __init__(self, msgs: list[BaseMessage]):
        """以已有消息作为初始内容，后续方法追加的消息会在 build() 时接在其后"""
        
        self.initial: list[BaseMessage] = msgs      # 构造时的最初状态
        self.buff: list[BaseMessage] = []           # buffer，追加后等待合入的消息


    # --- Base --- #


    def system(self, msg: str) -> Self:
        """追加一条 system 消息"""
        self.buff.append(SystemMessage(msg))

        return self

    def user(self, msg: str) -> Self:
        """追加一条 user 消息"""
        self.buff.append(UserMessage(msg))

        return self

    def ai(self, msg: str) -> Self:
        """追加一条 assistant 消息"""
        self.buff.append(AssistantMessage(msg))

        return self


    # --- Tool --- #


    def ai_calls(self, tool_calls: list[dict[str, Any]], content: Optional[str] = None) -> Self:
        """追加一条 assistant 消息，请求调用工具（OpenAI 格式的 tool_calls）"""
        self.buff.append(ToolCallMessage(tool_calls, content))

        return self

    def tool(self, tool_call_id: str, content: str) -> Self:
        """追加一条工具结果消息，按 tool_call_id 应答对应的 assistant 调用"""
        self.buff.append(ToolMessage(tool_call_id, content))

        return self


    def build(self) -> list[BaseMessage]:
        """结束追加，获得完整的消息列表"""
        return self.initial + self.buff


    @classmethod
    def from_messages(cls, msgs: list[BaseMessage]) -> Self:
        """以已有消息构造 builder"""
        return cls(msgs)

    @classmethod
    def from_empty(cls) -> Self:
        """以空列表构造 builder"""
        return cls([])
