# trimmer.py 定义了消息的裁剪组件

import json
from typing import Self

import tiktoken

from .base import BaseMessage, Role
from .tool import ToolCallMessage


_ENCODING_NAME = "cl100k_base"
_ENCODING = tiktoken.get_encoding(_ENCODING_NAME)

DEFAULT_CAP_TURNS = 20
DEFAULT_CAP_TOKENS = 1_000_000  # 1M 上下文


# MessageTrimmer 是 list[BaseMessage] 的裁剪组件，用于维持上下文窗口
class MessageTrimmer:

    def __init__(
        self,
        cap_turns: int,
        cap_tokens: int
    ) -> None:
        """
        Args:
            cap_turns: 最大允许的 turn 数，以 user 发出消息并获得完整回复计算一次 turn
            cap_tokens: 最大允许的 token 数
        """
        self.set_cap_turns(cap_turns)
        self.set_cap_tokens(cap_tokens)

    @property
    def cap_turns(self) -> int:
        return self._cap_turns

    @property
    def cap_tokens(self) -> int:
        return self._cap_tokens

    def set_cap_turns(self, turns: int) -> None:
        if turns <= 0:
            raise ValueError(f"cap turns must be positive, got {turns}")
        self._cap_turns = turns

    def set_cap_tokens(self, tokens: int) -> None:
        if tokens <= 0:
            raise ValueError(f"cap tokens must be positive, got {tokens}")
        self._cap_tokens = tokens

    def trim(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """执行裁剪"""
        kept = self._drop_turns(messages, self._cap_turns)

        while _turns(kept) > 1 and _tokens(kept) > self._cap_tokens:
            kept = self._drop_turns(kept, _turns(kept) - 1)

        return kept

    @staticmethod
    def _drop_turns(messages: list[BaseMessage], keep: int) -> list[BaseMessage]:
        """
        根据设定的 `cap_turns` 执行 turns 裁剪，较早的 turns 会被淘汰

        裁剪位置之前的 `SystemMessage` 会被保留下来，随结果一起返回
        """
        starts = _turn_starts(messages)
        if len(starts) <= keep:
            return messages

        cut = starts[len(starts) - keep]
        head = [m for m in messages[:cut] if m.role == Role.SYSTEM]
        return head + messages[cut:]

    @classmethod
    def default(cls) -> Self:
        """以默认值构造 Trimmer"""
        return cls(
            DEFAULT_CAP_TURNS,
            DEFAULT_CAP_TOKENS
        )



def _turn_starts(messages: list[BaseMessage]) -> list[int]:
    """返回每个 turn 的起始下标：turn 以一条 user 消息开始，包含它引发的后续消息"""
    return [i for i, m in enumerate(messages) if m.role == Role.USER]

def _turns(messages: list[BaseMessage]) -> int:
    """消息列表包含的 turn 数量"""
    return len(_turn_starts(messages))

def _tokens(messages: list[BaseMessage]) -> int:
    """消息列表按 cl100k_base 计算的总 token 数"""
    return sum(_count(message_text(m)) for m in messages)

def message_text(m: BaseMessage) -> str:
    """消息中会发送给模型、因而计入 token 的文本"""
    if isinstance(m, ToolCallMessage):
        calls = json.dumps(m.tool_calls, ensure_ascii=False)
        return "\n".join(part for part in (m.content, calls) if part)

    return m.content or ""

def _count(text: str) -> int:
    """按 cl100k_base 计算 `text` 的 token 数"""
    return len(_ENCODING.encode(text, disallowed_special=()))
