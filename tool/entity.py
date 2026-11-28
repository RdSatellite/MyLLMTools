# entity.py (tool.entity) 定义工具调用相关实体类

import json
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Self

from .util import py_type_to_json_type, tool_name_of


# 目前仅支持 Function 类型的工具调用
type Tools = list["Function"]


class Serialize(ABC):
    """Serialize 约定了 Tool 下各实体的转换方法"""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """转换为 dict 形式，以供 LLM 调用"""
        ...

    def serialize(self) -> str:
        """转换为 str 形式，以供日志、持久化等使用"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


# --- Tool entities (for LLM tool-call usages) --- #


@dataclass
class Property(Serialize):
    """Property 是 Function Tool 的单条参数"""

    type: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "description": self.description,
        }


@dataclass
class Parameters(Serialize):
    """Parameters 表示 Function Tool 的所有参数"""

    properties: dict[str, Property]
    required: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {name: prop.to_dict() for name, prop in self.properties.items()},
            "required": list(self.required),
        }


@dataclass
class Function(Serialize):
    """Function 表示可被 LLM 调用的工具函数（Function Tool）"""

    name: str
    description: str
    parameters: Parameters

    fn: Callable


    def __call__(self, args: dict[str, Any]) -> Any:
        """使用 args 调用底层函数"""
        sig = inspect.signature(self.fn)
        bound = sig.bind(**args)
        bound.apply_defaults()

        return self.fn(*bound.args, **bound.kwargs)


    def content_of(self, result: Any) -> str:
        """
        把工具结果转换为 `role: "tool"` 消息所需的字符串。

        适用于所有工具的通用规则：str 原样返回（provider 所需的形态），
        其他类型序列化为 JSON 文本。若某个工具需要更复杂的处理，可重写此方法。
        """
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


    # --- Impl Serialize --- #

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters.to_dict(),
            },
        }

    @classmethod
    def from_callable(
        cls,
        fn: Callable,
        tool_description: str,
        param_descriptions: dict[str, str],
        tool_name: str | None = None
    ) -> Self:
        """
        从可调用对象构造函数工具。

        Args:
            fn: 原始 Python 可调用对象
            tool_description: 面向 LLM 的工具描述
            param_descriptions: {参数名: 参数描述}，所有参数都必须提供描述
            tool_name: 覆盖工具名；为 None 时使用限定名并将点替换为下划线
        """
        sig = inspect.signature(fn)
        tool_name = tool_name or tool_name_of(fn)

        properties: dict[str, Property] = {}
        required: list[str] = []

        for name, param in sig.parameters.items():
            if name not in param_descriptions:
                raise ValueError(f"Missing manual description for parameter `{name}`")

            if param.default is inspect.Parameter.empty:
                required.append(name)

            annot = param.annotation
            if annot is inspect.Parameter.empty:
                raise ValueError(f"Missing param type for parameter `{name}`")
            else:
                json_type = py_type_to_json_type(annot)

            properties[name] = Property(
                type=json_type,
                description=param_descriptions[name]
            )

        param_entity = Parameters(properties=properties, required=required)
        return cls(
            name=tool_name,
            description=tool_description,
            parameters=param_entity,
            fn=fn
        )


# --- Tool-call Results --- #


@dataclass
class ToolCall(Serialize):
    """ToolCall 表示模型决定调用的一个工具，直接来自 provider 响应"""

    id: str
    name: str
    arguments: str

    def to_dict(self) -> dict[str, Any]:
        """provider 要求的格式，用于在 assistant 消息中回传该调用。"""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            }
        }

    @classmethod
    def from_provider(cls, call: Any) -> Self:
        """从 provider 的工具调用对象构造。"""
        if call.type != "function":
            raise ValueError(f"Unsupported tool call type {call.type!r}")

        return cls(
            id=call.id,
            name=call.function.name,
            arguments=call.function.arguments,
        )


@dataclass
class ToolCallDecision:
    """ToolCallDecision 表示模型在一轮中做出的决定"""

    content: str
    finish_reason: str
    calls: list[ToolCall]   # provider 以扁平形式返回 [(id, name, arguments)]。

    @classmethod
    def from_provider(cls, response: Any) -> Self:
        """从 provider 的 chat-completion 响应构造。"""
        choice = response.choices[0]
        message = choice.message

        return cls(
            content=message.content or "",  # 某些 provider 会返回 None，这里视为空字符串
            finish_reason=choice.finish_reason,
            calls=[ToolCall.from_provider(call) for call in message.tool_calls or []]
        )

    @classmethod
    def empty(cls) -> Self:
        """返回一个空的调用决定。"""
        return cls(
            content = "Empty call, it is seem to be stopped by system.",
            finish_reason = "",
            calls = []
        )

    @property
    def wants_tools(self) -> bool:
        """模型是否决定发起工具调用"""
        return self.finish_reason == "tool_calls"
