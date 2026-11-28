# lib.py 定义需要对外暴露的主要公共函数

from dataclasses import dataclass
from typing import Any, Self

import json
from openai import OpenAI

from .entity import Function, Tools, ToolCall, ToolCallDecision
from .error import ToolCallError, ToolParamError, UnknownToolError
from .util import json_type_matches
from llm.chat import ChatSettings
from message import BaseMessage, MessagesBuilder


@dataclass
class ToolRunContext:
    """
    ToolRunContext 承载一次工具调用任务使用的运行上下文
    
    中间结果也会被保存在 ToolRunContext 中，例如，模型的回答会被写入到 history 中
    """

    history: list[BaseMessage]
    settings: ChatSettings

    @classmethod
    def default(cls, settings: ChatSettings) -> Self:
        """获取默认的 ToolRunContext"""
        return cls(history=[], settings=settings)


def index_tools(tools: Tools) -> dict[str, Function]:
    """按名称索引 `tools`"""

    by_name: dict[str, Function] = {}
    for tool in tools:
        if tool.name in by_name:
            raise ValueError(f"two tools are both named {tool.name!r}")
        by_name[tool.name] = tool

    return by_name


def resolve_tool(call: ToolCall, by_name: dict[str, Function]) -> Function:
    """
    `call` 所指名的工具——名称必须通过的校验，
    正如 `deserialize_params` 是参数必须通过的校验。
    """

    tool = by_name.get(call.name)
    if tool is None:
        raise UnknownToolError(call, "no such tool was offered")

    return tool


def decide_callings(
    client: OpenAI,
    tools: Tools,
    ctx: ToolRunContext,
) -> ToolCallDecision:
    """根据 `ctx.history`，询问模型想调用 `tools` 中的哪些工具。"""

    settings = ctx.settings
    messages = [
        m.to_dict()
        for m in MessagesBuilder.from_messages(ctx.history).build()
    ]

    response = client.chat.completions.create(
        model=settings.model,
        messages=messages,
        tools=[tool.to_dict() for tool in tools],
        stream=False,   # 忽略流式配置，以确保拿到完整回复而不是分片
        reasoning_effort=settings.reasoning_effort,
        extra_body=settings.extra_body,
    )

    return ToolCallDecision.from_provider(response)


def execute(
    tool: Function,
    call: ToolCall,
    ctx: ToolRunContext
) -> Any:
    """
    执行一个名称与参数均已通过校验的调用。

    同时会更新 ToolRunContext。
    """

    try:
        args = deserialize_params(call, tool)
        result = tool(args)
        content = tool.content_of(result)
    except ToolCallError as exc:
        content = f"Error: {exc}"
    except Exception as exc:
        content = f"Error: the tool raised {type(exc).__name__}: {exc}"

    ctx.history[:] = (
        MessagesBuilder.from_messages(ctx.history)
        .tool(call.id, content)
        .build()
    )

    return content


def deserialize_params(call: ToolCall, tool: Function) -> dict[str, Any]:
    """将工具调用的原始 JSON 参数转换为经过校验的 Python 参数。"""
    try:
        raw = json.loads(call.arguments)    # 实际上这里已基本完成转换。
    except json.JSONDecodeError as exc:
        raise ToolParamError(call, f"arguments are not valid JSON: {exc}") from exc

    # 防止返回 "[]" 形式的 JSON（正常应为 "{}"）。
    if not isinstance(raw, dict):
        raise ToolParamError(
            call, f"arguments must be a JSON object, got {type(raw).__name__}"
        )

    schema = tool.parameters.properties

    unknown = sorted(set(raw) - set(schema))
    if unknown:
        raise ToolParamError(call, f"unknown parameter(s) {unknown}")

    missing = sorted(set(tool.parameters.required) - set(raw))
    if missing:
        raise ToolParamError(call, f"missing required parameters(s) {missing}")

    for name, value in raw.items():
        expected = schema[name].type
        if not json_type_matches(expected, value):
            raise ToolParamError(
                call,
                f"parameter {name!r} expects {expected}, "
                f"got {type(value).__name__}: {value!r}",
            )

    return raw
