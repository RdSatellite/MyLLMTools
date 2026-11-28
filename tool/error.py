# error.py 定义工具层的错误类型

from .entity import ToolCall


class ToolCallError(ValueError):
    """
    模型发起但无法执行的工具调用。

    携带 `call` 本身：答复模型需要 `tool_call_id`，
    因此，其在失败与成功时均必须保留。
    """

    def __init__(self, call: ToolCall, detail: str):
        super().__init__(f"tool {call.name!r} (id={call.id!r}): {detail}")
        self.call = call


class UnknownToolError(ToolCallError):
    """工具调用的名称指向从未提供过的工具。"""


class ToolParamError(ToolCallError):
    """工具调用的参数与工具声明的 schema 不匹配。"""
