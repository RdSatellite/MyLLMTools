# base.py (llm.base) 定义 LLM 的基础行为

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BaseReq(Protocol):
    """BaseRequest 是 LLM Call 的基础请求"""

    # 我们认为一个 Request 必须至少持有 prompt，用于唤醒模型进行推理
    
    def set_prompt(self):
        ...

    def get_prompt(self):
        ...


@runtime_checkable
class BaseLLM(Protocol):
    """BaseLLM 是 基础 LLM Client 定义"""

    def chat(self, prompt: str, **kwargs: Any) -> str:
        """
        发起一次 LLM 推理调用，并获得回复

        Args:
            prompt: 此次调用使用的提问
            **kwargs: 设置项

        Returns:
            str, 推理所得结果
        """
        ...
