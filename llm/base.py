# base.py (llm.base) 定义 LLM 的基础行为

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BaseLLM(Protocol):

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
