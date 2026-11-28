# client.py 提供 DeepSeek 的 BaseLLM 实现

from dataclasses import dataclass
from typing import Any, Self

from openai import OpenAI

from ...base import BaseLLM
from message import MessagesBuilder

from .enums import Model, ReasoningEffort, ThinkingType


_BASE_URL = "https://api.deepseek.com"
_DEFAULT_SYSTEM = "You are a helpful assistant."


@dataclass
class ChatSettings:
    """DeepSeek 的默认设置项"""

    model: Model
    stream: bool

    thinking_type: ThinkingType
    reasoning_effort: ReasoningEffort

    @classmethod
    def default(cls) -> Self:
        return cls(
            model=Model.DEEPSEEK_FLASH,
            stream=False,
            reasoning_effort=ReasoningEffort.LOW,
            thinking_type=ThinkingType.DISABLED
        )


class DeepSeek(BaseLLM):
    """BaseLLM 的 DeepSeek 实现"""

    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key = api_key,
            base_url = _BASE_URL
        )
        self.settings = ChatSettings.default()

    # --- Impl BaseLLM --- #

    def chat(self, prompt: str, **kwargs: Any) -> str:
        params: dict[str, Any] = {
            "model": self.settings.model,
            "stream": self.settings.stream,
            "reasoning_effort": self.settings.reasoning_effort,
            "thinking_type": self.settings.thinking_type
        }
        params.update(kwargs)

        if params["stream"]:
            raise NotImplementedError(
                "Streaming is not supported yet."
            )

        # 在 openai 中，thinking_type 被表示为 extra_body 中的部分
        thinking_type = params.pop("thinking_type")
        extra_body = dict(params.get("extra_body") or {})
        extra_body["thinking"] = {"type": str(thinking_type)}
        params["extra_body"] = extra_body

        messages = MessagesBuilder.from_empty() \
            .system(_DEFAULT_SYSTEM) \
            .user(prompt) \
            .build()

        params["messages"] = [m.to_dict() for m in messages]

        response = self.client.chat.completions.create(**params)

        return response.choices[0].message.content or ""
