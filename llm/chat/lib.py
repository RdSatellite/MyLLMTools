# lib.py 定义对外开放的集成 api 和相关工具

from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from .settings import ChatSettings
from message import MessagesBuilder


_DEFAULT_SYSTEM = "You are a helpful assistant."


def chat(client: OpenAI, prompt: str, settings: ChatSettings) -> Optional[str]:
    """
    发送 prompt 并获得一次回复
    
    Args:
        client: OpenAI 形式的 Client
        prompt: 用户的问题输入
        settings: ChatSettings, 会话设置项

    Returns:
        str, LLM 的回复
    """

    messages = MessagesBuilder.from_empty() \
        .system(_DEFAULT_SYSTEM) \
        .user(prompt) \
        .build()

    messages_dict = [m.to_dict() for m in messages]

    # TODO: 补充流式支持
    response = client.chat.completions.create(
        model=settings.model,
        messages=messages_dict,
        stream=settings.stream,
        reasoning_effort=settings.reasoning_effort,
        extra_body=settings.extra_body
    )

    return response.choices[0].message.content
