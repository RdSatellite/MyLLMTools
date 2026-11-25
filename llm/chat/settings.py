# settings.py (chat.settings) 给出客户端配置类定义

from typing import Dict, Self
from dataclasses import dataclass


@dataclass
class ChatSettings:
    model: str
    stream: bool
    reasoning_effort: str
    extra_body: Dict

    @classmethod
    def default_deepseek(cls) -> Self:
        return cls(
            model = "deepseek-flash",
            stream = False,
            reasoning_effort = "low",
            extra_body = {"thinking": {"type": "disabled"}}
        )
