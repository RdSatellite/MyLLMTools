# settings.py (chat.settings) 给出客户端配置类定义

from typing import Any, Self
from dataclasses import dataclass


@dataclass
class ChatSettings:
    model: str
    stream: bool
    reasoning_effort: str
    extra_body: dict[str, Any]
