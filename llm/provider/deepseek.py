# deepseek.py (provider.deepseek) 提供了接入 deepseek 所需的构造方法

from openai import OpenAI
from ..chat.settings import ChatSettings


_BASE_URL = "https://api.deepseek.com"


def get_deepseek_client(api_key: str) -> OpenAI:
    """获得 OpenAI 形式的 DeepSeek Client"""

    return OpenAI(
        api_key = api_key,
        base_url = _BASE_URL
    )


def get_deepseek_default_settings() -> ChatSettings:
    """获得 DeepSeek Client 的默认配置"""
    
    return ChatSettings(
        model = "deepseek-flash",
        stream = False,
        reasoning_effort = "low",
        extra_body = {"thinking": {"type": "disabled"}}
    )
