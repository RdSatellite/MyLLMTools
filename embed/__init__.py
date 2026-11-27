# package embed 定义嵌入 (embedding) 相关的工具和类

from .base import BaseEmbeddingFunction, Vector

__all__ = [
    "BaseEmbeddingFunction",
    "Vector",
]

# 需要引用供应商实现时，采用以下代码：
#
#   from embed.dashscope import DashScopeEmbeddingFunction
#   from embed.openai import OpenAIEmbeddingFunction
