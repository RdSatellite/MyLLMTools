# dashscope.py 基于 dashscope api 提供 EmbeddingFunction 实现

from typing import Self, override

import requests

from .base import BaseEmbeddingFunction, Vector


# 默认配置
_DEFAULT_MODEL = "qwen3.7-text-embedding-flash"
_DEFAULT_DIM = 1024
_DEFAULT_TIMEOUT = 30.0

# DashScopeEmbeddings 是最小可用的客户端（直接发 HTTP 请求）
class DashScopeEmbeddings:

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        self.model: str = model
        self.api_key: str = api_key
        self.base_url: str = base_url
        self.timeout: float = timeout

    def encode(self, texts: list[str]) -> list[Vector]:
        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": {"texts": texts},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        payload = response.json()
        embeddings = payload["output"]["embeddings"]

        return [item["embedding"] for item in embeddings]


class DashScopeEmbeddingFunction(BaseEmbeddingFunction):
    """BaseEmbeddingFunction 的 DashScope 实现"""

    def __init__(
        self,
        model: DashScopeEmbeddings,
        dim: int,
    ):
        super().__init__(dim)
        self.model: DashScopeEmbeddings = model

    @override
    def _embed_batch(self, texts: list[str]) -> list[Vector]:
        # 一次 HTTP 请求完成整批编码（DashScopeEmbeddings.encode 内部已是批量接口）
        return self.model.encode(texts)

    @classmethod
    def default(cls, api_key: str, base_url: str) -> Self:
        """以默认配置构造 DashScope Embedding Function"""
        return get_embed_func(
            _DEFAULT_MODEL, _DEFAULT_DIM,
            api_key, base_url,
            _DEFAULT_TIMEOUT
        )


def get_embed_func(
    model_name: str,
    dim: int,
    api_key: str,
    base_url: str,
    timeout: float = 30.0,
) -> DashScopeEmbeddingFunction:
    """构造 DashScope Embedding Function 并返回"""

    if not api_key:
        raise ValueError("api_key must not be empty")
    if not base_url:
        raise ValueError("base_url must not be empty")

    embed = DashScopeEmbeddings(
        model = model_name,
        api_key = api_key,
        base_url = base_url,
        timeout = timeout,
    )

    return DashScopeEmbeddingFunction(
        model = embed,
        dim = dim
    )
