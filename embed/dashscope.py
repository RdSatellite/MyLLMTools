# dashscope.py 基于 dashscope api 提供 EmbeddingFunction 实现

from typing import Self, override

import requests

from .base import BaseEmbeddingFunction


# 默认配置
_DEFAULT_MODEL = "qwen3.7-text-embedding-flash"
_DEFAULT_DIM = 1024


# DashScopeEmbeddings 是最小可用的客户端（直接发 HTTP 请求）
class DashScopeEmbeddings:

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float = 30.0,
    ):
        self.model: str = model
        self.api_key: str = api_key
        self.base_url: str = base_url
        self.timeout: float = timeout

    def encode(self, texts: list[str]) -> list[list[float]]:
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
    """BaseEmbeddingFunction 的实现，用于编码 query"""

    def __init__(
        self,
        model: DashScopeEmbeddings,
        dim: int
    ):
        self.model: DashScopeEmbeddings = model
        self._dimension = dim

    @override
    def __call__(self, texts: list[str]) -> list[list[float]]:
        self._validate_input(texts)
        embeddings = self.model.encode(texts)
        self._validate_output(texts, embeddings)

        return embeddings

    def _validate_input(self, texts: list[str]) -> None:
        """
        确保 texts 确实是 list[str]：不能是单个 str，不能为空等。

        防止污染数据库（Python 会静默遍历 str 并生成无意义的 embedding）。
        """
        if type(texts) is not list:
            raise TypeError(
                "texts must be list[str]. "
                "Passing str directly is forbidden. "
            )

        if not texts:
            raise ValueError("texts is empty")

        for i, text in enumerate(texts):
            if type(text) is not str:
                raise TypeError(
                    f"texts[{i}] is not str."
                )

            if not text.strip():
                raise ValueError(
                    f"texts[{i}] is empty."
                )

    def _validate_output(self, texts: list[str], embeddings: list[list[float]]) -> None:
        """
        确保每条 text 都正确生成了 embedding。

        校验不通过的数据不应写入数据库。
        """
        if len(embeddings) != len(texts):
            raise RuntimeError(
                "Embedding result count mismatch."
            )

        for i, embed in enumerate(embeddings):
            if len(embed) != self.dim:
                raise RuntimeError(
                    f"Embedding dimension mismatch at index {i}: "
                    f"expected {self.dim}, got {len(embed)}"
                )

    @override
    def encode_queries(self, queries: list[str]) -> list[list[float]]:
        return self(queries)

    # NOTE: 文档是否需要额外的处理逻辑？尚未可知。
    # def encode_documents(self, docs: list[str]) -> list[list[float]]:
    #     ...

    @property
    def dim(self) -> int:
        return self._dimension

    @classmethod
    def default(cls, api_key: str, base_url: str, timeout: float = 30.0) -> Self:
        """以默认配置构造 DashScope Embedding Function"""
        return get_embed_func(
            _DEFAULT_MODEL, _DEFAULT_DIM,
            api_key, base_url,
            timeout
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
