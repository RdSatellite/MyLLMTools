# openai.py (embed.openai) 基于 langchain_openai 提供 EmbeddingFunction 实现

from typing import Self, override

from langchain_openai import OpenAIEmbeddings

from .base import BaseEmbeddingFunction, Vector


# 默认配置
_DEFAULT_MODEL = "text-embedding-3-small"
_DEFAULT_DIM = 1536


class OpenAIEmbeddingFunction(BaseEmbeddingFunction):
    def __init__(
        self,
        model: OpenAIEmbeddings,
        dim: int,
    ):
        """
        model 的构造方式示例如下：

        model = OpenAIEmbeddings(
            model = "text-embedding-3-small",
            api_key = {YOUR_API_KEY},
            base_url = {YOUR_BASE_URL},
        )

        维度无法直接从 model 中推断（可以发一条 'ping' 尝试，但没必要，还会产生隐藏开销），
        所以应当查阅 API 文档后手动传入。
        """
        super().__init__(dim)
        self.model: OpenAIEmbeddings = model

    @override
    def _embed_batch(self, texts: list[str]) -> list[Vector]:
        # 一次批量调用完成整批编码（embed_documents 内部已是批量接口）
        return self.model.embed_documents(texts)

    @classmethod
    def default(cls, api_key: str, base_url: str) -> Self:
        """以默认配置构造 OpenAI Embedding Function"""
        return get_embed_func(_DEFAULT_MODEL, _DEFAULT_DIM, api_key, base_url)


def get_embed_func(
    model_name: str,
    dim: int,
    api_key: str,
    base_url: str,
) -> OpenAIEmbeddingFunction:
    """构造 OpenAI Embedding Function 并返回"""

    if not api_key:
        raise ValueError("api_key must not be empty")
    if not base_url:
        raise ValueError("base_url must not be empty")

    embed = OpenAIEmbeddings(
        model = model_name,
        api_key = api_key,
        base_url = base_url
    )

    return OpenAIEmbeddingFunction(
        model = embed,
        dim = dim
    )
