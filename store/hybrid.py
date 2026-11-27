# hybrid.py 定义混合检索（HybridSearch）相关特性。

from typing import Any, Protocol, runtime_checkable

from .text import HasText
from .vector import Vector, VectorData


# --- Data --- #


@runtime_checkable
class HybridData[T](VectorData[T], HasText, Protocol):
    """HybridData 即支持混合检索的数据。同时能够以文本和嵌入向量描述。"""
    ...


# --- Store --- #


@runtime_checkable
class HybridStore(Protocol):
    """HybridStore 表示后端支持混合检索"""

    def get_by_hybrid[T: HybridData](
        self,
        collection_name: str,
        query_text: str,
        query_vector: Vector,
        data_cls: type[T],
        limit: int = 10,
        reranker: Any = None,
    ) -> list[tuple[T, float]]:
        """
        获取同时按文本与向量召回、融合后最接近的 `limit` 行。

        结果按分数降序排列，索引 0 最接近。

        `data_cls` 必须实现 `HybridData`，且 collection 必须已包含
        `data_cls.text_column()` 指定的列。`reranker` 由后端决定；
        传 None 时由后端选用默认实现。
        """
        ...
