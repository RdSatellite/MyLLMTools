# vector.py 定义 vectordb 的向量特性

from typing import Any, Protocol, runtime_checkable

from .base import BaseData


# 类型别名
Vector = list[float]


# --- Data --- #


@runtime_checkable
class HasVector(Protocol):
    """HasVector 表示数据可以被嵌入向量表示"""

    @classmethod
    def get_vector(cls, payload: Any) -> Vector:
        """获取嵌入向量"""
        raise NotImplementedError


@runtime_checkable
class VectorData[T](BaseData[T], HasVector, Protocol):
    ...


# --- Store --- #


@runtime_checkable
class VectorStore(Protocol):
    """VectorStore 的存储契约"""

    def get_by_sim[T: VectorData](
        self,
        collection_name: str,
        query_vector: Vector,
        data_cls: type[T],
        limit: int = 10
    ) -> list[tuple[T, float]]:
        """
        获取与 `query_vector` 最相似的 `limit` 条记录。

        返回 `(数据, 相似度)` 列表，应认为分数越高越相似。
        """
        ...


def require_vector_data(data_cls: type) -> None:
    """检查数据类是否实现了 VectorData"""
    if not issubclass(data_cls, VectorData):
        raise TypeError(
            f"data_cls must implement VectorData, got {data_cls.__name__}"
        )
