# kv.py 定义 vectordb 的键值（key-value）特性。

from typing import Protocol, runtime_checkable

from .base import BaseData


# --- Store --- #


@runtime_checkable
class KvStore(Protocol):
    """KvStore 表示后端支持键值存储"""

    def set(self, collection_name: str, data: BaseData) -> int:
        """写入并返回其 id。"""
        return self.batch_set(collection_name, [data])[0]

    def batch_set(
        self,
        collection_name: str,
        data: list[BaseData]
    ) -> list[int]:
        """批量写入，按入参顺序返回 id。"""
        ...

    def get_by_id[T: BaseData](
        self,
        collection_name: str,
        id: int,
        data_cls: type[T]
    ) -> T:
        """按 id 获取，并反序列化为 `data_cls`。"""
        return self.batch_get_by_id(collection_name, [id], data_cls)[0]

    def batch_get_by_id[T: BaseData](
        self,
        collection_name: str,
        ids: list[int],
        data_cls: type[T]
    ) -> list[T]:
        """按 id 批量获取。每个 id 一条记录，按入参顺序返回。"""
        ...

    def delete_by_id(self, collection_name: str, id: int) -> None:
        """按 id 删除一条记录。"""
        self.batch_delete_by_id(collection_name, [id])

    def batch_delete_by_id(self, collection_name: str, ids: list[int]) -> None:
        """按 id 批量删除。"""
        ...
