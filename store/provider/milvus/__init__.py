# milvus/__init__.py (store.provider.milvus) 基于 Milvus 提供 Store 实现

from dataclasses import dataclass
from typing import Self
from pathlib import Path

from pymilvus import MilvusClient

from ...base import BaseData
from ...kv import KvStore
from ...vector import Vector, VectorData, VectorStore, require_vector_data


@dataclass
class MilvusLiteSettings:
    """MilvusLiteSettings 是 MilvusLiteStore 的设置项，用于初始化"""

    path: Path

    @classmethod
    def default(cls, path: Path) -> Self:
        directory = path / ".milvus"
        directory.mkdir(parents=True, exist_ok=True)

        return cls(
            path = directory / "vdb.db"
        )


class MilvusLiteStore(KvStore, VectorStore):
    """MilvusLiteStore 即 MilvusLite 实现的存储后端"""

    def __init__(self, settings: MilvusLiteSettings):
        self.path = settings.path
        self.client = MilvusClient(str(self.path))

    def has_collection(self, name: str) -> bool:
        """检查 collection 是否存在。"""
        return self.client.has_collection(name)


    # --- Impl KvStore --- #


    def batch_set(
        self,
        collection_name: str,
        data: list[BaseData]
    ) -> list[int]:
        """批量写入，按入参顺序返回 id。"""
        if not data:
            return []

        items: list[VectorData] = []
        for d in data:
            if not isinstance(d, VectorData):
                raise TypeError(
                    f"MilvusLiteStore stores VectorData rows, "
                    f"got {type(d).__name__}"
                )
            items.append(d)

        rows = []
        for d in items:
            payload = d.get_payload()
            rows.append({
                "vector": d.get_vector(payload),
                "payload": d.serialize_payload(payload)
            })

        result = self.client.insert(
            collection_name = collection_name,
            data = rows
        )

        return [int(i) for i in result["ids"]]


    def batch_get_by_id[T: BaseData](
        self,
        collection_name: str,
        ids: list[int],
        data_cls: type[T]
    ) -> list[T]:
        """按 id 批量获取。每个 id 一条记录，按入参顺序返回。

        任一 id 缺失时抛 KeyError，保证返回列表始终与 `ids` 位置对齐。
        """
        require_vector_data(data_cls)

        if not ids:
            return []

        rows = self.client.get(
            collection_name = collection_name,
            ids = ids,
            output_fields = ["vector", "payload"]
        )

        by_id = {row["id"]: row for row in rows}

        missing = [i for i in ids if i not in by_id]
        if missing:
            raise KeyError(
                f"ids not found in collection {collection_name!r}: {missing}"
            )

        return [
            data_cls(
                id = i,
                payload = data_cls.deserialize_payload(by_id[i]["payload"])
            )
            for i in ids
        ]


    def batch_delete_by_id(self, collection_name: str, ids: list[int]) -> None:
        """按 id 批量删除。"""
        if not ids:
            return

        self.client.delete(
            collection_name = collection_name,
            ids = ids
        )


    # --- Impl VectorStore --- #


    def get_by_sim[T: VectorData](
        self,
        collection_name: str,
        query_vector: Vector,
        data_cls: type[T],
        limit: int = 10
    ) -> list[tuple[T, float]]:
        """
        获取与 `query_vector` 最相似的 `limit` 条记录。

        返回 `(数据, 余弦相似度)` 列表，分数越高越相似。
        要求 collection 的向量索引以 COSINE 度量建立。
        """
        require_vector_data(data_cls)

        results = self.client.search(
            collection_name=collection_name,
            data = [query_vector],
            limit = limit,
            output_fields = ["vector", "payload"],
            search_params = {"metric_type": "COSINE"}
        )

        return [
            (
                data_cls(
                    id = hit["id"],
                    payload = data_cls.deserialize_payload(hit["entity"]["payload"])
                ),
                hit["distance"],
            )
            for hit in results[0]
        ]

    @classmethod
    def default(cls, path: Path) -> Self:
        return cls(MilvusLiteSettings.default(path))
