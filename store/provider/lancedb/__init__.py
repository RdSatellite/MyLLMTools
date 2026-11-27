# lancedb/__init__.py (store.provider.lancedb) 基于 LanceDB 提供 Store 实现

from dataclasses import dataclass
from typing import Any, Self
from pathlib import Path
import json

import lancedb
from lancedb.rerankers import RRFReranker
import pyarrow as pa
import pyarrow.compute as pc

from ...base import BaseData
from ...hybrid import HybridData, HybridStore
from ...kv import KvStore
from ...text import HasText
from ...vector import Vector, VectorData, VectorStore, require_vector_data


@dataclass
class LanceDBSettings:
    """LanceDBSettings 是 LanceDB 存储后端的设置项，用于初始化"""

    path: Path

    @classmethod
    def default(cls, path: Path) -> Self:
        directory = path / ".lancedb"
        directory.mkdir(parents=True, exist_ok=True)

        return cls(
            path = directory
        )


def _has_text(
    collection_name: str,
    table,
    data: list[VectorData],
) -> bool:
    """检查数据类是否实现了 HasText。"""
    data_cls = type(data[0])

    if any(type(d) is not data_cls for d in data):
        classes = sorted({type(d).__name__ for d in data})
        raise TypeError(f"one batch stores one class, got {classes}")

    if not issubclass(data_cls, HasText):
        return False

    text_column = data_cls.text_column()
    if text_column not in table.schema.names:
        raise ValueError(
            f"collection {collection_name!r} has no {text_column!r} column: "
            f"create it before storing {data_cls.__name__}"
        )

    return True


class LanceDBStore(KvStore, VectorStore, HybridStore):
    """LanceDBStore 即 LanceDB 存储后端"""

    def __init__(self, settings: LanceDBSettings):
        self.path = settings.path
        self.client = lancedb.connect(str(self.path))

    def has_collection(self, name: str) -> bool:
        """检查 collection 是否存在。"""
        return name in self.client

    def _table(self, name: str):
        """获取已存在 table 的句柄。"""
        return self.client.open_table(name)


    # --- Impl KvStore --- #


    def _next_id(self, table) -> int:
        """表中最大 id 的下一个值。"""
        ids = table.to_arrow().column("id").to_pylist()

        return max(ids) + 1 if ids else 0


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
                    f"LanceDBStore stores VectorData rows, "
                    f"got {type(d).__name__}"
                )
            items.append(d)

        table = self._table(collection_name)
        start = self._next_id(table)
        ids = list(range(start, start + len(items)))

        data_cls = type(items[0])
        has_text = _has_text(collection_name, table, items)

        if has_text:
            text_column = data_cls.text_column()

        rows = []
        for id, d in zip(ids, items):
            payload = d.get_payload()
            row = {
                "id": id,
                "payload": json.dumps(
                    d.serialize_payload(payload), ensure_ascii=False
                ),
                "vector": data_cls.get_vector(payload),
            }

            if has_text:
                row[text_column] = data_cls.get_text(payload)
            rows.append(row)

        table.add(rows)

        return ids


    def batch_get_by_id[T: BaseData](
        self,
        collection_name: str,
        ids: list[int],
        data_cls: type[T]
    ) -> list[T]:
        """按 id 批量获取。每个 id 一条记录，按入参顺序返回。"""
        require_vector_data(data_cls)

        if not ids:
            return []

        rows = self._rows_by_id(self._table(collection_name), ids)

        missing = [i for i in ids if i not in rows]
        if missing:
            raise KeyError(
                f"ids not found in collection {collection_name!r}: {missing}"
            )

        return [
            data_cls(
                id = i,
                payload = data_cls.deserialize_payload(
                    json.loads(rows[i]["payload"])
                )
            )
            for i in ids
        ]

    def _rows_by_id(self, table, ids: list[int]) -> dict[int, dict[str, Any]]:
        """id 在 `ids` 中的行，按 id 索引。"""
        arrow = table.to_arrow()
        mask = pc.is_in(
            arrow["id"], value_set = pa.array(ids, type = pa.int64())
        )

        return {row["id"]: row for row in arrow.filter(mask).to_pylist()}


    def batch_delete_by_id(self, collection_name: str, ids: list[int]) -> None:
        """按 id 批量删除。"""
        if not ids:
            return

        predicate = ", ".join(str(i) for i in ids)
        self._table(collection_name).delete(f"id IN ({predicate})")


    # --- Impl VectorStore--- #


    def get_by_sim[T: VectorData](
        self,
        collection_name: str,
        query_vector: Vector,
        data_cls: type[T],
        limit: int = 10
    ) -> list[tuple[T, float]]:
        """获取与 `query_vector` 最相似的 `limit` 条记录。

        返回 `(数据, 余弦相似度)` 列表，分数越高越相似。
        """
        require_vector_data(data_cls)

        rows = (
            self._table(collection_name)
            .search(query_vector)
            .metric("cosine")
            .limit(limit)
            .to_list()
        )

        return [
            (
                data_cls(
                    id = row["id"],
                    payload = data_cls.deserialize_payload(
                        json.loads(row["payload"])
                    )
                ),
                1.0 - float(row["_distance"]),  # cosine 距离 -> 余弦相似度
            )
            for row in rows
        ]

    # --- Impl HybridStore --- #

    def get_by_hybrid[T: HybridData](
        self,
        collection_name: str,
        query_text: str,
        query_vector: Vector,
        data_cls: type[T],
        limit: int = 10,
        reranker = None,
    ) -> list[tuple[T, float]]:
        """获取同时按文本与向量召回、融合后最接近的 `limit` 行。"""
        if not issubclass(data_cls, HybridData):
            raise TypeError(
                f"data_cls must implement HybridData for a hybrid query, "
                f"got {data_cls.__name__}"
            )

        table = self._table(collection_name)
        text_column = data_cls.text_column()

        if text_column not in table.schema.names:
            raise ValueError(
                f"collection {collection_name!r} has no {text_column!r} column: "
                f"a hybrid search over it would be vector-only"
            )

        rows = (
            table.search(query_type="hybrid", fts_columns=text_column)
            .vector(query_vector)
            .text(query_text)
            .metric("cosine")
            .rerank(RRFReranker() if reranker is None else reranker)
            .limit(limit)
            .to_list()
        )

        return [
            (
                data_cls(
                    id = row["id"],
                    payload = data_cls.deserialize_payload(
                        json.loads(row["payload"])
                    )
                ),
                float(row["_relevance_score"]),
            )
            for row in rows
        ]

    @classmethod
    def default(cls, path: Path) -> Self:
        return cls(LanceDBSettings.default(path))
