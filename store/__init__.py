# package store 定义存储契约

from .base import BaseData, BaseStore
from .vector import Vector, HasVector, VectorData, VectorStore, require_vector_data
from .text import HasText
from .hybrid import HybridData, HybridStore
from .kv import KvStore

__all__ = [
    "BaseData",
    "BaseStore",
    "Vector",
    "HasVector",
    "VectorData",
    "VectorStore",
    "require_vector_data",
    "HasText",
    "HybridData",
    "HybridStore",
    "KvStore",
]


# 使用 provider 实现时，显式引用子包：
#
#   from store.provider.lancedb import LanceDBSettings, LanceDBStore
#   from store.provider.milvus import MilvusLiteSettings, MilvusLiteStore

