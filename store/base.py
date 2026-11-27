# base.py (store.base) 定义数据库基础契约

from typing import Any, Protocol, runtime_checkable


# --- Data --- #


@runtime_checkable
class BaseData[T](Protocol):
    """BaseData 即所要存储的数据"""

    def get_payload(self) -> T:
        """获取 payload"""
        raise NotImplementedError

    def get_id(self) -> int | None:
        """获取 id。数据尚未存储前为 `None`。"""
        raise NotImplementedError

    @classmethod
    def serialize_payload(cls, payload: T) -> Any:
        """序列化 payload（用于存储）"""
        raise NotImplementedError

    @classmethod
    def deserialize_payload(cls, raw: Any) -> T:
        """反序列化 payload（用于存储）"""
        raise NotImplementedError


# --- Store --- #


@runtime_checkable
class BaseStore(Protocol):
    """BaseStore 即存储后端"""

    # 作为 vectordb 时，采用 collection ，若为 rdb 则等价为 table
    def has_collection(self, name: str) -> bool:
        """检查 collection 是否存在。"""
        ...
