# text.py 定义文本相关特性。

from typing import Any, Protocol, runtime_checkable


# --- Data --- #


@runtime_checkable
class HasText(Protocol):
    """HasText 表示数据能够被纯文本表示"""

    @classmethod
    def get_text(cls, payload: Any) -> str:
        """该数据的文本，即词法侧看到的内容。"""
        raise NotImplementedError

    @classmethod
    def text_column(cls) -> str:
        """文本数据存储的列名。"""
        raise NotImplementedError
