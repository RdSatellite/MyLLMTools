# base.py 定义嵌入的基础类

from abc import ABC, abstractmethod


# 嵌入向量
type Vector = list[float]


class BaseEmbeddingFunction(ABC):
    """BaseEmbeddingFunction 作为 Embedding Function 的基类"""

    def __init__(self, dim: int):
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self._dimension = dim

    @property
    def dim(self) -> int:
        """嵌入向量的维度"""
        return self._dimension

    # 考虑到远程调用需要在单条请求中批量处理以提高网络效率，此处要求子类直接提供批量处理版本
    # 对于单条 text 的处理，依然委托到此函数
    @abstractmethod
    def _embed_batch(self, texts: list[str]) -> list[Vector]:
        """
        批量将文本转换为向量

        Args:
            texts: list[str]，待编码的文本列表
        
        Returns:
            编码后的嵌入向量列表
        """
        ...

    def __call__(self, texts: str | list[str]) -> Vector | list[Vector]:
        """
        编码文本

        Args:
            texts: 单条文本，或文本列表
        
        Returns:
            传入为单条文本 (str) 时，返回为 Vector;
            文本列表 (list[str]) 时，则返回 list[Vector]
        """
        self._validate_input(texts)

        is_single = isinstance(texts, str)
        items = [texts] if is_single else texts

        vectors = self._embed_batch(items)
        self._validate_output(items, vectors)

        return vectors[0] if is_single else vectors

    @staticmethod
    def _validate_input(texts: str | list[str]) -> None:
        """对入参执行合法性校验"""

        if isinstance(texts, str):
            if not texts.strip():
                raise ValueError("text is empty")
            return

        if type(texts) is not list:
            raise TypeError(
                "texts must be str or list[str]"
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

    def _validate_output(self, texts: list[str], vectors: list[Vector]) -> None:
        """确保所得结果格式上的正确性"""

        if len(vectors) != len(texts):
            raise RuntimeError(
                "Embedding result count mismatch."
            )

        for i, vector in enumerate(vectors):
            if len(vector) != self.dim:
                raise RuntimeError(
                    f"Embedding dimension mismatch at index {i}: "
                    f"expected {self.dim}, got {len(vector)}"
                )


# 以 BaseEmbeddingFunction 作为 EmbeddingFunction 的基类实现，例如：
#
#   from [this_file] import BaseEmbeddingFunction
#
#   class YourFunc(BaseEmbeddingFunction):
#       ...
