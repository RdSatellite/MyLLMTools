# base.py 定义嵌入的基础类

from abc import ABC, abstractmethod


# 来自 pymilvus.model.base.BaseEmbeddingFunction
# （pymilvus/model/base.py）

# BaseEmbeddingFunction 作为 Embedding Function 的基类
class BaseEmbeddingFunction(ABC):
    """Embedding Function 的基类"""

    @abstractmethod
    def __call__(self, texts: list[str]):
        """ """

    @abstractmethod
    def encode_queries(self, queries: list[str]):
        """ """


# 以 BaseEmbeddingFunction 作为 EmbeddingFunction 的基类实现，例如：
#
#   from [this_file] import BaseEmbeddingFunction
#
#   class YourFunc(BaseEmbeddingFunction):
#       ...
