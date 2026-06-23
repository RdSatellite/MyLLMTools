from .base import InvokeContext, InvokeOption
from .structured import StructuredOption


def with_structured(schema: type[T]) -> StructuredOption:
    return StructuredOption(schema)
