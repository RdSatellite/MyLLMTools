from typing import Any
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class InvokeContext:
    prompt: str             # Prompt原文
    raw_response: str = ""  # 响应原文
    response: Any = None    # 处理完成后的响应

class InvokeOption(ABC):

    @abstractmethod
    def pre_handle(
        self,
        ctx: InvokeContext,
    ):
        ...
    
    @abstractmethod
    def post_handle(
        self,
        ctx: InvokeContext
    ):
        ...
