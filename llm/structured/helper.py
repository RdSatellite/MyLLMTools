# helper.py 定义 package 暴露的 helper 类

from typing import TypeVar
from pydantic import BaseModel

from langchain_core.output_parsers import PydanticOutputParser

from .json import extract_json


class Helper[T]:
    """Helper[T] 确定需要处理的类型 T，并提供 JSON schema 的构造方法和解析方法入口"""

    @staticmethod
    def get_json_schema(schema: type[T]) -> str:
        """构造 JSON schema

        Args:
            schema (type[T]): 目标类型

        Returns:
            str: JSON Schema
        """
        parser = PydanticOutputParser(pydantic_object=schema)
        return parser.get_format_instructions()

    @staticmethod
    def parse(raw: str, schema: type[T]) -> T:
        # 1. 假设 raw 为纯 json，尝试解析
        try:
            return schema.model_validate_json(raw)
        except Exception:
            pass
        
        # 2. 尝试使用 RE 解析
        json_text = extract_json(raw)
        try:
            return schema.model_validate_json(json_text)
        except Exception:
            pass

        raise ValueError(
            f"Failed to parse structured output:\n{raw}"
        )
