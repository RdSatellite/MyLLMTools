from typing import TypeVar
from pydantic import BaseModel

from langchain_core.output_parsers import PydanticOutputParser

from .json import extract_json

T = TypeVar("T", bound=BaseModel)


class Helper:
    @staticmethod
    def get_json_schema(schema: type[T]) -> str:
        """Build JSON schema

        Args:
            schema (type[T]): Target type

        Returns:
            str: JSON Schema
        """
        parser = PydanticOutputParser(pydantic_object=schema)
        return parser.get_format_instructions()

    @staticmethod
    def parse(raw: str, schema: type[T]) -> T:
        # 1. Assume pure json
        try:
            return schema.model_validate_json(raw)
        except Exception:
            pass
        
        # 2. Fallback to RE
        json_text = extract_json(raw)
        try:
            return schema.model_validate_json(json_text)
        except Exception:
            pass

        raise ValueError(
            f"Failed to parse structured output:\n{raw}"
        )
